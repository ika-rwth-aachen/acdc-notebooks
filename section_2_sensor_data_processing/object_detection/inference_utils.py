### NOTE ###
# This function is a copy from the open PointPillars repository https://github.com/ika-rwth-aachen/PointPillars (forked from https://github.com/tyagi-iiitv/PointPillars)
### NOTE ###
 
import numpy as np
import cv2 as cv
from typing import List
from config import Parameters
from readers import DataReader
from processors import DataProcessor


class BBox(tuple):
    """ bounding box tuple that can easily be accessed while being compatible to cv2 rotational rects """

    def __new__(cls, bb_x, bb_y, bb_z, bb_length, bb_width, bb_height, bb_yaw, bb_heading, bb_cls, bb_conf):
        bbx_tuple = ((float(bb_x), float(bb_y)), (float(bb_length), float(bb_width)), float(np.rad2deg(bb_yaw)))
        return super(BBox, cls).__new__(cls, tuple(bbx_tuple))

    def __init__(self, bb_x, bb_y, bb_z, bb_length, bb_width, bb_height, bb_yaw, bb_heading, bb_cls, bb_conf):
        self.x = bb_x
        self.y = bb_y
        self.z = bb_z
        self.length = bb_length
        self.width = bb_width
        self.height = bb_height
        self.yaw = bb_yaw
        self.heading = bb_heading
        self.cls = bb_cls
        self.conf = bb_conf

    def __str__(self):
        return "BB | Cls: %s, x: %f, y: %f, l: %f, w: %f, yaw: %f" % (
            self.cls, self.x, self.y, self.length, self.width, self.yaw)


def rotational_nms(set_boxes, confidences, score_threshold, iou_threshold):
    """ rotational NMS
    set_boxes = size NSeqs list of size NDet lists of tuples. each tuple has the form ((pos, pos), (size, size), angle)
    confidences = size NSeqs list of lists containing NDet floats, i.e. one per detection
    """
    assert len(set_boxes) == len(confidences) and 0 < score_threshold < 1 and 0 < iou_threshold < 1
    if not len(set_boxes):
        return []
    nms_boxes = []
    for boxes, confs in zip(set_boxes, confidences):
        assert len(boxes) == len(confs)
        indices = cv.dnn.NMSBoxesRotated(boxes, confs, score_threshold, iou_threshold)

        if len(indices) > 0:
            indices = indices.reshape(len(indices)).tolist()
            nms_boxes.append([boxes[i] for i in indices])
        else:
            nms_boxes.append([])
    return nms_boxes


def generate_bboxes_from_pred(occ, pos, siz, ang, hdg, clf, anchor_dims, occ_threshold=0.5):
    """ Generating the bounding boxes based on the regression targets """

    # Get only the boxes where occupancy is greater or equal threshold.
    real_boxes = np.where(occ >= occ_threshold)
    # Get the indices of the occupancy array
    coordinates = list(zip(real_boxes[0], real_boxes[1], real_boxes[2]))
    # Assign anchor dimensions as original bounding box coordinates which will eventually be changed
    # according to the predicted regression targets
    anchor_dims = anchor_dims
    real_anchors = np.random.rand(len(coordinates), len(anchor_dims[0]))

    for i, value in enumerate(real_boxes[2]):
        real_anchors[i, ...] = anchor_dims[value]

    # Change the anchor boxes based on regression targets, this is the inverse of the operations given in
    # createPillarTargets function (src/PointPillars.cpp)
    predicted_boxes = []
    for i, value in enumerate(coordinates):
        real_diag = np.sqrt(np.square(real_anchors[i][0]) + np.square(real_anchors[i][1]))
        real_x = value[0] * Parameters.x_step * Parameters.downscaling_factor + Parameters.x_min
        real_y = value[1] * Parameters.y_step * Parameters.downscaling_factor + Parameters.y_min
        bb_x = pos[value][0] * real_diag + real_x
        bb_y = pos[value][1] * real_diag + real_y
        bb_z = pos[value][2] * real_anchors[i][2] + real_anchors[i][3]
        # print(position[value], real_x, real_y, real_diag)
        bb_length = np.exp(siz[value][0]) * real_anchors[i][0]
        bb_width = np.exp(siz[value][1]) * real_anchors[i][1]
        bb_height = np.exp(siz[value][2]) * real_anchors[i][2]
        bb_yaw = np.arcsin(np.clip(ang[value], -1, 1)) + real_anchors[i][4]
        bb_heading = np.round(hdg[value])
        bb_cls = np.argmax(clf[value])
        bb_conf = occ[value]
        predicted_boxes.append(BBox(bb_x, bb_y, bb_z, bb_length, bb_width, bb_height,
                                    bb_yaw, bb_heading, bb_cls, bb_conf))

    return predicted_boxes


class GroundTruthGenerator(DataProcessor):
    """ Multiprocessing-safe data generator for training, validation or testing, without fancy augmentation """

    def __init__(self, data_reader: DataReader, label_files: List[str], calibration_files: List[str] = None,
                 network_format: bool = False):
        super(GroundTruthGenerator, self).__init__()
        self.data_reader = data_reader
        self.label_files = label_files
        self.calibration_files = calibration_files
        self.network_format = network_format

    def __len__(self):
        return len(self.label_files)

    def __getitem__(self, file_id: int):
        label = self.data_reader.read_label(self.label_files[file_id])
        R, t = self.data_reader.read_calibration(self.calibration_files[file_id])
        label_transformed = self.transform_labels_into_lidar_coordinates(label, R, t)
        if self.network_format:
            occupancy, position, size, angle, heading, classification = self.make_ground_truth(label_transformed)
            occupancy = np.array(occupancy)
            position = np.array(position)
            size = np.array(size)
            angle = np.array(angle)
            heading = np.array(heading)
            classification = np.array(classification)
            return [occupancy, position, size, angle, heading, classification]
        return label_transformed


def focal_loss_checker(y_true, y_pred, n_occs=-1):
    y_true = np.stack(np.where(y_true == 1))
    if n_occs == -1:
        n_occs = y_true.shape[1]
    occ_thr = np.sort(y_pred.flatten())[-n_occs]
    y_pred = np.stack(np.where(y_pred >= occ_thr))
    p = 0
    for gt in range(y_true.shape[1]):
        for pr in range(y_pred.shape[1]):
            if np.all(y_true[:, gt] == y_pred[:, pr]):
                p += 1
                break
    print("#matched gt: ", p, " #unmatched gt: ", y_true.shape[1] - p, " #unmatched pred: ", y_pred.shape[1] - p,
          " occupancy threshold: ", occ_thr)
