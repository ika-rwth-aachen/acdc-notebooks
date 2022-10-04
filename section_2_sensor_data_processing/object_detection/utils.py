import numpy as np
import math
import tensorflow as tf
import cv2 as cv
import matplotlib.pyplot as plt

from config import Parameters

def transform_labels_into_lidar_coordinates(labels, R, t):
    """ transform_labels_into_lidar_coordinates
    from PointPillars repository
    """
    transformed = []
    for label in labels:
        label.centroid = label.centroid @ np.linalg.inv(R).T - t
        label.dimension = label.dimension[[2, 1, 0]]
        label.yaw -= np.pi / 2
        while label.yaw < -np.pi:
            label.yaw += (np.pi * 2)
        while label.yaw > np.pi:
            label.yaw -= (np.pi * 2)
        transformed.append(label)
        
    return labels

def plot2DBox(ax, box):

    dx_l = np.cos(box.yaw) * box.length / 2
    dy_l = np.sin(box.yaw) * box.length / 2
    
    dx_w = np.sin(box.yaw) * box.width / 2
    dy_w = np.cos(box.yaw) * box.width / 2
    
    FL = [box.x + dx_l + dx_w, box.y - dy_l + dy_w, box.z]
    FR = [box.x + dx_l - dx_w, box.y - dy_l - dy_w, box.z]
    RL = [box.x - dx_l + dx_w, box.y + dy_l + dy_w, box.z]
    RR = [box.x - dx_l - dx_w, box.y + dy_l - dy_w, box.z]
    
    x = [FL[0], FR[0], RR[0], RL[0], FL[0]]
    y = [FL[1], FR[1], RR[1], RL[1], FL[1]]
    z = [FL[2], FR[2], RR[2], RL[2], FL[2]]
    ax.plot(x, y, z, 'b-')

def plot2DLabel(ax, label):

    x = label.centroid[0] 
    y = label.centroid[1]
    z = label.centroid[2]
    length = label.dimension[0]
    width = label.dimension[1]
    height = label.dimension[2]
    yaw = label.yaw

    dx_l = np.cos(yaw) * length / 2
    dy_l = np.sin(yaw) * length / 2
    
    dx_w = np.sin(yaw) * width / 2
    dy_w = np.cos(yaw) * width / 2
    
    FL = [x + dx_l + dx_w, y - dy_l + dy_w, z]
    FR = [x + dx_l - dx_w, y - dy_l - dy_w, z]
    RL = [x - dx_l + dx_w, y + dy_l + dy_w, z]
    RR = [x - dx_l - dx_w, y + dy_l - dy_w, z]
    
    x = [FL[0], FR[0], RR[0], RL[0], FL[0]]
    y = [FL[1], FR[1], RR[1], RL[1], FL[1]]
    z = [FL[2], FR[2], RR[2], RL[2], FL[2]]
    ax.plot(x, y, z, 'g-')

def plot2DVertices(ax, FL, FR, RR, RL):
    x = [FL[0], FR[0], RR[0], RL[0], FL[0]]
    y = [FL[1], FR[1], RR[1], RL[1], FL[1]]
    z = [FL[2], FR[2], RR[2], RL[2], FL[2]]
    
    ax.plot(x, y, z)
    
    return ax

def setUpPlot():
    fig = plt.figure("Sample Point Cloud and Labels")
    fig.set_size_inches(10,5)
    fig.set_facecolor('black')

    ax = plt.axes(projection='3d')
    ax.set_facecolor('black') 
    ax.grid(False) 
    ax.w_xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.w_yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.w_zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.set_xlim([5,25])
    ax.set_ylim([-5,5])
    ax.view_init(elev=90, azim=-90)

    return ax;

def build_pillar_net(input_pillars, input_indices, params):

    # extract required parameters
    max_pillars = int(params.max_pillars)
    max_points  = int(params.max_points_per_pillar)
    nb_features = int(params.nb_features)
    nb_channels = int(params.nb_channels)
    batch_size  = int(params.batch_size)
    image_size  = tuple([params.Xn, params.Yn])
    nb_classes  = int(params.nb_classes)
    nb_anchors  = len(params.anchor_dims)

    def correct_batch_indices(tensor, batch_size):
        array = np.zeros((batch_size, max_pillars, 3), dtype=np.float32)
        for i in range(batch_size):
            array[i, :, 0] = i
        return tensor + tf.constant(array, dtype=tf.int32)

    if batch_size > 1:
            corrected_indices = tf.keras.layers.Lambda(lambda t: correct_batch_indices(t, batch_size))(input_indices)
    else:
        corrected_indices = input_indices

    # pillars
    x = tf.keras.layers.Conv2D(nb_channels, (1, 1), activation='linear', use_bias=False, name="pillars/conv2d")(input_pillars)
    x = tf.keras.layers.BatchNormalization(name="pillars/batchnorm", fused=True, epsilon=1e-3, momentum=0.99)(x)
    x = tf.keras.layers.Activation("relu", name="pillars/relu")(x)
    x = tf.keras.layers.MaxPool2D((1, max_points), name="pillars/maxpooling2d")(x)

    if tf.keras.backend.image_data_format() == "channels_first":
        reshape_shape = (nb_channels, max_pillars)
    else:
        reshape_shape = (max_pillars, nb_channels)

    x = tf.keras.layers.Reshape(reshape_shape, name="pillars/reshape")(x)
    pillars = tf.keras.layers.Lambda(lambda inp: tf.scatter_nd(inp[0], inp[1],
                                                               (batch_size,) + image_size + (nb_channels,)),
                                     name="pillars/scatter_nd")([corrected_indices, x])

    return pillars

def build_detection_head(concat, input_pillars, input_indices, params):

    # extract required parameters
    max_pillars = int(params.max_pillars)
    max_points  = int(params.max_points_per_pillar)
    nb_features = int(params.nb_features)
    nb_channels = int(params.nb_channels)
    batch_size  = int(params.batch_size)
    image_size  = tuple([params.Xn, params.Yn])
    nb_classes  = int(params.nb_classes)
    nb_anchors  = len(params.anchor_dims)

    # Detection head
    occ = tf.keras.layers.Conv2D(nb_anchors, (1, 1), name="occupancy/conv2d", activation="sigmoid")(concat)

    loc = tf.keras.layers.Conv2D(nb_anchors * 3, (1, 1), name="loc/conv2d", kernel_initializer=tf.keras.initializers.TruncatedNormal(0, 0.001))(concat)
    loc = tf.keras.layers.Reshape(tuple(i//2 for i in image_size) + (nb_anchors, 3), name="loc/reshape")(loc)

    size = tf.keras.layers.Conv2D(nb_anchors * 3, (1, 1), name="size/conv2d", kernel_initializer=tf.keras.initializers.TruncatedNormal(0, 0.001))(concat)
    size = tf.keras.layers.Reshape(tuple(i//2 for i in image_size) + (nb_anchors, 3), name="size/reshape")(size)

    angle = tf.keras.layers.Conv2D(nb_anchors, (1, 1), name="angle/conv2d")(concat)

    heading = tf.keras.layers.Conv2D(nb_anchors, (1, 1), name="heading/conv2d", activation="sigmoid")(concat)

    clf = tf.keras.layers.Conv2D(nb_anchors * nb_classes, (1, 1), name="clf/conv2d")(concat)
    clf = tf.keras.layers.Reshape(tuple(i // 2 for i in image_size) + (nb_anchors, nb_classes), name="clf/reshape")(clf)

    pillar_net = tf.keras.models.Model([input_pillars, input_indices], [occ, loc, size, angle, heading, clf])

    return pillar_net
