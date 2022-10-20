import numpy as np
import cv2
import tensorflow as tf


def grid_map_to_img(grid_map):
    dims = tf.shape(grid_map).numpy()
    dims[-1] = 1
    image = tf.concat([tf.reverse(grid_map, axis=[-1]), tf.zeros(dims)], axis=-1)
    grid_map = tf.cast(255*image, dtype=tf.uint8)
    
    return grid_map


def lidar_to_bird_view_img(pointcloud: np.ndarray,
                           x_min,
                           x_max,
                           y_min,
                           y_max,
                           step_x_size,
                           step_y_size,
                           intensity_threshold=None,
                           factor=1):
    # Input:
    #   pointcloud: (N, 4) with N points [x, y, z, intensity], intensity in [0,1]
    # Output:
    #   birdview: ((x_max-x_min)/step_x_size)*factor, ((y_max-y_min)/step_y_size)*factor, 3)

    if intensity_threshold is not None:
        pointcloud[:, 3] = np.clip(pointcloud[:, 3] / intensity_threshold,
                                   0.0,
                                   1.0,
                                   dtype=np.float32)

    size_x = int((x_max - x_min) / step_x_size)
    size_y = int((y_max - y_min) / step_y_size)
    birdview = np.zeros((size_x * factor, size_y * factor, 1), dtype=np.uint8)

    for point in pointcloud:
        x, y = point[0:2]
        # scale with minimum intensity for visibility in image
        i = 55 + point[3] * 200
        if not 0 <= i <= 255:
            raise ValueError("Intensity out of range [0,1].")
        if x_min < x < x_max and y_min < y < y_max:
            x = int((x - x_min) / step_x_size * factor)
            y = int((y - y_min) / step_y_size * factor)
            cv2.circle(birdview, ((size_y * factor - y, size_x * factor - x)),
                       radius=0,
                       color=(i))
    birdview = cv2.applyColorMap(birdview, cv2.COLORMAP_HOT)
    birdview = cv2.cvtColor(birdview, cv2.COLOR_BGR2RGB)

    return birdview

