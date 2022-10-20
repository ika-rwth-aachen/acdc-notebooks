import tensorflow as tf
import numpy as np


def segmentation_map_to_rgb_encoding(segmentation_map, class_color_map):
    """
    Converts a segmentation class map into a RGB encoding
    
    Arguments:
    segmentation_map -- Numpy ndarray of shape [height, width] or [height, width, 1] containing class IDs of the defined classes
    class_color_map -- Numpy ndarray of shape [NUM_CLASSES, 3] which contains the RGB values for each class.
    
    Returns:
    rgb_encoding -- Numpy ndarray of shape [height, width, 3] which contains the RGB encoding of the segmentation map
    """
    
    # Store the original shape of the segmentation map
    shape = segmentation_map.shape
    
    # Flatten segmentation map 
    segmentation_map = segmentation_map.flatten() 
    
    # Convert the segmentation map into np.int32
    segmentation_map = segmentation_map.astype(np.int32)

    # Extract RGB values
    rgb_encoding = class_color_map[segmentation_map]
    
    # Reshape to original width and height
    rgb_encoding = rgb_encoding.reshape([shape[0], shape[1], 3])
    
    return rgb_encoding
