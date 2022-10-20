# ==============================================================================
# Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
#
# This work is licensed under the terms of the GPL v3.0 license.
# For a copy, see https://github.com/fferroni/PointPillars/blob/master/LICENSE.
# ==============================================================================

# ==============================================================================
# https://github.com/fferroni/PointPillars
# Commit: ab4d1e0eb47b073ea9d7a4451418c1f2347e80d2
# Modifications:
#   - removed function "correct_batch_indices"
#   - do not use lambda function for tf.scatter_nd
#   - removed detection head
#   - reverse dimensions to match grid map coordinates
#   - added dropout of 0.1 in loops of Block1, Block2 and Block3
# ==============================================================================

import tensorflow as tf


def getPointPillarsModel(image_size, max_pillars, max_points, nb_features, nb_channels):

    if tf.keras.backend.image_data_format() == "channels_first":
        raise NotImplementedError
    else:
        input_shape = (max_pillars, max_points, nb_features)

    input_pillars = tf.keras.layers.Input(input_shape,
                                          name="pillars/input")
    input_indices = tf.keras.layers.Input((max_pillars, 3),
                                          name="pillars/indices",
                                          dtype=tf.int32)

    # Pillar Feature Net
    x = tf.keras.layers.Conv2D(nb_channels, (1, 1),
                               activation='linear',
                               use_bias=False,
                               name="pillars/conv2d")(input_pillars)
    x = tf.keras.layers.BatchNormalization(name="pillars/batchnorm",
                                           fused=True,
                                           epsilon=1e-3,
                                           momentum=0.99)(x)
    x = tf.keras.layers.Activation("relu", name="pillars/relu")(x)
    x = tf.keras.layers.MaxPool2D((1, max_points),
                                  name="pillars/maxpooling2d")(x)

    if tf.keras.backend.image_data_format() == "channels_first":
        reshape_shape = (nb_channels, max_pillars)
    else:
        reshape_shape = (max_pillars, nb_channels)

    x = tf.keras.layers.Reshape(reshape_shape, name="pillars/reshape")(x)

    pillars = tf.scatter_nd(input_indices, x,
                            (tf.shape(input_indices)[0], ) + image_size + (nb_channels, ))

    # reverse dimensions (x,y) to match OGM coordinates (size_x-x, size_y-y)
    pillars = tf.reverse(pillars, [1, 2])

    # 2D CNN backbone

    # Block1(S, 4, C)
    x = pillars
    for n in range(4):
        S = (2, 2) if n == 0 else (1, 1)
        x = tf.keras.layers.Conv2D(nb_channels, (3, 3),
                                   strides=S,
                                   padding="same",
                                   activation="relu",
                                   name="cnn/block1/conv2d%i" % n)(x)
        x = tf.keras.layers.BatchNormalization(name="cnn/block1/bn%i" % n,
                                               fused=True)(x)
        x = tf.keras.layers.Dropout(0.1, name="cnn/block1/dropout%i" % n)(x)
    x1 = x

    # Block2(2S, 6, 2C)
    for n in range(6):
        S = (2, 2) if n == 0 else (1, 1)
        x = tf.keras.layers.Conv2D(2 * nb_channels, (3, 3),
                                   strides=S,
                                   padding="same",
                                   activation="relu",
                                   name="cnn/block2/conv2d%i" % n)(x)
        x = tf.keras.layers.BatchNormalization(name="cnn/block2/bn%i" % n,
                                               fused=True)(x)
        x = tf.keras.layers.Dropout(0.1, name="cnn/block2/dropout%i" % n)(x)
    x2 = x

    # Block3(4S, 6, 4C)
    for n in range(6):
        S = (2, 2) if n == 0 else (1, 1)
        x = tf.keras.layers.Conv2D(4 * nb_channels, (3, 3),
                                   strides=S,
                                   padding="same",
                                   activation="relu",
                                   name="cnn/block3/conv2d%i" % n)(x)
        x = tf.keras.layers.BatchNormalization(name="cnn/block3/bn%i" % n,
                                               fused=True)(x)
        x = tf.keras.layers.Dropout(0.1, name="cnn/block3/dropout%i" % n)(x)
    x3 = x

    # Up1 (S, S, 2C)
    up1 = tf.keras.layers.Conv2DTranspose(2 * nb_channels, (3, 3),
                                          strides=(1, 1),
                                          padding="same",
                                          activation="relu",
                                          name="cnn/up1/conv2dt")(x1)
    up1 = tf.keras.layers.BatchNormalization(name="cnn/up1/bn",
                                             fused=True)(up1)

    # Up2 (2S, S, 2C)
    up2 = tf.keras.layers.Conv2DTranspose(2 * nb_channels, (3, 3),
                                          strides=(2, 2),
                                          padding="same",
                                          activation="relu",
                                          name="cnn/up2/conv2dt")(x2)
    up2 = tf.keras.layers.BatchNormalization(name="cnn/up2/bn",
                                             fused=True)(up2)

    # Up3 (4S, S, 2C)
    up3 = tf.keras.layers.Conv2DTranspose(2 * nb_channels, (3, 3),
                                          strides=(4, 4),
                                          padding="same",
                                          activation="relu",
                                          name="cnn/up3/conv2dt")(x3)
    up3 = tf.keras.layers.BatchNormalization(name="cnn/up3/bn",
                                             fused=True)(up3)

    # Concat
    concat = tf.keras.layers.Concatenate(name="cnn/concatenate")(
        [up1, up2, up3])

    return input_pillars, input_indices, concat
