import tensorflow as tf
import numpy as np
from config import Parameters


def build_point_pillar_graph(params: Parameters):

    # extract required parameters
    max_pillars = int(params.max_pillars)
    max_points  = int(params.max_points_per_pillar)
    nb_features = int(params.nb_features)
    nb_channels = int(params.nb_channels)
    batch_size  = int(params.batch_size)
    image_size  = tuple([params.Xn, params.Yn])
    nb_classes  = int(params.nb_classes)
    nb_anchors  = len(params.anchor_dims)

    if tf.keras.backend.image_data_format() == "channels_first":
        raise NotImplementedError
    else:
        input_shape = (max_pillars, max_points, nb_features)

    input_pillars = tf.keras.layers.Input(input_shape, batch_size=batch_size, name="pillars/input")
    input_indices = tf.keras.layers.Input((max_pillars, 3), batch_size=batch_size, name="pillars/indices",
                                          dtype=tf.int32)

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

    # 2d cnn backbone

    # Block1(S, 4, C)
    x = pillars
    for n in range(4):
        S = (2, 2) if n == 0 else (1, 1)
        x = tf.keras.layers.Conv2D(nb_channels, (3, 3), strides=S, padding="same", activation="relu",
                                   name="cnn/block1/conv2d%i" % n)(x)
        x = tf.keras.layers.BatchNormalization(name="cnn/block1/bn%i" % n, fused=True)(x)
    x1 = x

    # Block2(2S, 6, 2C)
    for n in range(6):
        S = (2, 2) if n == 0 else (1, 1)
        x = tf.keras.layers.Conv2D(2 * nb_channels, (3, 3), strides=S, padding="same", activation="relu",
                                   name="cnn/block2/conv2d%i" % n)(x)
        x = tf.keras.layers.BatchNormalization(name="cnn/block2/bn%i" % n, fused=True)(x)
    x2 = x

    # Block3(4S, 6, 4C)
    for n in range(6):
        S = (2, 2) if n == 0 else (1, 1)
        x = tf.keras.layers.Conv2D(2 * nb_channels, (3, 3), strides=S, padding="same", activation="relu",
                                   name="cnn/block3/conv2d%i" % n)(x)
        x = tf.keras.layers.BatchNormalization(name="cnn/block3/bn%i" % n, fused=True)(x)
    x3 = x

    # Up1 (S, S, 2C)
    up1 = tf.keras.layers.Conv2DTranspose(2 * nb_channels, (3, 3), strides=(1, 1), padding="same", activation="relu",
                                          name="cnn/up1/conv2dt")(x1)
    up1 = tf.keras.layers.BatchNormalization(name="cnn/up1/bn", fused=True)(up1)

    # Up2 (2S, S, 2C)
    up2 = tf.keras.layers.Conv2DTranspose(2 * nb_channels, (3, 3), strides=(2, 2), padding="same", activation="relu",
                                          name="cnn/up2/conv2dt")(x2)
    up2 = tf.keras.layers.BatchNormalization(name="cnn/up2/bn", fused=True)(up2)

    # Up3 (4S, S, 2C)
    up3 = tf.keras.layers.Conv2DTranspose(2 * nb_channels, (3, 3), strides=(4, 4), padding="same", activation="relu",
                                          name="cnn/up3/conv2dt")(x3)
    up3 = tf.keras.layers.BatchNormalization(name="cnn/up3/bn", fused=True)(up3)

    # Concat
    concat = tf.keras.layers.Concatenate(name="cnn/concatenate")([up1, up2, up3])

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
#     print(pillar_net.summary())

    return pillar_net
