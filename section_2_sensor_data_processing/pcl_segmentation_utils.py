import tensorflow as tf

class CAM(tf.keras.layers.Layer):
    """Context Aggregation Module"""

    def __init__(self, in_channels, reduction_factor=16, bn_momentum=0.999, l2=0.0001):
        super(CAM, self).__init__()
        self.in_channels = in_channels
        self.reduction_factor = reduction_factor
        self.bn_momentum = bn_momentum
        self.l2 = l2

        self.pool = tf.keras.layers.MaxPool2D(
            pool_size=7,
            strides=1,
            padding='SAME'
        )

        self.squeeze = tf.keras.layers.Conv2D(
            filters=(self.in_channels // self.reduction_factor),
            kernel_size=1,
            strides=1,
            padding='SAME',
            kernel_initializer='glorot_uniform',
            kernel_regularizer=tf.keras.regularizers.L2(l2=self.l2)
        )
        self.squeeze_bn = tf.keras.layers.BatchNormalization(momentum=self.bn_momentum)

        self.excitation = tf.keras.layers.Conv2D(
            filters=self.in_channels,
            kernel_size=1,
            strides=1,
            padding='SAME',
            kernel_initializer='glorot_uniform',
            kernel_regularizer=tf.keras.regularizers.L2(l2=self.l2)
        )
        self.excitation_bn = tf.keras.layers.BatchNormalization(momentum=self.bn_momentum)

    def call(self, inputs, training=False):
        pool = self.pool(inputs)
        squeeze = tf.nn.relu(self.squeeze_bn(self.squeeze(pool)))
        excitation = tf.nn.sigmoid(self.excitation_bn(self.excitation(squeeze)))
        return inputs * excitation

    
class FIRE(tf.keras.layers.Layer):
    """FIRE MODULE"""

    def __init__(self, sq1x1_planes, ex1x1_planes, ex3x3_planes, bn_momentum=0.999, l2=0.0001):
        super(FIRE, self).__init__()
        self.sq1x1_planes = sq1x1_planes
        self.ex1x1_planes = ex1x1_planes
        self.ex3x3_planes = ex3x3_planes
        self.bn_momentum = bn_momentum
        self.l2 = l2

        self.squeeze = tf.keras.layers.Conv2D(
            filters=self.sq1x1_planes,
            kernel_size=1,
            strides=1,
            padding='SAME',
            kernel_regularizer=tf.keras.regularizers.L2(l2=self.l2)
        )
        self.squeeze_bn = tf.keras.layers.BatchNormalization(momentum=self.bn_momentum)

        self.expand1x1 = tf.keras.layers.Conv2D(
            filters=self.ex1x1_planes,
            kernel_size=1,
            strides=1,
            padding='SAME',
            kernel_regularizer=tf.keras.regularizers.L2(l2=self.l2)
        )
        self.expand1x1_bn = tf.keras.layers.BatchNormalization(momentum=self.bn_momentum)

        self.expand3x3 = tf.keras.layers.Conv2D(
            filters=self.ex3x3_planes,
            kernel_size=3,
            strides=1,
            padding='SAME',
            kernel_regularizer=tf.keras.regularizers.L2(l2=self.l2)
        )
        self.expand3x3_bn = tf.keras.layers.BatchNormalization(momentum=self.bn_momentum)

    def call(self, inputs, training=False):
        squeeze = tf.nn.relu(self.squeeze_bn(self.squeeze(inputs), training))
        expand1x1 = tf.nn.relu(self.expand1x1_bn(self.expand1x1(squeeze), training))
        expand3x3 = tf.nn.relu(self.expand3x3_bn(self.expand3x3(squeeze), training))
        return tf.concat([expand1x1, expand3x3], axis=3)


class FIREUP(tf.keras.layers.Layer):
    """FIRE MODULE WITH TRANSPOSE CONVOLUTION"""

    def __init__(self, sq1x1_planes, ex1x1_planes, ex3x3_planes, stride, bn_momentum=0.99, l2=0.0001):
        super(FIREUP, self).__init__()
        self.sq1x1_planes = sq1x1_planes
        self.ex1x1_planes = ex1x1_planes
        self.ex3x3_planes = ex3x3_planes
        self.stride = stride
        self.bn_momentum = bn_momentum
        self.l2 = l2

        self.squeeze = tf.keras.layers.Conv2D(
            filters=self.sq1x1_planes,
            kernel_size=1,
            strides=1,
            padding='SAME',
            kernel_regularizer=tf.keras.regularizers.L2(l2=self.l2)
        )
        self.squeeze_bn = tf.keras.layers.BatchNormalization(momentum=self.bn_momentum)

        if self.stride == 2:
            self.upconv = tf.keras.layers.Conv2DTranspose(
                filters=self.sq1x1_planes,
                kernel_size=[1, 4],
                strides=[1, 2],
                padding='SAME',
                kernel_regularizer=tf.keras.regularizers.L2(l2=self.l2)
            )

        self.expand1x1 = tf.keras.layers.Conv2D(
            filters=self.ex1x1_planes,
            kernel_size=1,
            strides=1,
            padding='SAME',
            kernel_regularizer=tf.keras.regularizers.L2(l2=self.l2)
        )
        self.expand1x1_bn = tf.keras.layers.BatchNormalization(momentum=self.bn_momentum)

        self.expand3x3 = tf.keras.layers.Conv2D(
            filters=self.ex3x3_planes,
            kernel_size=3,
            strides=1,
            padding='SAME',
            kernel_regularizer=tf.keras.regularizers.L2(l2=self.l2)
        )
        self.expand3x3_bn = tf.keras.layers.BatchNormalization(momentum=self.bn_momentum)

    def call(self, inputs, training=False):
        squeeze = tf.nn.relu(self.squeeze_bn(self.squeeze(inputs), training))
        if self.stride == 2:
            upconv = tf.nn.relu(self.upconv(squeeze))
        else:
            upconv = squeeze
        expand1x1 = tf.nn.relu(self.expand1x1_bn(self.expand1x1(upconv), training))
        expand3x3 = tf.nn.relu(self.expand3x3_bn(self.expand3x3(upconv), training))
        return tf.concat([expand1x1, expand3x3], axis=3)