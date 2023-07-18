import tensorflow as tf

from keras_core.models import *
from keras_core.layers import *


class DeepONet(Model):
    """
    The Deep Operator Networks (DeepONet) proposed by Lu et al. 2020.

    :param trunk_sizes: The sizes of the layers in the trunk
    :param branch_sizes: The sizes of the layers in the branch
    :param activation: The activation function
    """
    def __init__(self, trunk_sizes, branch_sizes, activation="swish", *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert trunk_sizes[-1] == branch_sizes[-1], "Last layer of trunk and branch must have same size"

        self.trunk_sizes = trunk_sizes
        self.branch_sizes = branch_sizes

        self.activation = activation

        # the trunk network
        self.trunk = Sequential(
            [
                Dense(x, activation=activation) for x in self.trunk_sizes
            ]
        )

        # the branch network
        self.branch = Sequential(
            [
                Dense(x, activation=activation) for x in self.branch_sizes
            ]
        )

        # the bias term in the output
        self.bias = tf.Variable(tf.zeros(1, dtype=tf.float32))

    def call(self, inputs, training=None, mask=None):
        sensors, y = inputs

        a = self.branch(sensors)
        b = self.trunk(y)

        return tf.tensordot(a, b) + self.bias
