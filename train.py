import numpy as np
import tensorflow as tf

from scipy.io import loadmat
from layers import FourierLayer
from losses import physics_loss
from models import FourierNeuralOperator
from utils import BoundaryCondition

r"""
# loading the data
data = loadmat(r"C:\Users\jedli\Downloads\burgers_data_R10.mat")

x = data['a']  # input
y = data['u']  # target

x = np.expand_dims(x, axis=-1)
y = np.expand_dims(y, axis=-1)

x_train, y_train = x[:-len(x) // 10], y[:-len(y) // 10]
x_test, y_test = x[-len(x) // 10:], y[-len(y) // 10:]

ds = tf.data.Dataset.from_tensor_slices(x_train.astype("float32")).batch(16).map(
    lambda x: tf.repeat(tf.expand_dims(x[::8192 // 512], axis=1), 100, axis=1)
)

val_ds = tf.data.Dataset.from_tensor_slices(x_test.astype("float32")).batch(16).map(
    lambda x: tf.repeat(tf.expand_dims(x[::8192 // 512], axis=1), 100, axis=1)
)
"""

# build the model
model = FourierNeuralOperator(
    num_params=0,
    input_shape=(100, 512, 1),
    periodic=[False, True],
    size=[1, 1],
    num_layers=4,
    k_max=32,
    fourier_layer=FourierLayer,
    physics_loss=physics_loss(
        lambda inputs, output, lst: lst[1][0] + lst[0][0] * lst[0][1] - 0.1 * lst[0][2],
        [1, 2], [1, 1], [[[BoundaryCondition.DIRICHLET, lambda inputs: inputs[:, 0]], None], BoundaryCondition.PERIODIC]
    )
)

model.compile(optimizer="adam")

# model.fit(ds, epochs=100, batch_size=1, validation_data=val_ds)
