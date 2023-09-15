import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.append('/mnt/c/users/jedli/onedrive - nus high school/documents/computing studies/navier_stonks')

import numpy as np
import tensorflow as tf

from data_generation import burgers_data_generator #, tf_serialize_example


r"""
def data_generator():
    root = r"C:\Users\jedli\Downloads\data"

    for i in range(1600):
        try:
            ic_file = f"{root}/ics/ic_{i}.npy"
            sol_file = f"{root}/sol/sol_{i}.npy"
            print(i)

            yield tf_serialize_example(
                np.load(ic_file).astype("float32"),
                np.load(sol_file).astype("float32")
            )
        except Exception as e: print(e)
"""

serialized_features_dataset = tf.data.Dataset.from_generator(
    burgers_data_generator, output_types=tf.string, output_shapes=()
)

filename = f'../neumann.tfrecord'
writer = tf.io.TFRecordWriter(filename)
writer.write(serialized_features_dataset)
