import sys, os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.append('/mnt/c/users/jedli/onedrive - nus high school/documents/computing studies/navier_stonks')

import tensorflow as tf

from multiprocessing import Pool
from data_generation import burgers_data_generator


def serialize_example(x1, y1):
    """
    Creates a tf.train.Example message ready to be written to a file.
    """

    # Create a dictionary mapping the feature name to the tf.train.Example-compatible data type.
    feature = {
        "x": tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(x1).numpy()])),
        "y": tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(y1).numpy()]))
    }

    # Create a Features message using tf.train.Example.
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def tf_serialize_example(x, y):
    tf_string = tf.py_function(
        serialize_example,
        (x, y),
        tf.string
    )
    return tf.reshape(tf_string, ())


serialized_features_dataset = tf.data.Dataset.from_generator(
    burgers_data_generator, output_types=tf.string, output_shapes=()
)

filename = f'../neumann.tfrecord'
writer = tf.data.experimental.TFRecordWriter(filename)
writer.write(serialized_features_dataset)
