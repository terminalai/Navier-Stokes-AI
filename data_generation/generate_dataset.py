import os

from data_generation.burgers_data import data_generator

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import tensorflow as tf

from multiprocessing import Pool


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


def main():
    num_threads = 2
    for i in range(100):
        p = Pool(num_threads)
        p.map(data_generator, [x*32+i*32*num_threads for x in range(num_threads)])
        p.join()


if __name__ == "__main__":
    main()

"""
serialized_features_dataset = tf.data.Dataset.from_generator(
    data_generator, output_types=tf.string, output_shapes=()
)

filename = f'../dirichlet_0_0.tfrecord'
writer = tf.data.experimental.TFRecordWriter(filename)
writer.write(serialized_features_dataset)
"""
