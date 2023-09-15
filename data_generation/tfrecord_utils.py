import tensorflow as tf


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