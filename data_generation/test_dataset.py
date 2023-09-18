import tensorflow as tf

feature_desc = {
    'x': tf.io.FixedLenFeature([], tf.string),
    'y': tf.io.FixedLenFeature([], tf.string)
}


def _parse_example(example_proto):
    res = tf.io.parse_single_example(example_proto, feature_desc)
    x = tf.io.parse_tensor(res['x'], out_type=tf.double)
    y = tf.io.parse_tensor(res['y'], out_type=tf.double)
    x.set_shape((2048,))
    y.set_shape((999, 2048))
    return x, y


ds = (tf.data.TFRecordDataset(r"C:\Users\jedli\Downloads\neumann.tfrecord")
        .map(_parse_example)
        .batch(16))
