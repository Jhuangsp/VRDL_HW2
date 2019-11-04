import tensorflow as tf

import os


def load_data(path, buffer_size, batch_size):

    if os.path.isfile(path):
        train_dataset = tf.data.TFRecordDataset(path)
        feature_description={
            'image_raw' : tf.io.FixedLenFeature([], tf.string)
        }
        def _parse_function(exam_proto):
            return tf.io.parse_single_example(exam_proto, feature_description)
        def _pre_process(exam_proto):
            exam_proto = tf.io.decode_raw(exam_proto['image_raw'], 'uint8')
            return (tf.dtypes.cast(tf.reshape(exam_proto, [64,64,3]), tf.float32) - 127.5) / 127.5

        train_dataset = train_dataset.map(_parse_function)
        train_dataset = train_dataset.map(_pre_process)
    else:
        print(path, 'not exist')
        os._exit(0)

    # Batch and shuffle the data
    train_dataset = train_dataset.shuffle(buffer_size).batch(batch_size)
    # Data augmentation
    def flip(x):
        return tf.image.random_flip_left_right(x)
    train_dataset = train_dataset.map(lambda x: tf.cond(tf.random.uniform([], 0, 1) > 0.75, lambda: flip(x), lambda: x), num_parallel_calls=4)

    return train_dataset