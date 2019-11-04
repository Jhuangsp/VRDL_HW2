import cv2
import os
import glob
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Transform dataset to TFRecord
DIR = 'D:/dataset/CelebA/Img/img_align_celeba/'
EVAL_PATH = 'front_face.txt'
OUT = DIR + 'Train.tfrecords'
with tf.io.TFRecordWriter(OUT) as TFWriter, open(EVAL_PATH, 'r') as f:
    print('Transform start...')
    image_list = f.readlines()
    num_images = len(image_list)

    face_width = face_height = 108
    image = cv2.imread(image_list[0][:-1])
    h, w, _ = image.shape
    i = (h - face_width) // 2
    j = (w - face_height) // 2
    for name in image_list:
        image = cv2.imread(name[:-1])
        image = cv2.resize(
            image[i:i+face_width, j:j+face_height], (64, 64), cv2.INTER_CUBIC)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_raw = image.tostring()
        ftrs = tf.train.Features(
            feature={'image_raw': tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[image_raw]))}
        )
        example = tf.train.Example(features=ftrs)
        TFWriter.write(example.SerializeToString())
    print('Transform done!')

data = tf.data.TFRecordDataset(OUT)
print(data)

feature_description = {
    'image_raw': tf.io.FixedLenFeature([], tf.string)
}


def _parse_function(exam_proto):
    return tf.io.parse_single_example(exam_proto, feature_description)


data = data.map(_parse_function)

fig = plt.figure()
for i, row in enumerate(data.take(9)):
    plt.subplot(3, 3, i+1)
    img = tf.reshape(tf.io.decode_raw(
        row['image_raw'], 'uint8'), [64, 64, 3]).numpy()
    plt.imshow(img)
plt.show()
