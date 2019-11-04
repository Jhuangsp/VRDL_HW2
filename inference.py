import tensorflow as tf
import numpy as np
import cv2
import helper
import matplotlib.pyplot as plt
import os


def output_fig(images_array, file_name="./results"):
    # the shape of your images_array should be (9, width, height, 3),  28 <= width, height <= 112
    plt.figure(figsize=(6, 6), dpi=100)
    plt.imshow(helper.images_square_grid(images_array))
    plt.axis("off")
    plt.savefig(file_name+'.png', bbox_inches='tight', pad_inches=0)


noise_dim = 100
num_imgs_in_example = 9
num_examples_to_generate = 500
seed = tf.random.normal(
    [num_examples_to_generate*num_imgs_in_example, noise_dim])
ckpt_dir = 'training_checkpoints'
date = '20191028-083249'
file = 'ckpt-126'
if not os.path.isdir(os.path.join('answer', date, file)):
    os.makedirs(os.path.join('answer', date, file))

generator = tf.keras.models.load_model(
    os.path.join(ckpt_dir, date, file+'.h5'), compile=False)
generator.summary()

splits = 10
size = num_imgs_in_example*num_examples_to_generate//splits
for b in range(splits):
    print('Split {} start'.format(b))
    predictions = generator(seed[b*size:(b+1)*size], training=False).numpy(
    ).reshape(size//num_imgs_in_example, num_imgs_in_example, 64, 64, 3)

    for i, imgs in enumerate(predictions):
        output_fig(imgs, file_name="answer2/{}_image".format(
            str.zfill(str(i+b*size//num_imgs_in_example), 3)))
