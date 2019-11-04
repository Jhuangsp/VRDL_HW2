import tensorflow as tf
import glob
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import imageio
import os
import datetime
import time
import PIL
import pprint
from dataloader import *
from dcgan import *
from solver import *

from cv2 import imread, cvtColor, resize, INTER_CUBIC, COLOR_BGR2RGB, blur
from tensorflow.keras import layers
pp = pprint.PrettyPrinter(indent=4)
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
print('current_time:', current_time)

# Load data
DATA_PATH = 'D:/dataset/CelebA/Img/img_align_celeba/'
FILE_NAME = 'Train.tfrecords'
BUFFER_SIZE = 60000
BATCH_SIZE = 256
train_dataset = load_data(DATA_PATH+FILE_NAME, BUFFER_SIZE, BATCH_SIZE)

# Tensorboard
Gen_log_dir = 'log/' + current_time + '/Gen'
Dis_log_dir = 'log/' + current_time + '/Dis'
Gen_writer = tf.summary.create_file_writer(Gen_log_dir)
Dis_writer = tf.summary.create_file_writer(Dis_log_dir)

# Check generator
generator = make_generator_model()
noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)
print(generated_image.shape)
plt.imshow(((generated_image[0].numpy()+1.0)*127.5).astype(np.uint8))
plt.show()

# Check discriminator
discriminator = make_discriminator_model()
decision = discriminator(generated_image)
print(decision)


# Optimizers
noise_dim = 100
solver = Solver(generator, discriminator, BATCH_SIZE, noise_dim)
learning_rate = 0.0002
solver.create_optimizer(learning_rate)


def generate_and_save_images(model, epoch, test_input):
    print('generate_and_save_images')
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(((predictions[i].numpy()+1.0)*127.5).astype(np.uint8))
        plt.axis('off')

    dir_path = os.path.join(checkpoint_dir, current_time,
                            'image_at_epoch_{:04d}.png'.format(epoch))
    plt.savefig(dir_path)
    plt.show()


# Define the training loop
EPOCHS = 200
num_examples_to_generate = 16
checkpoint_dir = './training_checkpoints'
os.makedirs(os.path.join(checkpoint_dir, current_time))
checkpoint_prefix = os.path.join(checkpoint_dir, current_time, "ckpt")
seed = tf.random.normal([num_examples_to_generate, noise_dim])


def train(solver, dataset, epochs):
    steps = 0
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            steps += 1
            gen_loss, disc_loss = solver.train_step(image_batch)
            if steps % 50 == 0:
                tf.print('Gen_loss: ', gen_loss, 'Disc_loss: ', disc_loss)
                with Gen_writer.as_default():
                    tf.summary.scalar('loss', gen_loss, step=steps)
                with Dis_writer.as_default():
                    tf.summary.scalar('loss', disc_loss, step=steps)

        generate_and_save_images(solver.generator,
                                 epoch + 1,
                                 seed)

        solver.checkpoint.save(file_prefix=checkpoint_prefix)

        print('Time for epoch {} is {} sec'.format(
            epoch + 1, time.time()-start))

    generate_and_save_images(solver.generator,
                             epochs,
                             seed)


train(solver, train_dataset, EPOCHS)
