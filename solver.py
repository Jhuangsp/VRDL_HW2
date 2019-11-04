import tensorflow as tf


class Solver(object):
    """docstring for Solver"""

    def __init__(self, generator, discriminator, batch_size, noise_dim, smooth=0.9):

        self.generator = generator
        self.discriminator = discriminator
        self.smooth = smooth
        self.batch_size = batch_size
        self.noise_dim = noise_dim
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(
            from_logits=True)

    def create_optimizer(self, learning_rate):
        beta1 = 0.5
        self.generator_optimizer = tf.keras.optimizers.Adam(
            learning_rate, beta_1=beta1)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(
            learning_rate, beta_1=beta1)

        # Save checkpoints
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                              discriminator_optimizer=self.discriminator_optimizer,
                                              generator=self.generator,
                                              discriminator=self.discriminator)
        pass

    # Generator loss
    def generator_loss(self, fake_output):
        return self.cross_entropy(tf.ones_like(fake_output)*self.smooth, fake_output)

    # Discriminator loss
    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(
            real_output)*self.smooth, real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    @tf.function
    def train_step(self, images):
        noise = tf.random.normal([self.batch_size, self.noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)

            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

            gradients_of_generator = gen_tape.gradient(
                gen_loss, self.generator.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient(
                disc_loss, self.discriminator.trainable_variables)

            tf.cond(gen_loss > disc_loss,
                    lambda: self.generator_optimizer.apply_gradients(
                        zip(gradients_of_generator, self.generator.trainable_variables)),
                    lambda: self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables)))

        return gen_loss, disc_loss
