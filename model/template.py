import tensorflow as tf
import datetime

# from progress.bar import Bar
# from IPython import display


class EncoderDecoder:
    """ A template to (Neural Network) steganography learning framework.
        Supports Steganography Autoencoder learning and is a base to more complicated model templates.

        Virtual member functions:
        - gen_encoder_model
        - gen_decoder_model
        They should return an object of type tf.keras.Model.
    """
    encoder: tf.keras.Model
    decoder: tf.keras.Model
    name: str
    _generator_optimizer: tf.keras.optimizers.Optimizer
    _covered_image_loss: tf.function
    _decoder_loss: tf.function

    def __init__(self):
        self.encoder = self.gen_encoder_model()
        self.decoder = self.gen_decoder_model()

        self._generator_optimizer = None
        self._covered_image_loss = None
        self._decoder_loss = None

        self.name = None

    @staticmethod
    def encoder_input_prepare(arguments: dict):
        """
        Returns input to the self.encoder.
        Should be overwritten if encoder takes non-standard arguments.

        :param arguments: dictionary with batch data and all calculated values so far.
        :return: list with tensors (barches) of color images and gray images
        """
        return [arguments['color'], arguments['gray']]

    @staticmethod
    def decoder_input_prepare(arguments: dict):
        """
        Returns input to the self.decoder.
        Should be overwritten, if decoder has non-standard input.

        :param arguments: dictionary with batch data and all calculated values so far.
        :return: tf.Tensor with encoder output
        """
        return arguments['encoder_output']

    def gen_encoder_model(self):
        """
        Function returning encoder, first part of the Model.
        Should be defined in inheriting class.

        :return: tf.keras.Model
        """
        raise Exception("Method gen_encoder_model not implemented!")

    def gen_decoder_model(self):
        """
        Function returning decoder, second part of the Model.
        Should be defined in inheriting class.

        :return: tf.keras.Model
        """
        raise Exception("Method gen_decoder_model not implemented!")

    def get_covered_image_loss(self):
        """
        Returns self._covered_image_loss.
        To change a loss function of a image with hidden data, change that variable.
        Used by:
        - gen_generator_loss
        - score

        If the variable is None, sets default value (MSE).

        :return: tf.function calculating loss of the image with hidden data.
        """
        if self._covered_image_loss is None:
            self._covered_image_loss = tf.keras.losses.MSE

        return self._covered_image_loss

    def get_decoder_loss(self):
        """
        Returns self._decoder_loss.
        To change a decoder loss function, change that variable.

        If the variable is None, sets default value (MSE).

        :return: tf.function calculating decoder loss
        """
        if self._decoder_loss is None:
            self._decoder_loss = tf.keras.losses.MSE

        return self._decoder_loss

    def gen_generator_loss(self):
        """
        Creates Lambda function calculating a loss value of the Autoencoder Model.

        :return: tf.keras.layers.Lambda calculating loss value.
        """
        encoder_loss = self.get_covered_image_loss()
        decoder_loss = self.get_decoder_loss()

        return tf.keras.layers.Lambda(
            lambda x: encoder_loss(x['color'], x['encoder_output']) + decoder_loss(x['gray'], x['decoder_output']),
            trainable=False,
            name='generator_loss'
        )

    def get_generator_optimizer(self):
        """
        Returns self._generator_optimizer.
        To change an optimizer in model, change that variable.

        If the variable is None, sets default value (Adam).

        :return: tf.keras.optimizers.Adam
        """
        if self._generator_optimizer is None:
            self._generator_optimizer = tf.keras.optimizers.Adam(
                learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, name='generator_optimizer'
            )

        return self._generator_optimizer

    def score(self, dataset: tf.keras.utils.Sequence):
        """
        Calculates model score over dataset.
        Encoder score and decoder score are calculated separately.

        :param dataset: iterable with batches.
        :return:
        """
        encoder_score = 0.
        decoder_score = 0.

        covered_loss = self.get_covered_image_loss()
        decoder_loss = self.get_decoder_loss()

        for batch in dataset:
            arguments = batch.copy()

            encoder_output = self.encoder.predict(self.encoder_input_prepare(arguments))
            arguments.update({'encoder_output': encoder_output})

            decoder_output = self.decoder.predict(self.decoder_input_prepare(arguments))
            arguments.update({'decoder_output': decoder_output})

            color, gray = arguments['color'], arguments['gray']
            encoder_score += tf.math.reduce_mean(covered_loss(color, encoder_output)).numpy()
            decoder_score += tf.math.reduce_mean(decoder_loss(gray, decoder_output)).numpy()

        print("Score: covered_image {}, recovered_image {}".format(encoder_score/len(dataset), decoder_score/len(dataset)))

    def gen_train_step(self):
        """
        Returns compiled TensorFlow function, responsible for a single learning step.
        Used at the beginning of every training, if training_step argument is not specified.

        :return: @tf.function performing a single (batch) learning step
        """
        loss = self.gen_generator_loss()
        optimizer = self.get_generator_optimizer()

        @tf.function
        def train_step(batch: dict):
            arguments = batch.copy()

            with tf.GradientTape() as tape:
                encoder_output = self.encoder(self.encoder_input_prepare(arguments), training=True)
                arguments.update({'encoder_output': encoder_output})

                decoder_output = self.decoder(self.decoder_input_prepare(arguments), training=True)
                arguments.update({'decoder_output': decoder_output})

                loss_value = loss(arguments)

            trainable = self.encoder.trainable_variables + self.decoder.trainable_variables
            gradients = tape.gradient(loss_value, trainable)

            optimizer.apply_gradients(zip(gradients, trainable))

        return train_step

    def train(self, dataset: tf.keras.utils.Sequence, epochs : int, fast: bool = True,
              train_step: tf.function = None, autosave: bool = None):
        """
        Function responsible for Model training. May work with different types of models.
        Should not be overwritten.
        Change other member functions to affect the model instead.

        :param dataset: iterable with batches
        :param epochs: number of epochs
        :param fast: if False, after every epoch, score will be calculated
        :param train_step: TensorFlow function responsible for training step, if none, self.gen_train_step will be used
        :param autosave: defines if model should be saved after every learning step
        :return:
        """

        if autosave is None:
            autosave = epochs > 10

        if train_step is None:
            train_step = self.gen_train_step()

        for epoch_id in range(epochs):
            start_time = datetime.datetime.now()
            for i, batch in enumerate(dataset):
                print("Epoch {}/{}, batch {}/{}".format(epoch_id+1, epochs, i+1, len(dataset)))
                train_step(batch)
            dataset.on_epoch_end()

            print("Time:", (datetime.datetime.now() - start_time).seconds)
            if not fast:
                self.score(dataset)

            if autosave:
                self.save()

    def save(self, name: str = None):
        """
        Function saving encoder, decoder to disk.
        The Autoencoder Model will be in './save/{name}/' directory.

        If name is not given, self.name will be used.
        If both are None, exception will bre raised.

        :param name: name of the Autoencoder model, optional.
        :return:
        """
        if name is None and self.name is None:
            raise Exception("Model name is unknown in function save.")

        if name is None:
            name = self.name

        self.encoder.save_weights('save/{}/encoder'.format(name), overwrite=True)
        self.decoder.save_weights('save/{}/decoder'.format(name), overwrite=True)

    def load(self, name=None):
        """
        Function loading encoder, decoder from files created by save member function.

        If name is not given, self.name will be used.
        If both are None, exception will bre raised.

        :param name: name of the Autoencoder model, optional.
        :return:
        """
        if name is None and self.name is None:
            raise Exception("Model name is unknown in function save.")

        if name is None:
            name = self.name

        self.encoder.load_weights('save/{}/encoder'.format(name))
        self.decoder.load_weights('save/{}/decoder'.format(name))

    def summary(self):
        """
        Prints summary of encoder and decoder.

        :return:
        """

        self.encoder.summary()
        self.decoder.summary()


class GAN(EncoderDecoder):
    """ A template to (Neural Network) steganography learning framework.
        Supports Steganography GANs learning and may be a base to more complicated model templates.

        Virtual member functions:
        - discriminator_input_prepare
        - all from EncoderDecoder class
        They should be defined before use.
    """
    discriminator: tf.keras.Model
    _discriminator_optimizer: tf.keras.optimizers.Optimizer
    _discriminator_value_loss: tf.function

    def __init__(self):
        super().__init__()

        self._discriminator_optimizer = None
        self._discriminator_value_loss = None

        self.discriminator = self.gen_discriminator_model()

    @staticmethod
    def discriminator_input_prepare(arguments: dict):
        """
        Returns input to the self.decoder.
        Should be overwritten, if discriminator has non-standard input.

        :param arguments: dictionary with batch data and all calculated values so far.
        :return: tf.Tensor with encoder output
        """

        return arguments['encoder_output']

    def discriminator_salted_input_prepare(self, arguments: dict):
        """
        That's a wrapper to self.discriminator_input_prepare.
        Function returns picture without information as those with hidden image.
        Use to learn discriminator when image does not contain hidden data.

        :param arguments: dictionary with batch data and all calculated values so far.
        :return: same as self.discriminator_input_prepare
        """
        salted = arguments.copy()

        salted['encoder_output'] = tf.keras.layers.GaussianNoise(1 / 255., trainable=False)(salted['color'])
        salted['color'] = tf.keras.layers.GaussianNoise(1 / 255., trainable=False)(salted['color'])
        # salted['encoder_output'] = salted['color']

        return self.discriminator_input_prepare(salted)

    def gen_discriminator_model(self):
        """
        Function returning encoder, first part of the Model.
        Should be defined in inheriting class.

        :return: tf.keras.Model
        """
        raise Exception("Method gen_discriminator_model not implemented!")

    def gen_generator_loss(self):
        """
        Creates Lambda function calculating a loss value of the GAN Model.

        :return: tf.keras.layers.Lambda calculating loss value.
        """
        encoder_loss = self.get_covered_image_loss()
        decoder_loss = self.get_decoder_loss()

        return tf.keras.layers.Lambda(
            lambda x:
                encoder_loss(x['color'], x['encoder_output']) +
                decoder_loss(x['gray'], x['decoder_output']) +
                tf.expand_dims(x['discriminator_output'], axis=1),
            trainable=False,
            name='generator_loss'
        )

    def get_discriminator_value_loss(self):
        """
        Returns self._discriminator_value_loss.
        To change a loss function of the discriminator, change that variable.

        Function is used to calculate distance from zero/one vector during learning.

        Used by:
        - gen_discriminator_loss
        - score

        If the variable is None, sets default value (MSE).

        :return: tf.function calculating loss of the image with hidden data.
        """
        if self._discriminator_value_loss is None:
            self._discriminator_value_loss = tf.keras.losses.MeanSquaredError()

        return self._discriminator_value_loss

    def gen_discriminator_loss(self):
        """
        Creates Lambda function calculating a loss value of the discriminator inside the Model.

        :return: tf.keras.layers.Lambda calculating discriminator loss value.
        """
        loss = self.get_discriminator_value_loss()

        return tf.keras.layers.Lambda(
            lambda x:
                loss(tf.zeros_like(x['discriminator_real_output']), x['discriminator_real_output']) +
                loss(tf.ones_like(x['discriminator_output']), x['discriminator_output']),
            trainable=False,
            name='discriminator_loss'
        )

    def get_discriminator_optimizer(self):
        """
        Returns self._discriminator_optimizer.
        To change the optimizer in model, change that variable.

        If the variable is None, sets default value (Adam).

        :return: tf.keras.optimizers.Adam
        """
        if self._discriminator_optimizer is None:
            self._discriminator_optimizer = tf.keras.optimizers.Adam(
                learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-07,
                amsgrad=True, name='discriminator_optimizer'
            )

        return self._discriminator_optimizer

    def score(self, dataset: tf.keras.utils.Sequence):
        """
        Calculates model score over dataset.
        Encoder score and decoder score are calculated separately.
        As well as discriminator socre and discriminator socre for "clean" pictures.

        :param dataset: iterable with batches.
        :return:
        """
        encoder_score = 0.
        decoder_score = 0.
        discriminator_score = 0.

        covered_loss = self.get_covered_image_loss()
        decoder_loss = self.get_decoder_loss()

        for batch in dataset:
            arguments = batch.copy()

            encoder_output = self.encoder.predict(self.encoder_input_prepare(arguments))
            arguments.update({'encoder_output': encoder_output})

            decoder_output = self.decoder.predict(self.decoder_input_prepare(arguments))
            arguments.update({'decoder_output': decoder_output})

            discriminator_output = self.discriminator.predict(self.discriminator_input_prepare(arguments))

            color, gray = arguments['color'], arguments['gray']
            encoder_score += tf.math.reduce_mean(covered_loss(color, encoder_output)).numpy()
            decoder_score += tf.math.reduce_mean(decoder_loss(gray, decoder_output)).numpy()
            discriminator_score += tf.math.reduce_mean(discriminator_output).numpy()

        print("Score: covered_image {}, recovered_image {}, discriminator {}".format(
            encoder_score / len(dataset),
            decoder_score / len(dataset),
            discriminator_score / len(dataset)
        ))

        discriminator_score = 0.
        for batch in dataset:
            arguments = batch.copy()

            discriminator_output = self.discriminator.predict(self.discriminator_salted_input_prepare(arguments))
            discriminator_score += tf.math.reduce_mean(discriminator_output).numpy()

        print("Score: real discriminator mean output {}".format(discriminator_score / len(dataset)))

    def gen_train_step(self):
        """
        Returns compiled TensorFlow function, responsible for a single learning step.
        Used at the beginning of every training, if training_step argument is not specified.

        :return: @tf.function performing a single (batch) learning step
        """
        generator_loss = self.gen_generator_loss()
        discriminator_loss = self.gen_discriminator_loss()

        generator_optimizer = self.get_generator_optimizer()
        discriminator_optimizer = self.get_discriminator_optimizer()

        @tf.function
        def train_step(batch: dict):
            arguments = batch.copy()

            with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:
                encoder_output = self.encoder(self.encoder_input_prepare(arguments), training=True)
                arguments.update({'encoder_output': encoder_output})

                decoder_output = self.decoder(self.decoder_input_prepare(arguments), training=True)
                arguments.update({'decoder_output': decoder_output})

                discriminator_output = self.discriminator(self.discriminator_input_prepare(arguments), training=True)
                arguments.update({'discriminator_output': discriminator_output})

                discriminator_real_output = self.discriminator(self.discriminator_salted_input_prepare(arguments),
                                                               training=True)
                arguments.update({'discriminator_real_output': discriminator_real_output})

                generator_loss_value = generator_loss(arguments)
                discriminator_loss_value = discriminator_loss(arguments)

            generator_trainable = self.encoder.trainable_variables + self.decoder.trainable_variables
            discriminator_trainable = self.discriminator.trainable_variables

            generator_gradients = generator_tape.gradient(generator_loss_value, generator_trainable)
            discriminator_gradients = discriminator_tape.gradient(discriminator_loss_value, discriminator_trainable)

            generator_optimizer.apply_gradients(zip(generator_gradients, generator_trainable))
            discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator_trainable))

        return train_step

    def gen_discriminator_train_step(self):
        """
        Returns compiled TensorFlow function, responsible for a single learning step.
        The function modifies (learns) ONLY discriminator.
        Result may be used as training_step argument in train member function.

        :return: @tf.function performing a single (batch) learning step
        """
        discriminator_loss = self.gen_discriminator_loss()
        discriminator_optimizer = self.get_discriminator_optimizer()

        @tf.function
        def train_step(batch):
            arguments = batch.copy()

            with tf.GradientTape() as tape:
                encoder_output = self.encoder(self.encoder_input_prepare(arguments), training=False)
                arguments.update({'encoder_output': encoder_output})

                discriminator_output = self.discriminator(self.discriminator_input_prepare(arguments), training=True)
                arguments.update({'discriminator_output': discriminator_output})

                discriminator_real_output = self.discriminator(self.discriminator_salted_input_prepare(arguments),
                                                               training=True)
                arguments.update({'discriminator_real_output': discriminator_real_output})

                discriminator_loss_value = discriminator_loss(arguments)

            discriminator_trainable = self.discriminator.trainable_variables

            discriminator_gradients = tape.gradient(discriminator_loss_value, discriminator_trainable)

            discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator_trainable))

        return train_step

    def save(self, name: str = None):
        """
         Function saving encoder, decoder and discriminator weights to disk.
         The GAN Model will be in './save/{name}/' directory.

         If name is not given, self.name will be used.
         If both are None, exception will bre raised.

         :param name: name of the Autoencoder model, optional.
         :return:
         """
        super(GAN, self).save(name)

        if name is None:
            name = self.name

        self.discriminator.save_weights('save/{}/discriminator'.format(name), overwrite=True)

    def load(self, name: str = None):
        """
        Function loading encoder, decoder and discriminator from files created by save member function.

        If name is not given, self.name will be used.
        If both are None, exception will bre raised.

        :param name: name of the model, optional.
        :return:
        """
        super(GAN, self).load(name)

        if name is None:
            name = self.name

        self.discriminator.load_weights('save/{}/discriminator'.format(name))

    def summary(self):
        """
        Prints summary of Autoencoder and discriminator.

        :return:
        """
        super(GAN, self).summary()

        self.discriminator.summary()