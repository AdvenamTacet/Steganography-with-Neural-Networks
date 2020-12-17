import tensorflow as tf
import datetime


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
    _creation_time: str
    _generator_optimizer: tf.keras.optimizers.Optimizer
    _covered_image_loss: tf.function
    _decoder_loss: tf.function

    def __init__(self):
        """
        Initialize encryptor and decryptor model.
        Should be called when
        - gen_encoder_model
        - gen_decoder_model
        - get_name
        may be called without an error.

        Function will try to call load, yet if an error occurs, will be printed and ignored.
        """
        self.name = self.get_name()
        self._epoch_counter = 0

        self.encoder = self.gen_encoder_model()
        self.decoder = self.gen_decoder_model()

        self._generator_optimizer = None
        self._covered_image_loss = None
        self._decoder_loss = None

        self._train_file_writer = tf.summary.create_file_writer('logs/{}/train'.format(self.name))
        self._test_file_writer = tf.summary.create_file_writer('logs/{}/test'.format(self.name))

        try:
            self.load(self.name)
        except Exception as e:
            print(e)

    def gen_graphs(self):
        """
        Creates graphs for TesnorBoard, you may need to recreate model after that.
        :param self:
        :return:
        """
        encoder_file_writer = tf.summary.create_file_writer('logs/{}/graphs/encoder'.format(self.name))
        with encoder_file_writer.as_default():
            encoder_graph = tf.Graph()
            with encoder_graph.as_default():
                self.gen_encoder_model()
            tf.summary.graph(encoder_graph)

        decoder_file_writer = tf.summary.create_file_writer(
            'logs/{}/graphs/decoder'.format(self.name))
        with decoder_file_writer.as_default():
            decoder_graph = tf.Graph()
            with decoder_graph.as_default():
                self.gen_decoder_model()
            tf.summary.graph(decoder_graph)

        encoder_file_writer.flush()
        decoder_file_writer.flush()

    @staticmethod
    @tf.function
    def encoder_input_prepare(arguments: dict):
        """
        Returns input to the self.encoder.
        Should be overwritten if encoder takes non-standard arguments.

        :param arguments: dictionary with batch data and all calculated values so far.
        :return: list with tensors (barches) of color images and gray images
        """
        return [arguments['color'], arguments['gray']]

    @staticmethod
    @tf.function
    def decoder_input_prepare(arguments: dict):
        """
        Returns input to the self.decoder.
        Should be overwritten, if decoder has non-standard input.

        :param arguments: dictionary with batch data and all calculated values so far.
        :return: tf.Tensor with encoder output
        """
        return arguments['encoder_output']

    def get_name(self):
        """
        Function returning model name.
        Should be defined in inheriting class.

        :return: str
        """

        raise Exception("Method get_name not implemented!")

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
            lambda x: (
                encoder_loss(x['color'], x['encoder_output']),
                decoder_loss(x['gray'], x['decoder_output']),
                1.25 * encoder_loss(x['color'], x['encoder_output']) + decoder_loss(x['gray'], x['decoder_output'])
            ),
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
            learning_rate = lambda: 3.25 * (0.001 if self._epoch_counter < 25 else
                                           (0.0001 if self._epoch_counter < 50 else
                                            (0.00007 if self._epoch_counter < 125 else 3e-5))
            )

            self._generator_optimizer = tf.keras.optimizers.Adam(
                learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07,
                amsgrad=False, name='generator_optimizer'
            )

        return self._generator_optimizer

    def score(self, dataset: tf.keras.utils.Sequence, verbose: bool = False):
        """
        Calculates model score over dataset.
        Encoder score and decoder score are calculated separately.


        :param dataset: iterable with batches.
        :param verbose: if value will be printed
        :param skip_prob: chance to skip a batch
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

        if verbose:
            print("Score: covered_image {}, recovered_image {}".format(encoder_score/len(dataset),
                                                                       decoder_score/len(dataset)))

        return encoder_score/len(dataset), decoder_score/len(dataset)

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

                encoder_loss, decoder_loss, generator_loss = loss(arguments)

            trainable = self.encoder.trainable_variables + self.decoder.trainable_variables
            gradients = tape.gradient(generator_loss, trainable)

            optimizer.apply_gradients(zip(gradients, trainable))

            return tf.reduce_mean(encoder_loss), tf.reduce_mean(decoder_loss), tf.reduce_mean(generator_loss), None

        return train_step

    def train(self, dataset: tf.keras.utils.Sequence, epochs : int, visual_batch: dict = None, fast: bool = True,
              train_step: tf.function = None, autosave: bool = None, verify_dataset: tf.keras.utils.Sequence = None):
        """
        Function responsible for Model training. May work with different types of models.
        Should not be overwritten.
        Change other member functions to affect the model instead.

        :param dataset: iterable with batches
        :param epochs: number of epochs
        :param visual_batch: batch used for visualisation purposes
        :param fast: if False, after every epoch, score will be calculated
        :param train_step: TensorFlow function responsible for training step, if none, self.gen_train_step will be used
        :param autosave: defines if model should be saved after every learning step
        :param verify_dataset: to calculate loss after every epoch
        :return:
        """
        def visualise(visual_batch, file_writer, save_inputs):
            with file_writer.as_default():
                _sample_batch = visual_batch.copy()

                if save_inputs:
                    tf.summary.image("Color images", _sample_batch['color'], step=self._epoch_counter)
                    tf.summary.image("Gray images", _sample_batch['gray'], step=self._epoch_counter)

                _sample_batch.update({'encoder_output': self.encoder(
                    self.encoder_input_prepare(_sample_batch), training=False)})
                tf.summary.image("Connected images", _sample_batch['encoder_output'], step=self._epoch_counter)

                tf.summary.image("Connected diffs", tf.math.sigmoid(_sample_batch['encoder_output'] -
                                                                    _sample_batch['color']), step=self._epoch_counter)

                _sample_batch.update({'decoder_output': self.decoder(
                    self.decoder_input_prepare(_sample_batch), training=False)})
                tf.summary.image("Restored images", _sample_batch['decoder_output'], step=self._epoch_counter)

                tf.summary.image("Restored diffs", tf.math.sigmoid(_sample_batch['decoder_output'] -
                                                                   _sample_batch['gray']), step=self._epoch_counter)

        if self.name is None:
            raise Exception("Model name is unknown in function train.")

        if autosave is None:
            autosave = epochs > 2

        if train_step is None:
            train_step = self.gen_train_step()

        for epoch_id in range(epochs):
            start_time = datetime.datetime.now()
            loss_values = {'encoder': [], 'decoder': [], 'generator': []}
            for i, batch in enumerate(dataset):
                print("Epoch {}/{}, batch {}/{}".format(epoch_id+1, epochs, i+1, len(dataset)))
                print("Real: {}".format(1 + self._epoch_counter))

                encoder_loss, decoder_loss, generator_loss, discriminator_score = train_step(batch)
                step_losses = {'encoder': encoder_loss, 'decoder': decoder_loss, 'generator': generator_loss}
                if discriminator_score is not None:
                    discriminator_loss_value, discriminator_output, discriminator_real_output = discriminator_score
                    step_losses.update({'discriminator': discriminator_loss_value, 'discriminator_output': discriminator_output,
                                        'discriminator_real': discriminator_real_output}
                    )
                    if 'discriminator' not in loss_values:
                        loss_values.update({'discriminator': [], 'discriminator_output': [], 'discriminator_real': []})

                for key in step_losses:
                    if step_losses[key] is not None:
                        loss_values[key].append(step_losses[key])

            with self._train_file_writer.as_default():
                for key in loss_values:
                    if len(loss_values[key]) > 0:
                        value_name = (key + ' loss' if '_' not in key else key)
                        tf.summary.scalar(value_name, tf.reduce_mean(tf.stack(loss_values[key])), step=self._epoch_counter)

            if visual_batch is not None:
                if 'train' in visual_batch:
                    visualise(visual_batch['train'], self._train_file_writer, epoch_id == 0)
                if 'test' in visual_batch:
                    visualise(visual_batch['test'], self._test_file_writer, epoch_id == 0)

            dataset.on_epoch_end()

            print("Time:", (datetime.datetime.now() - start_time).seconds)

            if verify_dataset is not None:
                verify_start_time = datetime.datetime.now()
                encoder_score, decoder_score = self.score(verify_dataset)

                with self._test_file_writer.as_default():
                    tf.summary.scalar('encoder loss', encoder_score, step=self._epoch_counter)
                    tf.summary.scalar('decoder loss', decoder_score, step=self._epoch_counter)

                print("Additional time:", (datetime.datetime.now() - verify_start_time).seconds)

            self._epoch_counter += 1

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

        with open('logs/{}/counter.data'.format(name), 'w') as f:
            f.write(str(self._epoch_counter))

    def load(self, name=None, only_generator: bool = True):
        """
        Function loading encoder, decoder from files created by save member function.

        If name is not given, self.name will be used.
        If both are None, exception will bre raised.

        :param name: name of the Autoencoder model, optional.
        :param only_generator: ignored
        :return:
        """
        if name is None and self.name is None:
            raise Exception("Model name is unknown in function save.")

        if name is None:
            name = self.name

        self.encoder.load_weights('save/{}/encoder'.format(name))
        self.decoder.load_weights('save/{}/decoder'.format(name))

        with open('logs/{}/counter.data'.format(name), 'r') as f:
            self._epoch_counter = int(f.read())

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
        """
        Starts with creating discriminator model.
        Should be called when 'gen_discriminator_model' won't rise an error.
        At the end, calls EncoderDecoder initializer.

        Function will try to call load, yet if an error occurs, will be printed and ignored.
        """
        self._discriminator_optimizer = None
        self._discriminator_value_loss = None

        self.discriminator = self.gen_discriminator_model()
        super().__init__()

    def gen_graphs(self):
        """
        Creates graphs for TesnorBoard, you may need to recreate model after that.
        :param self:
        :return:
        """
        super().gen_graphs()

        discriminator_file_writer = tf.summary.create_file_writer('logs/{}/graphs/discriminator'.format(self.name))
        with discriminator_file_writer.as_default():
            discriminator_graph = tf.Graph()
            with discriminator_graph.as_default():
                self.gen_discriminator_model()
            tf.summary.graph(discriminator_graph)

        discriminator_file_writer.flush()

    @staticmethod
    @tf.function
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

        salted['encoder_output'] = tf.keras.layers.GaussianNoise(0.5 / 255., trainable=False)(salted['color'])

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
            lambda x:(
                encoder_loss(x['color'], x['encoder_output']),
                decoder_loss(x['gray'], x['decoder_output']),
                encoder_loss(x['color'], x['encoder_output']) +
                1.25*decoder_loss(x['gray'], x['decoder_output']) +
                3 * tf.expand_dims(x['discriminator_output'], axis=1)/1000.
            ),
            trainable=False,
            name='generator_loss'
        )

    def get_discriminator_loss_for_a_set(self):
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
        loss = self.get_discriminator_loss_for_a_set()

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
            learning_rate = lambda: 28.5 * (0.001 if self._epoch_counter < 25 else
                                            (0.0001 if self._epoch_counter < 50 else
                                             (0.00007 if self._epoch_counter < 125 else 4e-5))
            )

            self._discriminator_optimizer = tf.keras.optimizers.Adam(
                learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07,
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

                encoder_loss, decoder_loss, generator_loss_value = generator_loss(arguments)
                discriminator_loss_value = discriminator_loss(arguments)
            generator_trainable = self.encoder.trainable_variables + self.decoder.trainable_variables
            discriminator_trainable = self.discriminator.trainable_variables

            generator_gradients = generator_tape.gradient(generator_loss_value, generator_trainable)
            discriminator_gradients = discriminator_tape.gradient(discriminator_loss_value, discriminator_trainable)

            generator_optimizer.apply_gradients(zip(generator_gradients, generator_trainable))
            discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator_trainable))

            return (
                tf.reduce_mean(encoder_loss),
                tf.reduce_mean(decoder_loss),
                tf.reduce_mean(generator_loss_value),
                (discriminator_loss_value,
                 tf.reduce_mean(discriminator_output), tf.reduce_mean(discriminator_real_output))
            )

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

            return (None, None, None, (tf.reduce_mean(discriminator_loss_value),
                                       tf.reduce_mean(discriminator_output), tf.reduce_mean(discriminator_real_output))
            )

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

    def load(self, name: str = None, only_generator: bool = False, **kwargs):
        """
        Function loading encoder, decoder and discriminator from files created by save member function.

        If name is not given, self.name will be used.
        If both are None, exception will bre raised.

        :param name: name of the model, optional.
        :param only_generator: boolean, set true to not load discriminator, otpional
        :return:
        """
        super(GAN, self).load(name)

        if name is None:
            name = self.name

        if not only_generator:
            self.discriminator.load_weights('save/{}/discriminator'.format(name))

    def summary(self):
        """
        Prints summary of Autoencoder and discriminator.

        :return:
        """
        super(GAN, self).summary()

        self.discriminator.summary()
