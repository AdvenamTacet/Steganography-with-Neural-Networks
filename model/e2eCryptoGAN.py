import tensorflow as tf
from tensorflow.keras import layers
import model.template as template
import model.e2eCNN
from model.e2eGAN import e2eGAN


class e2eCryptoGAN(template.GAN):
    """
    Model same as e2eGAN, but discriminator also takes color picture (the one in which we try to hide data).
    """
    image_W = 160
    image_H = 160

    def __init__(self, bks: int = 3, dr: int = 1, ddr: int = 1):
        """
        Initialize the model.
        Saves a size of the kernel and dilation rate of the model.

        :param bks: this value will be returned by self.basic_kernel_size
        :param dr: this value will be returned by self.basic_dilation_rate
        """
        self.__bks = bks
        self.__dr = dr
        self.__ddr = ddr
        super().__init__()

    def get_name(self):
        return 'e2eCryptoGAN_' + str(self.basic_kernel_size()) + '_' + str(self.basic_dilation_rate())

    @staticmethod
    @tf.function
    def encoder_input_prepare(arguments: dict):
        return [arguments['color'], arguments['gray'], arguments['key']]

    @staticmethod
    @tf.function
    def decoder_input_prepare(arguments: dict):
        return [arguments['encoder_output'],  arguments['key']]

    @staticmethod
    @tf.function
    def discriminator_input_prepare(arguments):
        return [arguments['encoder_output'], arguments['color'], arguments['gray'], arguments['partial_key']]

    def basic_kernel_size(self):
        """
        Returns kernel size for most of layers. Default is three. Value is set in constructor.

        :return: int
        """
        return self.__bks

    def basic_dilation_rate(self):
        """
        Returns dilation rate for most of layers. Default is one. Value is set in constructor.

        :return: int
        """
        return self.__dr

    def discriminator_dilation_rate(self):
        """
        Returns dilation rate for most of layers in discriminator. Default is one. Value is set in constructor.

        :return: int
        """
        return self.__ddr

    def gen_encoder_model(self):
        return model.e2eCNN.e2eCNN.gen_encoder_model(self)

    def gen_decoder_model(self):
        return model.e2eCNN.e2eCNN.gen_decoder_model(self)

    def gen_generator_loss(self):
        encoder_loss = self.get_covered_image_loss()
        decoder_loss = self.get_decoder_loss()

        return tf.keras.layers.Lambda(
            lambda x:(
                encoder_loss(x['color'], x['encoder_output']),
                decoder_loss(x['gray'], x['decoder_output']),
                1 * encoder_loss(x['color'], x['encoder_output']) +
                decoder_loss(x['gray'], x['decoder_output']) +
                5.5 * tf.expand_dims(x['discriminator_output'], axis=1)/1000.
            ),
            trainable=False,
            name='generator_loss'
        )

    def gen_encoder_model(self):
        bks = self.basic_kernel_size()
        dr = self.basic_dilation_rate()
        cover_image = layers.Input(shape=(self.image_H, self.image_W, 3), name='cover_image')
        hide_image = layers.Input(shape=(self.image_H, self.image_W, 1), name='hide_image')
        key = layers.Input(shape=(self.image_H, self.image_W, 1), name="key")

        gray_layer = layers.Concatenate(axis=3, name='gray_start_in_cg')([hide_image, key])
        gray_layer = layers.Conv2D(filters=1, kernel_size=1, padding='same', activation='relu', name='gray_start_cg')(
            gray_layer)

        conn_layer = layers.Conv2D(filters=3, kernel_size=1, padding='same', activation='relu', name='color_start')(
            cover_image)

        N = 4
        for i in range(N):
            conn_layer_in = layers.Concatenate(axis=3, name='encoder_in_' + str(i))([gray_layer, conn_layer])

            conn_layer = layers.Conv2D(filters=16, kernel_size=bks, dilation_rate=dr, padding='same', activation='relu',
                                       name='encoder_color_A_' + str(i))(conn_layer_in)

            conn_layer = layers.Conv2D(filters=16, kernel_size=(bks if i + 1 < N else 1),
                                       dilation_rate=(dr if i + 1 < N else 1),
                                       activation='relu', padding='same', name='encoder_color_B_' + str(i))(conn_layer)

            if i + 1 < N:
                gray_layer = layers.Conv2D(filters=16, kernel_size=bks, dilation_rate=dr, padding='same',
                                           activation='relu',  # kernel_initializer=tf.keras.initializers.Ones(),
                                           name='encoder_gray_A_' + str(i))(gray_layer)

                gray_layer = layers.Conv2D(filters=16, kernel_size=bks, dilation_rate=dr,
                                           padding='same', activation='relu',
                                           # kernel_initializer=tf.keras.initializers.Ones(),
                                           name='encoder_gray_B_' + str(i))(gray_layer)

        layer_1 = layers.Conv2D(filters=8, kernel_size=1, padding='same', activation='relu',
                                name='encoder_l_1')(conn_layer)

        out = layers.Conv2D(filters=3, kernel_size=1, padding='same',
                            name='encoded_image')(layer_1)
        return tf.keras.Model([cover_image, hide_image, key], [out], name="encoder_model")

    def gen_decoder_model(self):
        bks = self.basic_kernel_size()
        dr = self.basic_dilation_rate()
        covered_image = layers.Input(shape=(self.image_H, self.image_W, 3), name='connected_images')
        key = layers.Input(shape=(self.image_H, self.image_W, 1), name="key")

        start = layers.Concatenate(axis=3, name='start_cg')([covered_image, key])

        layer_0 = layers.Conv2D(filters=3, kernel_size=1, activation='relu',
                                padding='same', name='decoder_l_0_cg')(start)

        layer_1 = layers.Conv2D(filters=16, kernel_size=bks, dilation_rate=dr, activation='relu',
                                padding='same', name='decoder_l_1')(layer_0)

        layer_2 = layers.Conv2D(filters=16, kernel_size=bks, dilation_rate=dr, activation='relu',
                                padding='same', name='decoder_l_2')(layer_1)

        layer_3 = layers.Conv2D(filters=8, kernel_size=bks, dilation_rate=dr, activation='relu',
                                padding='same', name='decoder_l_3')(layer_2)

        layer_4 = layers.Conv2D(filters=8, kernel_size=bks, dilation_rate=dr, activation='relu',
                                padding='same', name='decoder_l_4')(layer_3)

        layer_5 = layers.Conv2D(filters=3, kernel_size=bks, dilation_rate=dr, activation='relu',
                                padding='same', name='decoder_l_5')(layer_4)

        out = layers.Conv2D(filters=1, kernel_size=1, padding='same',
                            name='recovered_image')(layer_5)

        return tf.keras.Model([covered_image, key], [out], name="decoder_model")

    def gen_discriminator_model(self):
        bks = self.basic_kernel_size()
        dr = self.basic_dilation_rate()

        connected_images_in = layers.Input(shape=(self.image_H, self.image_W, 3), name="connected_images")
        color_image_in = layers.Input(shape=(self.image_H, self.image_W, 3), name="color_image_0")
        gray_image_in = layers.Input(shape=(self.image_H, self.image_W, 1), name="gray_image_0")
        partial_key = layers.Input(shape=(self.image_H, self.image_W, 1), name="partial_key")

        connected_images = layers.Conv2D(filters=3, kernel_size=1, dilation_rate=dr, activation='relu',
                                         padding='same', name='connected_images_scale')(connected_images_in)
        cover_image = layers.Concatenate(axis=3, name='connect_0')([color_image_in, gray_image_in])

        layer_in = layers.Concatenate(axis=3, name='in_extended_layer_1')([connected_images, cover_image, partial_key])
        layer = layers.Conv2D(filters=16, kernel_size=bks, dilation_rate=dr, strides=min(bks, 2), activation='relu',
                              padding='valid', name='discriminator_l_0')(layer_in)

        layer = layers.Conv2D(filters=16, kernel_size=bks, dilation_rate=dr, strides=min(bks, 2), activation='relu',
                              padding='valid', name='discriminator_l_1')(layer)

        layer = layers.MaxPooling2D(pool_size=(2, 2), name='discriminator_pooling_1')(layer)

        if bks > 1:
            cover_image = layers.AveragePooling2D(pool_size=bks, strides=2, padding='valid',
                                                  name='cover_pool_a')(cover_image)
            cover_image = layers.AveragePooling2D(pool_size=bks, strides=2, padding='valid',
                                                  name='cover_pool_b')(cover_image)

        cover_image = layers.AveragePooling2D(pool_size=(2, 2), name='cover_pool_1')(cover_image)
        layer_in = layers.Concatenate(axis=3, name='in_extended_layer_2')([layer, cover_image])

        layer = layers.Conv2D(filters=16, kernel_size=bks, dilation_rate=dr, activation='relu',
                              padding='valid', name='discriminator_l_2')(layer_in)

        layer = layers.Conv2D(filters=8, kernel_size=bks, dilation_rate=dr, activation='relu',
                              padding='valid', name='discriminator_l_3')(layer)

        layer = layers.Conv2D(filters=3, kernel_size=bks, dilation_rate=dr, activation='relu',
                              padding='valid', name='discriminator_l_4')(layer)

        layer = layers.Conv2D(filters=1, kernel_size=bks, dilation_rate=dr, activation='relu',
                              padding='valid', name='discriminator_l_5')(layer)

        layer = layers.Flatten(name='discriminator_flatten_1')(layer)

        layer = layers.Dense(32, name='discriminator_dense_1')(layer)

        colors = layers.AveragePooling2D(pool_size=(32, 32), name='colors_pooling')(connected_images)
        colors = layers.Conv2D(filters=1, kernel_size=2, activation='relu',
                               padding='valid', name='colors_l_1')(colors)
        colors = layers.Flatten(name='colors_flatten')(colors)

        layer = layers.Dense(8, name='discriminator_dense_2')(layer)
        colors = layers.Dense(8, activation='relu', name='discriminator_colors_2')(colors)

        final_layer = layers.Concatenate(axis=1, name='discriminator_connected_pipes')([layer, colors])

        layer_out = layers.Dense(6, activation='sigmoid', name='discriminator_connected_pipes_l')(final_layer)
        layer_out = layers.Dense(1, activation='sigmoid', name='discriminator_evaluation')(layer_out)

        return tf.keras.Model([connected_images_in, color_image_in, gray_image_in, partial_key],
                              [layer_out], name="decoder_model")


class e2eCryptoTest(e2eCryptoGAN):

    def get_name(self):
        return 'e2eCryptoTest_' + str(self.basic_kernel_size()) + '_' + str(self.basic_dilation_rate())

    @staticmethod
    @tf.function
    def discriminator_input_prepare(arguments: dict):
        return e2eGAN.gen_discriminator_model()

    def gen_discriminator_model(self):
        return e2eGAN.gen_discriminator_model(self)