import tensorflow as tf
from tensorflow.keras import layers
import model.template as template
import model.e2eCNN


class e2eGAN(template.GAN):
    """
    Model from e2eCNN extended with discriminator.
    """
    image_W = 160
    image_H = 160

    def __init__(self, bks=3, dr=1):
        """
        Initialize the model.
        Saves a size of the kernel and dilation rate of the model.

        :param bks: this value will be returned by self.basic_kernel_size
        :param dr: this value will be returned by self.basic_dilation_rate
        """
        self.__bks = bks
        self.__dr = dr
        super().__init__()

    def get_name(self):
        return 'e2eGAN_' + str(self.basic_kernel_size()) + '_' + str(self.basic_dilation_rate())

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

    def gen_encoder_model(self):
        return model.e2eCNN.e2eCNN.gen_encoder_model(self)

    def gen_decoder_model(self):
        return model.e2eCNN.e2eCNN.gen_decoder_model(self)

    def gen_discriminator_model(self):
        bks = self.basic_kernel_size()
        dr = self.basic_dilation_rate()

        connected_images_in = layers.Input(shape=(self.image_H, self.image_W, 3), name="connected_images")

        connected_images = layers.Conv2D(filters=3, kernel_size=1, dilation_rate=dr, activation='relu',
                                    padding='same', name='connected_images_scale')(connected_images_in)

        layer = layers.Conv2D(filters=16, kernel_size=bks, dilation_rate=dr, strides=min(bks, 2), activation='relu',
                              padding='valid', name='discriminator_l_0')(connected_images)

        layer = layers.Conv2D(filters=16, kernel_size=bks, dilation_rate=dr, strides=min(bks, 2), activation='relu',
                              padding='valid', name='discriminator_l_1')(layer)

        layer = layers.MaxPooling2D(pool_size=(2, 2), name='discriminator_pooling_1')(layer)

        layer = layers.Conv2D(filters=16, kernel_size=bks, dilation_rate=dr, activation='relu',
                              padding='valid', name='discriminator_l_2')(layer)

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

        return tf.keras.Model([connected_images_in], [layer_out], name="decoder_model")