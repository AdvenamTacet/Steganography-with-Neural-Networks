import tensorflow as tf
from tensorflow.keras import layers
import model.template as template
import model.e2eCNN

class e2eGAN(template.GAN):
    """
    Model from e2eCNN extended with discriminator.

    Discriminator does not support dilation convolutions.
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

        self.name = 'e2eGAN_' + str(self.basic_kernel_size()) + '_' + str(self.basic_dilation_rate())
        try:
            self.load(self.name)
        except Exception as e:
            print(e)

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
        covered_image = layers.Input(shape=(self.image_H, self.image_W, 3), name="covered_image")

        clean_image = layers.Conv2D(filters=3, kernel_size=1, activation='relu',
                                    padding='same', name='clean_image_0')(covered_image)
        layer = layers.Conv2D(filters=3, kernel_size=bks, activation='relu',
                              padding='same', name='discriminator_0')(clean_image)

        N = 4
        for i in range(N):
            layer_in = layers.Concatenate(axis=3, name='discriminator_in_' + str(i+1))([layer, clean_image])
            layer = layers.Conv2D(filters=3, kernel_size=bks, activation='relu',
                                  padding='same', name='discriminator_' + str(i+1))(layer_in)
            if i + 1 < N:
                layer = layers.MaxPool2D(pool_size=(2, 2), name='discriminator_pool_{}'.format(i+1))(layer)
                clean_image = layers.AveragePooling2D(pool_size=(2, 2), name='clean_image_{}'.format(i+1))(clean_image)

        layer_flat = layers.Flatten(name='discriminator_flatten')(layer)
        layer_d1 = layers.Dense(128, activation='relu', name='discriminator_dense_1')(layer_flat)

        layer_d2 = layers.Dense(64, activation='softmax', name='discriminator_dense_2')(layer_d1)

        layer_d3 = layers.Dense(8, activation='softmax', name='discriminator_dense_3')(layer_d2)

        layer_out = layers.Dense(1, activation='sigmoid', name='discriminator_evaluation')(layer_d3)

        return tf.keras.Model([covered_image], [layer_out], name="decoder_model")