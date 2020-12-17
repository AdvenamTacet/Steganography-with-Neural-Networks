import tensorflow as tf
from tensorflow.keras import layers
import model.template as template


class SGAN(template.GAN):
    """
    Generative adversarial network model.

    Network model is inspired by the paper:
        "SteganoGAN: High Capacity Image Steganography with GANs"
        by
        Kevin Alex Zhang, Alfredo Cuesta-Infante, Lei Xu, Kalyan Veeramachaneni
        https://arxiv.org/abs/1901.03892

    Yet, I intentionally skipped a few parts of the model, so if you want to check
    real performance of SteganGAN mode, check original implemntation by authors:
        https://github.com/DAI-Lab/SteganoGAN/tree/master/steganogan
    """
    image_W = 160
    image_H = 160

    def __init__(self):
        super().__init__()

    def get_name(self):
        return 'SGAN'

    @staticmethod
    def discriminator_input_prepare(arguments):
        return [arguments['encoder_output'], arguments['color']]

    def gen_encoder_model(self):
        cover_image = layers.Input(shape=(self.image_H, self.image_W, 3), name='cover_image')
        hide_image  = layers.Input(shape=(self.image_H, self.image_W, 1), name='hide_image')

        gray_layer_1 = layers.Conv2D(filters=1, kernel_size=1, padding='same', activation='relu', name='encoder_1_gray')(hide_image)
        conn_layer_1 = layers.Conv2D(filters=3, kernel_size=1, padding='same', activation='relu', name='encoder_1_conn')(cover_image)

        gray_layer_2 = layers.Conv2D(filters=6, kernel_size=4, padding='same', activation='relu', name='encoder_2_gray')(gray_layer_1)
        conn_layer_2 = layers.Conv2D(filters=6, kernel_size=4, padding='same', activation='relu', name='encoder_2_conn')(conn_layer_1)

        layer_3_in  = layers.Concatenate(axis=3, name='encoder_3_in')([gray_layer_2, conn_layer_2])
        layer_3_out = layers.Conv2D(filters=6, kernel_size=4, padding='same', activation='relu', name='encoder_3_out')(layer_3_in)

        layer_4 = layers.Conv2D(filters=6, kernel_size=4, padding='same', activation='relu', name='encoder_4')(layer_3_out)

        layer_5 = layers.Conv2D(filters=3, kernel_size=4, padding='same', activation='relu', name='encoder_5')(layer_4)

        layer_6_in = layers.Concatenate(axis=3, name='encoder_6_in')([layer_5, cover_image])
        layer_6_out = layers.Conv2D(filters=3, kernel_size=4, padding='same', name='layer_6_out')(layer_6_in)

        out = layers.Conv2D(filters=3, kernel_size=1, padding='same', name='encoded_image')(layer_6_out)

        return tf.keras.Model([cover_image, hide_image], [out], name="encoder_model")

    def gen_decoder_model(self):
        covered_image = layers.Input(shape=(self.image_H, self.image_W, 3))

        layer_0 = layers.Conv2D(filters=3, kernel_size=1,
                                padding='same', name='decoder_0')(covered_image)

        layer_1 = layers.Conv2D(filters=16, kernel_size=4, activation='relu',
                                padding='same', name='decoder_1')(layer_0)

        layer_2 = layers.Conv2D(filters=16, kernel_size=4, activation='relu',
                                padding='same', name='decoder_2')(layer_1)

        layer_3 = layers.Conv2D(filters=8, kernel_size=4, activation='relu',
                                padding='same', name='decoder_3')(layer_2)

        layer_4 = layers.Conv2D(filters=8, kernel_size=4, activation='relu',
                                padding='same', name='decoder_4')(layer_3)

        layer_5 = layers.Conv2D(filters=3, kernel_size=4, activation='relu',
                                padding='same', name='decoder_5')(layer_4)

        out = layers.Conv2D(filters=1, kernel_size=1, padding='same', name='recovered_image')(layer_5)

        return tf.keras.Model([covered_image], [out], name="decoder_model")

    def gen_discriminator_model(self):
        covered_image = layers.Input(shape=(self.image_H, self.image_W, 3), name='discriminator_input_a')
        cover_image = layers.Input(shape=(self.image_H, self.image_W, 3), name='discriminator_input_b')

        layer_0_a = layers.Conv2D(filters=3, kernel_size=1, activation='relu',
                                  padding='same', name='discriminator_0_a')(covered_image)
        layer_0_b = layers.Conv2D(filters=3, kernel_size=1, activation='relu',
                                  padding='same', name='discriminator_0_b')(cover_image)

        layer_0_in = layers.Concatenate(name='discriminator_1_in')([layer_0_a, layer_0_b])
        layer_1_out = layers.Conv2D(filters=16, kernel_size=4, activation='relu',
                                    padding='same', name='discriminator_1_out')(layer_0_in)

        layer_2 = layers.Conv2D(filters=16, kernel_size=4, activation='relu',
                                padding='same', name='discriminator_2')(layer_1_out)

        layer_3 = layers.Conv2D(filters=8, kernel_size=4, activation='relu',
                                padding='same', name='discriminator_3')(layer_2)

        layer_4 = layers.Conv2D(filters=8, kernel_size=4, activation='relu',
                                padding='same', name='discriminator_4')(layer_3)

        layer_5 = layers.Conv2D(filters=3, kernel_size=4, activation='relu',
                                padding='same', name='discriminator_5')(layer_4)

        layer_6 = layers.Conv2D(filters=1, kernel_size=1, padding='same', name='discriminator_6')(layer_5)

        layer_7_in = layers.Flatten(name='discriminator_7_flatten')(layer_6)
        layer_7_out = layers.Dense(1,  name='discriminator_evaluation')(layer_7_in)

        return tf.keras.Model([covered_image, cover_image], [layer_7_out], name="discriminator_model")