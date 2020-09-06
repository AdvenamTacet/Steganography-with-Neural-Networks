import tensorflow as tf
from tensorflow.keras import layers
import model.template as template


class e2eCNN(template.EncoderDecoder):
    """
    Simple Autoencoder model.

    Network model is based on the paper:
        "End-to-end Trained CNN Encode-Decoder Networks for Image Steganography"
        by
        Atique ur Rehman, Rafia Rahim, M Shahroz Nadeem, Sibt ul Hussain
        https://arxiv.org/abs/1711.07201
    """
    image_W = 160
    image_H = 160

    def __init__(self, bks=3):
        """
        Initialize the model.
        Saves a size of the kernel.

        :param bks: this value will be returned by self.basic_kernel_size and describes kernel size in some layers
        """
        self.bks = bks
        super().__init__()

        self.name = 'e2eCNN_' + str(self.basic_kernel_size())
        try:
            self.load(self.name)
        except Exception as e:
            print(e)

    def basic_kernel_size(self):
        return self.bks

    def gen_encoder_model(self):
        bks = self.basic_kernel_size()
        cover_image = layers.Input(shape=(self.image_H, self.image_W, 3), name='cover_image')
        hide_image  = layers.Input(shape=(self.image_H, self.image_W, 1), name='hide_image')

        gray_layer = layers.Conv2D(filters=1, kernel_size=1, padding='same', activation='relu', name='gray_start')(
            hide_image)
        conn_layer = layers.Conv2D(filters=3, kernel_size=1, padding='same', activation='relu', name='color_start')(
            cover_image)

        N = 4
        for i in range(N):
            conn_layer_in = layers.Concatenate(axis=3, name='encoder_in_' + str(i))([gray_layer, conn_layer])

            conn_layer = layers.Conv2D(filters=16, kernel_size=bks, padding='same', activation='relu',
                                       name='encoder_color_A_' + str(i))(conn_layer_in)
            gray_layer = layers.Conv2D(filters=16, kernel_size=bks, padding='same', activation='relu',
                                       name='encoder_gray_A_' + str(i))(gray_layer)

            conn_layer = layers.Conv2D(filters=16, kernel_size=(bks if i+1 < N else 1), activation='relu',
                                       padding='same', name='encoder_color_B_' + str(i))(conn_layer)

            if i + 1 < N:
                gray_layer = layers.Conv2D(filters=16, kernel_size=bks, padding='same', activation='relu',
                                           name='encoder_gray_B_' + str(i))(gray_layer)

        layer_1 = layers.Conv2D(filters=8, kernel_size=1, padding='same', activation='relu',
                                name='encoder_l_1')(conn_layer)

        out = layers.Conv2D(filters=3, kernel_size=1, padding='same',
                            name='encoded_image')(layer_1)
        return tf.keras.Model([cover_image, hide_image], [out], name="encoder_model")

    def gen_decoder_model(self):
        bks = self.basic_kernel_size()
        covered_image = layers.Input(shape=(self.image_H, self.image_W, 3))

        layer_0 = layers.Conv2D(filters=3, kernel_size=1, activation='relu',
                                padding='same', name='decoder_l_0')(covered_image)

        layer_1 = layers.Conv2D(filters=16, kernel_size=bks, activation='relu',
                                padding='same', name='decoder_l_1')(layer_0)

        layer_2 = layers.Conv2D(filters=16, kernel_size=bks, activation='relu',
                                padding='same', name='decoder_l_2')(layer_1)

        layer_3 = layers.Conv2D(filters=8, kernel_size=bks, activation='relu',
                                padding='same', name='decoder_l_3')(layer_2)

        layer_4 = layers.Conv2D(filters=8, kernel_size=bks, activation='relu',
                                padding='same', name='decoder_l_4')(layer_3)

        layer_5 = layers.Conv2D(filters=3, kernel_size=bks, activation='relu',
                                padding='same', name='decoder_l_5')(layer_4)

        out = layers.Conv2D(filters=1, kernel_size=1, padding='same', name='recovered_image')(layer_5)

        return tf.keras.Model([covered_image], [out], name="decoder_model")
