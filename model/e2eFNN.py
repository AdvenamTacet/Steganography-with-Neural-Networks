import tensorflow as tf
from tensorflow.keras import layers
import model.template as template


class e2eFNN(template.EncoderDecoder):
    """
    Simple Fully Connected model.

    Because of (obvious) memory issues, hidden layers have small size
    """
    image_W = 160
    image_H = 160

    def __init__(self):
        super().__init__()
        self.name = 'e2eFNN'
        try:
            self.load(self.name)
        except Exception as e:
            print(e)

    def gen_encoder_model(self):
        cover_image = layers.Input(shape=(self.image_H, self.image_W, 3), name='cover_image')
        hide_image  = layers.Input(shape=(self.image_H, self.image_W, 1), name='hide_image')
        n_pixels = self.image_W * self.image_H * 3

        gray_layer = layers.Conv2D(filters=1, kernel_size=1, padding='same', activation='relu', name='gray_start')(
            hide_image)
        colo_layer = layers.Conv2D(filters=3, kernel_size=1, padding='same', activation='relu', name='color_start')(
            cover_image)

        conn_layer_in = layers.Concatenate(axis=3, name='encoder_connected')([gray_layer, colo_layer])
        conn_layer = layers.Flatten(name="encoder_flatten")(conn_layer_in)

        N = 3
        for i in range(N):
            conn_layer = layers.Dense(n_pixels//(160 if i + 1 < N else 1 ), activation='relu', name="encoder_dense_{}".format(i))(conn_layer)

        out_in = layers.Reshape((self.image_W, self.image_H, 3), name="encoder_reshaped")(conn_layer)

        out = layers.Conv2D(filters=3, kernel_size=1, padding='same',
                            name='encoded_image')(out_in)
        return tf.keras.Model([cover_image, hide_image], [out], name="encoder_model")

    def gen_decoder_model(self):
        covered_image = layers.Input(shape=(self.image_H, self.image_W, 3))
        n_pixels = self.image_W * self.image_H * 3

        layer_0 = layers.Conv2D(filters=3, kernel_size=1, activation='relu',
                                padding='same', name='decoder_0')(covered_image)

        layer = layers.Flatten(name="decoder_flatten")(layer_0)

        N = 3
        for i in range(N):
            layer = layers.Dense(n_pixels//(160 if i + 1 < N else 1 ), activation='relu', name="decoder_dense_{}".format(1+i))(layer)

        out_in = layers.Reshape((self.image_W, self.image_H, 3), name="decoder_reshaped")(layer)

        out = layers.Conv2D(filters=1, kernel_size=1, padding='same', name='recovered_image')(out_in)

        return tf.keras.Model([covered_image], [out], name="decoder_model")

