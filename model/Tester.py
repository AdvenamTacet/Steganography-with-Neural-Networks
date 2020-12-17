import tensorflow as tf
from model.e2eGAN import e2eGAN


class Tester():
    """
        This is alpha-version of class template to evaluate quality of a model.

        Idea:
            Class to test: T
            - Creates two models T1, T2 of type T and model M  to test.
            - Model M learns to detect hidden data by model T1.
            - Type T is evaluated based on T2 (against M) performance.

        At the current form, model M is same as e2eGAN discriminator, which is not
        good choice for that purpose. Other type of model M should be implemented.

        For good ideas and better solutions, check:
            Deep Learning Hierarchical Representations for Image Steganalysis, Jiangqun Ni,
            Jian Ye, Yang YI, IEEE Transactions on Information Forensics and Security,
            06.2017,
            https://www.researchgate.net/publication/317294735_Deep_Learning_Hierarchical_Representations_for_Image_Steganalysis.
    """

    @staticmethod
    def name_with_prefix_class(T, prefix):
        """
        Returns class T' with overwritten member function gen_name
        returning prefix + T.get_name().

        :param T: base class
        :param prefix: prefix to the new name
        :return:
        """
        class PrefixedClass(T):
            def get_name(self):
                return prefix + super().get_name()

        return PrefixedClass

    @staticmethod
    def create_evaluator_class(T, base_name):
        """
        Creates class M from the Tester description, class to evaluate steganographical type T.

        :param T: type to evaluate
        :param base_name: name of evaluator model, default constructor sets to T name without prefix.
        :return:
        """
        class Evaluator(e2eGAN):
            def get_name(self):
                return 'Testers/Eval_{}'.format(base_name)

            def gen_encoder_model(self):
                return T.gen_encoder_model(self)

            def gen_decoder_model(self):
                return T.gen_decoder_model(self)

        return Evaluator

    def __init__(self, T, **kwargs):
        """
        Initialize the model.

        :param T: type to evaluate.
        :param kwargs: passed to T instances constructor.
        """

        prefix='Testers/Training_'
        self._training_instance = self.name_with_prefix_class(T, prefix)(**kwargs)
        base_name = self._training_instance.get_name()[len(prefix):]

        self._evaluator_instance = self.create_evaluator_class(T, base_name)(**kwargs)
        super().__init__()

    def train_instance(self, dataset: tf.keras.utils.Sequence, epochs : int, **kwargs):
        """
        Trains T1 model. (Look to the Tester class description.)

        :param dataset: training dataset
        :param epochs: number of epochs to train
        :param kwargs: passed to train function
        :return: pass returned value by train
        """
        return self._training_instance.train(dataset, epochs, **kwargs)

    def train_evaluator(self, dataset: tf.keras.utils.Sequence, epochs : int, **kwargs):
        """
        Train the M model. (Look to the Tester class description.)
        autosave=True

        :param dataset: training dataset
        :param epochs: number of epochs to train
        :param kwargs: passed to train function
        :return: pass returned value by train
        """
        try:
            self._evaluator_instance.load(self._training_instance.get_name(), only_generator=True)
        except:
            pass

        return self._evaluator_instance.train(dataset, epochs,
                                              train_step=self._evaluator_instance.gen_discriminator_train_step(),
                                              autosave=True,
                                              **kwargs)

    def test(self, m, dataset: tf.keras.utils.Sequence):
        """
        Takes model T2 and evaluates type T.

        :param m: model T2 of a type T,
        :param dataset: test dataset
        :return: pass returned value by train
        """
        self._evaluator_instance.load(m.get_name(), only_generator=True)

        return e2eGAN.score(self._evaluator_instance, dataset)