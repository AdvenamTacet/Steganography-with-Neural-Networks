import tensorflow as tf
import math


class ShuffleSequence(tf.keras.utils.Sequence):

    def __init__(self, data, batch_size, gen_keys: bool = False, dropout_rate: float = 0.7):
        self.color, self.gray = data[0].copy(), data[1].copy()

        self.batch_size = batch_size
        self.gen_keys = gen_keys
        self.dropout_rate = dropout_rate

    def __len__(self):
        return math.ceil(len(self.color) / self.batch_size)

    def __getitem__(self, idx):
        from PIL import Image
        import numpy as np

        def path_to_tensor(path):
            img = Image.open(path).resize((240, 160))
            arr = np.array(img)
            tensor = tf.convert_to_tensor(arr[:160, :160], dtype=float)
            img.close()

            return tensor

        batch_color = self.color[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_gray  = self.gray[ idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_color = list(map(path_to_tensor, batch_color))
        batch_gray  = list(map(path_to_tensor, batch_gray))
        ret = {'color': tf.stack(batch_color) / 255., 'gray': tf.image.rgb_to_grayscale(tf.stack(batch_gray)) / 255.}

        if self.gen_keys:
            ret.update({'key': tf.random.normal(ret['gray'].shape)})
            ret.update({'partial_key': tf.nn.dropout(ret['key'], self.dropout_rate)})

        return ret

    def on_epoch_end(self):
        self.gray = self.gray[-1:] + self.gray[:-1]

    def take(self, n=None):
        if n is not None:
            assert self.batch_size >= n, "Cannot take more samples than in one batch"

        ret = self.__getitem__(0)

        if n is None:
            return ret

        return {key: ret[key][:n] for key in ret}
        # return {'color' : ret['color'][:n], 'gray' : ret['gray'][:n]}


def parse_dataset(dataset_name='challenge2018'):
    import os.path
    import glob
    from PIL import Image
    data_dir = '../data'
    index_file = '{}/{}.index'.format(data_dir, dataset_name)
    if os.path.isfile(index_file):
        with open(index_file, 'r') as f:
            return [(path, (int(x), int(y))) for path, x, y in [line.split(';') for line in f.read().split('\n') if len(line) > 1] ]

    ret = []

    for path in glob.iglob('{}/{}/*.jpg'.format(data_dir, dataset_name)):
        img = Image.open(path)
        ret.append((path, img.size))
        img.close()

    with open(index_file, 'w') as f:
        for path, (x, y) in ret:
            f.write('{};{};{}\n'.format(path, x, y))

    return ret


def get_data(test_fraction=0.3):
    metadata = parse_dataset()
    paths = [path for path, size in metadata if size == (1024, 682)]

    train_n = int(len(paths) * (1-test_fraction))
    train, test = paths[:train_n], paths[train_n:]

    if len(train) % 2 == 1:
        train = train[:-1]
    if len(test) % 2 == 1:
        test = test[:-1]

    return (train[::2], train[1::2]), (test[::2], test[1::2])
