import itertools
import numpy as np


def print_one_transformation(imgs):
    import matplotlib.pyplot as plt

    assert len(imgs) == 4, "4 images needed"

    fig, axes = plt.subplots(2, 2)

    imgs = [ np.squeeze(x) for x in imgs]

    axes[0, 0].imshow(imgs[0])
    axes[0, 1].imshow(imgs[1], cmap='gray')
    axes[1, 0].imshow(imgs[2])
    axes[1, 1].imshow(imgs[3], cmap='gray')


def print_batch(color, gray, covered, recovered, n=10, skip=0):
    for imgs in itertools.islice(zip(color, gray, covered, recovered), skip, skip+n):
        print_one_transformation(imgs)


def print_from_set(m, S, n=10, skip=0):
    subset = S.take(n)
    covered = m.encoder(subset)
    recovered = m.decoder(covered)
    color, gray = subset['color'], subset['gray']

    print_batch(color, gray, covered, recovered, n=n, skip=skip)