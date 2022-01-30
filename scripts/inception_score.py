# calculate inception score for cifar-10 in Keras
from math import floor
from numpy import expand_dims
from numpy import log
from numpy import mean
from numpy import std
from numpy import exp
from numpy.random import shuffle
from matplotlib import pyplot
import numpy as np

# scale an array of images to a new size
from configs.inception_train_config import inception_fm_net_config
from dataloader.fashion_mnist import FashionMNISTInceptionDataset
from models.inception_model import FashionInception

WEIGHTS_PATH = "../checkpoints/inception/best/cp.ckpt"


def calc_class_frequencies(images, n_split=100):
    class_frequencies = {}
    model = FashionInception(inception_fm_net_config)
    model.load_weights(WEIGHTS_PATH).expect_partial()
    n_part = floor(images.shape[0] / n_split)
    count = 0
    for i in range(n_split):
        ix_start, ix_end = i * n_part, (i + 1) * n_part
        subset = images[ix_start:ix_end]
        count += len(subset)
        p_yx = model.predict(subset)
        for arr in p_yx:
            try:
                class_frequencies[int(np.where(arr == np.amax(arr))[0])] += 1
            except KeyError:
                class_frequencies[int(np.where(arr == np.amax(arr))[0])] = 1
    print(f"Processed {count} images...")
    print("Predicted class distribution: ", class_frequencies)


def _calc_is(p_yx, eps=1E-16):
    """
    :param p_yx: Array of marginal probabilities like
    p_yx = asarray([[0.33, 0.33, 0.33], [0.33, 0.33, 0.33], [0.33, 0.33, 0.33]])
    :param eps:
    :return:
    """
    # calculate p(y)
    p_y = expand_dims(p_yx.mean(axis=0), 0)
    # kl divergence for each image
    kl_d = p_yx * (log(p_yx + eps) - log(p_y + eps))
    # sum over classes
    sum_kl_d = kl_d.sum(axis=1)
    # average over images
    avg_kl_d = mean(sum_kl_d)
    # undo the logs
    is_score = exp(avg_kl_d)
    return is_score


# assumes images have any shape and pixels in [0,255]
def calculate_inception_score(images, n_split):
    # load inception v3 model
    model = FashionInception(inception_fm_net_config)
    model.load_weights(WEIGHTS_PATH).expect_partial()
    # enumerate splits of images/predictions
    scores = list()
    n_part = floor(images.shape[0] / n_split)
    for i in range(n_split):
        # retrieve images
        ix_start, ix_end = i * n_part, (i + 1) * n_part
        subset = images[ix_start:ix_end]
        # predict p(y|x)
        p_yx = model.predict(subset)
        is_score = _calc_is(p_yx)
        # store
        scores.append(is_score)
    # average across images
    is_avg, is_std = mean(scores), std(scores)
    return is_avg, is_std


images = FashionMNISTInceptionDataset().images
for i in range(9):
    # define subplot
    pyplot.subplot(330 + 1 + i)
    # plot raw pixel data
    pyplot.imshow(images[i], cmap=pyplot.get_cmap('gray'))
pyplot.show()

shuffle(images)
batch_size = 5000
splits = int(len(images) / batch_size)
print('loaded', images.shape)
print(f"Creating {splits} splits with {batch_size} images per split.")
# calculate inception score
is_avg, is_std = calculate_inception_score(images, n_split=splits)
print('Inception score', is_avg, "Standard deviation", is_std)
calc_class_frequencies(images, splits)
