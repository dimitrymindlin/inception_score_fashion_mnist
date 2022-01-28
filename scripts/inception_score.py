# calculate inception score for cifar-10 in Keras
from math import floor
from numpy import expand_dims, ones
from numpy import log
from numpy import mean
from numpy import std
from numpy import exp
from numpy.random import shuffle
from keras.applications.inception_v3 import preprocess_input
from skimage.transform import resize
from numpy import asarray

# scale an array of images to a new size
from configs.inception_train_config import inception_fm_net_config
from dataloader.fashion_mnist import FashionMNISTInceptionDataset
from models.inception_model import FashionInception

WEIGHTS_PATH = "checkpoints/inception/best/cp.ckpt"


def calc_class_frequency(n_split=100, eps=1E-16):
    model = FashionInception(inception_fm_net_config)
    model.load_weights(WEIGHTS_PATH).expect_partial()
    # enumerate splits of images/predictions
    scores = list()
    n_part = floor(images.shape[0] / n_split)
    for i in range(n_split):
        if i == 5:
            break
        # retrieve images
        ix_start, ix_end = i * n_part, (i + 1) * n_part
        subset = images[ix_start:ix_end]
        # convert from uint8 to float32
        subset = subset.astype('float32')
        # scale images to the required size
        subset = scale_images(subset, (75, 75, 3))
        # pre-process images, scale to [-1,1]
        subset = preprocess_input(subset)
        # predict p(y|x)
        p_yx = model.predict(subset)
        print(p_yx)


def scale_images(images, new_shape):
    images_list = list()
    for image in images:
        # resize with nearest neighbor interpolation
        new_image = resize(image, new_shape, 0)
        # store
        images_list.append(new_image)
    return asarray(images_list)


# assumes images have any shape and pixels in [0,255]
def calculate_inception_score(images, n_split=100000, eps=1E-16):
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
        # convert from uint8 to float32
        subset = subset.astype('float32')
        # scale images to the required size
        subset = scale_images(subset, (75, 75, 3))
        # pre-process images, scale to [-1,1]
        subset = preprocess_input(subset)
        # predict p(y|x)
        p_yx = model.predict(subset)  # TODO: Not working on my mac...
        # calculate p(y)
        p_y = expand_dims(p_yx.mean(axis=0), 0)
        # calculate KL divergence using log probabilities
        kl_d = p_yx * (log(p_yx + eps) - log(p_y + eps))
        # sum over classes
        sum_kl_d = kl_d.sum(axis=1)
        # average over images
        avg_kl_d = mean(sum_kl_d)
        # undo the log
        is_score = exp(avg_kl_d)
        # store
        scores.append(is_score)
    # average across images
    is_avg, is_std = mean(scores), std(scores)
    return is_avg, is_std


print("Crunch")
images = FashionMNISTInceptionDataset().images
# shuffle images
shuffle(images)
batch_size = 8
splits = (len(images) + 15 - 1) // batch_size
print(f"Trying to create {splits} splits.")
print('loaded', images.shape)
# calculate inception score
is_avg, is_std = calculate_inception_score(images, n_split=splits)
print('score', is_avg, is_std)
