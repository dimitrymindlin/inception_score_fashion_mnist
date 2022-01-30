import numpy as np
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy.random import shuffle
from scipy.linalg import sqrtm

from configs.inception_train_config import inception_fm_net_config
from dataloader.fashion_mnist import FashionMNISTInceptionDataset
from models.inception_model import FashionInception

WEIGHTS_PATH = "../checkpoints/inception/best/cp.ckpt"


def load_model_without_clf_layer():
    model1 = FashionInception(inception_fm_net_config, include_top=False)
    model1.load_weights(WEIGHTS_PATH).expect_partial()
    return model1


def calculate_fid(model, images1, images2):
    # calculate activations
    act1 = model.predict(images1)
    act2 = model.predict(images2)
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


model = load_model_without_clf_layer()
images = FashionMNISTInceptionDataset().images

shuffle(images)
fid = calculate_fid(model, images[:500], images[500:1000])
print(fid)
