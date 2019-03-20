import os
import sys
import random
import warnings

import numpy as np

import pandas as pd
import tensorflow as tf

from tqdm import tqdm
from itertools import chain

import matplotlib.pyplot as plt
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras.losses import binary_crossentropy
from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.losses import binary_crossentropy


# Custom Dice coef metric
def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


# Custom loss function
def bce_dice_loss(y_true, y_pred):
    return 0.5 * binary_crossentropy(y_true, y_pred) - dice_coef(y_true, y_pred)


def dice(im1, im2):
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / (im1.sum() + im2.sum())


# Set some parameters
IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3
TRAIN_PATH = './input/stage1_train/'
TEST_PATH = './input/stage1_test/'

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')

# Set seed values
seed = 42
random.seed = seed
np.random.seed(seed=seed)

# Get train IDs
train_ids = next(os.walk(TRAIN_PATH))[1]

# Get and resize train images and masks
X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
print('Getting and resizing train images and masks ... ')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    path = TRAIN_PATH + id_
    img = imread('{0}/{1}/images/{1}.png'.format(TRAIN_PATH, id_))[:, :, :IMG_CHANNELS]
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_train[n] = img
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    for mask_file in next(os.walk(path + '/masks/'))[2]:
        mask_ = imread(path + '/masks/' + mask_file)
        mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant',
                                      preserve_range=True), axis=-1)
        mask = np.maximum(mask, mask_)
    Y_train[n] = mask

validation_split = 0.1
X_train, X_val, Y_train, Y_val = train_test_split(X_train,
                                                  Y_train,
                                                  train_size=1 - validation_split,
                                                  test_size=validation_split,
                                                  random_state=seed)

model = load_model('model-dsbowl2018-1.h5', custom_objects={'bce_dice_loss': bce_dice_loss, 'dice_coef': dice_coef})

Y_val_predicted = model.predict(X_val, verbose=1)
Y_val_predicted = Y_val_predicted > 0.5

dice_coefs = []

for i in range(Y_val.shape[0]):
    dice_coefs.append(dice(Y_val[i], Y_val_predicted[i]))

dice_coefs = pd.Series(dice_coefs)

print('Evaluation of predicted masks on validation set(dice coefficients):')
print(dice_coefs.describe())

print('Number of dice coefficients below 0.4: {0}'.format((dice_coefs < 0.4).sum()))
