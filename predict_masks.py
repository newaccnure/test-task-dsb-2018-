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


# Set some parameters
IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3
TEST_PATH = './input/stage1_test/'
warnings.filterwarnings('ignore', category=UserWarning, module='skimage')

model = load_model('model-dsbowl2018-1.h5', custom_objects={'bce_dice_loss': bce_dice_loss, 'dice_coef': dice_coef})

test_ids = next(os.walk(TEST_PATH))[1]
# Get and resize test images
X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)

print('Getting and resizing test images ... ')

sys.stdout.flush()
for i, test_id in enumerate(os.listdir(TEST_PATH)):  # loop through test_ids in the test_path
    img = imread('{0}/{1}/images/{1}.png'.format(TEST_PATH, test_id))[:, :, :IMG_CHANNELS]
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_test[i] = img

print('Done!')

print('Predicting masks ...')

# Predict on val
Y_test_predicted = model.predict(X_test, verbose=1)
# Threshold predictions
Y_test_predicted = (Y_test_predicted > 0.5).astype(np.uint8)

print('Done!')

print('Upsampling masks to original size and saving results ...')

# Upsample Y_test back to the original X_test size (height and width)
Y_test_upsampled = []
for i, test_id in enumerate(os.listdir(TEST_PATH)):  # loop through test_ids in the test_path
    img = imread('{0}/{1}/images/{1}.png'.format(TEST_PATH, test_id))  # read original test image directly from path
    img_upscaled = resize(Y_test_predicted[i], (img.shape[0], img.shape[1]), mode='constant',
                          preserve_range=True)  # upscale Y_test image according to original test image
    Y_test_upsampled.append(img_upscaled)  # append upscaled image to Y_test_upsampled
    plt.imsave('./output/{0}.jpg'.format(test_id), np.dstack((img_upscaled, img_upscaled, img_upscaled)))

print('Done!')

#
original_images = []
for i, test_id in enumerate(os.listdir(TEST_PATH)):  # loop through test_ids in the test_path
    img = imread('{0}/{1}/images/{1}.png'.format(TEST_PATH, test_id))  # read original test image directly from path
    original_images.append(img)  # append upscaled image to Y_test_upsampled


def show_images(images, masks):
    plt.close('all')

    nrows = 5
    idx = random.sample(range(0, len(images)), nrows)
    fig, ax = plt.subplots(nrows=nrows, ncols=2, figsize=(8, 6))
    ax[0, 0].set_title('Images')
    ax[0, 1].set_title('Masks')
    for subplot_idx, img_idx in enumerate(idx):
        ax[subplot_idx, 0].imshow(images[img_idx])

        plt.gray()
        ax[subplot_idx, 1].imshow(masks[img_idx].reshape(masks[img_idx].shape[0:2]))


show_images(original_images, Y_test_upsampled)
plt.show()
