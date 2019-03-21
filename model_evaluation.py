import os
import random
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import backend as K
from keras.losses import binary_crossentropy
from keras.models import load_model
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from tqdm import tqdm


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


def show_images(images, masks):
    plt.close('all')

    nrows = 5
    idx = random.sample(range(0, len(images)), nrows)
    fig, ax = plt.subplots(nrows=nrows, ncols=2, figsize=(8, 6))
    ax[0, 0].set_title('Masks')
    ax[0, 1].set_title('Predicted masks')
    for subplot_idx, img_idx in enumerate(idx):
        plt.gray()
        ax[subplot_idx, 0].imshow(images[img_idx].reshape(masks[img_idx].shape[0:2]))

        plt.gray()
        ax[subplot_idx, 1].imshow(masks[img_idx].reshape(masks[img_idx].shape[0:2]))
    plt.show()


show_images(Y_val, Y_val_predicted)
