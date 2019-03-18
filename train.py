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


# Build U-Net model
def Conv2d_3x3(filters):
    return Conv2D(filters, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')


def Conv2DTranspose_2x2(filters):
    return Conv2DTranspose(filters, kernel_size=(2, 2), strides=(2, 2), padding='same')


inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
s = Lambda(lambda x: x / 255)(inputs)

c1 = Conv2d_3x3(16)(s)
c1 = Dropout(0.1)(c1)
c1 = Conv2d_3x3(16)(c1)
p1 = MaxPooling2D((2, 2))(c1)

c2 = Conv2d_3x3(32)(p1)
c2 = Dropout(0.1)(c2)
c2 = Conv2d_3x3(32)(c2)
p2 = MaxPooling2D((2, 2))(c2)

c3 = Conv2d_3x3(64)(p2)
c3 = Dropout(0.2)(c3)
c3 = Conv2d_3x3(64)(c3)
p3 = MaxPooling2D((2, 2))(c3)

c4 = Conv2d_3x3(128)(p3)
c4 = Dropout(0.2)(c4)
c4 = Conv2d_3x3(128)(c4)
p4 = MaxPooling2D(pool_size=(2, 2))(c4)

c5 = Conv2d_3x3(256)(p4)
c5 = Dropout(0.3)(c5)
c5 = Conv2d_3x3(256)(c5)

u6 = Conv2DTranspose_2x2(128)(c5)
u6 = concatenate([u6, c4])
c6 = Conv2d_3x3(128)(u6)
c6 = Dropout(0.2)(c6)
c6 = Conv2d_3x3(128)(c6)

u7 = Conv2DTranspose_2x2(64)(c6)
u7 = concatenate([u7, c3])
c7 = Conv2d_3x3(64)(u7)
c7 = Dropout(0.2)(c7)
c7 = Conv2d_3x3(64)(c7)

u8 = Conv2DTranspose_2x2(32)(c7)
u8 = concatenate([u8, c2])
c8 = Conv2d_3x3(32)(u8)
c8 = Dropout(0.1)(c8)
c8 = Conv2d_3x3(32)(c8)

u9 = Conv2DTranspose_2x2(16)(c8)
u9 = concatenate([u9, c1], axis=3)
c9 = Conv2d_3x3(16)(u9)
c9 = Dropout(0.1)(c9)
c9 = Conv2d_3x3(16)(c9)

outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

model = Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss=bce_dice_loss, metrics=[dice_coef])
model.summary()


# Runtime data augmentation
def get_train_val_augmented(X_data, Y_data, validation_split=0.1, batch_size=32, seed=seed):
    X_train, X_val, Y_train, Y_val = train_test_split(X_data,
                                                      Y_data,
                                                      train_size=1 - validation_split,
                                                      test_size=validation_split,
                                                      random_state=seed)

    # Image data generator distortion options
    data_gen_args = dict(rotation_range=45.,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         shear_range=0.2,
                         zoom_range=0.2,
                         horizontal_flip=True,
                         vertical_flip=True,
                         fill_mode='reflect')

    # Train data, provide the same seed and keyword arguments to the fit and flow methods
    X_datagen = ImageDataGenerator(**data_gen_args)
    Y_datagen = ImageDataGenerator(**data_gen_args)
    X_datagen.fit(X_train, augment=True, seed=seed)
    Y_datagen.fit(Y_train, augment=True, seed=seed)
    X_train_augmented = X_datagen.flow(X_train, batch_size=batch_size, shuffle=True, seed=seed)
    Y_train_augmented = Y_datagen.flow(Y_train, batch_size=batch_size, shuffle=True, seed=seed)

    # Validation data, no data augmentation, but we create a generator anyway
    X_datagen_val = ImageDataGenerator()
    Y_datagen_val = ImageDataGenerator()
    X_datagen_val.fit(X_val, augment=True, seed=seed)
    Y_datagen_val.fit(Y_val, augment=True, seed=seed)
    X_val_augmented = X_datagen_val.flow(X_val, batch_size=batch_size, shuffle=True, seed=seed)
    Y_val_augmented = Y_datagen_val.flow(Y_val, batch_size=batch_size, shuffle=True, seed=seed)

    # combine generators into one which yields image and masks
    train_generator = zip(X_train_augmented, Y_train_augmented)
    val_generator = zip(X_val_augmented, Y_val_augmented)

    return train_generator, val_generator, X_train, X_val, Y_train, Y_val


batch_size = 16

train_generator, val_generator, X_train, X_val, Y_train, Y_val = get_train_val_augmented(X_data=X_train, Y_data=Y_train,
                                                                                         validation_split=0.1,
                                                                                         batch_size=batch_size)

validation_steps = 0.1 * len(X_train) / batch_size
steps_per_epoch = len(X_train) / (batch_size * 2)
checkpointer = ModelCheckpoint('model-dsbowl2018-1.h5', verbose=1, save_best_only=True)

model.fit_generator(train_generator, validation_data=val_generator,
                    validation_steps=validation_steps, steps_per_epoch=steps_per_epoch,
                    epochs=30, callbacks=[checkpointer])


def dice(im1, im2):
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / (im1.sum() + im2.sum())


Y_val_predicted = model.predict(X_val, verbose=1)
Y_val_predicted = Y_val_predicted > 0.5

dice_coefs = np.array([])
for i in range(Y_val.shape[0]):
    dice_coefs = np.append(dice_coefs, dice(Y_val[i], Y_val_predicted[i]))

id = np.argmin(dice_coefs)

imshow(X_val[id])
plt.show()
imshow(Y_val[id][:, :, 0])
plt.show()
imshow(Y_val_predicted[id][:, :, 0])
plt.show()
