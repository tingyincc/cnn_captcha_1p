#!/usr/bin/env python

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from PIL import Image
import numpy as np
import csv
import os
import cv2
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utils import hueChange
import datetime

LETTERSTR = "123456789ABCDEFGHJKLMNPQRSTUVWXYZefqw"


def toonehot(text):
    labellist = []
    for letter in text:
        onehot = [0 for _ in range(37)]
        num = LETTERSTR.find(letter)
        onehot[num] = 1
        labellist.append(onehot)
    return labellist


inputData = Input((20, 20, 1))
out = inputData
out = Conv2D(filters=32, kernel_size=(3, 3),
             padding='same', activation='relu')(out)
out = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(out)
out = BatchNormalization()(out)
out = MaxPooling2D(pool_size=(2, 2))(out)
out = Dropout(0.5)(out)
out = Conv2D(filters=64, kernel_size=(3, 3),
             padding='same', activation='relu')(out)
out = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(out)
out = BatchNormalization()(out)
out = MaxPooling2D(pool_size=(2, 2))(out)
out = Dropout(0.5)(out)
out = Conv2D(filters=128, kernel_size=(3, 3),
             padding='same', activation='relu')(out)
out = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(out)
out = BatchNormalization()(out)
out = MaxPooling2D(pool_size=(1, 1))(out)
out = Dropout(0.5)(out)
out = Conv2D(filters=256, kernel_size=(1, 1), activation='relu')(out)
out = BatchNormalization()(out)
out = MaxPooling2D(pool_size=(1, 1))(out)
out = Flatten()(out)
out = Dropout(0.5)(out)
out = [Dense(37, name='digit', activation='softmax')(out)]
model = Model(inputs=inputData, outputs=out)
model.compile(loss='categorical_crossentropy',
              optimizer='Adam', metrics=['accuracy'])
model.summary()


datagen = ImageDataGenerator(
    rotation_range=0,
    width_shift_range=0,
    height_shift_range=0,
    shear_range=0,
    zoom_range=0,
    horizontal_flip=False,
    vertical_flip=False,
    preprocessing_function=hueChange,
    fill_mode='nearest')


train_data = []
train_label = [[] for _ in range(1)]
source_dir = './train_data/'
imageFiles = os.walk(source_dir)
for dirpath, dirs, files in imageFiles:
    for f in files:
        fullpath = os.path.join(dirpath, f)
        image = cv2.imread(fullpath, 0)
        nparr = np.array(image)
        nparr = np.expand_dims(nparr, axis=2)
        nparr = nparr / 255.0
        train_data.append(nparr)

        dirname = dirpath.split(os.path.sep)[-1][-1]
        onehot = toonehot(dirname)
        train_label[0].append(onehot[0])
train_data = np.stack(train_data)
print(train_data.shape)
train_label = [arr for arr in np.asarray(train_label)]
print(np.array(train_label).shape)


vali_data = []
vali_label = [[] for _ in range(1)]
source_dir = './valid_data/'
imageFiles = os.walk(source_dir)
for dirpath, dirs, files in imageFiles:
    for f in files:
        fullpath = os.path.join(dirpath, f)
        image = cv2.imread(fullpath, 0)
        nparr = np.array(image)  # 轉成np array
        nparr = np.expand_dims(nparr, axis=2)
        nparr = nparr / 255.0
        vali_data.append(nparr)
        dirname = dirpath.split(os.path.sep)[-1][-1]
        onehot = toonehot(dirname)
        vali_label[0].append(onehot[0])
vali_data = np.stack(vali_data)
vali_label = [arr for arr in np.asarray(vali_label)]

filepath = "./model.h5"
checkpoint = ModelCheckpoint('ep{epoch: 03d}-loss{loss: .3f}-val_loss{val_loss: .3f}.h5',
                             monitor='val_loss', save_best_only=True, save_weight_only=False, period=1)
# earlystop = EarlyStopping( monitor = 'val_loss', patience = 5, verbose = 1, mode = 'auto')
log_dir = r"logs/train/plugins/profile/" + \
    datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
try:
    os.makedirs(log_dir)
except FileExistsError:
    pass
tensorBoard = TensorBoard(log_dir="./logs", histogram_freq=1)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', factor=0.3, patience=30, verbose=1, min_lr=1e-5)
callbacks_list = [checkpoint, tensorBoard]


# datagen.fit(train_data)
# train_history = model.fit_generator(datagen.flow(train_data, train_label[0], batch_size=1000), samples_per_epoch=(len(train_data)*1000),epochs=50, validation_data=(vali_data, vali_label[0]), callbacks=callbacks_list)

model.fit(train_data, train_label, batch_size=40, epochs=300, verbose=2,
          validation_data=(vali_data, vali_label), callbacks=callbacks_list, shuffle=True)
