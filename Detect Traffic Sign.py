# Detect Traffic Sign

import os
from keras.layers.core import Dropout
import numpy as np
import pandas as pd
import cv2
from tensorflow.keras.utils import to_categorical
import random
from keras import Sequential
from keras.layers import Dense, Flatten, MaxPooling2D, Conv2D, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

path = 'D:\MaDoc\GTSRB\Final_Training\Images'
pixels = []
labels = []

for dir in os.listdir(path):
    class_dir = os.path.join(path, dir)
    print(os.path.join(path, dir))
    info_class = os.path.join(class_dir, 'GT-' + dir + '.csv')
    print(info_class)
    dataset = pd.read_csv(info_class, sep=';')
    print(dataset)
    # for row in dataset.iterrows():
    #     print(row)
    for row in dataset.iterrows():
        pixel = cv2.imread(os.path.join(class_dir, row[1]['Filename']))
        pixel = pixel[row[1]['Roi.X1']:row[1]['Roi.X2'],
                      row[1]['Roi.Y1']: row[1]['Roi.Y2']]
        img = cv2.resize(pixel, (64, 64))
        pixels.append(img)
        labels.append(row[1]['ClassId'])

pixels = np.array(pixels)
labels = np.array(labels)
labels = to_categorical(labels)

randomize = np.arange(len(pixels))
np.random.shuffle(randomize)

X = pixels[randomize]
Y = labels[randomize]

train_size = int(X.shape[0] * 0.6)
X_train, X_val_test = X[:train_size], X[train_size:]
Y_train, Y_val_test = Y[:train_size], Y[train_size:]

test_size = int(X_val_test.shape[0] * 0.5)
X_val, X_test = X_val_test[:test_size], X_val_test[test_size:]
Y_val, Y_test = Y_val_test[: test_size], Y_val_test[test_size:]

filter_size = (3, 3)
pool_size = (2, 2)

model = Sequential()
model.add(Conv2D(16, filter_size, activation='relu',
          input_shape=(64, 64, 3), padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(16, filter_size, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.2))
model.add(Conv2D(32, filter_size, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(32, filter_size, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.2))
model.add(Conv2D(64, filter_size, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(64, filter_size, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=1e-4), metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=20, batch_size=16,
          validation_data=(X_val, Y_val))

X_new = np.array(X_test[78])
y_pre = model.predict(X_test[78:79])
y_pre = np.argmax(y_pre, axis=1)
print(y_pre)
plt.imshow(X_new)
plt.show()

# Finish
