import os
import numpy as np
import pickle

from sklearn.model_selection import train_test_split

import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.callbacks import ModelCheckpoint

from model import create_model

batch_size = 32
num_classes = 10
epochs = 200
data_augmentation = True
dirname = 'model1'

# create directories to save the model
if not os.path.exists('saved_models/' + dirname):
    os.makedirs('saved_models/' + dirname)

model_path = 'saved_models/' + dirname + '/cifar10_cnn.hdf5'

# load and split the data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train,
                                                      test_size=0.2,
                                                      stratify=y_train,
                                                      shuffle=True,
                                                      random_state=42)
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_valid.shape[0], 'valid samples')
print(x_test.shape[0], 'test samples')

y_train = keras.utils.to_categorical(y_train, num_classes)
y_valid = keras.utils.to_categorical(y_valid, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x_train = x_train.astype('float32')
x_valid = x_valid.astype('float32')
x_test = x_test.astype('float32')

# standardization
x_train /= 255
x_valid /= 255
x_test /= 255
x_train_mean = np.mean(x_train, axis=0)
x_train -= x_train_mean
x_valid -= x_train_mean
x_test -= x_train_mean

# construct the model
model = create_model(x_train.shape[1:], num_classes=num_classes)
opt = keras.optimizers.Adam()
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
checkpoint = ModelCheckpoint(model_path, monitor='val_loss', verbose=2,
                             save_best_only=True, mode='auto')

if not data_augmentation:
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        callbacks=[checkpoint],
                        validation_data=(x_valid, y_valid),
                        shuffle=True, verbose=2)
else:
    datagen = ImageDataGenerator(featurewise_center=False,
                                 samplewise_center=False,
                                 featurewise_std_normalization=False,
                                 samplewise_std_normalization=False,
                                 zca_whitening=False,
                                 rotation_range=0.,
                                 width_shift_range=0.1,
                                 height_shift_range=0.1,
                                 horizontal_flip=True,
                                 vertical_flip=False)

    datagen.fit(x_train)
    history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                                  steps_per_epoch=x_train.shape[0] // batch_size,
                                  epochs=epochs,
                                  callbacks=[checkpoint],
                                  validation_data=(x_valid, y_valid),
                                  verbose=2)

print(model.evaluate(x_train, y_train))
print(model.evaluate(x_valid, y_valid))
print(model.evaluate(x_test, y_test))
pickle.dump(history.history, open('saved_models/' + dirname + '/history.pkl', 'wb'), 2)
