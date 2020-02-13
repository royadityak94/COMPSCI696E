import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Conv3D, Flatten, MaxPool2D, AveragePooling2D, BatchNormalization
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.python.keras import metrics
from keras import regularizers
from keras.constraints import unit_norm

import os
import numpy as np
from skimage.transform import rescale, resize
import matplotlib.pyplot as plt


# Global variables
MAX_CIFAR = 10000
MAX_TRAIN, MAX_VAL, MAX_TEST = 10000, 3000, 1000 
final_img_size = (128, 128, 3)
MULTIPLIER = int(final_img_size[0]/32)


# Helper Modules
def get_random_data(data1, data2, low, high, max_samples=100, multiplier=1):
    _, H1, W1, C1 = data1.shape
    _, N = data2.shape
    suff_data1 = np.empty((max_samples, H1*multiplier, W1*multiplier, C1))
    suff_data2 = np.empty((max_samples, N))
    shuffles = np.random.randint(low, high, max_samples)
    for idx in range(shuffles.shape[0]):
        suff_data1[idx] = rescale(data1[shuffles[idx], :, :, :], multiplier, anti_aliasing=False)
        suff_data2[idx] = data2[shuffles[idx], :]
    return suff_data1, suff_data2

def load_formatted_image(input_dir, limit=None, img_size=final_img_size):
    all_files = os.listdir(input_dir)
    if limit is not None:
        np.random.shuffle(all_files)
        all_files = all_files[:limit]
    img_arr = np.zeros((len(all_files), img_size[0], img_size[1], img_size[2]))
    for idx in range(len(all_files)):
        img_arr[idx] = resize(plt.imread(os.path.join(input_dir, all_files[idx])), img_size, anti_aliasing=False)
        
    return img_arr
    
# Loading CIFAR dataset
cifar = tf.keras.datasets.cifar10 
(x_train, y_train), (x_test, y_test) = cifar.load_data()
X, y = np.concatenate((x_train, x_test), axis=0), np.concatenate((y_train, y_test), axis=0)
res_X, _ = get_random_data(X, y, 0, X.shape[0], MAX_CIFAR, multiplier=MULTIPLIER)

# Loading KAIST and Soda dataset
kaist_arr = load_formatted_image('../data/KAIST/')
soda_arr = load_formatted_image('../data/all_soda_bottles/')

# Merging KAIST and CIFAR dataset and creating training, test splits
y_others = np.zeros(((kaist_arr.shape[0]+res_X.shape[0]), 1))
y_soda = np.ones((soda_arr.shape[0], 1))

pre_X, pre_y = np.concatenate((soda_arr, kaist_arr, res_X)), np.concatenate((y_soda, y_others))
X, y = get_random_data(pre_X, pre_y, 0, pre_X.shape[0], pre_X.shape[0])

X_train, y_train = X[:MAX_TRAIN], y[:MAX_TRAIN]
X_val, y_val = X[MAX_TRAIN:MAX_TRAIN+MAX_VAL], y[MAX_TRAIN:MAX_TRAIN+MAX_VAL]
X_test, y_test = X[MAX_TRAIN+MAX_VAL:MAX_TRAIN+MAX_VAL+MAX_TEST], y[MAX_TRAIN+MAX_VAL:MAX_TRAIN+MAX_VAL+MAX_TEST]

# Model hyperparameters
activation_func = 'relu'
weight_decay = 1e-4
curr_optimizer = 'Adam'

model = Sequential()

#Instantiating Layer 1
model.add(Conv2D(6, kernel_size=(3, 3), strides=(1, 1), activation=activation_func, padding='valid', 
                 kernel_regularizer=regularizers.l2(weight_decay)))
model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1'))
model.add(BatchNormalization())

# #Instantiating Layer 2
model.add(Conv2D(16, kernel_size=(3, 3), strides=(1, 1), activation=activation_func, padding='valid', 
                kernel_regularizer=regularizers.l2(weight_decay)))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='pool2'))
model.add(BatchNormalization())

# #Instantiating Layer 3
model.add(Conv2D(120, kernel_size=(3, 3), strides=(1, 1), activation=activation_func, padding='valid', 
                kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Flatten())

#Instantiating Layer 4
model.add(Dense(84, activation=activation_func)) 

#Output Layer
model.add(Dense(1, activation='sigmoid'))    

model.compile(loss="binary_crossentropy", optimizer=curr_optimizer, validation_data=(X_val, y_val),
                  metrics=["accuracy", "mae", "mse"])
                  
model.fit(X_train, y_train, epochs=12, batch_size=1024)
score = model.evaluate(X_test, y_test, batch_size=100)
print ("Score = {}".format(score))
