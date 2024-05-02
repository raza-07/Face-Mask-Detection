#Importing libraries
# import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import sys
from keras.preprocessing import image
from tensorflow.keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Input
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpointcessing
# import image
# from tensorflow.keras.utils import plot_model



CNN_aug_new = Sequential()

CNN_aug_new.add(Input(shape=(75, 75, 3)))

#Specify a list of the number of filters for each convolutional layer

for n_filters in [16,32, 64]:
    CNN_aug_new.add(Conv2D(n_filters,strides=(2, 2), kernel_size=3, activation='relu'))

# Fill in the layer needed between our 2d convolutional layers and the dense layer
CNN_aug_new.add(Flatten())

#Specify the number of nodes in the dense layer before the output
CNN_aug_new.add(Dense(128, activation='relu'))

#Specify the output layer
CNN_aug_new.add(Dense(2, activation='softmax'))
 
#Compiling the model
CNN_aug_new.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

CNN_aug_new.load_weights('./model_weights.h5')

img = image.load_img('./278806330_1428620074259588_3384862339238655338_n.jpg',target_size=(75, 75))
# img = image.load_img('/data/Face Mask Dataset/Test/WithoutMask/1000.png',target_size=(75, 75))
img = image.img_to_array(img)
img = np.array([img])
prediction=CNN_aug_new.predict(img)
print(prediction[0])
result = 'withmask' if np.argmax(prediction)==0 else "withoutmask"
print(result)
