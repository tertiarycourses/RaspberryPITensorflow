# Module 9 Keras
# CNN Model on MNIST dataaset
import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


# Parameters
n_classes = 10
learning_rate = 0.5
training_epochs = 2
batch_size = 100


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

# Step 1 Load the Data
# import tflearn.datasets.mnist as mnist
# X_train, y_train, X_test, y_test = mnist.load_data(one_hot=True)
#
# X_train = X_train.reshape(-1, 28, 28, 1)
# X_test = X_test.reshape(-1, 28, 28, 1)

from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
y_train = keras.utils.to_categorical(y_train, n_classes)
y_test = keras.utils.to_categorical(y_test, n_classes)

# # Step 2: Build the Network
# model = Sequential()
# model.add(Conv2D(32,(3, 3),activation='relu',input_shape=(28,28,1),padding='same'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(64,(3, 3),activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(n_classes, activation='softmax'))
# model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
# print(model.summary())
# # Step 3: Training
# model.fit(X_train, y_train, epochs=training_epochs,batch_size=batch_size)

from keras.models import model_from_json
import cv2
import numpy as np
from PIL import Image, ImageDraw
from PIL import ImageFilter
### Load architecture
with open('cnn_model.json', 'r') as architecture_file:    
    model_architecture = json.load(architecture_file)
model = model_from_json(model_architecture)
 
### Load weights
model.load_weights('cnn_wt.h5')
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])

# Step 4: Evaluation
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# img = cv2.imread('mnist5.jpg')
# img = cv2.resize(img, (28, 28))
# arr = np.array(img).reshape((28,28,3))
# arr = np.expand_dims(arr, axis=0)
img = Image.open('mnist5.jpg').convert('L')
img = img.resize((28, 28), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
data = [(255 - x) * 1.0 / 255.0 for x in list(img.getdata())]
im = np.reshape(data, (-1, 28, 28, 1))
print(im.shape)
prediction = model.predict(im)[0]
print(prediction)

# prediction = model.predict(im)[0]
bestclass = ''
bestconf = -1
for n in [0,1,2,3,4,5,6,7,8,9]:
	if (prediction[n] > bestconf):
		bestclass = str(n)
		bestconf = prediction[n]
print('I think this digit is a ' + bestclass + ' with ' + str(bestconf * 100) + ' confidence.')