# Module 9 Keras
# Challenge InceptionV3 Transfer Learning

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Input

import tflearn.datasets.oxflower17 as oxflower17
from sklearn.model_selection import train_test_split
X, Y = oxflower17.load_data(one_hot=True, resize_pics=(224, 224))
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

from keras.models import model_from_json
from keras.optimizers import SGD

# # Step 1: Create the base pre-trained model
# input_tensor = Input(shape=(224, 224, 3))
# base_model = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=False)
#
# # Step 2: Create a new model with dense and softamx layers
# x = base_model.output
# x = GlobalAveragePooling2D()(x)
# x = Dense(1024, activation='relu')(x)
# predictions = Dense(17, activation='softmax')(x)
# model = Model(inputs=base_model.input, outputs=predictions)
#
# # Step 3: Freeze all pre-trained layers and train the top layers with new dataaset
# for layer in base_model.layers:
#     layer.trainable = False
# model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
# model.fit(X_train, Y_train, batch_size=50, epochs=2)
#
# # Step 4: Unfreeze some pre-trained layers and train with new dataset
# for layer in model.layers[:5]:
#     layer.trainable = False
# for layer in model.layers[5:]:
#     layer.trainable = True
#
# from keras.optimizers import SGD
# model.compile(optimizer=SGD(lr=0.001, momentum=0.9), loss='categorical_crossentropy',metrics=['accuracy'])
# model.fit(X_train, Y_train, batch_size=25, epochs=2)
#
# #serialize model to JSON
# model_json = model.to_json()
# with open("./tmp_inc2/model.json", "w") as json_file:
#     json_file.write(model_json)
# # serialize weights to HDF5
# model.save_weights("./tmp_inc2/model.h5")
# print("Saved model to disk")

# load json and create model
json_file = open('./tmp_inc2/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("./tmp_inc2/model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(optimizer=SGD(lr=0.001, momentum=0.9), loss='categorical_crossentropy',metrics=['accuracy'])
# score = loaded_model.evaluate(X, Y, verbose=0)
# print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))
print("Loaded model compile")

# Step 4: Evaluation
# score = loaded_model.evaluate(X_test, Y_test, batch_size=25)
# print("\nTraining Accuracy = ",score[1],"Loss",score[0])

print(X_test[0:1].shape)
print(Y_test[0:1])
preds = loaded_model.predict(X_test[0:1])
print(preds)
sess = tf.Session()
print(sess.run(tf.argmax(preds, 1)))

#---------------------------------------------------------
from PIL import Image
import numpy as np

img = Image.open('./tmp_vgg/test/daisy10.jpg')
new_width  = 224
new_height = 224
img = img.resize((new_width, new_height), Image.ANTIALIAS)
# img.save('./tmp_vgg/test/test.jpg') # format may what u want ,*.png,*jpg,*.gif
#
# img = Image.open('./tmp_vgg/test/test.jpg')
#img = img.convert('RGB')
x = np.asarray(img, dtype='float32')
# x = x.transpose(2, 0, 1)
x = np.expand_dims(x, axis=0)
x = x.reshape(1,224,224,3)
preds = loaded_model.predict(x)
print(preds)
sess = tf.Session()
print(sess.run(tf.argmax(preds, 1)))


# from keras.applications.vgg16 import preprocess_input,decode_predictions
# from keras.preprocessing import image
# import numpy as np
# img_path = './tmp_vgg/test/image_1316.jpg'
# img = image.load_img(img_path, target_size=(224, 224))
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)
# print(x.shape)
# x = preprocess_input(x)
# print(x.shape)
#
# preds = loaded_model.predict(x)
# print(preds)
# sess = tf.Session()
# print(sess.run(tf.argmax(preds, 1)))