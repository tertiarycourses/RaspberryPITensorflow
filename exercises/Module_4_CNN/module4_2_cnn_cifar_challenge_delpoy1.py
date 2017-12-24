# Module 4_2 Convolutional Neural Network (CNN)
# Challenge : CIFAR-10 dataset

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Parameters

# learning_rate = 0.01
# training_epochs = 1
# batch_size = 100

# Parameters 1
learning_rate = 0.0001
training_epochs = 10
batch_size = 100

import tensorflow as tf
from tflearn.datasets import cifar10

from tflearn.data_utils import to_categorical
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
Y_train = to_categorical(Y_train, 10)
Y_test = to_categorical(Y_test, 10)

# Step 1: Initial Setup
X = tf.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.placeholder(tf.float32, [None, 10])
pkeep = tf.placeholder(tf.float32)

# three convolutional layers with their channel counts, and a
# fully connected layer (tha last layer has 10 softmax neurons)
L1 = 32  # first convolutional layer output depth
L2 = 64  # second convolutional layer output depth
L3 = 1024   # Fully connected layer

W1 = tf.Variable(tf.truncated_normal([5, 5, 3, L1], stddev=0.1))
B1 = tf.Variable(tf.zeros([L1]))
W2 = tf.Variable(tf.truncated_normal([3, 3, L1, L2], stddev=0.1))
B2 = tf.Variable(tf.zeros([L2]))
W3 = tf.Variable(tf.truncated_normal([8 * 8 * L2, L3], stddev=0.1))
B3 = tf.Variable(tf.zeros([L3]))
W4 = tf.Variable(tf.truncated_normal([L3, 10], stddev=0.1))
B4 = tf.Variable(tf.zeros([10]))

# Step 2: Setup Model
Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME') + B1)
Y1 = tf.nn.max_pool(Y1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, 1, 1, 1], padding='SAME') + B2)
Y2 = tf.nn.max_pool(Y2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# reshape the output from the third convolution for the fully connected layer
YY = tf.reshape(Y2, shape=[-1, 8 * 8 * L2])

Y3 = tf.nn.relu(tf.matmul(YY, W3) + B3)
YY3 = tf.nn.dropout(Y3, pkeep)
Ylogits = tf.matmul(Y3, W4) + B4
yhat = tf.nn.softmax(Ylogits)

# Step 3: Loss Functions
loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=y))

# Step 4: Optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
# optimizer = tf.train.AdamOptimizer(0.1)
train = optimizer.minimize(loss)

# accuracy of the trained model, between 0 (worst) and 1 (best)
is_correct = tf.equal(tf.argmax(y, 1), tf.argmax(yhat, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

init = tf.global_variables_initializer()
sess = tf.Session()
saver = tf.train.Saver()
sess.run(init)

# Step 5: restore
saver.restore(sess, "./tmp_cifar/cifar.ckpt")
print("Model Restore: ")


# Step 6: Evaluation
acc = []
for i in range(int(X_test.shape[0] / batch_size)):
    batch_X = X_test[(i*batch_size):((i+1)*batch_size)]
    batch_y = Y_test[(i*batch_size):((i+1)*batch_size)]
    test_data = {X: batch_X, y: batch_y}
    sess.run(train, feed_dict = test_data)
    acc.append(sess.run(accuracy, feed_dict = test_data))

print("Testing Accuracy/Loss = ", sess.run(tf.reduce_mean(acc)))

from random import randint
num = randint(0, X_test.shape[0])
print(num)
img = X_test[num:num+1]
print(Y_test[num:num+1])
classification = sess.run(tf.argmax(yhat, 1), feed_dict={X: img})
print('predicted', classification[0])

# import matplotlib.pyplot as plt
# from scipy.misc import toimage
# plt.imshow(toimage(X_test[num]))
# plt.show()

from PIL import Image
import numpy as np
imgnew = Image.open('newcat.jpg')
im = np.asarray(imgnew)
print(im.shape)
classification = sess.run(tf.argmax(yhat, 1), feed_dict={X: [im]})
print('predicted', classification[0])
