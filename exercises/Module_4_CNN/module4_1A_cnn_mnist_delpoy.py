




# Module 4_1 Deploy: Convolutional Neural Network (CNN)
# CNN model with dropout for MNIST dataset

# CNN structure:
# · · · · · · · · · ·      input data                                               X  [batch, 28, 28, 1]
# @ @ @ @ @ @ @ @ @ @   -- conv. layer 5x5x1x4  stride 1                            W1 [5, 5, 1, 4]
# ∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶                                                               Y1 [batch, 28, 28, 4]
#   @ @ @ @ @ @ @ @     -- conv. layer 5x5x4x8  with max pooling stride 2           W2 [5, 5, 4, 8]
#   ∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶                                                                 Y2 [batch, 14, 14, 8]
#     @ @ @ @ @ @       -- conv. layer 4x4x8x12 stride 2 with max pooling stride 2  W3 [4, 4, 8, 12]
#     ∶∶∶∶∶∶∶∶∶∶∶                                                                   Y3 [batch, 7, 7, 12]
#      \x/x\x\x/        -- fully connected layer (relu)                             W4 [7*7*12, 200]
#       · · · ·                                                                     Y4 [batch, 200]
#       \x/x\x/         -- fully connected layer (softmax)                          W5 [200, 10]
#        · · ·                                                                      Y [batch, 10]

import os
#os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Parameters
learning_rate = 0.0001
training_epochs = 10
batch_size = 50 #100

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("mnist", one_hot=True,reshape=False,validation_size=0)

# Step 1: Initial Setup
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
y = tf.placeholder(tf.float32, [None, 10])
pkeep = tf.placeholder(tf.float32)

L1 = 4 # first convolutional filters
L2 = 8 # second convolutional filters
L3 = 16 # third convolutional filters
L4 = 256 # fully connected neurons

W1 = tf.Variable(tf.truncated_normal([5,5,1,L1], stddev=0.1))
B1 = tf.Variable(tf.zeros([L1]))
W2 = tf.Variable(tf.truncated_normal([3,3,L1,L2], stddev=0.1))
B2 = tf.Variable(tf.zeros([L2]))
W3 = tf.Variable(tf.truncated_normal([3,3,L2,L3], stddev=0.1))
B3 = tf.Variable(tf.zeros([L3]))
W4 = tf.Variable(tf.truncated_normal([7*7*L3,L4], stddev=0.1))
B4 = tf.Variable(tf.zeros([L4]))
W5 = tf.Variable(tf.truncated_normal([L4, 10], stddev=0.1))
B5 = tf.Variable(tf.zeros([10]))

# Step 2: Setup Model
Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1,1,1,1], padding='SAME') + B1)# output is 28x28
Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1,1,1,1], padding='SAME') + B2)
Y2 = tf.nn.max_pool(Y2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME') # output is 14x14
Y2= tf.nn.dropout(Y2, 1.0)
Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1,1,1,1], padding='SAME') + B3)
Y3 = tf.nn.max_pool(Y3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME') # output is 7x7
Y3= tf.nn.dropout(Y3, 1.0)

# Flatten the third convolution for the fully connected layer
YY = tf.reshape(Y3, shape=[-1, 7 * 7 * L3])

Y4 = tf.nn.relu(tf.matmul(YY, W4) + B4)
YY4 = tf.nn.dropout(Y4, 1.0)
Ylogits = tf.matmul(Y4, W5) + B5
yhat = tf.nn.softmax(Ylogits)

# Step 3: Loss Functions
loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=y))

# Step 4: Optimizer
# optimizer = tf.train.GradientDescentOptimizer(learning_rate)
optimizer = tf.train.AdamOptimizer()
train = optimizer.minimize(loss)

# accuracy of the trained model, between 0 (worst) and 1 (best)
is_correct = tf.equal(tf.argmax(y,1),tf.argmax(yhat,1))
accuracy = tf.reduce_mean(tf.cast(is_correct,tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()
saver = tf.train.Saver()
sess = tf.Session()
sess.run(init)

# Step 5: Restore
saver.restore(sess, "./tmp_a/mnist.ckpt")
print("Model Restore: ")


# Step 6: Evaluation
test_data = {X:mnist.test.images,y:mnist.test.labels, pkeep: 0.5}
print("Testing Accuracy = ", sess.run(accuracy, feed_dict = test_data))

from matplotlib import pyplot as plt
from random import randint
num = randint(0, mnist.test.images.shape[0])
img = mnist.test.images[num:num+1]
classification = sess.run(tf.argmax(yhat, 1), feed_dict={X: img})
print('predicted', classification[0])
plt.imshow(img.reshape(28, 28), cmap=plt.cm.binary)
plt.show()

from PIL import Image
import numpy as np
imgnew = Image.open('mnist.jpg')
im = np.asarray(imgnew)
im = np.expand_dims(im, axis=0)
im = im.reshape(1,28,28,1)
print(im.shape)
classification = sess.run(tf.argmax(yhat, 1), feed_dict={X: im})
print('predicted', classification[0])
plt.imshow(im.reshape(28, 28), cmap=plt.cm.binary)
plt.show()