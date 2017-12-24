# Module 2_2: 
# Simple TF model on MINST dataset

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Parameters
learning_rate = 0.00001
batch_size = 100


import tensorflow as tf
tf.set_random_seed(25)

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("mnist", one_hot=True)

# Step 1: Initial Setup
X = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.truncated_normal([784, 10],stddev=0.1))
b = tf.Variable(tf.zeros([10]))

# Step 2: Setup Model
yhat = tf.nn.softmax(tf.matmul(X,W)+b)
y = tf.placeholder(tf.float32, [None, 10]) # placeholder for correct answers

# Step 3: Cross Entropy Loss Functions
loss = -tf.reduce_sum(y*tf.log(yhat))

# Step 4: Optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(loss)

# % of correct answer found in batches
is_correct = tf.equal(tf.argmax(y,1),tf.argmax(yhat,1))
accuracy = tf.reduce_mean(tf.cast(is_correct,tf.float32))

sess = tf.Session()
saver = tf.train.Saver()
init = tf.global_variables_initializer()
sess.run(init)

# Step 5: Restore 
saver.restore(sess, "./tmp/mnist.ckpt")
print("Model Restore: ")

# Step 6: Evaluation
test_data = {X:mnist.test.images,y:mnist.test.labels}
print("Testing accuracy = ",sess.run(accuracy, feed_dict=test_data))

print("W=", sess.run(W).shape)
print("b=", sess.run(b).shape)

X_test = mnist.test.images
y_test = mnist.test.labels

print(type(X_test[1:2]))
print(y_test[1])
print(y_test[2])

test_data = {X: X_test[1:3]}
print(sess.run(yhat, feed_dict=test_data))

from matplotlib import pyplot as plt
from random import randint
num = randint(0, mnist.test.images.shape[0])
img = mnist.test.images[num:num+1]
classification = sess.run(tf.argmax(yhat, 1), feed_dict={X: img})
print('predicted', classification[0])
plt.imshow(img.reshape(28, 28), cmap=plt.cm.binary)
plt.show()