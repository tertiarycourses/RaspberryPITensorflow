# Module 3_1 Deploy: Neural Network and Deep Learning
# NN model for MNIST dataset and save model and delpy


import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Parameters
learning_rate = 0.5
training_epochs = 2
batch_size = 100

import tensorflow as tf
tf.set_random_seed(25)

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("mnist", one_hot=True)

# # Step 1: Initial Setup
# X = tf.placeholder(tf.float32, [None, 784])
# y = tf.placeholder(tf.float32, [None, 10])
#
# L1 = 200
# L2 = 100
# L3 = 60
# L4 = 30
#
# #[784,L1]  input , number neutron
# W1 = tf.Variable(tf.truncated_normal([784, L1], stddev=0.1))
# B1 = tf.Variable(tf.zeros([L1]))
# W2 = tf.Variable(tf.truncated_normal([L1, L2], stddev=0.1))
# B2 = tf.Variable(tf.zeros([L2]))
# W3 = tf.Variable(tf.truncated_normal([L2, L3], stddev=0.1))
# B3 = tf.Variable(tf.zeros([L3]))
# W4 = tf.Variable(tf.truncated_normal([L3, L4], stddev=0.1))
# B4 = tf.Variable(tf.zeros([L4]))
# W5 = tf.Variable(tf.truncated_normal([L4, 10], stddev=0.1))
# B5 = tf.Variable(tf.zeros([10]))
#
# # Step 2: Setup Model
# Y1 = tf.nn.relu(tf.matmul(X, W1) + B1)
# Y2 = tf.nn.relu(tf.matmul(Y1, W2) + B2)
# Y3 = tf.nn.relu(tf.matmul(Y2, W3) + B3)
# Y4 = tf.nn.relu(tf.matmul(Y3, W4) + B4)
# Ylogits = tf.matmul(Y4, W5) + B5
# yhat = tf.nn.softmax(Ylogits)
#
# # Step 3: Loss Functions
# loss = tf.reduce_mean(
#    tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=Ylogits))
#
# # Step 4: Optimizer
# #optimizer = tf.train.GradientDescentOptimizer(learning_rate)
# optimizer = tf.train.AdamOptimizer()
# train = optimizer.minimize(loss)
#
# # accuracy of the trained model, between 0 (worst) and 1 (best)
# is_correct = tf.equal(tf.argmax(y,1),tf.argmax(yhat,1))
# accuracy = tf.reduce_mean(tf.cast(is_correct,tf.float32))

sess = tf.Session()
# saver = tf.train.Saver()
init = tf.global_variables_initializer()
sess.run(init)

# Step 5: Restore
# saver.restore(sess, "./tmp/mnist.ckpt")
#First let's load meta graph and restore weights
saver = tf.train.import_meta_graph('./tmp_a/mnist.ckpt.meta')
saver.restore(sess,tf.train.latest_checkpoint('./tmp_a'))
print("Model Restore: ")

graph = tf.get_default_graph()
X = graph.get_tensor_by_name("X_input:0")
#Now, access the op that you want to run.
result = graph.get_tensor_by_name("yhat_output:0")

# # Step 6: Evaluation
# test_data = {X:mnist.test.images,y:mnist.test.labels}
# print("Testing Accuracy = ", sess.run('acc:0', feed_dict = test_data))

from matplotlib import pyplot as plt
from random import randint
num = randint(0, mnist.test.images.shape[0])
img = mnist.test.images[num:num+1]
classification = sess.run(tf.argmax(result, 1), feed_dict={X: img})
print('predicted', classification[0])
plt.imshow(img.reshape(28, 28), cmap=plt.cm.binary)
plt.show()