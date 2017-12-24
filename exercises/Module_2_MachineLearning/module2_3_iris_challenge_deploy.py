# Module 4: Simple TF Models
# Challenge: Iris flower dataset

import os
#os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Parameters
learning_rate = 0.05
training_epochs = 20

import tensorflow as tf
import numpy as np
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data
target = iris.target

# Convert the label into one-hot vector
num_labels = len(np.unique(target))
Y = np.eye(num_labels)[target]

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

# Step 1: Initial Setup
X = tf.placeholder(tf.float32, [None, 4])
W = tf.Variable(tf.truncated_normal([4, 3],stddev=0.1))
b = tf.Variable(tf.truncated_normal([3],stddev=0.1))

# Step 2: Define Model
yhat = tf.matmul(X, W) + b
y = tf.placeholder(tf.float32, [None, 3]) # Placeholder for correct answer

# Step 3: Loss Function
loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))

# Step 4: Optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(loss)

correct_prediction = tf.equal(tf.argmax(yhat, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()
sess = tf.Session()
saver = tf.train.Saver()
sess.run(init)

# Step 5: Restore
saver.restore(sess, "./tmp_iris/iris.ckpt")
print("Model Restore: ")

# Step 6: Evaluation
test_data = {X: X_test, y: y_test}
print("Training Accuracy = ", sess.run(accuracy, feed_dict = test_data))

from random import randint
num = randint(0, X_test.shape[0])
testitem = X_test[num:num+1]
print(y_test[num:num+1])
classification = sess.run(tf.argmax(yhat, 1), feed_dict={X: testitem})
print('predicted', classification[0])
