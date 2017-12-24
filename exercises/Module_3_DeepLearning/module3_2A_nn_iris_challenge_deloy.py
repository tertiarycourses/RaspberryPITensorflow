# Module 3_2_deloy: Neural Network and Deep Learning
# Challenge: Iris Flower dataset

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Parameters
n_features = 4
n_classes = 3
learning_rate = 0.05
training_epochs = 20

import tensorflow as tf
tf.set_random_seed(25)

import numpy as np
from sklearn import datasets
tf.set_random_seed(25)

iris = datasets.load_iris()
X = iris.data
target = iris.target

# Convert the label into one-hot vector
num_labels = len(np.unique(target))
Y = np.eye(num_labels)[target]

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)


init = tf.global_variables_initializer()
sess = tf.Session()
#saver = tf.train.Saver()
sess.run(init)

# Step 5: Restore
#saver.restore(sess, "./tmp_iris/iris.ckpt")
saver = tf.train.import_meta_graph('./tmp_a_iris/iris.ckpt.meta')
saver.restore(sess,tf.train.latest_checkpoint('./tmp_a_iris'))
print("Model Restore: ")

graph = tf.get_default_graph()
X = graph.get_tensor_by_name("X_input:0")
Y = graph.get_tensor_by_name("Y_input:0")
result = graph.get_tensor_by_name("yhat_output:0")
accuracy= graph.get_tensor_by_name("acc:0")
# Step 6: Evaluation
test_data = {X: X_test, Y: y_test}
print("Test Accuracy = ", sess.run(accuracy, feed_dict = test_data))

from random import randint
num = randint(0, X_test.shape[0])
testitem = X_test[num:num+1]
print(y_test[num:num+1])
classification = sess.run(tf.argmax(result, 1), feed_dict={X: testitem})
print('predicted', classification[0])