



import os
#os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Parameters
learning_rate = 0.0001
training_epochs = 10
batch_size = 50 #100

import tensorflow as tf
from tensorflow.python.platform import gfile

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("mnist", one_hot=True,reshape=False,validation_size=0)


sess = tf.Session()
# init = tf.global_variables_initializer()
# sess.run(init)
saver = tf.train.import_meta_graph('./tmp_b/mnist.ckpt.meta')
saver.restore(sess,tf.train.latest_checkpoint('./tmp_b'))
print("Model Restore: ")
graph = tf.get_default_graph()
X = graph.get_tensor_by_name("X_input:0")
result = graph.get_tensor_by_name("yhat_output:0")



from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
imgnew = Image.open('mnist.jpg')
im = np.asarray(imgnew)
im = np.expand_dims(im, axis=0)
im = im.reshape(1,28,28,1)
print(im.shape)
classification = sess.run(tf.argmax(result, 1), {X: im})
print('predicted', classification[0])
plt.imshow(im.reshape(28, 28), cmap=plt.cm.binary)
plt.show()