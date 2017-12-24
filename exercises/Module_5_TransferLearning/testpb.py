import tensorflow as tf
import os
import numpy as np
from tensorflow.python.platform import gfile

data = np.arange(10,dtype=np.int32)
with tf.Session() as sess:
  print("# build graph and run")
  input1= tf.placeholder(tf.int32, [10], name="input")
  output1= tf.add(input1, tf.constant(100,dtype=tf.int32), name="output") #  data depends on the input data
  saved_result= tf.Variable(data, name="saved_result")
  do_save=tf.assign(saved_result,output1)
  tf.global_variables_initializer()
  #os.system("rm -rf /tmp/load")
  tf.train.write_graph(sess.graph_def, "./", "test.pb", False) #proto
  # now set the data:
  data = np.arange(10)
  result,_=sess.run(output1, {input1: data}) # calculate output1 and assign to 'saved_result'
  print(result)
  saver = tf.train.Saver(tf.all_variables())
  saver.save(sess,"checkpoint.data")

with tf.Session() as persisted_sess:
  print("load graph")
  with gfile.FastGFile("./test.pb",'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    persisted_sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')
  print("map variables")
  # persisted_result = persisted_sess.graph.get_tensor_by_name("saved_result:0")
  # tf.add_to_collection(tf.GraphKeys.VARIABLES,persisted_result)
  # try:
  #   saver = tf.train.Saver(tf.all_variables()) # 'Saver' misnomer! Better: Persister!
  # except:pass
  # print("load data")
  # saver.restore(persisted_sess, "checkpoint.data")  # now OK
  # print(persisted_result.eval())
  # print("DONE")

  get_input = persisted_sess.graph.get_tensor_by_name('input:0')
  get_output = persisted_sess.graph.get_tensor_by_name('output:0')
  predictions = sess.run(get_output, {get_input: [1,2,3,4,5,6,7,8,9]})
  prubt(predictions)