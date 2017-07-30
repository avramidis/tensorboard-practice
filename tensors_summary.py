import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf

logs_path = 'logs'
# Set the logs folder and delete any files in it
import glob, os
r = glob.glob(logs_path + '/*')
for i in r:
    os.remove(i)

a = tf.Variable(1)
b = tf.placeholder(tf.float32)
c = b + 1

init = tf.global_variables_initializer()
tf.summary.scalar("test_variable", a)
tf.summary.scalar("test_variable_2", c)
merged_summary_op = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(init)

    summary_writer = tf.summary.FileWriter(logs_path, sess.graph)

    for i in range(10):
        _, summary = sess.run([c, merged_summary_op], feed_dict={b:10})
        summary_writer.add_summary(summary, i)
