import argparse
import sys
import cv2
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import tensorflow as tf



def solve():
    X = np.zeros((1, 840), np.int32)
    im = cv2.imread("./test/test.png")
    im = np.array(im)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).flatten()
    X[0] = im


    prediction = 0

    # machine learning to read the numbers
    with tf.Session() as session:
        # restore the model
        saver = tf.train.import_meta_graph("./model/mod.meta")
        saver.restore(session, tf.train.latest_checkpoint("./model/"))
        graph = tf.get_default_graph()
        x = tf.placeholder(tf.float32, [None, 840])
        W = graph.get_tensor_by_name("weights:0")
        b = graph.get_tensor_by_name("biases:0")
        y = tf.argmax(tf.nn.softmax(tf.matmul(x, W) + b), axis=1)

        P = session.run(y, feed_dict={x: X})
        prediction = P[0] + 1 # +1 because everything is 0 indexed

    print("detected : "+str(prediction))


solve()