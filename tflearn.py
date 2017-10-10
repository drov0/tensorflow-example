from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import cv2
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import tensorflow as tf
import math
FLAGS = None

class Capchat:
    data_dir = "images/"
    nb_categories = 9
    X_train = None
    Y_train = None

    train_nb = 0

    X_test = None
    Y_test = None
    test_nb = 0

    index = 0



    def readimg(self, file, label, train = True):
        im = cv2.imread(file);
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).flatten()

        if train :
            self.X_train[self.index] = im
            self.Y_train[self.index][label - 1] = 1
        else :
            self.X_test[self.index] = im
            self.Y_test[self.index][label - 1] = 1

        self.index += 1

    def __init__(self):
        total_size = [f for f in listdir(self.data_dir + "1/") if isfile(join(self.data_dir + "1/", f))].__len__()
        self.train_nb = math.floor(total_size * 0.8)
        self.test_nb = math.ceil(total_size *0.2)

        self.X_train = np.zeros((self.train_nb*self.nb_categories, 840), np.int32)
        self.Y_train = np.zeros((self.train_nb*self.nb_categories, 9), np.int32)

        self.X_test = np.zeros((self.test_nb*self.nb_categories, 840), np.int32)
        self.Y_test = np.zeros((self.test_nb*self.nb_categories, 9), np.int32)



        files_1 = [f for f in listdir(self.data_dir+"1/") if isfile(join(self.data_dir+"1/", f))]
        files_2 = [f for f in listdir(self.data_dir+"2/") if isfile(join(self.data_dir+"2/", f))]
        files_3 = [f for f in listdir(self.data_dir+"3/") if isfile(join(self.data_dir+"3/", f))]
        files_4 = [f for f in listdir(self.data_dir+"4/") if isfile(join(self.data_dir+"4/", f))]
        files_5 = [f for f in listdir(self.data_dir+"5/") if isfile(join(self.data_dir+"5/", f))]
        files_6 = [f for f in listdir(self.data_dir+"6/") if isfile(join(self.data_dir+"6/", f))]
        files_7 = [f for f in listdir(self.data_dir+"7/") if isfile(join(self.data_dir+"7/", f))]
        files_8 = [f for f in listdir(self.data_dir+"8/") if isfile(join(self.data_dir+"8/", f))]
        files_9 = [f for f in listdir(self.data_dir + "0/") if isfile(join(self.data_dir + "0/", f))]


        for i in range(self.train_nb):

            self.readimg(self.data_dir+"1/"+files_1[i], 1)
            self.readimg(self.data_dir+"2/"+files_2[i], 2)
            self.readimg(self.data_dir+"3/"+files_3[i], 3)
            self.readimg(self.data_dir+"4/"+files_4[i], 4)
            self.readimg(self.data_dir+"5/"+files_5[i], 5)
            self.readimg(self.data_dir+"6/"+files_6[i], 6)
            self.readimg(self.data_dir+"7/"+files_7[i], 7)
            self.readimg(self.data_dir+"8/"+files_8[i], 8)
            self.readimg(self.data_dir + "0/" + files_9[i], 9)

        self.index = 0

        for i  in range (self.train_nb, self.train_nb + self.test_nb):

            self.readimg(self.data_dir+"1/" + files_1[i], 1, False)
            self.readimg(self.data_dir+"2/" + files_2[i], 2, False)
            self.readimg(self.data_dir+"3/" + files_3[i], 3, False)
            self.readimg(self.data_dir+"4/" + files_4[i], 4, False)
            self.readimg(self.data_dir+"5/" + files_5[i], 5, False)
            self.readimg(self.data_dir+"6/" + files_6[i], 6, False)
            self.readimg(self.data_dir+"7/" + files_7[i], 7, False)
            self.readimg(self.data_dir+"8/" + files_8[i], 8, False)
            self.readimg(self.data_dir+"0/" + files_9[i], 9, False)




def main(_):
  # Import data
  cap = Capchat()
  # Create the model
  x = tf.placeholder(tf.float32, [None, 840])
  W = tf.Variable(tf.zeros([840, 9]), name="weights")
  b = tf.Variable(tf.zeros([9]), name="biases")
  inter = tf.matmul(x, W)
  y = tf.add(inter, b, name="calc")

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 9])

  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

  saver = tf.train.Saver()

  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()
  # Train

  for _ in range(1000):

    sess.run(train_step, feed_dict={x: cap.X_train, y_: cap.Y_train})

  # Test trained model
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print(sess.run(accuracy, feed_dict={x: cap.X_test,
                                      y_: cap.Y_test}))
  # this will put a few files in your local directory, remove all the files from the directory "model"
  # and put them there
  saver.save(sess, "./mod")


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main)
