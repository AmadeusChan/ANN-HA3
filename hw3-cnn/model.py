# -*- coding: utf-8 -*-

import tensorflow as tf


class Model:
    def __init__(self,
                 is_train,
                 learning_rate=0.001,
                 learning_rate_decay_factor=0.9995):
        self.x_ = tf.placeholder(tf.float32, [None, 1, 28, 28])
        self.y_ = tf.placeholder(tf.int32, [None])
        self.keep_prob = tf.placeholder(tf.float32)

        x = tf.reshape(self.x_, [-1, 28, 28, 1])

        # TODO: implement input -- Conv -- BN -- ReLU -- MaxPool -- Conv -- BN -- ReLU -- MaxPool -- Linear -- loss
        #        the 10-class prediction output is named as "logits"
        # logits = tf.Variable(tf.constant(0.0, shape=[100, 10]))  # deleted this line after you implement above layers

        # 1st convolution layer
        self.W_conv1 = weight_variable(shape = [5, 5, 1, 32])
        self.b_conv1 = bias_variable(shape = [32])

        self.u1 = tf.nn.conv2d(x, self.W_conv1, strides = [1, 1, 1, 1], padding = "SAME") + self.b_conv1
        self.y1 = tf.nn.relu(self.u1)

        # 1st max-pool layer
        self.pool1 = tf.nn.max_pool(self.y1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME") # output: batch_size x 14 x 14 x 32

        # 2nd convolution layer
        self.W_conv2 = weight_variable(shape = [5, 5, 32, 256])
        self.b_conv2 = bias_variable(shape = [256])

        self.u2 = tf.nn.conv2d(self.pool1, self.W_conv2, strides = [1, 1, 1, 1], padding = "SAME") + self.b_conv2
        self.y2 = tf.nn.relu(self.u2)

        # 2nd max-pool layer
        self.pool2 = tf.nn.max_pool(self.y2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")
        # output: batch_size x 7 x 7 x 256

        self.pool2_reshape = tf.reshape(self.pool2, [-1, 7 * 7 * 256])
        self.pool2_reshape_drop = tf.nn.dropout(self.pool2_reshape, keep_prob = self.keep_prob)

        # classification layer
        self.W1 = weight_variable(shape = [7 * 7 * 256, 10])
        self.b1 = bias_variable(shape = [10])
        logits = tf.matmul(self.pool2_reshape_drop, self.W1) + self.b1

        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_, logits=logits))
        self.correct_pred = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), self.y_)
        self.pred = tf.argmax(logits, 1)
        self.acc = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

        self.learning_rate = tf.Variable(float(learning_rate), trainable=False,
                                         dtype=tf.float32)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)

        self.global_step = tf.Variable(0, trainable=False)
        self.params = tf.trainable_variables()
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step,
                                                                            var_list=self.params)

        self.saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2,
                                    max_to_keep=3, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)


def weight_variable(shape):  # you can use this func to build new variables
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):  # you can use this func to build new variables
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def batch_normalization_layer(inputs, isTrain=True):
    # TODO: implemented the batch normalization func and applied it on conv and fully-connected layers
    # hint: you can add extra parameters (e.g., shape) if necessary
    return inputs
