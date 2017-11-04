# -*- coding: utf-8 -*-

import tensorflow as tf

class Model:
    def __init__(self,
                 is_train,
                 learning_rate=0.001,
                 learning_rate_decay_factor=0.9995,
                 ):
        self.is_train = is_train

        self.x_ = tf.placeholder(tf.float32, [None, 28*28], name = "x_input")
        self.y_ = tf.placeholder(tf.int32, [None], "y_input")

        self.keep_prob = tf.placeholder(tf.float32)

        # TODO:  implement input -- Linear -- BN -- ReLU -- Linear -- loss
        #        the 10-class prediction output is named as "logits"

        self.W1 = weight_variable(shape = [784, 1024], name = "linear_W1")
        self.b1 = bias_variable(shape = [1024], name = "linear_b1")

        self.u1 = tf.matmul(self.x_, self.W1) + self.b1
        self.y1 = tf.nn.relu(self.u1)

        self.y1_drop = tf.nn.dropout(self.y1, keep_prob = self.keep_prob)

        self.W2 = weight_variable(shape = [1024, 10], name = "linear_W2")
        self.b2 = bias_variable(shape = [10], name = "linear_b2")

        self.logits = tf.matmul(self.y1_drop, self.W2) + self.b2

        # logits = tf.Variable(tf.constant(0.0, shape=[100, 10]))  # deleted this line after you implement above layers

        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_, logits=self.logits))
        self.correct_pred = tf.equal(tf.cast(tf.argmax(self.logits, 1), tf.int32), self.y_)
        self.pred = tf.argmax(self.logits, 1)  # Calculate the prediction result
        self.acc = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))  # Calculate the accuracy in this mini-batch

        self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=tf.float32)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)  # Learning rate decay

        self.global_step = tf.Variable(0, trainable=False)
        self.params = tf.trainable_variables()
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step,
                                                                            var_list=self.params)  # Use Adam Optimizer

        self.saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2,
                                    max_to_keep=3, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)


def weight_variable(shape, name):  # you can use this func to build new variables
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name = name)


def bias_variable(shape, name):  # you can use this func to build new variables
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name = name)


def batch_normalization_layer(inputs, isTrain=True):
    # TODO: implemented the batch normalization func and applied it on fully-connected layers
    return inputs



