# -*- coding: utf-8 -*-

import tensorflow as tf

class Model:
    def __init__(self,
                 is_train,
                 learning_rate=0.001,
                 learning_rate_decay_factor=0.9995,
		 mean_var_decay = 0.999
                 ):
        self.is_train = is_train

        self.x_ = tf.placeholder(tf.float32, [None, 28*28], name = "x_input")
        self.y_ = tf.placeholder(tf.int32, [None], "y_input")

        self.keep_prob = tf.placeholder(tf.float32)

        # TODO:  implement input -- Linear -- BN -- ReLU -- Linear -- loss
        #        the 10-class prediction output is named as "logits"

        self.W1 = weight_variable(shape = [784, 2048], name = "linear_W1")
        self.b1 = bias_variable(shape = [2048], name = "linear_b1")

        self.u1 = tf.matmul(self.x_, self.W1) + self.b1
        self.u1_bn = batch_normalization_layer(self.u1, 2048, isTrain = is_train)

        self.y1 = tf.nn.relu(self.u1_bn)
        self.y1_drop = tf.nn.dropout(self.y1, keep_prob = self.keep_prob)

        self.W2 = weight_variable(shape = [2048, 10], name = "linear_W2")
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

def batch_normalization_layer(inputs, depth, decay = 0.99, isTrain=True):
    # TODO: implemented the batch normalization func and applied it on fully-connected layers

    bat_sz = tf.to_float(tf.shape(inputs)[0])
    m, v = tf.nn.moments(inputs, axes = 0, keep_dims = True)
    ave_m = tf.Variable(tf.constant(0., shape = [1, depth]), trainable = False)
    ave_v = tf.Variable(tf.constant(0., shape = [1, depth]), trainable = False)
    upd_m = ave_m.assign(ave_m * decay + m * (1. - decay))
    upd_v = ave_v.assign(ave_v * decay + v * bat_sz / (bat_sz - 1.) * (1. - decay))
    tf.add_to_collection("update_op", upd_m)
    tf.add_to_collection("update_op", upd_v)
    scale = tf.Variable(tf.constant(1., shape = [1, depth]))
    offset = tf.Variable(tf.constant(0., shape = [1, depth]))
    if isTrain:
        return tf.nn.batch_normalization(inputs, mean = m, variance = v, offset = offset, scale = scale, variance_epsilon = tf.constant(1e-10))
    else:
        return tf.nn.batch_normalization(inputs, mean = ave_m, variance = ave_v, offset = offset, scale = scale, variance_epsilon = tf.constant(1e-10))

