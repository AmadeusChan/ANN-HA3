# -*- coding: utf-8 -*-

import tensorflow as tf


class Model:
    def __init__(self,
                 is_train,
                 learning_rate= 1e-3,
                 learning_rate_decay_factor=.99,
                 mean_var_decay=0.99,
		 weight_decay=1e-6):
        with tf.name_scope("input"):
            self.x_ = tf.placeholder(tf.float32, [None, 1, 28, 28], name = "x_input")
            self.y_ = tf.placeholder(tf.int32, [None], name = "y_input")
            x = tf.reshape(self.x_, [-1, 28, 28, 1])
            # x = tf.nn.dropout(x, keep_prob = 0.9)

        self.keep_prob = tf.placeholder(tf.float32)

        # TODO: implement input -- Conv -- BN -- ReLU -- MaxPool -- Conv -- BN -- ReLU -- MaxPool -- Linear -- loss
        #        the 10-class prediction output is named as "logits"
        # logits = tf.Variable(tf.constant(0.0, shape=[100, 10]))  # deleted this line after you implement above layers

        # 1st convolution layer
        # the 1st branch
        with tf.name_scope("conv-pool1"):
            with tf.name_scope("conv"):
                self.W_conv1 = weight_variable(shape = [5, 5, 1, 32])
                self.b_conv1 = bias_variable(shape = [32])
                self.u1 = tf.nn.conv2d(x, self.W_conv1, strides = [1, 1, 1, 1], padding = "SAME") + self.b_conv1
            with tf.name_scope("bn"):
                self.u1_bn = batch_normalization_layer(self.u1, mean_var_decay, 32, isTrain = is_train)
            with tf.name_scope("relu"):
                self.y1 = tf.nn.relu(self.u1_bn, name = "relu1")
            # 1st max-pool layer
            with tf.name_scope("max_pool"):
                self.pool1 = tf.nn.max_pool(self.y1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME", name = "max_pool1") # output: batch_size x 14 x 14 x 32

        # 2nd convolution layer
        with tf.name_scope("conv-pool2"):
            with tf.name_scope("conv"):
                self.W_conv2 = weight_variable(shape = [5, 5, 32, 256])
                self.b_conv2 = bias_variable(shape = [256])
                self.u2 = tf.nn.conv2d(self.pool1, self.W_conv2, strides = [1, 1, 1, 1], padding = "SAME") + self.b_conv2
            with tf.name_scope("bn"):
                self.u2_bn = batch_normalization_layer(self.u2, mean_var_decay, 256, isTrain = is_train)
            with tf.name_scope("relu"):
                self.y2 = tf.nn.relu(self.u2_bn, name = "relu2")
            # 2nd max-pool layer
            with tf.name_scope("max_pool"):
                self.pool2 = tf.nn.max_pool(self.y2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME", name = "max_pool2")
            # output: batch_size x 7 x 7 x 256

        with tf.name_scope("classification_layer"):
            with tf.name_scope("reshape"):
                self.pool2_reshape = tf.reshape(self.pool2, [-1, 7 * 7 * 256])

            with tf.name_scope("dropout"):
                self.pool2_reshape_drop = tf.nn.dropout(self.pool2_reshape, keep_prob = self.keep_prob, name = "drop1")
            # classification layer

            with tf.name_scope("linear"):
                self.W3 = weight_variable(shape = [7 * 7 * 256, 128])
                self.b3 = bias_variable(shape = [128])
                self.u3 = tf.matmul(self.pool2_reshape_drop, self.W3) + self.b3
                self.y3 = tf.nn.relu(self.u3)
                
                self.W4 = weight_variable(shape = [128, 10])
                self.b4 = bias_variable(shape = [10])
                logits = tf.matmul(self.y3, self.W4) + self.b4

        # the 2nd branch
        with tf.name_scope("conv-pool1-branch1"): # output: 12 x 12 x 32
            with tf.name_scope("conv"):
                self.W_conv1_b1 = weight_variable(shape = [5, 5, 1, 32])
                self.b_conv1_b1 = bias_variable(shape = [32])
                self.u1_b1 = tf.nn.conv2d(x, self.W_conv1_b1, strides = [1, 1, 1, 1], padding = "VALID") + self.b_conv1_b1
            with tf.name_scope("bn"):
                self.u1_bn_b1 = batch_normalization_layer(self.u1_b1, mean_var_decay, 32, isTrain = is_train)
            with tf.name_scope("relu"):
                self.y1_b1 = tf.nn.relu(self.u1_bn_b1)
            with tf.name_scope("max_pool"):
                self.p1_b1 = tf.nn.max_pool(self.y1_b1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

        with tf.name_scope("conv-pool2-branch1"): # output: 4 x 4 x 256
            with tf.name_scope("conv"):
                self.W_conv2_b1 = weight_variable(shape = [5, 5, 32, 256])
                self.b_conv2_b1 = bias_variable(shape = [256])
                self.u2_b1 = tf.nn.conv2d(self.p1_b1, self.W_conv2_b1, strides = [1, 1, 1, 1], padding = "VALID") + self.b_conv2_b1
            with tf.name_scope("bn"):
                self.u2_bn_b1 = batch_normalization_layer(self.u2_b1, mean_var_decay, 256, isTrain = is_train)
            with tf.name_scope("relu"):
                self.y2_b1 = tf.nn.relu(self.u2_bn_b1)
            with tf.name_scope("max_pool"):
                self.p2_b1 = tf.nn.max_pool(self.y2_b1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

        with tf.name_scope("classification_layer-branch1"):
            with tf.name_scope("reshape"):
                self.p2_rep_b1 = tf.reshape(self.p2_b1, [-1, 4096])
            with tf.name_scope("dropout"):
                self.p2_rep_dp_b1 = tf.nn.dropout(self.p2_rep_b1, keep_prob = self.keep_prob)
            with tf.name_scope("linear-relu1"):
                self.W1_b1 = weight_variable(shape = [4096, 256])
                self.b1_b1 = bias_variable(shape = [256])
                self.lu1_b1 = tf.matmul(self.p2_rep_dp_b1, self.W1_b1) + self.b1_b1
                self.ly1_b1 = tf.nn.relu(self.lu1_b1)
            with tf.name_scope("linear"):
                self.W2_b1 = weight_variable(shape = [256, 10])
                self.b2_b1 = bias_variable(shape = [10])
                logits_b1 = tf.matmul(self.ly1_b1, self.W2_b1) + self.b2_b1

        # the 3rd branch
        with tf.name_scope("conv-pool1-branch2"): # output: 14 x 14 x 32
            with tf.name_scope("conv"):
                self.W_conv1_b2 = weight_variable(shape = [5, 5, 1, 32])
                self.b_conv1_b2 = bias_variable(shape = [32])
                self.u1_b2 = tf.nn.conv2d(x, self.W_conv1_b2, strides = [1, 1, 1, 1], padding = "SAME") + self.b_conv1_b2
            with tf.name_scope("bn"):
                self.u1_bn_b2 = batch_normalization_layer(self.u1_b2, mean_var_decay, 32, isTrain = is_train)
            with tf.name_scope("relu"):
                self.y1_b2 = tf.nn.relu(self.u1_bn_b2)
            with tf.name_scope("max_pool"):
                self.p1_b2 = tf.nn.max_pool(self.y1_b2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

        with tf.name_scope("conv-pool2-branch2"): # output: 7 x 7 x 64
            with tf.name_scope("conv"):
                self.W_conv2_b2 = weight_variable(shape = [5, 5, 32, 64])
                self.b_conv2_b2 = bias_variable(shape = [64])
                self.u2_b2 = tf.nn.conv2d(self.p1_b2, self.W_conv2_b2, strides = [1, 1, 1, 1], padding = "SAME") + self.b_conv2_b2
            with tf.name_scope("bn"):
                self.u2_bn_b2 = batch_normalization_layer(self.u2_b2, mean_var_decay, 64, isTrain = is_train)
            with tf.name_scope("relu"):
                self.y2_b2 = tf.nn.relu(self.u2_bn_b2)
            with tf.name_scope("max_pool"):
                self.p2_b2 = tf.nn.max_pool(self.y2_b2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

        with tf.name_scope("classification_layer-branch2"):
            with tf.name_scope("reshape"):
                self.p2_rep_b2 = tf.reshape(self.p2_b2, [-1, 7 * 7 * 64])
            with tf.name_scope("dropout"):
                self.p2_rep_dp_b2 = tf.nn.dropout(self.p2_rep_b2, keep_prob = self.keep_prob)
            with tf.name_scope("linear-relu1"):
                self.W1_b2 = weight_variable(shape = [7 * 7 * 64, 1024])
                self.b1_b2 = bias_variable(shape = [1024])
                self.lu1_b2 = tf.matmul(self.p2_rep_dp_b2, self.W1_b2) + self.b1_b2
                self.ly1_b2 = tf.nn.relu(self.lu1_b2)
            with tf.name_scope("linear"):
                self.W2_b2 = weight_variable(shape = [1024, 10])
                self.b2_b2 = bias_variable(shape = [10])
                logits_b2 = tf.matmul(self.ly1_b2, self.W2_b2) + self.b2_b2

        # the 4td branch
        with tf.name_scope("conv-pool1-branch3"): # output: 14 x 14 x 32
            with tf.name_scope("conv"):
                self.W_conv1_b3 = weight_variable(shape = [5, 5, 1, 16])
                self.b_conv1_b3 = bias_variable(shape = [16])
                self.u1_b3 = tf.nn.conv2d(x, self.W_conv1_b3, strides = [1, 1, 1, 1], padding = "VALID") + self.b_conv1_b3
            with tf.name_scope("bn"):
                self.u1_bn_b3 = batch_normalization_layer(self.u1_b3, mean_var_decay, 16, isTrain = is_train)
            with tf.name_scope("relu"):
                self.y1_b3 = tf.nn.relu(self.u1_bn_b3)
            with tf.name_scope("max_pool"):
                self.p1_b3 = tf.nn.max_pool(self.y1_b3, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

        with tf.name_scope("conv-pool2-branch3"): # output: 6 x 6 x 64
            with tf.name_scope("conv"):
                self.W_conv2_b3 = weight_variable(shape = [5, 5, 16, 64])
                self.b_conv2_b3 = bias_variable(shape = [64])
                self.u2_b3 = tf.nn.conv2d(self.p1_b3, self.W_conv2_b3, strides = [1, 1, 1, 1], padding = "SAME") + self.b_conv2_b3
            with tf.name_scope("bn"):
                self.u2_bn_b3 = batch_normalization_layer(self.u2_b3, mean_var_decay, 64, isTrain = is_train)
            with tf.name_scope("relu"):
                self.y2_b3 = tf.nn.relu(self.u2_bn_b3)
            with tf.name_scope("max_pool"):
                self.p2_b3 = tf.nn.max_pool(self.y2_b3, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

        with tf.name_scope("conv-pool3-branch3"): # output: 3 x 3 x 128
            with tf.name_scope("conv"):
                self.W_conv3_b3 = weight_variable(shape = [5, 5, 64, 128])
                self.b_conv3_b3 = bias_variable(shape = [128])
                self.u3_b3 = tf.nn.conv2d(self.p2_b3, self.W_conv3_b3, strides = [1, 1, 1, 1], padding = "SAME") + self.b_conv3_b3
            with tf.name_scope("bn"):
                self.u3_bn_b3 = batch_normalization_layer(self.u3_b3, mean_var_decay, 128, isTrain = is_train)
            with tf.name_scope("relu"):
                self.y3_b3 = tf.nn.relu(self.u3_bn_b3)
            with tf.name_scope("max_pool"):
                self.p3_b3 = tf.nn.max_pool(self.y3_b3, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

        with tf.name_scope("classification_layer-branch3"):
            with tf.name_scope("reshape"):
                self.p3_rep_b3 = tf.reshape(self.p3_b3, [-1, 3 * 3 * 128])
            with tf.name_scope("dropout"):
                self.p3_rep_dp_b3 = tf.nn.dropout(self.p3_rep_b3, keep_prob = self.keep_prob)
            with tf.name_scope("linear-relu1"):
                self.W1_b3 = weight_variable(shape = [3 * 3 * 128, 1024])
                self.b1_b3 = bias_variable(shape = [1024])
                self.lu1_b3 = tf.matmul(self.p3_rep_dp_b3, self.W1_b3) + self.b1_b3
                self.ly1_b3 = tf.nn.relu(self.lu1_b3)
            with tf.name_scope("linear"):
                self.W2_b3 = weight_variable(shape = [1024, 10])
                self.b2_b3 = bias_variable(shape = [10])
                logits_b3 = tf.matmul(self.ly1_b3, self.W2_b3) + self.b2_b3

        '''
	self.w = tf.Variable(1.)
	self.w_b1 = tf.Variable(1.)
        self.w_b2 = tf.Variable(1.)
        '''
        self.logits = logits + logits_b1 + logits_b2 + logits_b3

	vars   = tf.trainable_variables() 
	self.lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in vars ]) * weight_decay

        with tf.name_scope("loss"):
            self.loss1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_, logits=logits), name = "loss1")
            self.loss2 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_, logits=logits_b1), name = "loss2")
            self.loss3 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_, logits=logits_b2), name = "loss3")
            self.loss4 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_, logits=logits_b3), name = "loss4")
            # self.loss_vote = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_, logits=self.logits), name = "loss_vote")
	    self.loss = self.loss1 + self.loss2 + self.loss3 + self.loss4 + self.lossL2

        with tf.name_scope("pred-acc"):
            self.correct_pred = tf.equal(tf.cast(tf.argmax(self.logits, 1), tf.int32), self.y_)
            self.pred = tf.argmax(self.logits, 1)
            self.acc = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

	self.weight_decay = tf.Variable(float(weight_decay), trainable=False, dtype = tf.float32)
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False,
                                         dtype=tf.float32)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)
	self.wd_decay_op = self.weight_decay.assign(self.weight_decay * 1.)

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


def batch_normalization_layer(inputs, mean_var_decay, channel, isTrain=True):
    # TODO: implemented the batch normalization func and applied it on conv and fully-connected layers
    # hint: you can add extra parameters (e.g., shape) if necessary
    mean, var = tf.nn.moments(inputs, axes = [0, 1, 2], keep_dims = False)
    ave_m = tf.Variable(tf.constant(0., shape = [channel]), trainable = False)
    ave_v = tf.Variable(tf.constant(0., shape = [channel]), trainable = False)
    batch_size = tf.to_float(tf.shape(inputs)[0])
    update_m = ave_m.assign(ave_m * mean_var_decay + mean * (1. - mean_var_decay))
    update_v = ave_v.assign(ave_v * mean_var_decay + var * batch_size / (batch_size - 1.) * (1. - mean_var_decay))
    tf.add_to_collection("update_mean_var_op", update_m)
    tf.add_to_collection("update_mean_var_op", update_v)
    scale = tf.Variable(tf.constant(1., shape = ave_m.shape))
    offset = tf.Variable(tf.constant(0., shape = ave_m.shape))
    
    if isTrain:
        m, v = tf.nn.moments(inputs, axes = [0, 1, 2], keep_dims = False)
    else:
        m, v = ave_m, ave_v
    return tf.nn.batch_normalization(inputs, m, v, offset, scale, tf.constant(1e-10))

