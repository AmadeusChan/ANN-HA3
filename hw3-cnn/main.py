# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import os
import time
from model import Model
from load_data import load_mnist_4d
import sys
from scipy import misc
from scipy import ndimage

tf.app.flags.DEFINE_integer("batch_size", 150, "batch size for training")
tf.app.flags.DEFINE_integer("num_epochs", 200, "number of epochs")
tf.app.flags.DEFINE_float("keep_prob", 0.5, "drop out rate")
tf.app.flags.DEFINE_boolean("is_train", True, "False to inference")
tf.app.flags.DEFINE_string("data_dir", "./MNIST_data", "data dir")
tf.app.flags.DEFINE_string("train_dir", "./train", "training dir")
tf.app.flags.DEFINE_integer("inference_version", 0, "the param version for inference")
FLAGS = tf.app.flags.FLAGS


def shuffle(X, y, shuffle_parts):
    chunk_size = len(X) / shuffle_parts
    shuffled_range = range(chunk_size)

    X_buffer = np.copy(X[0:chunk_size])
    y_buffer = np.copy(y[0:chunk_size])

    for k in range(shuffle_parts):
        np.random.shuffle(shuffled_range)
        for i in range(chunk_size):
            X_buffer[i] = X[k * chunk_size + shuffled_range[i]]
            y_buffer[i] = y[k * chunk_size + shuffled_range[i]]

        X[k * chunk_size:(k + 1) * chunk_size] = X_buffer
        y[k * chunk_size:(k + 1) * chunk_size] = y_buffer

    return X, y

iteration = 0

def train_epoch(model, sess, X, y):
    global iteration
    loss, acc = 0.0, 0.0
    st, ed, times = 0, FLAGS.batch_size, 0
    while st < len(X) and ed <= len(X):
        X_batch, y_batch = X[st:ed], y[st:ed]
        feed = {model.x_: X_batch, model.y_: y_batch, model.keep_prob: FLAGS.keep_prob}
        loss_, acc_, _, __= sess.run([model.loss, model.acc, model.train_op, tf.get_collection("update_mean_var_op")], feed)
        loss += loss_
        acc += acc_
        st, ed = ed, ed+FLAGS.batch_size
        times += 1

        iteration += 1
        with open(train_file, "a") as f:
            f.write(str(iteration) + " " + str(loss_) + " " + str(acc_) + "\n")
    loss /= times
    acc /= times
    return acc, loss


def valid_epoch(model, sess, X, y):
    loss, acc = 0.0, 0.0
    st, ed, times = 0, FLAGS.batch_size, 0
    while st < len(X) and ed <= len(X):
        X_batch, y_batch = X[st:ed], y[st:ed]
        feed = {model.x_: X_batch, model.y_: y_batch, model.keep_prob: 1.0}
        loss_, acc_ = sess.run([model.loss, model.acc], feed)
        loss += loss_
        acc += acc_
        st, ed = ed, ed+FLAGS.batch_size
        times += 1
    loss /= times
    acc /= times
    return acc, loss


def inference(model, sess, X):
    return sess.run([model.pred], {model.x_: X, model.keep_prob: 1.0})[0]

if not os.path.exists("result"):
    os.mkdir("result")

train_file = "result/train.txt"
valid_file = "result/valid.txt"
test_file = "result/test.txt"

if len(sys.argv)>1:
    train_file = "result/" + sys.argv[1]
if len(sys.argv)>2:
    valid_file = "result/" + sys.argv[2]
if len(sys.argv)>3:
    test_file = "result/" + sys.argv[3]

os.system("rm " + train_file)
os.system("touch " + train_file)
os.system("rm " + valid_file)
os.system("touch " + valid_file)
os.system("rm " + test_file)
os.system("touch " + test_file)

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4) 

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    if not os.path.exists(FLAGS.train_dir):
        os.mkdir(FLAGS.train_dir)
    if FLAGS.is_train:
        X_train, X_test, y_train, y_test = load_mnist_4d(FLAGS.data_dir)

        # to shuffle the trainning data
        temp = np.arange(X_train.shape[0])
        np.random.shuffle(temp)
        X_train = X_train[temp]
        y_train = y_train[temp]

        X_val, y_val = X_train[50000:], y_train[50000:]
        X_train, y_train = X_train[:50000], y_train[:50000]

        cnn_model = Model(is_train=True)

        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter("/tmp/mnist_logs", sess.graph)

        '''
        if tf.train.get_checkpoint_state(FLAGS.train_dir):
            cnn_model.saver.restore(sess, tf.train.latest_checkpoint(FLAGS.train_dir))
        else:
            tf.global_variables_initializer().run()
        '''
        tf.global_variables_initializer().run()

        pre_losses = [1e18] * 3
        best_val_acc = 0.0
        for epoch in range(FLAGS.num_epochs):
            start_time = time.time()
            train_acc, train_loss = train_epoch(cnn_model, sess, X_train, y_train)
            X_train, y_train = shuffle(X_train, y_train, 1)

            val_acc, val_loss = valid_epoch(cnn_model, sess, X_val, y_val)

            with open(valid_file, "a") as f:
                f.write(str(epoch) + " " + str(val_loss) + " " + str(val_acc) + "\n")

            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                test_acc, test_loss = valid_epoch(cnn_model, sess, X_test, y_test)
                cnn_model.saver.save(sess, '%s/checkpoint' % FLAGS.train_dir, global_step=cnn_model.global_step)

                with open(test_file, "a") as f:
                    f.write(str(test_loss) + " " + str(test_acc) + "\n")

            epoch_time = time.time() - start_time
            print("Epoch " + str(epoch + 1) + " of " + str(FLAGS.num_epochs) + " took " + str(epoch_time) + "s")
            print("  learning rate:                 " + str(cnn_model.learning_rate.eval()))
            print("  training loss:                 " + str(train_loss))
            print("  validation loss:               " + str(val_loss))
            print("  validation accuracy:           " + str(val_acc))
            print("  best epoch:                    " + str(best_epoch))
            print("  best validation accuracy:      " + str(best_val_acc))
            print("  test loss:                     " + str(test_loss))
            print("  test accuracy:                 " + str(test_acc))
            print "\n"

            if train_loss > max(pre_losses):
                sess.run(cnn_model.learning_rate_decay_op)
            pre_losses = pre_losses[1:] + [train_loss]

    else:
        cnn_model = Model(is_train=False)
        if FLAGS.inference_version == 0:
            model_path = tf.train.latest_checkpoint(FLAGS.train_dir)
        else:
            model_path = '%s/checkpoint-%08d' % (FLAGS.train_dir, FLAGS.inference_version)
        cnn_model.saver.restore(sess, model_path)
        X_train, X_test, y_train, y_test = load_mnist_4d(FLAGS.data_dir)

        count = 0
        for i in range(len(X_test)):
            test_image = X_test[i].reshape((1, 1, 28, 28))
            result = inference(cnn_model, sess, test_image)[0]
            if result == y_test[i]:
                count += 1
        print("test accuracy: {}".format(float(count) / len(X_test)))
