import matplotlib.pyplot as plt
import numpy as np

def read_valid(name):
    with open("result/" + name, "r") as f:
        epoch = []
        loss = []
        acc = []
        while True:
            line = f.readline()
            if line == None or len(line) == 0:
                break
            z = line.split(" ")
            epoch.append(int(z[0]))
            loss.append(float(z[1]))
            acc.append(float(z[2]))
    return epoch, loss, acc

def plot_valid(name):
    with open("result/" + name, "r") as f:
        epoch = []
        loss = []
        acc = []
        while True:
            line = f.readline()
            if line == None or len(line) == 0:
                break
            z = line.split(" ")
            epoch.append(int(z[0]))
            loss.append(float(z[1]))
            acc.append(float(z[2]))

        p1 = plt.subplot(1, 2, 1)

        plt.plot(epoch, loss)
        plt.xlabel("epoch")
        plt.ylabel("validation loss")

        p2 = plt.subplot(1, 2, 2)
        plt.plot(epoch, acc)
        plt.xlabel("epoch")
        plt.ylabel("validation accuracy")

        plt.show()

def read_train(name):
    with open("result/" + name, "r") as f:
        epoch = []
        loss = []
        acc = []

        loss_ = []
        acc_ = []
        while True:
            line = f.readline()
            if line == None or len(line) == 0:
                break
            z = line.split(" ")
            e = int(z[0])
            loss_.append(float(z[1]))
            acc_.append(float(z[2]))
            if e % 100 == 0:
                epoch.append(e)
                loss.append(np.mean(loss_))
                acc.append(np.mean(acc_))
                loss_ = []
                acc_ = []
    return epoch, loss, acc


def plot_train(name):
    with open("result/" + name, "r") as f:
        epoch = []
        loss = []
        acc = []

        loss_ = []
        acc_ = []
        while True:
            line = f.readline()
            if line == None or len(line) == 0:
                break
            z = line.split(" ")
            e = int(z[0])
            loss_.append(float(z[1]))
            acc_.append(float(z[2]))
            if e % 100 == 0:
                epoch.append(e)
                loss.append(np.mean(loss_))
                acc.append(np.mean(acc_))
                loss_ = []
                acc_ = []

        p1 = plt.subplot(1, 2, 1)

        plt.plot(epoch, loss)
        plt.xlabel("iteration")
        plt.ylabel("training loss")

        p2 = plt.subplot(1, 2, 2)
        plt.plot(epoch, acc)
        plt.xlabel("iteration")
        plt.ylabel("training accuracy")

        plt.show()

p1 = plt.subplot(1, 2, 1)
plt.xlabel("iteration")
plt.ylabel("loss")

p2 = plt.subplot(1, 2, 2)
plt.xlabel("iteration")
plt.ylabel("accuracy")

index, loss, acc = read_train("train_nobn.txt")
p1.plot(index, loss, label = "without batch normalization")
p2.plot(index, acc, label = "without batch normalization")

index, loss, acc = read_train("train_bn.txt")
p1.plot(index, loss, label = "with batch normalization")
p2.plot(index, acc, label = "with batch normalization")

p1.legend(loc='upper right')
p2.legend(loc='lower right')

plt.show()
