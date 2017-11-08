import matplotlib.pyplot as plt
import numpy as np

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
        plt.xlabel("epoch")
        plt.ylabel("training loss")

        p2 = plt.subplot(1, 2, 2)
        plt.plot(epoch, acc)
        plt.xlabel("epoch")
        plt.ylabel("training accuracy")

        plt.show()

