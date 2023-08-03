import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkio as nio
import cnn
import math
from tqdm import tqdm

x, y, y2 = [], [], []  # the dataset


def main():
    """
    this is the main thread
    """
    nio.load()  # load network from nio
    read()  # load dataset
    train(start=0, end=40000, batch_size=64, num_epochs=10)  # if batch_size=1, it is stochastic gradient descent
    print("training set accuracy: ")
    test(start=0, end=40000)
    print("dev set accuracy: ")
    test(start=40000, end=41000)
    print("test set accuracy: ")
    test(start=41000, end=42000)


def test(start, end):
    """
    :param start: the starting index of testing data
    :param end: the ending index of testing data
    prints the accuracy rate of the test, #correct answers, and total #answers
    """
    corr_times = 0
    progress_bar = tqdm(total=end - start)
    for sample_num in range(start, end):
        yhat = cnn.test(x[sample_num, :, :].reshape(1, x.shape[1], x.shape[2]))  # gets network's output
        answer = yhat.argmax(axis=0)  # the index whose element has the largest value
        if answer == y[0, sample_num]:
            corr_times += 1  # got it right
        progress_bar.update(1)
    progress_bar.close()
    print("accuracy: " + str(corr_times) + " out of " + str(end - start) + " times (" +
          str(corr_times / (end - start)) + ")")


# trains the network through gradient descent
def train(start, end, batch_size, num_epochs):
    """
    :param start: the index of the first training sample
    :param end: the index of the last training sample
    :param batch_size: the amount of training samples used in one step of gradient descent
    :param num_epochs: the number of passes through the entire training set
    """
    num_batches = math.ceil((end - start) / batch_size)
    progress_bar = tqdm(total=num_batches * num_epochs)
    for e in range(num_epochs):
        for b in range(num_batches):
            batch_start = start + batch_size * b
            batch_end = min(end, start + batch_size * (b + 1))  # start and end index of the current batch
            cnn.learn(x[batch_start: batch_end, :, :], y2[:, batch_start: batch_end])
            progress_bar.update(1)
        nio.save()  # autosave every epoch
    progress_bar.close()

    # display cnn's cost map
    x_points = range(len(cnn.cost_map))
    y_points = cnn.cost_map
    plt.plot(x_points, y_points)
    plt.show()


# read the dataset and define x, y and y2
def read():
    """
    defines x, y, and y2
    x: the image data, a numpy matrix of dimension (m, 28, 28)
    y: the image labels, a numpy matrix of (1, m)
    y2: the correct outputs, a numpy matrix of (10, m)
    """
    global x, y, y2
    df = pd.read_csv("train.csv")
    x_temp = df.to_numpy()  # read from csv and convert it to a numpy matrix
    y = x_temp[:, 0:1].T
    x_temp = x_temp[:, 1:].T  # separates data labels w/ data
    x_temp = x_temp / 255  # contains x between 0 and 1, makes training easier
    m = x_temp.shape[1]
    y2 = np.zeros((10, m))  # init y2, x.shape[1] = m = #samples
    y2[y.flatten(), range(x_temp.shape[1])] = 1.  # if the col index = the data label, have that element equals 1
    '''
    if we were working with only dense layers, then the initialized dataset is good to go
    but since we are starting with convolutional layers, we need to convert x from dimension (784, m) to (m, 28, 28)
    '''
    x = x_temp.T.reshape(m, 28, 28)


if __name__ == '__main__':
    main()
