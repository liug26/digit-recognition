import numpy as np

epsilon = 1e-8  # a very small number to avoid division by 0
network = []  # all the layers in the cnn
cost_map = []  # a list of cost function values calculated during training
t = 1  # the number of steps made in training
# hyper parameters
alpha = 0.01  # the learning rate 0.01


# computes a = activation(z), accepts numpy matrices
def activation(z, activation_func):
    """
    :param z: the unactivated matrix
    :param activation_func: points to one type of activation function
    :return: a = activation(z), a and z should be of the same dimension
    """
    if activation_func == "sigmoid":
        a = 1 / (1 + np.exp(-z))
        return a
    elif activation_func == "relu":
        return np.maximum(0, z)
    elif activation_func == "linear":
        return z
    else:
        print("unrecognized activation function: " + activation_func)


# computes a' = activation'(z), the derivative of activation(z), accepts numpy matrices
def activation_prime(z, activation_func):
    """
    :param z: the unactivated matrix
    :param activation_func: points to one type of activation function
    :return: a' = activation'(z), a' and z should be of the same dimension
    """
    if activation_func == "sigmoid":
        sigmoid = activation(z, "sigmoid")
        return sigmoid * (1 - sigmoid)
    elif activation_func == "relu":
        return (z > 0) * 1.0
    elif activation_func == "linear":
        return 1
    else:
        print("unrecognized activation function: " + activation_func)


class DenseLayer:
    def __init__(self, num_neurons, num_inputs, activation_func, initialization):
        """
        :param num_neurons: #neurons of this layer
        :param num_inputs: #inputs each neuron will receive, equals to #neurons in the previous layer
        :param activation_func: type of activation function
        :param initialization: type of weight initialization method
        """
        self.num_neurons = num_neurons
        self.num_inputs = num_inputs
        self.activation_func = activation_func
        self.x = np.zeros(num_inputs)
        self.z = np.zeros(num_neurons)
        # initialization
        if initialization == "rand":
            self.w = np.random.randn(num_neurons, num_inputs) * 0.01
            self.b = np.random.randn(num_neurons) * 0.01
        elif initialization == "kaiming":
            self.w = np.random.randn(num_neurons, num_inputs) * np.sqrt(2 / num_inputs)
            self.b = np.random.randn(num_neurons) * np.sqrt(2 / num_inputs)
        else:
            print("unrecognized initialization method: " + initialization)
        # gradient descent
        self.dw = np.zeros((num_neurons, num_inputs))
        self.db = np.zeros(num_neurons)

    # inputs a matrix x and outputs the weighted and activated matrix of x
    def forward_propagation(self, x):
        """
        :param x: the input matrix of dimension (#inputs, #samples)
        :return: a, the activation matrix of dimension (#neurons, #samples)
        """
        # x is a numpy array of dimension (#inputs)
        self.x = x  # store x for back prop
        for neuron_num in range(self.num_neurons):
            z = 0
            for weight_num in range(self.num_inputs):
                z += self.w[neuron_num][weight_num] * x[weight_num]
            z += self.b[neuron_num]
            self.z[neuron_num] = z  # stores z for back prop
        a = activation(self.z, self.activation_func)
        return a

    # inputs dJ/da of the current layer, updates its weights and b, outputs the dJ/da of the previous layer
    def back_propagation(self, da):
        """
        :param da: dJ/da of the current layer, a numpy array of dimension (#neurons)
        :return: da_prev: dJ/da of the previous layer, a numpy array of dimension (#inputs)
        """
        da_prev = np.zeros(self.num_inputs)
        for neuron_num in range(self.num_neurons):
            dz = da[neuron_num] * activation_prime(self.z[neuron_num], self.activation_func)
            for weight_num in range(self.num_inputs):
                self.dw[neuron_num][weight_num] += dz * self.x[weight_num]
                da_prev[weight_num] += dz * self.w[neuron_num][weight_num]
            self.db[neuron_num] += dz

        return da_prev

    # update weights and biases according to dw and db, and reset dw and db
    def update_weights(self, m_batch):
        """
        :param m_batch: the number of samples in the batch
        updates weights and bias, and reset dw and db
        """
        self.dw = self.dw / m_batch
        self.db = self.db / m_batch  # gets the mean of the gradients

        self.w = self.w - alpha * self.dw
        self.b = self.b - alpha * self.db

        # resets dw and db
        self.dw = np.zeros((self.num_neurons, self.num_inputs))
        self.db = np.zeros(self.num_neurons)


network.append(DenseLayer(num_neurons=96, num_inputs=784, activation_func="relu", initialization="kaiming"))
network.append(DenseLayer(num_neurons=80, num_inputs=96, activation_func="relu", initialization="kaiming"))
network.append(DenseLayer(num_neurons=10, num_inputs=80, activation_func="sigmoid", initialization="rand"))


# inputs an image and outputs the network's output
def test(x):
    """
    :param x: the input image, a numpy matrix of dimension (1, image_wdith, image_height)
    :return: y, the outputs of the network, a numpy matrix of dimension (#outputs, 1)
    """
    # do forward propagation across network
    for index, layer in enumerate(network):
        if index == 0:
            yhat = layer.forward_propagation(x)
        else:
            yhat = layer.forward_propagation(yhat)
    # the arrived yhat is the output of the network
    return yhat


# train the network with x and y2
def learn(x, y2):
    """
    :param x: the batch of input images, a numpy matrix of dimension (#inputs, m_batch)
    :param y2: the correct outputs y2, a numpy matrix of (10, m_batch)
    records the average cost of network's output
    """
    m_batch = x.shape[1]  # the #samples of the current batch
    aL_map = np.zeros((10, m_batch))  # records all aL to calculate cost, of dimension (#outputs, m)
    # do forward propagation for each sample
    for sample_num in range(m_batch):
        # forward propagate each sample across network
        for index, layer in enumerate(network):
            if index == 0:
                yhat = layer.forward_propagation(x[:, sample_num])
            else:
                yhat = layer.forward_propagation(yhat)

        aL_map[:, sample_num] = yhat.reshape(10, )  # store aL=yhat
        # calculate the dJ/daL of the current sample, of dimension (#outputs, 1)
        daL = -(np.divide(y2[:, sample_num], aL_map[:, sample_num] + epsilon) -
                np.divide(1 - y2[:, sample_num], 1 - aL_map[:, sample_num] + epsilon)).reshape(-1, 1)

        # do back propagation for each sample
        da = daL  # propagate da down the network
        for layer in reversed(network):
            da = layer.back_propagation(da)

    # calculate and record average cost J of this training step
    cost = -(np.sum(y2 * np.log(aL_map + epsilon) + (1 - y2) * np.log(1 - aL_map + epsilon)))
    cost_map.append(cost / m_batch)

    # update weights
    for layer in network:
        layer.update_weights(m_batch)

    # update t
    global t
    t += 1
