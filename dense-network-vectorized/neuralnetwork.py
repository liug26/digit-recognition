import numpy as np
epsilon = 1e-8  # a very small number to avoid division by 0
network = []  # all the layers in the cnn
cost_map = []  # a list of cost function values calculated during training
t = 1  # the number of steps made in training
# hyper parameters
alpha = 0.001  # the learning rate
# regularization
l2_reg_lambda = 0.001  # 0 disables l2 regularization, 0.001 is good
dropout_keep_prob = 0.8  # 1.0 disables dropout regularization, 0.8 is good
# adam optimizer
momentum_beta = 0.9  # 0 disables momentum
momentum_bias_corr = True
rms_prop_beta = 0.99  # 0 disables rms prop
rms_prop_bias_corr = True


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
        '''
        w: weight matrix of dimension (#neurons, #inputs)
        b: bias matrix of dimension (#neurons, 1)
        '''
        if initialization == "rand":
            self.w = np.random.randn(num_neurons, num_inputs) * 0.01
            self.b = np.random.randn(num_neurons, 1) * 0.01
        elif initialization == "kaiming":
            self.w = np.random.randn(num_neurons, num_inputs) * np.sqrt(2 / num_inputs)
            self.b = np.random.randn(num_neurons, 1) * np.sqrt(2 / num_inputs)
        else:
            print("unrecognized initialization method: " + initialization)

    # inputs a 2D matrix x and outputs the unactivated and activated matrix z and a
    def forward_propagation(self, x):
        """
        :param x: the input matrix of dimension (#inputs, #samples)
        :return: a, the activation matrix of dimension (#neurons, #samples)
        """
        assert x.shape[0] == self.num_inputs  # assumes x is in desired shape
        z = np.dot(self.w, x) + self.b
        a = activation(z, self.activation_func)
        return z, a


# initialize network
n_inputs = 784
n_neurons = [96, 80, 10]
activation_f = ["relu", "relu", "sigmoid"]
init = ["kaiming", "kaiming", "rand"]
v_dw, v_db, s_dw, s_db = [0] * len(n_neurons), [0] * len(n_neurons), [0] * len(n_neurons), [0] * len(n_neurons)
for i in range(len(n_neurons)):
    if i == 0:
        network.append(DenseLayer(n_neurons[i], n_inputs, activation_f[i], init[i]))
    else:
        network.append(DenseLayer(n_neurons[i], n_neurons[i - 1], activation_f[i], init[i]))


# inputs an image and returns the network's outputs
def test(x, y):
    """
    :param x: the input image, a numpy matrix of dimension (784, m)
    :param y, the labels of the input images, a numpy matrix of dimension (1, m)
    prints the correct times, total times, and the accuracy rate of the test
    """
    # do forward propagation across network
    for layer_num, layer in enumerate(network):
        if layer_num == 0:
            z, a = layer.forward_propagation(x)
        else:
            z, a = layer.forward_propagation(a)

    answers = np.argmax(a, axis=0, keepdims=True)  # return the index of the largest element in the column
    results = (answers == y).astype(int)  # if answers = labels
    correct_times = np.sum(results)  # count the times answers = labels
    all_times = x.shape[1]
    print("test result: " + str(correct_times) + " out of " + str(all_times) + " times (" +
          str(correct_times / all_times) + ")")


# train the network with x and y2
def learn(x, y):
    """
    :param x: the batch of input images, a numpy matrix of dimension (784, m_batch)
    :param y: actually main's y2, the correct outputs of the network, a numpy matrix of (10, m_batch)
    records the average cost of network's output
    """
    global v_dw, v_db, s_dw, s_db, t, cost_map
    z_cache, a_cache, d_cache = [], [], []  # these caches store the z, a, and dropout mask of each layer
    m = x.shape[1]  # really m_batch, but for convenience just call it m
    cost = 0  # the cost of this step

    # do forward propagation across network
    for i, layer in enumerate(network):
        if i == 0:
            z, a = layer.forward_propagation(x)
        else:
            z, a = layer.forward_propagation(a)

        # implement dropout (all masks will be all 1s if keep_prob = 1)
        if i == len(network) - 1:
            # no dropout implemented on the topmost layer
            d = np.ones(a.shape)
        else:
            # create dropout mask
            d = np.random.rand(a.shape[0], a.shape[1])
            d = (d < dropout_keep_prob).astype(int)
            a = a * d
            a = a / dropout_keep_prob

        # store z, a, and dropout mask
        d_cache.append(d)
        z_cache.append(z)
        a_cache.append(a)

        # l2 regularization extra step: include l2 cost in cost
        cost += l2_reg_lambda / 2 / m * np.sum(np.square(layer.w))

    # calculate and store average cost of this step
    cost += -1 / m * np.sum(y * np.log(a + epsilon) + (1 - y) * np.log(1 - a + epsilon))
    cost_map.append(cost)

    # place x at end of list so that a_cache[-1] will refer to x, x will be removed after gd
    a_cache.append(x)
    dw_cache, db_cache = [], []  # stores the gradients
    for l in reversed(range(len(network))):
        if l == len(network) - 1:  # if it's the topmost layer
            da = -1 / m * (np.divide(y, a + epsilon) - np.divide(1 - y,
                                     1 - a + epsilon))  # epsilon to prevent division by 0
        da = da * d_cache[l]  # apply the dropout mask
        da = da / dropout_keep_prob

        # gradients calculation
        dz = da * activation_prime(z_cache[l], network[l].activation_func)
        dw = 1 / m * np.dot(dz, a_cache[l - 1].T)
        db = 1 / m * np.sum(dz, axis=1, keepdims=True)
        dw = dw + l2_reg_lambda / m * network[l].w  # l2 regularization

        # stores the gradients
        dw_cache.insert(0, dw)
        db_cache.insert(0, db)

        # calculate the gradients for the previous layer
        da = np.dot(network[l].w.T, dz)
    del a_cache[-1]  # deletes the x in a_cache

    # update the weights
    for l in range(len(network)):
        # adam optimizer
        v_dw[l] = momentum_beta * v_dw[l] + (1 - momentum_beta) * dw_cache[l]
        v_db[l] = momentum_beta * v_db[l] + (1 - momentum_beta) * db_cache[l]
        s_dw[l] = rms_prop_beta * s_dw[l] + (1 - rms_prop_beta) * np.square(dw_cache[l])
        s_db[l] = rms_prop_beta * s_db[l] + (1 - rms_prop_beta) * np.square(db_cache[l])
        v_dw_corr, v_db_corr, s_dw_corr, s_db_corr = np.copy(v_dw[l]), np.copy(v_db[l]), np.copy(s_dw[l]),\
                                                     np.copy(s_db[l])
        if momentum_bias_corr:
            v_dw_corr = v_dw_corr / (1 - np.power(momentum_beta, t))
            v_db_corr = v_db_corr / (1 - np.power(momentum_beta, t))
        if rms_prop_bias_corr:
            s_dw_corr = s_dw_corr / (1 - np.power(rms_prop_beta, t))
            s_db_corr = s_db_corr / (1 - np.power(rms_prop_beta, t))

        if rms_prop_beta == 0:
            network[l].w = network[l].w - alpha * v_dw_corr
            network[l].b = network[l].b - alpha * v_db_corr
        else:
            network[l].w = network[l].w - alpha * v_dw_corr / (np.sqrt(s_dw_corr) + epsilon)
            network[l].b = network[l].b - alpha * v_db_corr / (np.sqrt(s_db_corr) + epsilon)

    # update t
    t += 1