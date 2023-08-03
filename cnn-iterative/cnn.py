import numpy as np
import math

epsilon = 1e-8  # a very small number to avoid division by 0
network = []  # all the layers in the cnn
cost_map = []  # a list of cost function values calculated during training
t = 1  # the number of steps made in training
# hyper parameters
alpha = 0.01  # the learning rate 0.01
# adam optimizer
momentum_beta = 0.9  # 0 disables momentum
momentum_bias_corr = True
rms_prop_beta = 0.99  # 0 disables rms prop
rms_prop_bias_corr = True
# regularization
dropout_keep_prob = 0.8  # 1 disables dropout, only implemented in dense layers


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


class ConvLayer:
    def __init__(self, x_shape, pad, filter_shape, num_filters, stride, activation_func):
        """
        :param x_shape: tuple denoting the shape of x (#channels, height, width)
        :param pad: size of zero padding on x
        :param filter_shape: tuple denoting the shape of a filter (height, width)
        :param num_filters: number of filters
        :param stride: how large a step the filter moves across x
        :param activation_func: type of activation function
        """
        self.num_channels = x_shape[0]
        self.x_h = x_shape[1]
        self.x_w = x_shape[2]
        self.pad = pad
        self.f_h = filter_shape[0]
        self.f_w = filter_shape[1]
        self.num_filters = num_filters
        self.stride = stride
        self.activation_func = activation_func
        # filters is a 4D numpy matrix of dimension (#filters, #channels, height, width)
        self.filters = np.random.randn(self.num_filters, self.num_channels, self.f_h, self.f_w) * 1.0
        # b is a 1D numpy matrix of dimension (#filters)
        self.b = np.random.randn(self.num_filters) * 0.01
        # gradient descent
        self.df = np.zeros((self.num_filters, self.num_channels, self.f_h, self.f_w))
        self.db = np.zeros((self.num_filters))
        # width and height of the output images
        self.y_w = math.floor((self.x_w - self.f_w + 2 * self.pad) / self.stride) + 1
        self.y_h = math.floor((self.x_h - self.f_h + 2 * self.pad) / self.stride) + 1
        self.z = np.zeros((self.num_filters, self.y_h, self.y_w))
        # adam optimizer
        self.vdf = np.zeros((self.num_filters, self.num_channels, self.f_h, self.f_w))
        self.sdf = np.zeros((self.num_filters, self.num_channels, self.f_h, self.f_w))
        self.vdb = np.zeros((self.num_filters))
        self.sdb = np.zeros((self.num_filters))

    # inputs a 3D matrix x and returns the convoluted 3D matrix
    def forward_propagation(self, x):
        """
        :param x: the input image x, a numpy matrix of dimension x_shape (#channels, x_h, x_w)
        :return: the output image a of dimension (#filters, y_h, y_w)
        """
        assert x.shape == (self.num_channels, self.x_h, self.x_w)  # assumes x's shape is desired
        self.x = x  # store x for back prop
        padded_x = np.pad(x, ((0, 0), (self.pad, self.pad), (self.pad, self.pad)), mode="constant")
        # z is the unactivated output image of dimension (#filters, y_h, y_w)
        self.z = np.zeros((self.num_filters, self.y_h, self.y_w))

        for filter_num in range(self.num_filters):
            # filter is the filter_num_th filter of dimension (#channels, f_h, f_w)
            filter = self.filters[filter_num, :, :, :]
            for y_row in range(self.y_h):
                for y_col in range(self.y_w):  # y_row and y_col points to every element on y
                    # a_slice is a (#channels, f_h, f_w) matrix that will go through convolution
                    a_slice = padded_x[:, y_row * self.stride: y_row * self.stride + self.f_h,
                              y_col * self.stride: y_col * self.stride + self.f_w]
                    conv_value = np.sum(a_slice * filter)  # convolution
                    z = conv_value + self.b[filter_num]
                    self.z[filter_num, y_row, y_col] = z  # stores z for back prop

        return activation(self.z, self.activation_func)

    # inputs dJ/da of the current layer, updates filters and b, outputs the dJ/da of the previous layer
    def back_propagation(self, da):
        """
        :param da: dJ/da of the current layer, a matrix of dimension (#filters, y_h, y_w)
        :return: dJ/da of the previous layer, a matrix of dimension (#channels, x_h, x_w)
        """
        # dz: change in unactivated output, of dimension (#filters, y_h, y_w)
        dz = da * activation_prime(self.z, self.activation_func)
        # pad x to return calculate a_slice
        padded_x = np.pad(self.x, ((0, 0), (self.pad, self.pad), (self.pad, self.pad)), mode="constant")
        # df: change in weights of filter
        df = np.zeros((self.num_filters, self.num_channels, self.f_h, self.f_w))
        # db: change in bias applied to each filter
        db = np.zeros((self.num_filters))
        # the padded dJ/da of the previous layer
        padded_da_prev = np.zeros((self.num_channels, self.x_h + 2 * self.pad, self.x_w + 2 * self.pad))
        for filter_num in range(self.num_filters):
            for channel_num in range(self.num_channels):
                for y_row in range(self.y_h):
                    for y_col in range(self.y_w):  # filter_row and filter_col points every element on every filter
                        a_slice = padded_x[channel_num, y_row * self.stride: y_row * self.stride + self.f_h,
                                  y_col * self.stride: y_col * self.stride + self.f_w]  # the a_slice of z[y_row, y_col]
                        df[filter_num, channel_num] += dz[filter_num, y_row, y_col] * a_slice
                        db[filter_num] += dz[filter_num, y_row, y_col]
                        padded_da_prev[channel_num, y_row * self.stride: y_row * self.stride + self.f_h,
                        y_col * self.stride: y_col * self.stride + self.f_w] \
                            += dz[filter_num, y_row, y_col] * self.filters[filter_num, channel_num]

        # get rid of the padding of padded dJ/da_prev
        if self.pad == 0:
            da_prev = padded_da_prev
        else:
            # if pad=0 it will return a matrix dimension of (:, 0:-0, 0:-0) = (:, 1, 1)
            da_prev = padded_da_prev[:, self.pad: -self.pad, self.pad: -self.pad]

        # record change in filter and bias
        self.df = self.df + df
        self.db = self.db + db

        return da_prev

    # update filters and biases according to df and db, and reset df and db
    def update_weights(self, m_batch):
        """
        :param m_batch: the number of samples in the batch
        updates filter and bias, and reset df and db
        """
        self.df = self.df / m_batch
        self.db = self.db / m_batch  # gets the mean of the gradients

        # adam optimizer
        self.vdf = momentum_beta * self.vdf + (1 - momentum_beta) * self.df
        self.vdb = momentum_beta * self.vdb + (1 - momentum_beta) * self.db
        self.sdf = rms_prop_beta * self.sdf + (1 - rms_prop_beta) * np.square(self.df)
        self.sdb = rms_prop_beta * self.sdb + (1 - rms_prop_beta) * np.square(self.db)
        vdf_corr, vdb_corr, sdf_corr, sdb_corr = np.copy(self.vdf), np.copy(self.vdb), np.copy(self.sdf), \
                                                 np.copy(self.sdb)
        if momentum_bias_corr:
            vdf_corr = vdf_corr / (1 - np.power(momentum_beta, t))
            vdb_corr = vdb_corr / (1 - np.power(momentum_beta, t))
        if rms_prop_bias_corr:
            sdf_corr = sdf_corr / (1 - np.power(rms_prop_beta, t))
            sdb_corr = sdb_corr / (1 - np.power(rms_prop_beta, t))

        if rms_prop_beta == 0:
            self.filters = self.filters - alpha * vdf_corr
            self.b = self.b - alpha * vdb_corr
        else:
            self.filters = self.filters - alpha * vdf_corr / (np.sqrt(sdf_corr) + epsilon)
            self.b = self.b - alpha * vdb_corr / (np.sqrt(sdb_corr) + epsilon)

        # resets dw and db
        self.df = np.zeros((self.num_filters, self.num_channels, self.f_h, self.f_w))
        self.db = np.zeros((self.num_filters))


class PoolLayer:
    def __init__(self, x_shape, filter_shape, stride, pool_type):
        """
        :param x_shape: tuple denoting the shape of x (#channels, height, width)
        :param filter_shape: tuple denoting the shape of a filter (height, width)
        :param stride: how large a step the filter moves across x
        :param pool_type: type of pooling, max or average
        """
        self.num_channels = x_shape[0]
        self.x_h = x_shape[1]
        self.x_w = x_shape[2]
        self.f_h = filter_shape[0]
        self.f_w = filter_shape[1]
        self.stride = stride
        self.pool_type = pool_type
        # width and height of the output images
        self.y_w = math.floor((self.x_w - self.f_w) / self.stride) + 1
        self.y_h = math.floor((self.x_h - self.f_h) / self.stride) + 1

    # inputs a 3D matrix x and returns the pooled 3D matrix y
    def forward_propagation(self, x):
        """
        :param x: the input image x, a numpy matrix of dimension x_shape (#channels, x_h, x_w)
        :return: y, the pooled image of dimension (#channels, y_h, y_w)
        """
        assert x.shape == (self.num_channels, self.x_h, self.x_w)  # assumes x's shape is desired
        self.x = x  # stores x for back prop
        y = np.zeros((self.num_channels, self.y_h, self.y_w))
        for y_row in range(self.y_h):
            for y_col in range(self.y_w):  # y_row and y_col points to every element on y
                # a_slice is a (#channels, f_h, f_w) matrix that will go through convolution
                a_slice = x[:, y_row * self.stride: y_row * self.stride + self.f_h,
                          y_col * self.stride: y_col * self.stride + self.f_w]
                # mask is a (#channels, f_h, f_w) that executes the pooling via convolution
                mask = self.create_mask(a_slice)
                pooled = np.sum(a_slice * mask, axis=(1, 2))  # pooling step
                y[:, y_row, y_col] = pooled

        return y

    # creates a mask that pools a_slice according to pool_type
    def create_mask(self, a_slice):
        """
        :param a_slice: a (#channels, f_h, f_w) matrix that will be pooled
        :return: a mask of dimension (#channels, f_h, f_w) that executes the pooling
        """
        if self.pool_type == "average":
            '''
            average pooling mask is a 3D matrix of the same dimension as a_slice, with all its values being 1 / 
            #elements-in-one-channel-of-a_slice
            '''
            return np.ones((self.num_channels, self.f_h, self.f_w)) * 1 / self.f_w / self.f_h
        elif self.pool_type == "maximum":
            '''
            raw_mask is a 3D matrix of the same dimension as a_slice, raw_mask is 1 if the corresponding element in
            a_slice is the maximum value in the 1-channel image, 0 otherwise.
            because the maximum value may appear more than 1 time in the 1-channel image, an additional step is made
            to make sure that all the values in the 1-channel mask add up to 1 by dividing all the appeared 1s
            '''
            raw_mask = (a_slice == np.max(a_slice, axis=(1, 2), keepdims=True)).astype(float)
            return raw_mask / np.sum(raw_mask, axis=(1, 2), keepdims=True)
        else:
            print("unrecognized pooling type: " + self.pool_type)

    # inputs dJ/da of the current layer, and outputs the dJ/da of the previous layer
    def back_propagation(self, da):
        """
        :param da: dJ/da of the current layer, a matrix of dimension (#channels, y_h, y_w)
        :return: dJ/da of the previous layer, a matrix of dimension (#channels, x_h, x_w)
        """
        # the dJ/da of the previous layer
        da_prev = np.zeros((self.num_channels, self.x_h, self.x_w))
        # no for loop for channel# is needed because it can be vectorized
        for y_row in range(self.y_h):
            for y_col in range(self.y_w):  # for each element in dJ/da
                # the a_slice that becomes y at y_row, y_col, of dimension (#channels, f_h, f_w)
                a_slice = self.x[:, y_row * self.stride: y_row * self.stride + self.f_h,
                          y_col * self.stride: y_col * self.stride + self.f_w]
                da_prev[:, y_row * self.stride: y_row * self.stride + self.f_h,
                y_col * self.stride: y_col * self.stride + self.f_w] \
                    = da[:, y_row, y_col].reshape(-1, 1, 1) * self.create_mask(a_slice)
        return da_prev

    # this function exists for uniformity
    def update_weights(self, m_batch):
        """
        :param m_batch: the number of samples in the batch
        does nothing
        """
        pass


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
        self.z = None
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
        # gradient descent
        self.dw = np.zeros((num_neurons, num_inputs))
        self.db = np.zeros((num_neurons, 1))
        # adam optimizer
        self.vdw = np.zeros((num_neurons, num_inputs))
        self.sdw = np.zeros((num_neurons, num_inputs))
        self.vdb = np.zeros((num_neurons, 1))
        self.sdb = np.zeros((num_neurons, 1))

    # inputs a 2D matrix x and outputs the weighted and activated matrix of x
    def forward_propagation(self, x):
        """
        :param x: the input matrix of dimension (#inputs, #samples)
        :return: a, the activation matrix of dimension (#neurons, #samples)
        """
        assert x.shape[0] == self.num_inputs  # assumes x is in desired shape
        self.x = x  # stores x for back prop
        self.z = np.dot(self.w, x) + self.b  # store z for back prop
        a = activation(self.z, self.activation_func)

        # dropout regularization
        if network.index(self) != len(network) - 1:
            self.d = np.random.rand(a.shape[0], a.shape[1])
            self.d = (self.d < dropout_keep_prob).astype(int)
            a = a * self.d
            a = a / dropout_keep_prob
        return a

    # inputs dJ/da of the current layer, updates its weights and b, outputs the dJ/da of the previous layer
    def back_propagation(self, da):
        """
        :param da: dJ/da of the current layer, a matrix of dimension (#neurons, #samples)
        :return: dJ/da of the previous layer, a matrix of dimension (#inputs, #samples)
        """
        # dropout regularization
        if network.index(self) != len(network) - 1:
            da = da * self.d
            da = da / dropout_keep_prob

        m_batch = da.shape[1]  # #samples
        dz = da * activation_prime(self.z, self.activation_func)
        dw = 1 / m_batch * np.dot(dz, self.x.T)
        db = 1 / m_batch * np.sum(dz, axis=1, keepdims=True)
        da_prev = np.dot(self.w.T, dz)

        # record dw and db
        self.dw = self.dw + dw
        self.db = self.db + db

        return da_prev

    # update weights and biases according to dw and db, and reset dw and db
    def update_weights(self, m_batch):
        """
        :param m_batch: the number of samples in the batch
        updates weights and bias, and reset dw and db
        """
        self.dw = self.dw / m_batch
        self.db = self.db / m_batch  # gets the mean of the gradients

        # adam optimizer
        self.vdw = momentum_beta * self.vdw + (1 - momentum_beta) * self.dw
        self.vdb = momentum_beta * self.vdb + (1 - momentum_beta) * self.db
        self.sdw = rms_prop_beta * self.sdw + (1 - rms_prop_beta) * np.square(self.dw)
        self.sdb = rms_prop_beta * self.sdb + (1 - rms_prop_beta) * np.square(self.db)
        vdw_corr, vdb_corr, sdw_corr, sdb_corr = np.copy(self.vdw), np.copy(self.vdb), np.copy(self.sdw), \
                                                 np.copy(self.sdb)
        if momentum_bias_corr:
            vdw_corr = vdw_corr / (1 - np.power(momentum_beta, t))
            vdb_corr = vdb_corr / (1 - np.power(momentum_beta, t))
        if rms_prop_bias_corr:
            sdw_corr = sdw_corr / (1 - np.power(rms_prop_beta, t))
            sdb_corr = sdb_corr / (1 - np.power(rms_prop_beta, t))

        if rms_prop_beta == 0:
            self.w = self.w - alpha * vdw_corr
            self.b = self.b - alpha * vdb_corr
        else:
            self.w = self.w - alpha * vdw_corr / (np.sqrt(sdw_corr) + epsilon)
            self.b = self.b - alpha * vdb_corr / (np.sqrt(sdb_corr) + epsilon)

        # resets dw and db
        self.dw = np.zeros((self.num_neurons, self.num_inputs))
        self.db = np.zeros((self.num_neurons, 1))


# used to connect convolution & pooling layers with dense layers
class FlattenLayer:
    def __init__(self, x_shape):
        """
        :param x_shape: tuple denoting the shape of x (#channels, height, width)
        """
        self.x_shape = x_shape

    def forward_propagation(self, x):
        """
        :param x: 3D matrix of dimension x_shape
        :return: y, the flattened x of dimension (#flattened_elements, 1)
        """
        assert x.shape == self.x_shape
        return x.reshape(-1, 1)

    # inputs dJ/da of the current layer, outputs the dJ/da of the previous layer
    def back_propagation(self, da):
        """
        :param da: dJ/da of the current layer, a matrix of dimension (#flattened_elements, 1)
        :return: dJ/da of the previous layer, a matrix of dimension (#channels, height, width)
        """
        return da.reshape(self.x_shape)

    # exists for uniformity
    def update_weights(self, m_batch):
        """
        :param m_batch: the number of samples in the batch
        does nothing
        """
        pass


# initialize cnn
# lenet structure, but using relu (because sigmoid doesn't do well with padding and x)
network.append(ConvLayer(x_shape=(1, 28, 28), pad=2, filter_shape=(5, 5), num_filters=6, stride=1,
                         activation_func="relu"))
network.append(PoolLayer(x_shape=(6, 28, 28), filter_shape=(2, 2), stride=2, pool_type="average"))
network.append(ConvLayer(x_shape=(6, 14, 14), pad=0, filter_shape=(5, 5), num_filters=16, stride=1,
                         activation_func="relu"))
network.append(PoolLayer(x_shape=(16, 10, 10), filter_shape=(2, 2), stride=2, pool_type="average"))
network.append(FlattenLayer(x_shape=(16, 5, 5)))
network.append(DenseLayer(num_neurons=120, num_inputs=400, activation_func="relu", initialization="kaiming"))
network.append(DenseLayer(num_neurons=84, num_inputs=120, activation_func="relu", initialization="kaiming"))
network.append(DenseLayer(num_neurons=10, num_inputs=84, activation_func="sigmoid", initialization="rand"))

# an all-dense network
'''
network.append(FlattenLayer(x_shape=(1, 28, 28)))
network.append(DenseLayer(num_neurons=120, num_inputs=784, activation_func="sigmoid", initialization="rand"))
network.append(DenseLayer(num_neurons=84, num_inputs=120, activation_func="sigmoid", initialization="rand"))
network.append(DenseLayer(num_neurons=10, num_inputs=84, activation_func="sigmoid", initialization="rand"))
'''


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
    :param x: the batch of input images, a numpy matrix of dimension (m_batch, image_wdith, image_height)
    :param y2: the correct outputs y2, a numpy matrix of (10, m_batch)
    records the average cost of network's output
    """
    m_batch = x.shape[0]  # the #samples of the current batch
    aL_map = np.zeros((10, m_batch))  # records all aL to calculate cost, of dimension (#outputs, m)
    # do forward propagation for each sample
    for sample_num in range(m_batch):
        # forward propagate each sample across network
        for index, layer in enumerate(network):
            if index == 0:
                yhat = layer.forward_propagation(x[sample_num].reshape(1, 28, 28))
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
