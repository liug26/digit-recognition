import numpy as np
import pickle
import cnn

pickle_path = "network.pickle"
txt_path = "network.txt"

'''
the pickle way of saving and reading network is convenient and fast in python, but it will be hard to transfer pickle 
files across different programming languages
'''


# reads and loads network at network_path
def save():
    with open(pickle_path, 'wb') as handle:
        pickle.dump(cnn.t, handle)
        pickle.dump(cnn.cost_map, handle)
        pickle.dump(cnn.network, handle)
    print("network saved")


# saves the network at network_path
def load():
    with open(pickle_path, 'rb') as handle:
        cnn.t = pickle.load(handle)
        cnn.cost_map = pickle.load(handle)
        cnn.network = pickle.load(handle)
    print("network loaded")