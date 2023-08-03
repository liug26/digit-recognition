import numpy as np
import pickle
import neuralnetwork

pickle_path = "network.pickle"
txt_path = "network.txt"

'''
the pickle way of saving and reading network is convenient and fast in python, but it will be hard to transfer pickle 
files across different programming languages
'''


# reads and loads network at network_path
def save():
    with open(pickle_path, 'wb') as handle:
        pickle.dump(neuralnetwork.t, handle)
        pickle.dump(neuralnetwork.cost_map, handle)
        pickle.dump(neuralnetwork.network, handle)
        pass
    print("network saved")


# saves the network at network_path
def load():
    with open(pickle_path, 'rb') as handle:
        neuralnetwork.t = pickle.load(handle)
        neuralnetwork.cost_map = pickle.load(handle)
        neuralnetwork.network = pickle.load(handle)
    print("network loaded")