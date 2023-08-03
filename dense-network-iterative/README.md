# Digit Recognition w/ Dense Layers from Scratch [Iterative]

## Summary
The contained python files provide a simple framework to implement and train a dense-layer neural network that recognizes hand-written digits. The framework is written from scratch (no TensorFlow, only NumPy). To make it simple, I incorporated only gradient descent. Everything is iterative using for loops, so training can get a bit slow, but much quicker than an iterative CNN!

## Structure
main.py -- where the main method is stored and where training/testing is called.  
neuralnetwork.py -- the network module containing the class DenseLayer, responsible for forward and backward propagation of the network.  
networkio.py -- stores and reads network and training information in files via pickle.  
training.csv -- the dataset.  

## Data source
The dataset (training.csv after unzipping training.zip) comes from: https://www.kaggle.com/c/digit-recognizer
