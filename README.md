---
title: Multi-Layer Perceptron regression in C++ code
---
# Multi-Layer Perceptrons in C++

Multi-layer perceptrons (MLP's), or artificial neural networks, are gaining in popularity across a variety of applications. From turbulence closure models in computational fluid dynamics to data regression applications, MLP's have been shown to be a valuable tool. This library was developed to easily allow for the evaluation of MLP's within a C++ code environment. To clarify: this library DOES NOT allow for the training or optimization of MLP architectures. It only allows for the evaluation of MLP's which already have been trained through an external tool like Python Tensorflow. 

# MLP class description
The MLP library can be downloaded from the git repository https://github.com/EvertBunschoten/MLPCpp.git. By including the header file CLookUp_ANN.hpp, it enables the use of multi-layer perceptrons for regression operations in C++ code. 
An MLP computes data by having its inputs manipulated by a series of operations, depending on the architecture of the network. Interpreting the network architecture and its respective input and output variables is therefore crucial for the MLP functionality. Information regarding the network architecture, input and output variables, and activation functions has to be provided via a .mlp input file, of which two examples are provided in the main library folder. More information regarding the file structure is provided in a later section. 

The main class governing  which can be used for look-up operations is the CLookUp_ANN class. This class allows to load one or multiple networks given a list of input files. Each of these files is read by the CReadNeuralNetwork class. This class reads the .mlp input file and stores the architectural information listed in it. It will also run some compatibility checks on the file format. For example, the total layer count should be provided before listing the activation functions. For every read .mlp file, an MLP class is generated using the CNeuralNetwork class. This class stores the input and output variables, network architecture, activation functions, and weights and biases of every synapse and neuron. Currently, the CNeuralNetwork class only supports simple, feed-forward, dense neural network types. Supported activation function types are:
1: linear (y = x)
2: relu
3: elu
4: swish
5: sigmoid
6: tanh
7: selu
8: gelu
9: exponential (y = exp(x))

It is possible to load multiple networks with different input and output variables. An error will be raised if none of the loaded MLP's contains all the input variables or if some of the desired outputs are missing from the MLP output variables.
In addition to loading multiple networks with different input and output variables, it is possible to load multple networks with the same input and output variables, but with different data ranges. When performing a regression operation, the CLookUp_ANN class will check which of the loaded MLPs with the right input and output variables has an input variable normalization range that includes the query point. The corresponding MLP will then be selected for regression. If the query point lies outside the data range of all loaded MLPs, extrapolation will be performed using the MLP with a data range close to the query point. 

# MLP Definition
In order to load an MLP architecture into the library, an input file needs to be provided describing the network architecture, as well as the input and output variable names and ranges. An example of such an input file is provided in this folder (MLP_1.mlp and MLP_2.mlp). This file can be generated from an MLP trained through Tensorflow using the "Tensorflow_Translation.py" script, which can be found under "src". Details regarding the functionality of this translation script can be found in the code itself.
