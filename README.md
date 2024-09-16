# Task-13-NN-Magic-Individual-
Neural Network from Scratch for MNIST Classification
Introduction:
This project implements a simple neural network from scratch based on the foundational concepts introduced in Chapter 1 of Michael Nielsen's book Neural Networks and Deep Learning. The network is designed to classify handwritten digits (0-9) from the MNIST dataset. The goal is to demonstrate a basic understanding of neural networks without using any deep learning libraries.

Approach:
Architecture:

Input Layer: 784 neurons (for 28x28 pixel images).
Hidden Layer: 64 neurons.
Output Layer: 10 neurons (one for each digit: 0-9).
Activation Function:

Sigmoid Activation Function is used in both the hidden and output layers.
Forward Propagation:

The input is fed through the network, and activations are computed layer by layer. The network applies the sigmoid activation function at each layer and outputs a prediction vector.
Loss Function:

The network uses Mean Squared Error (MSE) to calculate the difference between the predicted outputs and the actual labels.
Backpropagation:

Gradients of the loss with respect to the weights and biases are calculated using the chain rule, and the weights are updated using Stochastic Gradient Descent (SGD).
Improvements:
This basic implementation can be extended by adding more advanced techniques:

Better Optimizers: Switching from Stochastic Gradient Descent to optimizers like Adam or RMSprop.
Regularization: Adding L2 regularization or dropout to reduce overfitting.
Weight Initialization: Improving weight initialization to speed up convergence.

Running the Code:
Install NumPy:
pip install numpy
Run the neural_network.py file to train the model. The network is initialized with random weights and trained over 1000 epochs using random dummy data. You can replace the dummy data with actual MNIST dataset images and labels.

Files Included:
neural_network.py: The code for the neural network.
README.md: This file.

Conclusion:
This project demonstrates the construction of a basic neural network from scratch, applying fundamental concepts such as forward propagation, backpropagation, and gradient descent without using any high-level machine learning libraries.
