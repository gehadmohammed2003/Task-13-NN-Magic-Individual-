import numpy as np

# Define the class for the simple neural network
class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        # Initialize weights and biases
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * 0.01
        self.bias_hidden = np.zeros((1, hidden_size))
        self.weights_hidden_output = np.random.randn(hidden_size, output_size) * 0.01
        self.bias_output = np.zeros((1, output_size))
        self.learning_rate = learning_rate
    
    def sigmoid(self, z):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_derivative(self, z):
        """Derivative of sigmoid function"""
        return self.sigmoid(z) * (1 - self.sigmoid(z))
    
    def forward_propagation(self, X):
        """Forward pass of the network"""
        # Hidden layer computations
        self.hidden_layer_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_layer_output = self.sigmoid(self.hidden_layer_input)
        
        # Output layer computations
        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights_hidden_output) + self.bias_output
        self.output_layer_output = self.sigmoid(self.output_layer_input)
        
        return self.output_layer_output
    
    def compute_loss(self, y_true, y_pred):
        """Compute the mean squared error loss"""
        return np.mean((y_true - y_pred) ** 2)
    
    def backpropagation(self, X, y_true, y_pred):
        """Backward pass to compute gradients and update weights"""
        # Compute the error in output
        output_error = y_pred - y_true
        output_delta = output_error * self.sigmoid_derivative(self.output_layer_input)
        
        # Compute the error in the hidden layer
        hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_layer_input)
        
        # Update the weights and biases (SGD)
        self.weights_hidden_output -= self.learning_rate * np.dot(self.hidden_layer_output.T, output_delta)
        self.bias_output -= self.learning_rate * np.sum(output_delta, axis=0, keepdims=True)
        self.weights_input_hidden -= self.learning_rate * np.dot(X.T, hidden_delta)
        self.bias_hidden -= self.learning_rate * np.sum(hidden_delta, axis=0, keepdims=True)
    
    def train(self, X, y_true, epochs=1000):
        """Training loop for the neural network"""
        for epoch in range(epochs):
            # Forward pass
            y_pred = self.forward_propagation(X)
            
            # Compute loss
            loss = self.compute_loss(y_true, y_pred)
            
            # Backpropagation and weight updates
            self.backpropagation(X, y_true, y_pred)
            
            # Print loss every 100 epochs
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Example Usage
if __name__ == "__main__":
    input_size = 784  # 28x28 input size for MNIST images
    hidden_size = 64  # Hidden layer with 64 neurons
    output_size = 10  # 10 output neurons for digits 0-9

    # Initialize a simple neural network
    nn = SimpleNeuralNetwork(input_size, hidden_size, output_size, learning_rate=0.01)

    # Dummy data for testing the training function
    X_dummy = np.random.randn(10, input_size)  # 10 samples of flattened 28x28 images
    y_dummy = np.random.randn(10, output_size)  # 10 random labels (dummy)

    # Train the network with dummy data
    nn.train(X_dummy, y_dummy, epochs=1000)
