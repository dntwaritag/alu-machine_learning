#!/usr/bin/env python3
'''This class defines a single neuron for binary classification'''

import numpy as np
import matplotlib.pyplot as plt


class Neuron:
    '''Neuron class that defines a single neuron
    performing binary classification'''

    def __init__(self, nx):
        '''Initialize the neuron'''
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        self.__W = np.random.randn(1, nx)  
        self.__b = 0  # Bias initialization to 0
        self.__A = 0  # Activated output initialized to 0

    @property
    def W(self):
        '''Getter for the weights vector'''
        return self.__W

    @property
    def b(self):
        '''Getter for the bias'''
        return self.__b

    @property
    def A(self):
        '''Getter for the activated output'''
        return self.__A

    def forward_prop(self, X):
        '''Perform forward propagation and
        calculate the neuron's activated output'''
        Z = np.dot(self.__W, X) + self.__b 
        # Apply the sigmoid activation function
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        '''Calculate the cost (cross-entropy loss)'''
        m = Y.shape[1]  # Number of examples
        # Cross-entropy loss formula
        cost = - (1 / m) * np.sum(Y * np.log(A) +
                                  (1 - Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        '''Evaluate the neuron's predictions'''
        A = self.forward_prop(X)  # Get the activated output (predictions)
        # Convert probabilities (A) to binary predictions (0 or 1)
        predictions = np.where(A >= 0.5, 1, 0)
        cost = self.cost(Y, A)  # Calculate the cost based on predictions
        return predictions, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        '''Perform one pass of gradient descent to update weights and bias'''
        m = Y.shape[1]  # Number of examples
        error = A - Y  # Error between predicted and actual labels
        # Gradient of the weights
        dW = (1 / m) * np.dot(error, X.T)
        # Gradient of the bias
        db = (1 / m) * np.sum(error)
        # Update weights and bias
        self.__W -= alpha * dW
        self.__b -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        '''
        Train the neuron using gradient descent

        Args:
            X (numpy.ndarray): Input data of shape (nx, m), where nx is the
            number of input features and m is the number of examples.
            Y (numpy.ndarray): Correct labels of shape (1, m).
            iterations (int): Number of iterations to train over.
            alpha (float): Learning rate for gradient descent.
            verbose (bool): Whether or not to print training information.
            graph (bool): Whether or not to graph the training cost.
            step (int): Interval of iterations to print or graph information.

        Returns:
            tuple: The evaluation of the training data after iterations.
        '''

        if not isinstance(iterations, int):
            raise TypeError('iterations must be an integer')
        if iterations <= 0:
            raise ValueError('iterations must be a positive integer')
        if not isinstance(alpha, float):
            raise TypeError('alpha must be a float')
        if alpha <= 0:
            raise ValueError('alpha must be positive')

        if (verbose or graph):
            if not isinstance(step, int):
                raise TypeError('step must be an integer')
            if step <= 0 or step > iterations:
                raise ValueError('step must be positive and <= iterations')

        # Initialize lists to store the cost and iteration steps
        costs = []
        iteration_steps = []

        # Training loop for the specified number of iterations
        for i in range(iterations + 1):
            A = self.forward_prop(X)  # Perform forward propagation
            self.gradient_descent(X, Y, A, alpha)  # Update weights and bias

            # Every `step` iterations, record cost and optionally print/log it
            if i % step == 0 or i == iterations:
                cost = self.cost(Y, A)  # Calculate current cost
                costs.append(cost)
                iteration_steps.append(i)

                if verbose:
                    # Print cost information if verbose is True
                    print(f"Cost after {i} iterations: {cost}")

        if graph:
            # Plot the cost function over iterations if graph is True
            plt.plot(iteration_steps, costs, 'b')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        # Return the evaluation (predictions and cost)
        # after the final iteration
        return self.evaluate(X, Y)
