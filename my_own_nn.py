# created by clay on 05-22-20
import numpy as np
from matplotlib import pyplot as plt
import yfinance as yf

def scale_inputs(inputs):
    global range_of_inputs
    range_of_inputs = np.max(inputs) - np.min(inputs)
    return (inputs - np.min(inputs)) / range_of_inputs

def scale_outputs(outputs):
    global range_of_outputs
    range_of_outputs = np.max(outputs) - np.min(outputs)
    return (outputs - np.min(outputs)) / range_of_outputs

def scale(array):
    return (array - np.min(array)) / (np.max(array) - np.min(array))

class NeuralNetwork:
    
    def __init__(self, inputs, outputs, weights):
        self.inputs = inputs
        self.outputs = outputs
        self.weights = weights
        self.error_list = [] 
        self.epoch_list = []


    def sigmoid(self, x, derivative = False):
        if derivative == True:
            return x * (1 - x)
        return 1 / (1 + np.exp(-x))

    def reverse_sigmoid(self, y):
        return np.log(y / (1 - y))

    def feed_forward(self):
        self.hidden_layer = self.sigmoid(np.dot(self.inputs, self.weights), derivative=False)


    def back_propogate(self):
        self.cost = (self.outputs - self.hidden_layer) 
        stepsize = self.cost * self.sigmoid(self.hidden_layer, derivative=True)
        self.weights += np.dot(self.inputs.T, stepsize)


    def train(self, epochs=25000):
        for epoch in range(epochs):
            self.feed_forward()
            self.back_propogate()
            self.error_list = np.append(self.error_list, np.average(np.abs(self.cost)))
            self.epoch_list = np.append(self.epoch_list, epoch)

    def predict(self, new_data):
        return self.sigmoid(np.dot(new_data, self.weights))


data = yf.download('AAPL', start='2010-04-10', stop='2020-05-27')
data = np.array(data['High'])
scaled_data = scale(data)
# print(len(scaled_data))

def create_train_input(scaled_data):

    train_input = []
    train_size = 100
    for i in range(train_size,len(scaled_data)):
        input_layer = scaled_data[i-train_size:i]
        train_input = np.append(train_input,input_layer)
        
    return train_input

def create_train_output(scaled_data):

    train_size = 100
    train_output = []
    for i in range(train_size, len(scaled_data)):
        train_output = np.append(train_output, scaled_data[i])

    return train_output

print(create_train_input(scaled_data))
print(create_train_output(scaled_data))


# inputs = np.array([[1, 2, 3],
#                    [4, 5, 6],
#                    [7, 8, 9],
#                    [2, 3, 4]])

# outputs = np.array([[4], [7], [10], [5]])

# weights = np.array([[.8],[.4,],[.7]])

# inputs = scale_inputs(inputs)
# outputs = scale_outputs(outputs)

# Go = NeuralNetwork(inputs, outputs, weights)

# Go.train()

# test = np.array([[.3,.4,.5]])

# print(Go.predict(test))

# print(Go.epoch_list)
# print(Go.error_list)

# plt.plot(Go.epoch_list, Go.error_list)
# plt.show()
