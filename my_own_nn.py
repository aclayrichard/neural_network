# created by clay on 05-22-20
import numpy as np
from matplotlib import pyplot as plt
import yfinance as yf

def scale_inputs(inputs, outputs):

    if np.max(inputs) >= np.max(outputs):
        if np.min(inputs) <= np.min(outputs):

            range_inputs = np.max(inputs) - np.min(inputs)
            return (inputs / range_inputs) - (np.min(inputs) / range_inputs)
        else:

            range_inputs = np.max(inputs) - np.min(outputs)
            return (inputs / range_inputs) - (np.min(outputs) / range_inputs)
    else: 
        if np.min(inputs) <= np.min(outputs):

            range_inputs = np.max(outputs) - np.min(inputs)
            return (inputs / range_inputs) - (np.min(inputs) / range_inputs)
        else:

            range_inputs = np.max(outputs) - np.min(outputs)
            return (inputs / range_inputs) - (np.min(outputs) / range_inputs)

def scale_outputs(inputs, outputs):

    if np.max(inputs) >= np.max(outputs):
        if np.min(inputs) <= np.min(outputs):

            range_outputs = np.max(inputs) - np.min(inputs)
            return (outputs / range_outputs) - (np.min(inputs) / range_outputs)
        else:

            range_outputs = np.max(inputs) - np.min(outputs)
            return (outputs / range_outputs) - (np.min(outputs) / range_outputs)
    else:
        if np.min(inputs) <= np.min(outputs):

            range_outputs = np.max(outputs) - np.min(inputs)
            return (outputs / range_outputs) - (np.min(inputs) / range_outputs)
        else:

            range_outputs = np.max(outputs) - np.min(outputs)
            return (outputs / range_outputs) - (np.min(outputs) / range_outputs)

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


    def train(self, epochs=250000):
        for epoch in range(epochs):
            self.feed_forward()
            self.back_propogate()
            self.error_list = np.append(self.error_list, np.average(np.abs(self.cost)))
            self.epoch_list = np.append(self.epoch_list, epoch)


    def predict(self, new_data):
        return self.sigmoid(np.dot(new_data, self.weights))


data = yf.download('AAPL', start='2010-04-10', stop='2020-06-04')
data = np.array(data['High'])
scaled_data = scale(data)
# print(scaled_data)
# print(len(data))

def create_train_input(data):

    train_input = []
    train_size = 100
    for i in range(train_size,len(data)):
        
        train_input.append(data[i-train_size:i])
        
    return train_input

def create_train_output(data):

    train_output = []
    train_size = 100
    for i in range(train_size, len(data)):

        train_output.append(data[i])

    return train_output


train_size = 100

weights = []
for i in range(train_size):
    weights.append(np.random.uniform())
weights = np.reshape(weights, [-1, 1])
weights = weights.tolist()

inputs = create_train_input(data)
inputs = np.reshape(inputs,[len(inputs),train_size])
outputs = create_train_output(data)
outputs = np.reshape(outputs, [-1,1])

# print(inputs)
# print(len(outputs))
# print(weights)

scaled_inputs = scale_inputs(inputs, outputs)
scaled_outputs = scale_outputs(inputs, outputs)

# print(scaled_inputs, scaled_outputs)
# print(scaled_data[-100:])

# inputs = np.array([[1, 2, 3],
#                    [4, 5, 6],
#                    [7, 8, 9],
#                    [2, 3, 4]])

# weights = [[.8],[.7],[.4]]

# outputs = np.array([[4], [7], [10], [5]])

# print(inputs)
# print(outputs)
# print(type(weights))

# scaled_inputs = scale_inputs(inputs, outputs)
# scaled_outputs = scale_outputs(inputs, outputs)

## print(scaled_inputs)
## print(scaled_outputs)

Go = NeuralNetwork(scaled_inputs, scaled_outputs, weights)
Go.train()

# test = np.array([.1,.2,.3])
test = np.array(scaled_data[-100:])
print(Go.predict(test))

# print(Go.epoch_list)
# print(Go.error_list)

plt.plot(Go.epoch_list, Go.error_list)
plt.show()
