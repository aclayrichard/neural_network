# created by clay on 05-22-20
import numpy as np
from matplotlib import pyplot as plt

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


# inputs = np.array([[1,0,0],
#                    [1,0,1],
#                    [0,1,0],
#                    [0,1,1]])

inputs = np.array([[.1, .2, .3],
                   [.4, .5, .6],
                   [.7, .8, .9],
                   [.2, .3, .4]])

# outputs = np.array([[1],[1],[0],[0]])

outputs = np.array([[.4], [.7], [1.0], [.5]])


weights = np.array([[.5],[.5,],[.5]])

Go = NeuralNetwork(inputs, outputs, weights)

Go.train()

test = np.array([[.4, .5, .6]])

answer = Go.predict(test)
# final = Go.reverse_sigmoid(answer)
print(answer)

# print(Go.epoch_list)
# print(Go.error_list)

plt.plot(Go.epoch_list, Go.error_list)
plt.show()
