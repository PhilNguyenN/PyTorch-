import torch.nn as nn
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
import numpy as np

from data_train import Data, Output

# Prepare Data

X_Trainning = torch.tensor(Data[0:3640],dtype=torch.float)     #70% data set 3640
y_Trainning = torch.tensor(Output[0:3640],dtype=torch.float)

X_Validating = torch.tensor(Data[3640:4420],dtype=torch.float)     #15%
y_Validating = torch.tensor(Output[3640:4420],dtype=torch.float)

# X_Testing = torch.tensor(Data[4420:5200])       #15%
# y_Testing = torch.tensor(Output[4420:5200])


# print(X_Trainning.size())
# print(y_Trainning.size())

# Scale units
# X_Trainning_Max, _ = torch.max(X_Trainning, 0)  
# X_Validating_Max, _ = torch.max(X_Validating, 0)
# #Note: the max function returns both a tensor and the corresponding indices. 
# #So use _ to capture the indices which we won't use here 
# #because we are only interested in the max values to conduct the scaling

# X_Trainning   = torch.div(X_Trainning,X_Trainning_Max)
# X_Validating  = torch.div(X_Validating,X_Validating_Max)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # parameters for each layers
        self.inputSize = 2
        self.outputSize = 1
        self.hiddenSize = 10
        
        self.W1 = torch.randn(self.inputSize, self.hiddenSize) #  2 X 3 tensor
        self.W2 = torch.randn(self.hiddenSize, self.outputSize) # 3 X 1 tensor

    def forward(self, X):                           # Forward signals to each layers
        
        self.fc1 = torch.matmul(X, self.W1)         # Matrix product of two tensors for hidden layer
     
        self.fc2 = self.sigmoid(self.fc1)           # Activation func 1st Hidden layer
        
        self.fc3 = torch.matmul(self.fc2, self.W2)  # Matrix product of two tensors for output layer
        
        out = self.sigmoid(self.fc3)                # Activation function output layer
        
        return out   

    def sigmoid(self, s):
        return 1/(1 + torch.exp(-s))            # Modify later

    def sigmoidPrime(self, s):
        return s*(1 - s)
    
    def backward(self, X, y, out):
        lr = 0.01
        self.o_error = y - out # error in output
        self.o_delta = self.o_error * self.sigmoidPrime(out)          # derivative of sig to error
        
        self.f2_error = torch.matmul(self.o_delta, torch.t(self.W2))
        self.f2_delta = self.f2_error * self.sigmoidPrime(self.fc2)
        
        self.W1 -= lr*torch.matmul(torch.t(X), self.f2_delta)            #torch.t transpose 
        self.W2 -= lr*torch.matmul(torch.t(self.fc2), self.o_delta)
        
    def train(self, X, y):
        # forward + backward pass for training
        pass_signal = self.forward(X)
        self.backward(X, y, pass_signal)
        
    def saveWeights(self, model):
        # we will use the PyTorch internal storage functions
        torch.save(model, "NN")
        # you can reload model with all the weights and so forth with:
        # torch.load("NN")
        
    def predict(self):
        print ("Predicted data based on trained weights: ")
        print ("Output: \n" + str(self.forward(X_Validating)))


#Trainning

NN = NeuralNetwork()

# print(NN(X_Trainning[0]))
# print(torch.t(X_Trainning[1]))

Epoch = 10
Save_data = []
for i in range(Epoch):
    print("Epoch ---------------------------------------------------------", i)
    for i in range(0,3640, 3):
        print ("#" + str(i) + " Loss: " + str(torch.mean((y_Trainning[i:i+3] - NN(X_Trainning[i:i+3]))**2).detach().item()))  # mean sum squared loss
        Save_data.append(torch.mean((y_Trainning[i:i+3] - NN(X_Trainning[i:i+3]))**2))
        NN.train(X_Trainning[i:i+3], y_Trainning[i:i+3])



# NN.saveWeights(NN)
# NN.predict()