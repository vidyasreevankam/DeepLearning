# -*- coding: utf-8 -*-
"""pytorchtensors.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1y3yeedx-N6HVXGUwrcsm85o91pTjPDZP
"""


def activation(x):
    """ Sigmoid activation function 
    
        Arguments
        ---------
        x: torch.Tensor
    """
    return 1/(1+torch.exp(-x))

### To Generate some data
torch.manual_seed(4) # Set the random seed

# Features are 5 random normal variables
features = torch.randn((1, 5))#data is like 1row and 5columns randomly distributed follows normal distribution mean 0 and stdev 1
# True weights for our data, random normal variables again
weights = torch.randn_like(features)#creates other tensor with same shape of features
# bias term
bias = torch.randn((1, 1))
y = activation(torch.sum(features * weights) + bias)
#y = activation((features * weights).sum() + bias)


