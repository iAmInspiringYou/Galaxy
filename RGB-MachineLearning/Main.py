import torch
import torch.nn as nn
import RGBClassifier as classifier
import sys

#Importing the model, and loading the model paramters
model = classifier.LinearClassifier()
model.load_state_dict(torch.load('model.ckpt'))

#Testing the model by giving random RGB values
test_sample = torch.tensor([[1.0,1.0,1.0]]) #Change the input here. 
test_pred = model(test_sample)
output = torch.argmax(test_pred)


print("Light color") if output.item() == 1 else print("Dark color")