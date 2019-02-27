import torch
import torch.nn as nn

import pandas as pd
from sklearn.model_selection import train_test_split

x = torch.tensor([[0.69,0.13,0.13],[0.91,0.58,0.47],[0.18,0.30,0.30],[0.67,0.84,0.90],[0.58,0,0.82],[0.86,0.62,0.86],[0.09,0.09,0.43],[0.96,0.96,0.86],[0.18,0.54,0.34],[1,0.38,0.27]])
y = torch.tensor([0,1,0,1,0,1,0,1,0,1])

#Dark -0
#Light - 1

# dataset = pd.read_csv('dataset//RGB.csv')

# train_X, test_X, train_y, test_y = train_test_split(dataset[dataset.columns[0:3]].values,
#                                                     dataset.Combination.values)

# train_X = torch.Tensor(train_X).float()
# test_X = torch.Tensor(test_X).float()
# train_y = torch.Tensor(train_y).long()
# test_y = torch.Tensor(test_y).long()

class LinearClassifier(nn.Module):
	def __init__(self):
		super(LinearClassifier, self).__init__()
		self.h_layer = nn.Linear(3, 2)
		self.s_layer = nn.Softmax()

	def forward(self,x):
		y = self.h_layer(x)
		p = self.s_layer(y)
		return p


model = LinearClassifier() #declaring the classifier to an object
loss_fn = nn.CrossEntropyLoss() #calculates the loss
optim = torch.optim.SGD(model.parameters(), lr = 0.01)

all_losses = []
for num in range(5000): #5000 iterations
	pred = model(x) #predict
	loss = loss_fn(pred, y) #calculate loss
	all_losses.append(loss.data)
	optim.zero_grad() #zero gradients to not accumulate
	loss.backward() #update weights based on loss
	optim.step() #update optimiser for next iteration


#Save model
torch.save(model.state_dict(), 'model.ckpt')

