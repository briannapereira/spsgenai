import torch
import torch.nn as nn

#5 Given a mini-batch of 4 values: [6, 8, 10, 6], compute the normalized output using BatchNorm without γ and β (i.e., just zero-mean, unit-variance).
x = torch.tensor([[6.0], [8.0], [10.0], [6.0]])
bn = nn.BatchNorm1d(num_features=1, affine=False)
bn.train()
y = bn(x)

print("#5 Answers:", y.squeeze())