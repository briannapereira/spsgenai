#2 If the embedding dimension is 8, write out the values of the sinusoidal embedding vector for t = 1. Assume max period of 10000.
import numpy as np

t = 1
d = 8
max_period = 10000

half_dim = d // 2
freqs = np.exp(-np.log(max_period) * np.arange(half_dim) / (half_dim - 1))

sin_part = np.sin(t * freqs)
cos_part = np.cos(t * freqs)

embedding = np.concatenate([sin_part, cos_part])
print("PE(1)=",embedding)

#Question 6: Basic Gradient Calculationss
import torch
#Create a tensor with requires_grad=True
x = torch.tensor([2.0], requires_grad=True)
#Define a simple function y = x² + 3x
y = x**2 + 3 * x
#Backpropagate
y.backward()
#Print the gradient
print("x.grad =", x.grad)
#b What happens if you set requires grad=False on x?
x = torch.tensor([2.0], requires_grad=True)
y = x**2 + 3 * x
y.backward()
print("x.grad =", x.grad)

#Question 7: Introduce Weights
# Create a tensor with requires_grad=True
x = torch.tensor([2.0], requires_grad=True)
w = torch.tensor([1.0, 3.0], requires_grad=True) #question b
# Define a simple function y = x² + 3x
y = w[0] * x**2 + w[1] * x
# Backpropagate
y.backward()
# Print the gradient
print("x.grad =", x.grad)
print("w.grad =", w.grad)

#Question 8: Breaking the Graph
x = torch.tensor([1.0], requires_grad=True)
y = x * 3
z = y.clone() #changed to clone() after detach()
w = z * 2
w.backward()

#Question 9: Gradient Accumulation
x = torch.tensor([1.0], requires_grad=True)
y1 = x * 2
y1.backward()
print("After first backward: x.grad =", x.grad)

x.grad.zero_() #reset x.grad

y2 = x * 3
y2.backward()
print("After second backward: x.grad =", x.grad)