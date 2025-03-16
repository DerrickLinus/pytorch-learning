import torch
from torch import nn

class Derrick(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, input):
        output = input + 1
        return output
    
derrick = Derrick()
output = derrick(1.0)
print(output)
print(type(output))