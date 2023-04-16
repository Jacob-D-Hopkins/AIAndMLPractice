import torch

X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
Y = torch.tensor([[.5, .2, .1, .7]], dtype=torch.float32)

print(torch.max(Y,1)[1])