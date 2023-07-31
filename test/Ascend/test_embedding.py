import torch

embedding = torch.nn.Embedding(10, 3)

def fn(input):
    y = embedding(input)
    return y

opt_model = torch.compile(fn, backend='ascendgraph')
torch.manual_seed(1)
input = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]])
print(input)
y = opt_model(input)
print(y)
print(y.shape)