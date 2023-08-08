import torch_dipu
import torch
import torchvision.models as models

model = models.resnet18().cuda()
input = torch.randn(2, 3, 224, 224).cuda()
target = torch.zeros(2, 1000).cuda()

optimizers = [
    torch.optim.Adadelta(model.parameters(), lr=0.01),
    torch.optim.Adagrad(model.parameters(), lr=0.01),
    torch.optim.Adam(model.parameters(), lr=0.01),
    torch.optim.AdamW(model.parameters(), lr=0.01),
    torch.optim.Adamax(model.parameters(), lr=0.01),
    torch.optim.ASGD(model.parameters(), lr=0.01),
    torch.optim.NAdam(model.parameters(), lr=0.01),
    torch.optim.RAdam(model.parameters(), lr=0.01),
    torch.optim.RMSprop(model.parameters(), lr=0.01),
    torch.optim.Rprop(model.parameters(), lr=0.01),
    torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
]
loss_fn = torch.nn.CrossEntropyLoss()

for epoch in range(2):
    for optimizer in optimizers:
        optimizer.zero_grad()

    output = model(input)
    loss = loss_fn(output, target)
    loss.backward()

    for optimizer in optimizers:
        optimizer.step()