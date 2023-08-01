import torch
import numpy as np


class TinyModel(torch.nn.Module):

    def __init__(self):
        super(TinyModel, self).__init__()

    def forward(self, proposals, gt):
        px = (proposals[:, 0] + proposals[:, 2]) * 0.5
        py = (proposals[:, 1] + proposals[:, 3]) * 0.5
        pw = proposals[:, 2] - proposals[:, 0]
        ph = proposals[:, 3] - proposals[:, 1]

        gx = (gt[:, 0] + gt[:, 2]) * 0.5
        gy = (gt[:, 1] + gt[:, 3]) * 0.5
        gw = gt[:, 2] - gt[:, 0]
        gh = gt[:, 3] - gt[:, 1]

        dx = (gx - px) / pw
        dy = (gy - py) / ph
        dw = torch.log(gw / pw)
        dh = torch.log(gh / ph)
        deltas = torch.stack([dx, dy, dw, dh], dim=-1)

        return deltas


def bbox2delta(proposals, gt):
    px = (proposals[..., 0] + proposals[..., 2]) * 0.5
    py = (proposals[..., 1] + proposals[..., 3]) * 0.5
    pw = proposals[..., 2] - proposals[..., 0]
    ph = proposals[..., 3] - proposals[..., 1]

    gx = (gt[..., 0] + gt[..., 2]) * 0.5
    gy = (gt[..., 1] + gt[..., 3]) * 0.5
    gw = gt[..., 2] - gt[..., 0]
    gh = gt[..., 3] - gt[..., 1]

    dx = torch.tensor((gx - px) / pw)
    dy = torch.tensor((gy - py) / ph)
    dw = torch.log(torch.tensor(gw / pw))
    dh = torch.log(torch.tensor(gh / ph))
    deltas = torch.stack([dx, dy, dw, dh], dim=-1)

    return deltas

def check_ret(message, ret):
    if ret != 0:
        raise Exception("{} failed ret={}"
                        .format(message, ret))

if __name__ == '__main__':
    tinymodel = TinyModel()

    # make input data
    compiled_model = torch.compile(tinymodel.forward, backend='ascendgraph', dynamic=True)
    shape = (1,4)
    input_data1 = torch.randn(*shape, dtype=torch.float32)
    input_data2 = torch.randn(*shape, dtype=torch.float32)
    print('shape: ', shape)
    result = compiled_model(input_data1, input_data2)
    print(result)
    print("Reference Result:")
    print(bbox2delta(input_data1, input_data2))

    shape = (2,4)
    input_data1 = torch.randn(*shape, dtype=torch.float32)
    input_data2 = torch.randn(*shape, dtype=torch.float32)
    print('shape: ', shape)
    result = compiled_model(input_data1, input_data2)
    print(result)
    print("Reference Result:")
    print(bbox2delta(input_data1, input_data2))

    shape = (3,4)
    input_data1 = torch.randn(*shape, dtype=torch.float32)
    input_data2 = torch.randn(*shape, dtype=torch.float32)
    print('shape: ', shape)
    result = compiled_model(input_data1, input_data2)
    print(result)
    print("Reference Result:")
    print(bbox2delta(input_data1, input_data2))
    
    shape = (4,4)
    input_data1 = torch.randn(*shape, dtype=torch.float32)
    input_data2 = torch.randn(*shape, dtype=torch.float32)
    print('shape: ', shape)
    result = compiled_model(input_data1, input_data2)
    print(result)
    print("Reference Result:")
    print(bbox2delta(input_data1, input_data2))
