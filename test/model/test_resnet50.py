import pytest
import copy
import torch
import torch.nn as nn
import os
import os.path as osp
import torch._dynamo as dynamo
import mmcls
from mmcv import Config
from mmcls.models import build_classifier
from common import utils
import torch_dipu


def gen_mmcls_model():
    mmcls_root = osp.dirname(mmcls.__file__)
    mim_dir = osp.join(mmcls_root, ".mim")
    if osp.exists(mim_dir):
        config_parent = mim_dir
    else:
        config_parent = osp.dirname(mmcls_root)
    config_root = osp.join(config_parent, 'configs')

    resnet_cfg = Config.fromfile(
        osp.join(config_root, f'_base_/models/resnet50.py')
    )
    model = build_classifier(resnet_cfg.model)
    return model

def gen_fake_train_loader(shape, num_batches=100, num_classes=1000):
    npu_cache_data = [1] * num_batches
    for i in range(num_batches):
        images = torch.randn(*shape)
        target = torch.randint(low=0, high=num_classes, size=(shape[0],), dtype=torch.int64)
        npu_cache_data[i] = (images, target)
    return npu_cache_data


class TestResnet50():
    def test_forward_train(self, backend, dynamic, fake_batch_num=10, batch_size=32):
        utils.update_dynamo_config(dynamic=dynamic)
        device = utils.get_device()
        torch_dipu.dipu.set_device(device)
        cpu_model = gen_mmcls_model()
        dicp_model = copy.deepcopy(cpu_model)
        dicp_model = dicp_model.to(device)
        torch_dipu.current_stream().synchronize()

        cpu_model_forward = cpu_model.forward_train
        dicp_model_forward = dicp_model.forward_train

        cpu_optimizer = torch.optim.SGD(cpu_model.parameters(), lr=0.01,
                                        momentum=0.9, weight_decay=1e-4)
        dicp_optimizer = torch.optim.SGD(dicp_model.parameters(), lr=0.01,
                                        momentum=0.9, weight_decay=1e-4)
        
        dicp_compiled = torch.compile(dicp_model_forward, backend=backend, dynamic=dynamic)
        dicp_optimizer_step = torch.compile(dicp_optimizer.step, backend=backend, dynamic=dynamic)

        input_batch_shape = (batch_size, 3, 224, 224)
        cpu_train_loader = gen_fake_train_loader(input_batch_shape, num_batches=fake_batch_num)

        cpu_model.train()
        dicp_model.train()
        
        for _, (cpu_image, cpu_target) in enumerate(cpu_train_loader):
            # CPU
            cpu_loss = cpu_model_forward(cpu_image, cpu_target)
            cpu_real_loss = cpu_loss["loss"]
            cpu_optimizer.zero_grad()
            cpu_real_loss.backward()
            cpu_optimizer.step()

            # DICP
            dicp_image = cpu_image.to(device)
            dicp_target = cpu_target.to(device)
            torch_dipu.current_stream().synchronize()
            dicp_loss = dicp_compiled(dicp_image, dicp_target)
            torch_dipu.current_stream().synchronize()
            dicp_real_loss = dicp_loss["loss"]
            dicp_optimizer.zero_grad()
            dicp_real_loss.backward()
            dicp_optimizer.step()

            assert torch.allclose(cpu_real_loss.detach(), dicp_real_loss.cpu().detach(), atol=1e-02, equal_nan=True)

