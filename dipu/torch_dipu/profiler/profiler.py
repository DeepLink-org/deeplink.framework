import torch
from torch_dipu import _C

def dipu_kineto_available():
    return True

def apply_profiler_patch():
    setattr(torch.profiler.profiler, 'kineto_available', dipu_kineto_available)
    setattr(torch.autograd.profiler, 'kineto_available', dipu_kineto_available)
    setattr(torch.autograd.profiler, '_prepare_profiler', _C._prepare_profiler)
    setattr(torch.autograd.profiler, '_enable_profiler', _C._enable_profiler)
    setattr(torch.autograd.profiler, '_disable_profiler', _C._disable_profiler)
    setattr(torch.autograd.profiler, '_kineto_step', _C._kineto_step)
    setattr(torch.autograd.profiler, '_supported_activities', _C._supported_activities)
    setattr(torch.autograd, '_supported_activities', _C._supported_activities)
    setattr(torch.autograd, '_add_metadata_json', _C._add_metadata_json)