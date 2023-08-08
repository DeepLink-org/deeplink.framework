import torch
from torch.optim.optimizer import required
from typing import Optional

# DIPU does not support foreach operator which using in optimizers.
# So we monkey patch all optimizers that support foreach parameter,
# and set foreach to False.
# The optimizer that supports foreach parameters can be found in the link https://pytorch.org/docs/stable/optim.html

class Adadelta(torch.optim.Adadelta):
    def __init__(self, params, lr=1.0, rho=0.9, eps=1e-6, weight_decay=0,
                 foreach: Optional[bool] = None, *, maximize: bool = False, differentiable: bool = False):
        foreach = False
        super().__init__(params, lr, rho, eps, weight_decay, foreach,
                         maximize = maximize, differentiable = differentiable)


class Adagrad(torch.optim.Adagrad):
    def __init__(self, params, lr=1e-2, lr_decay=0, weight_decay=0, initial_accumulator_value=0,
                 eps=1e-10, foreach: Optional[bool] = None, *, maximize: bool = False, differentiable: bool = False):
        foreach = False
        super().__init__(params, lr, lr_decay, weight_decay, initial_accumulator_value, eps,
                         foreach, maximize = maximize, differentiable = differentiable)


class Adam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, *, foreach: Optional[bool] = None,
                 maximize: bool = False, capturable: bool = False,
                 differentiable: bool = False, fused: Optional[bool] = None):
        foreach = False
        super().__init__(params, lr, betas, eps, weight_decay, amsgrad,
                         foreach = foreach, maximize = maximize, capturable = capturable,
                         differentiable = differentiable, fused = fused)


class AdamW(torch.optim.AdamW):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, amsgrad=False,
        *, maximize: bool = False, foreach: Optional[bool] = None, capturable: bool = False,
        differentiable: bool = False, fused: Optional[bool] = None):
        foreach = False
        super().__init__(params, lr, betas, eps, weight_decay, amsgrad,
                         foreach = foreach, maximize = maximize, capturable = capturable,
                         differentiable = differentiable, fused = fused)


class Adamax(torch.optim.Adamax):
    def __init__(self, params, lr=2e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0,
                 foreach: Optional[bool] = None, *, maximize: bool = False, differentiable: bool = False):
        foreach = False
        super().__init__(params, lr, betas, eps, weight_decay, foreach,
                         maximize = maximize, differentiable = differentiable)


class ASGD(torch.optim.ASGD):
    def __init__(self, params, lr=1e-2, lambd=1e-4, alpha=0.75, t0=1e6, weight_decay=0,
                 foreach: Optional[bool] = None, maximize: bool = False, differentiable: bool = False):
        foreach = False
        super().__init__(params, lr, lambd, alpha, t0, weight_decay,
                         foreach, maximize, differentiable)


class NAdam(torch.optim.NAdam):
    def __init__(self, params, lr=2e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, momentum_decay=4e-3, *, foreach: Optional[bool] = None,
                 differentiable: bool = False):
        foreach = False
        super().__init__(params, lr, betas, eps, weight_decay, momentum_decay,
                         foreach = foreach, differentiable = differentiable)


class RAdam(torch.optim.RAdam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0,
                 *, foreach: Optional[bool] = None, differentiable: bool = False):
        foreach = False
        super().__init__(params, lr, betas, eps, weight_decay,
                         foreach = foreach, differentiable = differentiable)


class RMSprop(torch.optim.RMSprop):
    def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0,
                 momentum=0, centered=False, foreach: Optional[bool] = None,
                 maximize: bool = False, differentiable: bool = False):
        foreach = False
        super().__init__(params, lr, alpha, eps, weight_decay, momentum, centered,
                         foreach, maximize, differentiable)


class Rprop(torch.optim.Rprop):
    def __init__(self, params, lr=1e-2, etas=(0.5, 1.2), step_sizes=(1e-6, 50), *,
                 foreach: Optional[bool] = None,
                 maximize: bool = False,
                 differentiable: bool = False):
        foreach = False
        super().__init__(params, lr, etas, step_sizes, foreach = foreach,
                         maximize = maximize, differentiable = differentiable)


class SGD(torch.optim.SGD):
    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, *, maximize: bool = False, foreach: Optional[bool] = None,
                 differentiable: bool = False):
        foreach = False
        super().__init__(params, lr, momentum, dampening, weight_decay, nesterov,
                         maximize = maximize, foreach = foreach, differentiable = differentiable)


def apply_optim_patch():
    torch.optim.Adadelta = Adadelta
    torch.optim.Adagrad = Adagrad
    torch.optim.Adam = Adam
    torch.optim.AdamW = AdamW
    torch.optim.Adamax = Adamax
    torch.optim.ASGD = ASGD
    torch.optim.NAdam = NAdam
    torch.optim.RAdam = RAdam
    torch.optim.RMSprop = RMSprop
    torch.optim.Rprop = Rprop
    torch.optim.SGD = SGD