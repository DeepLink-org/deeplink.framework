import torch

class Operator():
    __name__: str
    def __init__(self, name_):
        super().__init__()
        self.__name__ = name_
    
    def __call__(self, *args, **kwds):
        return self.__name__

class Add(Operator):
    def __init__(self, a, b):
        super().__init__("add")
        self.a = a
        self.b = b

class Abs(Operator):
    def __init__(self, a):
        super().__init__("abs")
        self.a = a

@torch.fx.wrap
def mull(a, b) -> torch.Tensor:
    return torch.mul(a, b)

@torch.fx.wrap
def abs(a) -> torch.Tensor:
    return torch.abs(a)

@torch.fx.wrap
def add(a, b) -> torch.Tensor:
    return torch.add(a, b)