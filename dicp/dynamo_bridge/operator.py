import torch
from abc import ABC

from torch._functorch import config
from torch.utils._pytree import tree_map, tree_flatten
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from contextlib import nullcontext
from torch._subclasses import FakeTensor, FakeTensorMode
from dicp.dynamo_bridge.utils import TensorInfo


class Operator(ABC):
    __name__: str
    _singleton = None

    def __init__(self, name_):
        super().__init__()
        self.__name__ = name_
        if torch.__version__.startswith("2.0"):
            self.shape_env = ShapeEnv() if config.use_dynamic_shapes else None
            self.fake_mode = (
                FakeTensorMode(shape_env=self.shape_env)
                if config.use_fake_tensor
                else nullcontext()
            )
        elif torch.__version__.startswith("2.1"):
            self.shape_env = ShapeEnv() if torch._dynamo.config.dynamic_shapes else None
            self.fake_mode = (
                FakeTensorMode(shape_env=self.shape_env)
                if config.fake_tensor_allow_meta
                else nullcontext()
            )
        else:
            raise ValueError(
                f"unsupported dicp torch version: {torch.__version__}")

    @classmethod
    def get_singleton(cls):
        args = [None] * (cls.__init__.__code__.co_argcount - 1)
        if cls._singleton is None:
            cls._singleton = cls(*args)
        return cls._singleton

    def name(self):
        return self.__name__

    # @abstractmethod
    # def infer_result(self, *args, **kwargs):
    #     pass

    def __call__(self, *args, **kwargs):
        def get_meta(x):
            return x if not hasattr(x, 'meta') else x.meta['val']
        new_args = tree_map(get_meta, args)

        fake_mode = None
        tmp_args, _ = tree_flatten(new_args)
        for arg in tmp_args:
            if isinstance(arg, FakeTensor):
                fake_mode = arg.fake_mode
                break
        fake_mode = self.fake_mode if fake_mode is None else fake_mode

        def make_faketensor(x):
            if not isinstance(x, torch.Tensor) or (isinstance(x, FakeTensor)
                                                   and x.fake_mode == fake_mode):
                return x
            if isinstance(x, FakeTensor):
                x.fake_mode = fake_mode
                return x
            return FakeTensor.from_tensor(x, fake_mode)
        new_args = tree_map(make_faketensor, new_args)

        def make_cpu(x):
            if isinstance(x, torch.Tensor):
                return x.to('cpu')
            return x
        new_args = tree_map(make_cpu, new_args)

        with fake_mode:
            if hasattr(self, "infer_result"):
                info: TensorInfo = self.infer_result(*new_args, **kwargs)
                return torch.empty(info.shape, dtype=info.dtype, memory_format=info.memory_format)
            else:
                return self.torch_op(*new_args, **kwargs)
