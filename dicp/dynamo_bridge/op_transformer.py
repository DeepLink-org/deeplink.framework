import torch
import torch.fx
from torch.fx import replace_pattern
from torch.fx.node import Argument, Target
import torch.fx.traceback as fx_traceback
from torch.fx.proxy import Proxy
from typing import Any, Dict, Tuple

class OpSetTransformer:
    def __init__(self, patterns):
        self._patterns = patterns

    def transform(self, module: torch.fx.GraphModule):
        # first step: replace pattern
        for pat in self._patterns:
            replace_pattern(module, pat.pattern, pat.replacement)
        return module


class SingleOpTransformer(torch.fx.Transformer):
    def __init__(self, module, conversions):
        super().__init__(module)
        self._conversions = conversions

    def placeholder(self, target : 'Target', args : Tuple[Argument, ...], kwargs : Dict[str, Any]) -> Proxy:
        proxy = super().placeholder(target, args, kwargs)
        proxy.node.meta = fx_traceback.get_current_meta()
        return proxy

    def call_function(self, target : Target, args : Tuple[Argument, ...], kwargs : Dict[str, Any]) -> Any:
        if target in self._conversions:
            out = self._conversions[target](*args, **kwargs)
            proxy = self.tracer.create_proxy('call_function', out, args, kwargs)
            proxy.node.meta = fx_traceback.get_current_meta()
            return proxy
        return super().call_function(target, args, kwargs)

    def get_attr(self, target : 'Target', args : Tuple[Argument, ...], kwargs : Dict[str, Any]) -> Proxy:
        proxy = super().get_attr(target, args, kwargs)
        proxy.node.meta = fx_traceback.get_current_meta()
        if not 'val' in proxy.node.meta:
            proxy.node.meta['val'] = self.fetch_attr(target)
        return proxy
