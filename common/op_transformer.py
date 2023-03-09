import torch
from torch.fx import replace_pattern
from torch.fx.node import Argument, Target
from typing import Any, Dict, Tuple

class OpSetTransformer:
    def __init__(self, patterns, new_namespace, whitelist):
        self._patterns = patterns
        self._new_namespace = new_namespace
        self._whitelist = whitelist

    def transform(self, module: torch.fx.GraphModule):
        # first step: replace pattern
        for pat in self._patterns:
            replace_pattern(module, pat.pattern, pat.replacement)
        
        # second step: transform single operater
        NameSpaceTransformer(self._new_namespace,
                             self._whitelist).transform(module)
        return module

class SingleOpTransformer(torch.fx.Transformer):
    def __init__(self, module, conversions):
        super().__init__(module)
        self._conversions = conversions

    def call_function(self, target : Target, args : Tuple[Argument, ...], kwargs : Dict[str, Any]) -> Any:
        if target in self._conversions:
            out = self._conversions[target](*args, **kwargs)
            return self.tracer.create_proxy('call_function', out, args, kwargs)
        return super().call_function(target, args, kwargs)

class NameSpaceTransformer:
    def __init__(self, new_namespace, whitelist):
        self._new_namespace = new_namespace
        self._whitelist = whitelist

    def trans_node(self, n : torch.fx.Node):
        if str(n.target) in self._whitelist:
            if hasattr(n.target, '__module__'):
                if n.target.__module__.split(".")[-1] == 'aten':
                    newm = n.target.__module__.replace('aten', self._new_namespace)
                    setattr(n.target, '__module__', newm)

            if hasattr(n.target, '_name'):
                if n.target._name.split("::")[0] == 'aten':
                    newname = n.target._name.replace('aten', self._new_namespace)
                    setattr(n.target, '_name', newname)

    def transform(self, gm : torch.fx.GraphModule):
        for node in gm.graph.nodes:
            if node.op == "call_function":
                self.trans_node(node)