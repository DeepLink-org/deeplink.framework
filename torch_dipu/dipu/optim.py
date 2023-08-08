import torch

# DIPU does not support foreach operator which using in optimizers.
# So we monkey patch all optimizers that support foreach parameter,
# and set foreach to False.

def GetFuncProxy(rawfunc, pos, name):
    def patch_func(self, *args, **kwargs):
        if pos >= 0 and pos < len(args):
            argsList = list(args)
            argsList[pos] = False
            args = tuple(argsList)
        else:
            kwargs[name] = False
        rawfunc(self, *args, **kwargs)

    return patch_func

def apply_optim_patch():
    torch.optim.Adadelta.__init__ = GetFuncProxy(torch.optim.Adadelta.__init__, 5, "foreach")
    torch.optim.Adagrad.__init__ = GetFuncProxy(torch.optim.Adagrad.__init__, 6, "foreach")
    torch.optim.Adam.__init__ = GetFuncProxy(torch.optim.Adam.__init__, -1, "foreach")
    torch.optim.AdamW.__init__ = GetFuncProxy(torch.optim.AdamW.__init__, -1, "foreach")
    torch.optim.Adamax.__init__ = GetFuncProxy(torch.optim.Adamax.__init__, 5, "foreach")
    torch.optim.ASGD.__init__ = GetFuncProxy(torch.optim.ASGD.__init__, 6, "foreach")
    torch.optim.NAdam.__init__ = GetFuncProxy(torch.optim.NAdam.__init__, -1, "foreach")
    torch.optim.RAdam.__init__ = GetFuncProxy(torch.optim.RAdam.__init__, -1, "foreach")
    torch.optim.RMSprop.__init__ = GetFuncProxy(torch.optim.RMSprop.__init__, 7, "foreach")
    torch.optim.Rprop.__init__ = GetFuncProxy(torch.optim.Rprop.__init__, -1, "foreach")
    torch.optim.SGD.__init__ = GetFuncProxy(torch.optim.SGD.__init__, -1, "foreach")