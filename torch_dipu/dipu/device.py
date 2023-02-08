from torch import device
__dipu__ = 'dipu'
__diputype__ = 'privateuseone'
__vendor__ = 'mlu'


#  ---- change tensor.to to support device wrapper? 
# enhance *args, **kwargs
def _device(dev_name):
    if dev_name == __dipu__:
        dev_name = __diputype__
    new_device = device(dev_name)
    return new_device


