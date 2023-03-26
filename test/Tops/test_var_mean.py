import torch


def _validate_reduction_axis(x, axis):
    size = x.size()
    if isinstance(axis, int):
        axis = [axis]
    elif not axis:
        axis = range(len(size))
        
    axis = list(axis)
    print(f"axis---{axis} ---{len(axis)}")
    for i in range(len(axis)):
        if axis[i] < 0:
            axis[i] += len(size) if len(size) else 1
    return axis

def test_var_mean(inputs, dims, correction=0, keepdim=True):
    shapes = inputs.size()
    denom = 1
    dims = _validate_reduction_axis(inputs, dims)
    mean1=torch.ops.aten.mean.dim(inputs, dims, keepdim)
    diffs = torch.ops.aten.square.default(torch.ops.aten.sub.Tensor(inputs, mean1))
    sum_results = torch.ops.aten.sum.dim_IntList(diffs, dims, keepdim)

    for i in dims:
        denom = denom *shapes[i]
    denom = denom -correction
    x_var =  torch.ops.aten.div.Tensor(sum_results, denom)
    return x_var, mean1

if 1:
#def main():
    inputs = torch.rand(5, 8, 16, 16)
    print(f"***shape***{inputs.size()}")
    #dims = None
    #dims=int(0)
    dims=[0, 2, 3]
    var, mean = test_var_mean(inputs,dims,correction=0, keepdim=True)
    refvar, refmean = torch.ops.aten.var_mean(inputs, dims, correction=0,keepdim=True)
    #print(f"*****test var: {var}\n*****ref var: {refvar}")
    #print(f"*****test mean: {mean}\n*****ref mean: {refmean}")
    print(f"*****test-ref var: \n{var -refvar}\n********")
    #print(f"*****test-ref mean: {mean-refmean}")