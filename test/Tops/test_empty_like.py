import torch
import torch._dynamo as dynamo

def fn(a):
    x = torch.empty_like(a)
    return x

torch._dynamo.reset()
opt_fn = dynamo.optimize(backend='topsgraph', dynamic=False)(fn)

input_tensor = torch.abs(torch.randn(5, 4))
dynamo_result = opt_fn(input_tensor)

ori_result = fn(input_tensor)

test_res = 'passed' if torch.allclose(ori_result, dynamo_result, equal_nan=True) else 'failed'
print(f'Test {test_res}')
