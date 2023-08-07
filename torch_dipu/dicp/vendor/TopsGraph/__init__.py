from torch._dynamo import register_backend

def topsgraph(gm, fake_input_tensor):
    from torch_dipu.dicp.dynamo_bridge.compile_fx import compile_fx

    return compile_fx(gm, fake_input_tensor, "topsgraph")
