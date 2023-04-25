from torch._dynamo import register_backend

def ascendgraph(gm, fake_input_tensor):
    from dicp.common.compile_fx import compile_fx

    return compile_fx(gm, fake_input_tensor, "ascendgraph")
