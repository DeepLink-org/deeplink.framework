
from collections.abc import Sequence
from typing import Optional, Tuple,Union

import torch


def get_node_tensor_meta_val(x,req_dim=True,req_dtype=True) -> Tuple[any,list,Union[int,None],Union[torch.dtype,type,None]]:
    if hasattr(x, 'meta'):
        x = x.meta['val']
        x_shape=list(x.shape)
        x_dim=x.dim() if req_dim else None
        x_dtype=x.dtype if req_dtype else None
    else:
        x_shape = [1]
        x_dim=1 if req_dim else None
        x_dtype=type(x) if req_dtype else None

    return x,x_shape,x_dim,x_dtype



def get_broadcast_res_two_shape(shape1,shape2)->Optional[list]:
    len1 = len(shape1)
    len2 = len(shape2)

    max_len = max(len1, len2)

    result_shape = []

    for i in range(-1, -max_len - 1, -1):
        dim1 = shape1[i] if i >= -len1 else 1  
        dim2 = shape2[i] if i >= -len2 else 1  

        if dim1 == dim2 or dim1 == 1 or dim2 == 1:
            result_shape.insert(0, max(dim1, dim2))
        else:
            print(torch.randn(shape1).shape," ",torch.randn(shape2).shape,end=" ")
            assert False,"input shapes must be broadcastable!" 

    return result_shape


def get_cast_dtype(type1: Union[str, torch.dtype,type], type2: Union[str, torch.dtype,type]) -> Union[str, torch.dtype, None]:
    type_map={
        int:torch.int,
        float:torch.float,
        complex:torch.complex,
        bool:torch.bool,
    }
    
    type1 = torch.dtype(type1) if isinstance(type1, str) else type1 
    type2 = torch.dtype(type2) if isinstance(type2, str) else type2

    type1 = type_map[type1] if isinstance(type1, type) else type1 
    type2 = type_map[type2] if isinstance(type2, type) else type2

    if type1 == type2:
        return type1

    complex_list=[torch.complex32,torch.complex64,torch.complex128]
    float_list=[torch.float16,torch.float32,torch.float,torch.float64]
    int_list=[torch.int8,torch.int16,torch.int32,torch.int,torch.int64]

    if type1 in complex_list or type2 in complex_list:

        t1_idx=complex_list.index(type1) if type1 in complex_list else -1
        t2_idx=complex_list.index(type2) if type2 in complex_list else -1
        return complex_list[max(t1_idx,t2_idx)]  
    
    elif type1 == torch.double or type2 == torch.double :
        return torch.double
    elif type1 in float_list or type2 in float_list :
        
        t1_idx=float_list.index(type1) if type1 in float_list else -1
        t2_idx=float_list.index(type2) if type2 in float_list else -1
        return float_list[max(t1_idx,t2_idx)] 
    elif type1 == torch.bool or type2 == torch.bool:
        return torch.bool      
    elif type1 in int_list or type2 in int_list:
        
        t1_idx=int_list.index(type1) if type1 in int_list else -1
        t2_idx=int_list.index(type2) if type2 in int_list else -1
        return int_list[max(t1_idx,t2_idx)]


    assert False,str(type1)+" "+str(type2)+" can't cast these two types!" 
    return None

def analyze_memory_format(tensor: torch.Tensor, operation: str) -> torch.memory_format:
    original_format = tensor.memory_format

    if operation == 'transpose':
        # tensor = tensor.transpose(0, 1)
        ...
    elif operation == 'permute':
        # tensor = tensor.permute(0, 2, 3, 1)
        ...

    return tensor.memory_format if tensor.is_contiguous() else original_format



# def stack_ops_output_size():




def reduce_ops_output_size(x_shape,x_dim,dim:Union[None ,Sequence,int],keepdim=False):
    if dim is None or isinstance(dim,Sequence) and len(dim)==0:
        if keepdim is True:
            shape=[1]*x_dim
        else:
            shape=[]
    else:
        # rank_mask=[x_dim + rank if rank < 0 else rank for rank in dim]
        # rank_mask=list(set(rank_mask))
        # rank_mask=np.sort(rank_mask)[::-1].tolist()
        dim=[dim] if not isinstance(dim,Sequence) else dim

        if keepdim is True:
            shape = [1 if r in dim else ori_size for r, ori_size in enumerate(x_shape)]
        else:
            shape = [x_shape[r] for r in range(x_dim) if r not in dim and r-x_dim not in dim ]
    
    return shape