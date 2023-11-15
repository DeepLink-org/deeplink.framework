import typing
import torch
# from typing import Tuple
from dicp.dynamo_bridge.operator import Operator
import numpy as np
from collections.abc import Sequence
from dicp.vendor.AscendGraph.infer_res_utils import *

from dicp.dynamo_bridge.utils import TensorInfo, get_memory_format

aten = torch.ops.aten


def symint_in_shape(shape):
    for elem in shape:
        if isinstance(elem, torch.SymInt):
            return True
    return False


def negative_in_shape(shape):
    for elem in shape:
        if elem < 0:
            return True
    return False





# TODO:tell the differences with "add" ?（torch has no operator "adds" ?）
class Adds(Operator):
    def __init__(self):
        super().__init__("adds")


class Add(Operator):
    def __init__(self):
        super().__init__("add")
        self.torch_op = aten.add

    def infer_result(self,x1,x2):
        x1,x1_shape,x1_dim,x1_dtype=get_node_tensor_meta_val(x1,True)
        x2,x2_shape,x2_dim,x2_dtype=get_node_tensor_meta_val(x2,True)  

        return TensorInfo(shape=get_broadcast_res_two_shape(x1_shape,x2_shape), 
                          dtype=get_cast_dtype(x1_dtype,x2_dtype), 
                          memory_format=get_memory_format(x1))


    

    # def __call__(self, a, b):
    #     if hasattr(a, 'meta'):
    #         a = a.meta['val']
    #         a_shape = a.shape
    #     else:
    #         a_shape = [1]
    #     if hasattr(b, 'meta'):
    #         b = b.meta['val']
    #         b_shape = b.shape
    #     else:
    #         b_shape = [1]

    #     fake_mode = None
    #     for arg in [a, b]:
    #         if isinstance(arg, FakeTensor):
    #             fake_mode = arg.fake_mode
    #             break
    #     fake_mode = self.fake_mode if fake_mode is None else fake_mode

    #     # TODO! better to check
    #     # whether satisfy broadcast
    #     if np.prod(a_shape) > np.prod(b_shape):
    #         shape = a_shape
    #     else:
    #         shape = b_shape
    #     with fake_mode:
    #         return aten.empty(shape, dtype=a.dtype)


class BroadcastTo(Operator):
    def __init__(self):
        super().__init__("BroadcastTo")
        self.torch_op = aten.broadcast_to

    def infer_result(self, x, shape):
        if hasattr(x, 'meta'):
            x = x.meta['val']
            x_shape=x.shape
        else:
            x_shape = [1]
        
        assert len(x_shape) > 0,self.__class__.__name__+": scalar"
        
        dims=zip(reversed(shape),reversed(x_shape))
        
        for i,t in enumerate(dims):
            tar_dim,cur_dim=t
            if tar_dim == -1:
                shape[-(i+1)]=cur_dim
                continue 
            elif cur_dim == 1:
                continue
            
            assert cur_dim == tar_dim,self.__class__.__name__+": shape mismatch!"

                
        # broadcast keep get_memory_format
        return TensorInfo(shape, dtype=x.dtype, memory_format=get_memory_format(x))


    


class Range(Operator):
    def __init__(self):
        super().__init__("Range")
        self.torch_op = aten.arange # torch.arange:right open section


    def infer_result(self, start, limit=None,delta =None):

        if hasattr(start, 'meta'):
            start = start.meta['val']
        else:
            start = torch.tensor(start)

        if limit is None:
            limit=start
            start=torch.tensor(0)
        else:
            if hasattr(limit,'meta'):
                limit=limit.meta['val']
            else :
                limit =torch.tensor(limit)

        if delta is None:
            delta=torch.tensor(1)
        else:
            if hasattr(delta,'meta'):
                delta=delta.meta['val']
            else :
                delta =torch.tensor(delta)

        seq_len= (limit-start)//delta
        if abs(start + seq_len * delta- limit)>1e-4:
            seq_len+=1 

                
        return TensorInfo([seq_len.int()], 
                          dtype= get_cast_dtype(start,delta) , 
                          memory_format=torch.contiguous_format)

''' too simple'''
class CumSum(Operator):
    def __init__(self):
        super().__init__("Cumsum")

    def infer_result(self, x, axis,exclusive,reverse=False):
        if hasattr(x, 'meta'):
            x = x.meta['val']
        x_shape=list(x.shape)
        return TensorInfo(x_shape,dtype=x.dtype,memory_format=get_memory_format(x))



class MatMul(Operator):
    def __init__(self):
        super().__init__("MatMul")
        self.torch_op = aten.mm #TODO: only consider 2D (N,D(H*W*C))？

    def infer_result(self, a, b, trans_a=False, trans_b=False, change_input=False):
        if hasattr(a, 'meta'):
            a = a.meta['val']
        if hasattr(b, 'meta'):
            b = b.meta['val']
        if change_input:
            (a, b) = (b, a)

        trans_a_shape = list(reversed(a.shape)) if trans_a else list(a.shape)
        trans_b_shape = list(reversed(b.shape)) if trans_b else list(b.shape)

        assert trans_a_shape[1] == trans_b_shape[0],self.__class__.__name__+": shape mismatch!"

        return TensorInfo([trans_a_shape[0],trans_b_shape[1]], 
                          dtype=get_cast_dtype(a.dtype,b.dtype), 
                          memory_format=get_memory_format(a))
    
        # trans_a_shape = shape_functions.t(a.shape) if trans_a else a.shape
        # trans_b_shape = shape_functions.t(b.shape) if trans_b else b.shape
        # mm_shape = shape_functions.matmul(trans_a_shape, trans_b_shape)
        # return TensorInfo(mm_shape, dtype=a.dtype, memory_format=get_memory_format(a))


class BatchMatMul(Operator):
    def __init__(self):
        super().__init__("BatchMatMul")
        self.torch_op = aten.bmm # torch's bmm only support 3D input (but the ascend manual said "2D or higher...")


    # only support 3D input,TODO:?
    def infer_result(self, x1, x2, adj_x1=False, adj_x2=False):
        if hasattr(x1, 'meta'):
            x1 = x1.meta['val']
        if hasattr(x2, 'meta'):
            x2 = x2.meta['val']

        adj_x1_shape = [x1.shape[0]]+list(reversed(x1.shape[1:])) if adj_x1 else list(x1.shape)
        adj_x2_shape = [x2.shape[0]]+list(reversed(x2.shape[1:])) if adj_x2 else list(x2.shape)
        # print(adj_x1_shape," ",adj_x2_shape)
        assert adj_x1_shape[2] == adj_x2_shape[1],self.__class__.__name__+": shape mismatch!"

        return TensorInfo(adj_x1_shape[0:2]+[adj_x2_shape[2]], dtype=get_cast_dtype(x1.dtype,x2.dtype), memory_format=get_memory_format(x1))    


class Sub(Operator):
    def __init__(self):
        super().__init__("Sub")
        self.torch_op = aten.sub


    def infer_result(self,x1,x2):

        x1,x1_shape,x1_dim,x1_dtype=get_node_tensor_meta_val(x1,True)
        x2,x2_shape,x2_dim,x2_dtype=get_node_tensor_meta_val(x2,True)

        # print(x1_dtype," ",x2_dtype)

        return TensorInfo(shape=get_broadcast_res_two_shape(x1_shape,x2_shape), 
                          dtype=get_cast_dtype(x1_dtype,x2_dtype), 
                          memory_format=get_memory_format(x1))

        


class Mul(Operator):
    def __init__(self):
        super().__init__("Mul")
        self.torch_op = aten.mul

    def infer_result(self,x1,x2):

        x1,x1_shape,x1_dim,x1_dtype=get_node_tensor_meta_val(x1,True)
        x2,x2_shape,x2_dim,x2_dtype=get_node_tensor_meta_val(x2,True)

        return TensorInfo(shape=get_broadcast_res_two_shape(x1_shape,x2_shape), 
                          dtype=get_cast_dtype(x1_dtype,x2_dtype), 
                          memory_format=get_memory_format(x1))


class MulNoNan(Operator):
    def __init__(self):
        super().__init__("Mul")

    def infer_result(self,x1,x2):
        x1,x1_shape,x1_dim,x1_dtype=get_node_tensor_meta_val(x1,True)
        x2,x2_shape,x2_dim,x2_dtype=get_node_tensor_meta_val(x2,True)

        return TensorInfo(shape=get_broadcast_res_two_shape(x1_shape,x2_shape), 
                          dtype=get_cast_dtype(x1_dtype,x2_dtype), 
                          memory_format=get_memory_format(x1))

class Div(Operator):
    def __init__(self):
        super().__init__("Div")

    def infer_result(self,x1,x2):
        x1,x1_shape,x1_dim,x1_dtype=get_node_tensor_meta_val(x1,True)
        x2,x2_shape,x2_dim,x2_dtype=get_node_tensor_meta_val(x2,True)

        return TensorInfo(shape=get_broadcast_res_two_shape(x1_shape,x2_shape), 
                          dtype=get_cast_dtype(x1_dtype,x2_dtype), 
                          memory_format=get_memory_format(x1))

class DivNoNan(Operator):
    def __init__(self):
        super().__init__("DivNoNan")

    def infer_result(self,x1,x2):
        x1,x1_shape,x1_dim,x1_dtype=get_node_tensor_meta_val(x1,True)
        x2,x2_shape,x2_dim,x2_dtype=get_node_tensor_meta_val(x2,True)

        return TensorInfo(shape=get_broadcast_res_two_shape(x1_shape,x2_shape), 
                          dtype=get_cast_dtype(x1_dtype,x2_dtype), 
                          memory_format=get_memory_format(x1))


class Maximum(Operator):
    def __init__(self):
        super().__init__("Maximum")

    # must be the same shape
    def infer_result(self,x1,x2):
        x1,x1_shape,x1_dim,x1_dtype=get_node_tensor_meta_val(x1,True)
        x2,x2_shape,x2_dim,x2_dtype=get_node_tensor_meta_val(x2,True)

        assert x1_shape == x2_shape,self.__class__.__name__+": shape mismatch!"

        return TensorInfo(x1_shape, 
                          dtype=get_cast_dtype(x1_dtype,x2_dtype), 
                          memory_format=get_memory_format(x1))



class Rsqrt(Operator):
    def __init__(self):
        super().__init__("Rsqrt")

    def infer_result(self,x):
        if hasattr(x, 'meta'):
            x = x.meta['val']
            x_shape = x.shape
        else:
            x_shape=[1]

        return TensorInfo(x_shape, dtype=x.dtype, memory_format=get_memory_format(x))


class Sqrt(Operator):
    def __init__(self):
        super().__init__("Sqrt")

    def infer_result(self,x):
        if hasattr(x, 'meta'):
            x = x.meta['val']
            x_shape = x.shape
        else:
            x_shape=[1]

        return TensorInfo(x_shape, dtype=x.dtype, memory_format=get_memory_format(x))


class Log(Operator):
    def __init__(self):
        super().__init__("Log")

    def infer_result(self,x,base=-1.0,scale=1.0,shift=0.0):
        if hasattr(x, 'meta'):
            x = x.meta['val']
            x_shape = x.shape
        else:
            x_shape=[1]

        return TensorInfo(x_shape, dtype=x.dtype, memory_format=get_memory_format(x))



class Exp(Operator):
    def __init__(self):
        super().__init__("Exp")

    def infer_result(self,x,base=-1.0,scale=1.0,shift=0.0):
        if hasattr(x, 'meta'):
            x = x.meta['val']
            x_shape = x.shape
        else:
            x_shape=[1]

        return TensorInfo(x_shape, dtype=x.dtype, memory_format=get_memory_format(x))


class Neg(Operator):
    def __init__(self):
        super().__init__("Neg")

    def infer_result(self,x,base=-1.0,scale=1.0,shift=0.0):
        if hasattr(x, 'meta'):
            x = x.meta['val']
            x_shape = x.shape
        else:
            x_shape=[1]

        return TensorInfo(x_shape, dtype=x.dtype, memory_format=get_memory_format(x))


class Relu(Operator):
    def __init__(self):
        super().__init__("Relu")


    def infer_result(self,x,base=-1.0,scale=1.0,shift=0.0):
        if hasattr(x, 'meta'):
            x = x.meta['val']
            x_shape = x.shape
        else:
            x_shape=[1]

        return TensorInfo(x_shape, dtype=x.dtype, memory_format=get_memory_format(x))


class Swish(Operator):
    def __init__(self):
        super().__init__("Swish")
        self.torch_op = aten.silu # torch silu no param scale

    def infer_result(self,x,scale=1.0):
        if hasattr(x, 'meta'):
            x = x.meta['val']
            x_shape = x.shape
        else:
            x_shape=[1]

        return TensorInfo(x_shape, dtype=x.dtype, memory_format=get_memory_format(x))
    


class Transpose(Operator):
    def __init__(self):
        super().__init__("Transpose")
    # Manual: Compatible with the TensorFlow operator Transpose (not Torch)
        self.torch_op = aten.permute # in torch is permute?

    # def infer_result(self, input, dim0, dim1):
    #     if hasattr(input, 'meta'):
    #         input = input.meta['val']
    #     shape = list(input.shape)
    #     (shape[dim1], shape[dim0]) = (shape[dim0], shape[dim1])
    #     return TensorInfo(shape, dtype=input.dtype, memory_format=get_memory_format(input))
    
    def infer_result(self, x, perm):
        x,x_shape,x_dim,x_dtype=get_node_tensor_meta_val(x,True)
        
        assert len(perm)==x_dim,self.__class__.__name__+": shape mismatch!"

        for rank in perm:
            assert rank in range(-x_dim,x_dim)  ,self.__class__.__name__+": illegal rank!"+" rank: "+str(rank)+" x_dim: "+str(x_dim)

        shape= [x_shape[ new_r ]  for new_r in perm]

        # TODO: may change memory_format
        return TensorInfo(shape, dtype=x_dtype, memory_format=get_memory_format(x))


class SoftmaxV2(Operator):
    def __init__(self):
        super().__init__("SoftmaxV2")

    def infer_result(self, x, axes=None):
        if hasattr(x, 'meta'):
            x = x.meta['val']
            x_shape = x.shape
        else:
            x_shape=[1]

        axes=[-1] if axes is None else axes

        return TensorInfo(x_shape, dtype=x.dtype, memory_format=get_memory_format(x))


class ReduceSum(Operator):   
    def __init__(self):
        super().__init__("ReduceSum")  
        self.torch_op = aten.sum


    def infer_result(self, x, axes=None,keep_dims=False):
        x,x_shape,x_dim,x_dtype=get_node_tensor_meta_val(x,True)
        

        shape=reduce_ops_output_size(x_shape,x_dim,axes,keep_dims)
               

        return TensorInfo(shape, dtype=x_dtype, memory_format=get_memory_format(x))


class Unsqueeze(Operator):
    def __init__(self):
        super().__init__("Unsqueeze")

    def infer_result(self, x, axes=None):
        x,x_shape,x_dim,x_dtype=get_node_tensor_meta_val(x,True)

        assert axes is not None,self.__class__.__name__+": doesn't specify axis to unsqueeze!"


        x_shape=list(x_shape)
        x_shape.insert(axes+x_dim+1 if axes<0 else axes,1)

        return TensorInfo(x_shape, dtype=x_dtype, memory_format=get_memory_format(x))


class Squeeze(Operator):
    def __init__(self):
        super().__init__("Squeeze")
        self.torch_op = aten.squeeze # torch support squeeze input axis as array/list/seq...

    def infer_result(self, x, axis=None):
        if hasattr(x, 'meta'):
            x = x.meta['val']
            x_shape = x.shape
        else:
            x_shape=[1]

        if axis is None:
            shape = [i for i in x_shape if i != 1 ]
        else:
            axis=[axis] if not isinstance(axis,Sequence) else axis

            shape = list(x_shape)
            for i in axis:
                assert x_shape[i] == 1,self.__class__.__name__+": can only squeeze a dimension that is 1!"
                shape.pop(i)

        return TensorInfo(shape, dtype=x.dtype, memory_format=get_memory_format(x))




class Pack(Operator):
    def __init__(self):
        super().__init__("Pack")
        self.torch_op = aten.stack

    def infer_result(self, x: typing.Sequence[torch.tensor], dim):
        if hasattr(x, 'meta'):
            x = x.meta['val']
            x_shape=list(x[0].shape)
            x_dtype=x[0].dtype
        else:
            ...

        dim= dim+len(x_shape)+1 if dim<0 else dim
        assert dim in range(len(x_shape)+1),self.__class__.__name__+": Dimension out of range!"
        x_shape.insert(dim,len(x))

        # 
        return TensorInfo(x_shape, dtype=x_dtype, memory_format=get_memory_format(x[0]))


class Permute(Operator):
    def __init__(self):
        super().__init__("Permute")

    # TODO:the same as transpose?
    def infer_result(self, x, order):
        x,x_shape,x_dim,x_dtype=get_node_tensor_meta_val(x,True)
        
        assert len(order)==x_dim,self.__class__.__name__+": shape mismatch!"

        for rank in order:
            assert rank in range(-x_dim,x_dim)  ,self.__class__.__name__+": illegal rank!"+" rank: "+str(rank)+" x_dim: "+str(x_dim)

        shape= [x_shape[ new_r ]  for new_r in order]

        # TODO: may change memory_format
        return TensorInfo(shape, dtype=x_dtype, memory_format=get_memory_format(x))


class Expand(Operator):
    def __init__(self):
        super().__init__("Expand")

    def infer_result(self, x, shape):
        x,x_shape,x_dim,x_dtype=get_node_tensor_meta_val(x,True)

        # shape = [dim.meta['val'] if hasattr(dim, 'meta') else dim for dim in shape]
        
        assert x_dim > 0,self.__class__.__name__+": scalar"
        
        dims=zip(reversed(shape),reversed(x_shape))
        
        for i,t in enumerate(dims):
            tar_dim,cur_dim=t
            if tar_dim == -1:
                shape[-(i+1)]=cur_dim
                continue 
            elif cur_dim == 1:
                continue
            
            assert cur_dim == tar_dim,self.__class__.__name__+": shape mismatch!"

                
        # broadcast keep get_memory_format
        return TensorInfo(shape, dtype=x_dtype, memory_format=get_memory_format(x))


'''what is the difference between Expand and ExpandD?'''
class ExpandD(Operator):
    def __init__(self):
        super().__init__("ExpandD")


class Sort(Operator):
    def __init__(self):
        super().__init__("Sort")

    def infer_result(self, x,axis=None, descending=False):
        x,x_shape,_,x_dtype=get_node_tensor_meta_val(x,True)
        
        # broadcast keep get_memory_format
        values= TensorInfo(x_shape, dtype=x_dtype, memory_format=get_memory_format(x))
        # TODO:the type specified in huawei's source code: [at::kLong(torch.long/torch.in64)],while in manual is [int32]
        indices=TensorInfo(x_shape, dtype=torch.long, memory_format=get_memory_format(x)) 
        return values,indices


class TopK(Operator):
    def __init__(self):
        super().__init__("TopK")

    def infer_result(self, x,k,sorted=True, largest=True,dim=None):
        x,x_shape,x_dim,x_dtype=get_node_tensor_meta_val(x,True)

        dim = x_dim-1 if dim is None else dim 
        # print(k," ",dim)
        assert k<=x_shape[dim] and k>=0 , self.__class__.__name__+" :selected index k: "+str(k)+" out of range: [0,"+str(x_shape[dim])+"]"

        shape=list(x_shape)
        shape[dim]=k
        
        values= TensorInfo(shape, dtype=x_dtype, memory_format=get_memory_format(x))
        # TODO:the type specified in huawei's source code: [at::kLong(torch.long/torch.in64)],while in manual is [int32]
        indices=TensorInfo(shape, dtype=torch.long, memory_format=get_memory_format(x)) 
        return values,indices


class ScatterElements(Operator):
    def __init__(self):
        super().__init__("ScatterElements")

    def infer_result(self, x,index,src,dim=None,reduce=None):
        x,x_shape,_,x_dtype=get_node_tensor_meta_val(x,True)

        return TensorInfo(x_shape, dtype=x_dtype, memory_format=get_memory_format(x)) 
    


class ReduceMean(Operator):
    def __init__(self):
        super().__init__("ReduceMean")

    def infer_result(self, x,dim=None,keepdim=False):
        x,x_shape,x_dim,x_dtype=get_node_tensor_meta_val(x,True)

        shape=reduce_ops_output_size(x_shape,x_dim,dim,keepdim)

        return TensorInfo(shape, dtype=x_dtype, memory_format=get_memory_format(x))
    




class ReduceStdV2Update(Operator):
    def __init__(self):
        super().__init__("ReduceStdV2Update")
        self.torch_op = aten.var_mean

    def infer_result(self, x,dim=None,unbiased=True,correction=1,keepdim=False):# from torch 2.0: unbiased --> correction
        x,x_shape,x_dim,x_dtype=get_node_tensor_meta_val(x,True)

        shape=reduce_ops_output_size(x_shape,x_dim,dim,keepdim)

        variance= TensorInfo(shape, dtype=x_dtype, memory_format=get_memory_format(x))
        mean=TensorInfo(shape, dtype=x_dtype, memory_format=get_memory_format(x)) 
        return variance,mean

    


class ReduceMax(Operator):  # ReduceMaxD ?
    def __init__(self):
        super().__init__("ReduceMax") # ReduceMaxD ?
        self.torch_op = aten.amax

    def infer_result(self, x,dim=None,keepdim=False):
        x,x_shape,x_dim,x_dtype=get_node_tensor_meta_val(x,True)

        shape=reduce_ops_output_size(x_shape,x_dim,dim,keepdim)

        return TensorInfo(shape, dtype=x_dtype, memory_format=get_memory_format(x))


class Const(Operator):
    def __init__(self):
        super().__init__("Const")

    def infer_result(self, value,dtype=None):
        x,x_shape,_,x_dtype=get_node_tensor_meta_val(value)

        return TensorInfo(x_shape, dtype=x_dtype if dtype is None else dtype, memory_format=get_memory_format(x)) 
    


class Sigmoid(Operator):
    def __init__(self):
        super().__init__("Sigmoid")

    def infer_result(self, x):
        if hasattr(x, 'meta'):
            x = x.meta['val']
            x_shape=x.shape
        else:
            x_shape = [1]

        return TensorInfo(list(x_shape), dtype=x.type, memory_format=get_memory_format(x)) 


class Pow(Operator):
    def __init__(self):
        super().__init__("Pow")

    def infer_result(self, x1,x2):
        x1,x1_shape,x1_dim,x1_dtype=get_node_tensor_meta_val(x1,True)

        x2,x2_shape,x2_dim,x2_dtype=get_node_tensor_meta_val(x2,True)

        return TensorInfo(get_broadcast_res_two_shape(x1_shape,x2_shape), 
                          dtype=get_cast_dtype(x1_dtype,x2_dtype), 
                          memory_format=get_memory_format(x1))  


class Select(Operator):
    def __init__(self):
        super().__init__("Select")
        self.torch_op = aten.where

    
    def infer_result(self, x1,x2,condition):
        x1,x1_shape,x1_dim,x1_dtype=get_node_tensor_meta_val(x1)
        x2,x2_shape,x2_dim,x2_dtype=get_node_tensor_meta_val(x2,True)
        _,c_shape,_,_=get_node_tensor_meta_val(condition)


        return TensorInfo(get_broadcast_res_two_shape(get_broadcast_res_two_shape(x1_shape,c_shape),x2_shape), 
                          dtype=get_cast_dtype(x1_dtype,x2_dtype), 
                          memory_format=get_memory_format(x1))  


class LessEqual(Operator):
    def __init__(self):
        super().__init__("LessEqual")
        self.torch_op = aten.le

    def infer_result(self, x1,x2):
        x1,x1_shape,x1_dim,x1_dtype=get_node_tensor_meta_val(x1)
        x2,x2_shape,x2_dim,x2_dtype=get_node_tensor_meta_val(x2,True)

        return TensorInfo(get_broadcast_res_two_shape(x1_shape,x2_shape), 
                          dtype=torch.bool, 
                          memory_format=get_memory_format(x1))  


class Less(Operator):
    def __init__(self):
        super().__init__("Less")
        self.torch_op = aten.lt

    def infer_result(self, x1,x2):
        x1,x1_shape,x1_dim,x1_dtype=get_node_tensor_meta_val(x1)
        x2,x2_shape,x2_dim,x2_dtype=get_node_tensor_meta_val(x2,True)

        return TensorInfo(get_broadcast_res_two_shape(x1_shape,x2_shape), 
                          dtype=torch.bool, 
                          memory_format=get_memory_format(x1))    


class Equal(Operator):
    def __init__(self):
        super().__init__("Equal")
        self.torch_op = aten.eq

    def infer_result(self, x1,x2):
        x1,x1_shape,x1_dim,x1_dtype=get_node_tensor_meta_val(x1)
        x2,x2_shape,x2_dim,x2_dtype=get_node_tensor_meta_val(x2,True)

        return TensorInfo(get_broadcast_res_two_shape(x1_shape,x2_shape), 
                          dtype=torch.bool, 
                          memory_format=get_memory_format(x1))   


class Conv2D(Operator):
    def __init__(self):
        super().__init__("Conv2D")
        self.torch_op = aten.conv2d

    def infer_result(self, x, weight,bias=None,stride=1,padding=0,dilation=1,groups=1):
        x,x_shape,x_dim,x_dtype=get_node_tensor_meta_val(x)
        w,w_shape,w_dim,w_dtype=get_node_tensor_meta_val(weight)

        # if bias is not None:
        #     b,b_shape,b_dim,b_dtype=get_node_tensor_meta_val(bias)
        # TODO:bias shape (assert 

        dilation = (dilation,dilation) if not isinstance(dilation,Sequence) else dilation
        padding = (padding,padding) if not isinstance(padding,Sequence) else padding
        stride = (stride,stride) if not isinstance(stride,Sequence) else stride

        # print(x_shape)
        N = x_shape[0]
        H = x_shape[2]
        W = x_shape[3]

        Co= w_shape[0]
        kernel_size= w_shape[2], w_shape[3]

        Ho = (H + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0] + 1
        Wo = (W + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1] + 1

        out_shape=[N,Co,Ho,Wo]
        return TensorInfo(out_shape, 
                          dtype=x_dtype, 
                          memory_format=get_memory_format(x)) # TODO: torch format ?
        


class GreaterEqual(Operator):
    def __init__(self):
        super().__init__("GreaterEqual")
        self.torch_op = aten.ge

    def infer_result(self, x1,x2):
        x1,x1_shape,x1_dim,x1_dtype=get_node_tensor_meta_val(x1)
        x2,x2_shape,x2_dim,x2_dtype=get_node_tensor_meta_val(x2,True)

        return TensorInfo(get_broadcast_res_two_shape(x1_shape,x2_shape), 
                          dtype=torch.bool, 
                          memory_format=get_memory_format(x1))   

# TODO: InplaceAdd?
# class InAdd(Operator):
#     def __init__(self):
#         super().__init__("inadd")




# TODO: no cast in torch --> tensor.to/tensor.type ???
class Cast(Operator):
    def __init__(self):
        super().__init__("Cast")
        # self.torch_op = aten.to
        # self.torch_op = aten.type

    def infer_result(self, x,dtype):
        x,x_shape,x_dim,x_dtype=get_node_tensor_meta_val(x)


        return TensorInfo(x_shape, 
                          dtype=dtype, 
                          memory_format=get_memory_format(x))


class Identity(Operator):
    def __init__(self):
        super().__init__("Identity")
        # TODO: self.torch_op = ???

    def infer_result(self, x):
        x,x_shape,x_dim,x_dtype=get_node_tensor_meta_val(x)

        return TensorInfo(x_shape, 
                          dtype=x_dtype, 
                          memory_format=get_memory_format(x))


# TODO 
class IdentityN(Operator):
    def __init__(self):
        super().__init__("IdentityN")

    def infer_result(self, x):
        if hasattr(x, 'meta'):
            x = x.meta['val']
            x_shape=list(x.shape)
            x_dtype=x[0].dtype 

        return TensorInfo(x_shape, 
                          dtype=x_dtype, 
                          memory_format=get_memory_format(x))

class Empty(Operator):
    def __init__(self):
        super().__init__("Empty")


class GatherV2(Operator):
    def __init__(self):
        super().__init__("GatherV2")


class OnesLike(Operator):
    def __init__(self):
        super().__init__("OnesLike")


class Fill(Operator):
    def __init__(self):
        super().__init__("Fill")


class Conv2DBackpropInput(Operator):
    def __init__(self):
        super().__init__("Conv2DBackpropInput")


class Conv2DBackpropFilter(Operator):
    def __init__(self):
        super().__init__("Conv2DBackpropFilter")


class LogSoftmaxV2(Operator):
    def __init__(self):
        super().__init__("LogSoftmaxV2")


class LogSoftmaxGrad(Operator):
    def __init__(self):
        super().__init__("LogSoftmaxGrad")


class FillV2D(Operator):
    def __init__(self):
        super().__init__("FillV2D")


class NLLLoss(Operator):
    def __init__(self):
        super().__init__("NLLLoss")


class NLLLossGrad(Operator):
    def __init__(self):
        super().__init__("NLLLossGrad")


class BNTrainingReduce(Operator):
    def __init__(self):
        super().__init__("BNTrainingReduce")


class BNTrainingUpdate(Operator):
    def __init__(self):
        super().__init__("BNTrainingUpdate")


class BNTrainingUpdateGrad(Operator):
    def __init__(self):
        super().__init__("BNTrainingUpdateGrad")


class BNTrainingReduceGrad(Operator):
    def __init__(self):
        super().__init__("BNTrainingReduceGrad")


class ReluGrad(Operator):
    def __init__(self):
        super().__init__("ReluGrad")


class ThresholdGradV2D(Operator):
    def __init__(self):
        super().__init__("ThresholdGradV2D")


class ZerosLike(Operator):
    def __init__(self, x):
        super().__init__("ZerosLike")


class SplitD(Operator):
    def __init__(self):
        super().__init__("SplitD")


class Slice(Operator):
    def __init__(self):
        super().__init__("Slice")


class ConcatD(Operator):
    def __init__(self):
        super().__init__("ConcatD")


class MaskedFill(Operator):
    def __init__(self):
        super().__init__("MaskedFill")


class Reshape(Operator):
    def __init__(self):
        super().__init__("Reshape")


class Pad(Operator):
    def __init__(self):
        super().__init__("Pad")


class Fills(Operator):
    def __init__(self):
        super().__init__("Fills")


class SoftmaxGrad(Operator):
    def __init__(self):
        super().__init__("SoftmaxGrad")


class StatelessBernoulli(Operator):
    def __init__(self):
        super().__init__("StatelessBernoulli")
        # self.torch_op = aten.bernoulli.p


class Shape(Operator):
    def __init__(self):
        super().__init__("Shape")


class AddV2(Operator):
    def __init__(self):
        super().__init__("AddV2")


class StatelessRandomUniformV2(Operator):
    def __init__(self):
        super().__init__("StatelessRandomUniformV2")


class Greater(Operator):
    def __init__(self):
        super().__init__("Greater")
        self.torch_op = aten.gt

    def infer_result(self, x1,x2):
        x1,x1_shape,x1_dim,x1_dtype=get_node_tensor_meta_val(x1)
        x2,x2_shape,x2_dim,x2_dtype=get_node_tensor_meta_val(x2,True)

        return TensorInfo(get_broadcast_res_two_shape(x1_shape,x2_shape), 
                          dtype=torch.bool, 
                          memory_format=get_memory_format(x1))   


class Addcmul(Operator):
    def __init__(self):
        super().__init__("Addcmul")


class Reciprocal(Operator):
    def __init__(self):
        super().__init__("Reciprocal")


def ret_triple(a, b, c) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return a, b, c


def ret_tuple(a, b) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    return a, b
