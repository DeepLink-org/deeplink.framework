import contextlib
import dataclasses
import functools
import math
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, List

import sympy

import torch

from torch._prims_common import is_float_dtype
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
from torch.fx.node import Argument, Node, Target, map_arg, map_aggregate
from torch._inductor.sizevars import SimplifyIndexing
import torch.fx.traceback as fx_traceback

from torch._inductor import codecache, config, ir, metrics
from torch._inductor.utils import sympy_product, sympy_subs, sympy_symbol
from torch._inductor.virtualized import ops, V
from .common import (
    BracesBuffer,
    CppWrapperKernelArgs,
    DeferredIndentedBuffer,
    ExprPrinter,
    IndentedBuffer,
    Kernel,
    KernelArgs,
    OpOverrides,
)


DTYPE_TO_CPP = {
    torch.float32: "float",
    torch.float64: "double",
    torch.float16: "half",
    torch.int64: "long",
    torch.int32: "int",
    torch.int16: "short",
    torch.int8: "signed char",
    torch.uint8: "unsigned char",
    torch.bool: "bool",
    torch.bfloat16: "bfloat16",
}

DTYPE_TO_ATEN = {
    torch.float32: "at::ScalarType::Float",
    torch.float64: "at::ScalarType::Double",
    torch.float16: "at::ScalarType::Half",
    torch.int64: "at::ScalarType::Long",
    torch.int32: "at::ScalarType::Int",
    torch.int16: "at::ScalarType::Short",
    torch.int8: "at::ScalarType::Char",
    torch.uint8: "at::ScalarType::Byte",
    torch.bool: "at::ScalarType::Bool",
    torch.bfloat16: "at::ScalarType::BFloat16",
}

INDEX_TYPE = "long"

RTYPE_TO_CPP = {
    "sum": "+",
    "min": "min",
    "max": "max",
    "argmin": "argmin",
    "argmax": "argmax",
    "any": "||",
}

class EnflameCodegen(torch.fx.Interpreter):
    def __init__(self, graph):
        self.code = fixed_code
        self.src_code = ''
        
        self.args_dict = {}
        self.input_args =[]
        self.output_args = []
        self.axes_dict = {}

        self.type_count = 0
        self.const_count = 0
        self.bool_count = 0
        self.axes_count = 0
        self.addmm_count =0
        self.name_count = 0
        self.reshape_count = 0
        self.reducemean_count = 0
        self.gather_count = 0
        self.bn_count = 0
        self.getitem_count = 0
        self.max_pool_count = 0 

        self.graph = graph
        super().__init__(graph)
        self.override = EnflameOverrides

    def placeholder(self, name, target, args, kwargs):    
        self.args_dict[name] = 'tmp' + str(len(self.args_dict))
        self.input_args.append(self.cur_node)
        
        in_shape = self.get_shape
        self.src_code += f'  std::vector<int64_t> {self.args_dict[name]}_in_shape{in_shape_size};\n'

        data_type = self.cur_node.meta['val'].dtype.__str__()
        if data_type == "torch.float32":
            self.src_code += f'  builder::Type {self.args_dict[name]}_input_type({self.args_dict[name]}_in_shape, ptype);\n'
        elif data_type == "torch.int64":
            self.src_code += f'  builder::Type {self.args_dict[name]}_input_type({self.args_dict[name]}_in_shape, builder::PrimitiveType::S64());\n'
        else:
            raise ValueError(data_type)
            
        self.src_code += f'  builder::Op {self.args_dict[name]} = hlir_builder->CreateInput({self.args_dict[name]}_input_type);\n\n'

    def call_function(self, name, target, args, kwargs):   
        if name not in self.args_dict.keys():
            self.args_dict[name] = 'tmp' + str(len(self.args_dict))
        
        if 'convolution' in name:
            if 'backward' in name:
                self.conv_grad(name, target, args, kwargs)
                return
            else:
                self.conv(name, target, args, kwargs)
                return
    
        if "max_pool2d_with_indices" in name:
            if 'backward' in name:
                self.max_pool_grad(name, target, args, kwargs)
                return
            self.max_pool(name, target, args, kwargs)
            return
         
        if target.__name__.split('::')[-1].title().split('.')[0] in op_set.keys():
            operation = op_set[target.__name__.split('::')[-1].title().split('.')[0]]
        else:
            operation = target.__name__.split('::')[-1].title().split('.')[0]
        
        operation = operation.split('_')[0]
        
        args_str = self.para_args(operation, args)

        getattr(self, operation)(self.args_dict[name], args)
        
        if "gather" in name:
            shape = self.get_shape()
            self.src_code += f'  builder::Type gather_type{self.gather_count}({shape}, ptype);\n\n'
            
            self.src_code += f"  std::vector<int64_t> gather_offset_dim{self.gather_count};\n"
            self.src_code += f"  for (int64_t i = 0; i < {args[1]}; i++) {'{'}\n     gather_offset_dim{self.gather_count}.emplace_back(i);\n  {'}'}\n\n"
            self.src_code += f"  auto gather_data_shape{self.gather_count} = {args_str[0]}.GetType().GetShape();\n"
            self.src_code += f"  auto gather_indices_shape{self.gather_count} = {args_str[2]}.GetType().GetShape();\n"
            self.src_code += f"  for (int64_t i = {args[1]} + 1; i < gather_data_shape{self.gather_count}.size(); i++) {'{'}\n    gather_offset_dim{self.gather_count}.emplace_back(i - 1 + gather_indices_shape{self.gather_count}.size());\n  {'}'}\n"
            self.src_code += f"  std::vector<int64_t> gather_slice_size{self.gather_count}(gather_data_shape{self.gather_count});\n"
            self.src_code += f"  gather_slice_size{self.gather_count}[{args[1]}] = 1;\n\n"
            
            self.src_code += f"  builder::GatherDimensionNumbers gather_gnums{self.gather_count}(gather_offset_dim{self.gather_count}, {'{' + '1' + '}'}, {'{' + '1' + '}'}, gather_indices_shape{self.gather_count}.size());\n"
            
            self.src_code += f"  auto {self.args_dict[name]} = builder::Gather({args_str[0]}, {args_str[2]}, gather_gnums{self.gather_count}, gather_slice_size{self.gather_count}, false, gather_type{self.gather_count});\n"
            
            self.gather_count += 1

            return
        
        if 'batch_norm' in name:
            shape = []
            lenth = len(self.cur_node.meta['val'])
            for i in range (0, lenth):
                shape.append('{' + str(self.cur_node.meta['val'][i].shape).split('[')[-1].split(']')[0] + '}')
            
            # self.src_code += f'  builder::Type bn_type{self.gather_count}({shape}, ptype);\n\n'
            self.src_code += f"  std::vector<std::vector<int64_t>> tuple_shape{self.bn_count};\n"
            self.src_code += f"  std::vector<builder::PrimitiveType> tuple_dtype{self.bn_count};\n"
            # for i in range(0, lenth):
            #     self.src_code += f"    tuple_shape{self.bn_count}.push_back({shape[i]});\n"
            # self.src_code += f"  for (uint i = 0; i < {lenth}; i++) {'{'}\n"
            # # self.src_code += f"    tuple_shape{self.bn_count}.push_back({shape[i]});\n"
            # self.src_code += f"    tuple_dtype{self.bn_count}.push_back(builder::PrimitiveType::F32());\n  {'}'}\n"
            
            self.src_code += f"  builder::Type bn_type{self.bn_count}(tuple_shape{self.bn_count}, tuple_dtype{self.bn_count});\n"
            
            # self.src_code += f"  auto {self.args_dict[name]} = builder::BatchNormInference({args_str[0]}, {args_str[1]}, {args_str[2]}, {args_str[3]}, {args_str[4]}, 0.1, 3);\n"
            # self.src_code += f"  auto {self.args_dict[name]} = builder::BatchNormTraining({args_str[0]}, {args_str[1]}, {args_str[2]}, 0.1, 5);\n"
            
            self.src_code += f"  auto {self.args_dict[name]} = enflame::batch_norm(hlir_builder, {args_str[0]}, {args_str[1]}, {args_str[2]});\n"
            
            self.bn_count += 1
            return
        
        if 'threshold_backward' in name:
            self.src_code += f"  builder::Op {self.args_dict[name]} = builder::ReluGrad({', '.join(args_str)});\n\n"
            return
        
    def output(self, name, target, args, kwargs):
        for i in range(0, len(args[0])):
            self.output_args.append(args[0][i])   

        def run_node(self, n : Node) -> Any:
        self.cur_node = n
        op = n.op
        name = n.name
        target = n.target
        args = n.args
        kwargs = n.kwargs

        assert isinstance(args, tuple)
        assert isinstance(kwargs, dict)
            
        return getattr(self, op)(name, target, args, kwargs) 
    
    def codegen(self):
        self.run()
        
        tmp_str = []
        res_str = ''
        for i in range(0, len(self.output_args)):
            if isinstance(self.output_args[i], type(None)):
                continue
            else:
                tmp_str.append(self.args_dict[self.output_args[i].name])

        res_str += f'  hlir_builder->SetOutput({"{" + ", ".join(tmp_str) + "}"});\n'
        
        input_paras = ''
        for i in range(0, len(self.input_args)):
            input_paras += f'float* input_ptr{str(i)}, '
        
        output_paras = []
        for i in range(0, len(self.output_args)):
            output_paras.append(f'float* output_ptr{str(i)}')
        output_paras = ', '.join(output_paras)
        
        inputs = ''
        for i in range(0, len(self.input_args)):
            inputs += f'  input_ptrs.emplace_back(static_cast<void *>(input_ptr{str(i)}));\n'
            
        outputs = ''
        for i in range(0, len(self.output_args)):
            outputs += f'  output_ptrs.emplace_back(output_ptr{str(i)});\n'
        
        kernel_cal = self.generate_kernel_cal()
        
        # self.code = self.code.replace('{self.src_code}', self.src_code)
        # self.code = self.code.replace('{res_str}', res_str)
        # self.code = self.code.replace('{input_paras}', input_paras)
        # self.code = self.code.replace('{output_paras}', output_paras)
        # self.code = self.code.replace('{inputs}', inputs)
        # self.code = self.code.replace('{outputs}', outputs)
        # self.code = self.code.replace('{kernel_cal}', kernel_cal)
        
        # print(self.code)
        
        print(self.code.format(self.src_code = self.src_code
                                res_str = res_str, 
                                input_paras = input_paras,
                                output_paras = output_paras,
                                inputs = inputs,
                                outputs = outputs,
                                kernel_cal = kernel_cal))
        
        raise ValueError("code")
    
        return self.code.format(self.src_code = self.src_code
                                res_str = res_str, 
                                input_paras = input_paras,
                                output_paras = output_paras,
                                inputs = inputs,
                                outputs = outputs,
                                kernel_cal = kernel_cal)

    def para_args(self, operation, args):
        args_str = []
        for i in range(0, len(args)):
            
            if isinstance(args[i], Node):
                args_str.append(self.args_dict[args[i].name])
                    
            elif isinstance(args[i], bool):
                args_str.append(str(args[i]).lower())
            elif isinstance(args[i], torch.fx.immutable_collections.immutable_list):
                args_str.append(str(args[i]).replace('[', '{').replace(']', '}'))
            elif isinstance(args[i], torch.dtype):
                in_shape_size = self.get_shape()
                self.src_code += f'  std::vector<int64_t> type{self.type_count}_in_shape{in_shape_size};\n'
                self.src_code += f'  builder::Type type{self.type_count} = builder::Type(type{self.type_count}_in_shape, ptype);\n\n'
                args_str.append(f'type{self.type_count}')
                self.type_count += 1
            else:          
                if 'squeeze' in operation:
                    self.src_code += f'  builder::Type axes{str(self.axes_count)}_type({"{" + "1" + "}"}, builder::PrimitiveType::S64());\n'
                    self.src_code += f'  std::vector<int64_t> axes{self.axes_count}_data = {"{" + str(args[i]).split("[")[-1].split("]")[0] + "}"};\n'
                    self.src_code += f'  builder::Op axes{self.axes_count} = builder::Const(hlir_builder, (axes{self.axes_count}_data.data()), axes{self.axes_count}_type);\n'
                    args_str.append(f'axes{self.axes_count}')
                    shape = self.get_shape()
                    self.src_code += f"  builder::Type output{self.axes_count}_type({shape}, builder::PrimitiveType::F32());\n"
                    args_str.append(f"output{self.axes_count}_type")
                    self.axes_count +=1
                    
                    pass
                else:                                  
                    in_shape_size = '{1}'
                    if isinstance(type(args[i]), type(int)):
                        self.src_code += f'  int value{self.const_count} = {str(args[i])};\n'
                    else:
                        self.src_code += f'  float value{self.const_count} = {str(args[i])};\n'
                    self.src_code += f'  std::vector<int64_t> const{self.const_count}_in_shape{in_shape_size};\n'
                    self.src_code += f'  builder::Type value{self.const_count}_type(const{self.const_count}_in_shape, ptype);\n'
                    self.src_code += f'  builder::Op const{self.const_count} = builder::Const(hlir_builder, static_cast<void *>(&value{self.const_count}), value{self.const_count}_type);\n\n'
                    args_str.append(f'const{self.const_count}')
                    self.const_count += 1

        return args_str
    
    def get_shape(self):
        shape = '{' + str(self.cur_node.meta['val'].shape).split('[')[-1].split(']')[0] + '}'
        return
    
    def gen(self, res, operation, args):
        args_str = self.para_args(operation, args)
        self.src_code += f"  builder::Op {res} = builder::{operation}({', '.join(args_str)});\n\n"
    
    @staticmethod
    def Add(self, res, args):
        self.gen(res, 'Add', args_str)
        return

    @staticmethod
    def Sub(self, res, args):
        self.gen(res, 'Sub', args)
        return

    @staticmethod
    def Sqrt(self, res, args):
        self.gen(res, 'Sqrt', args)
        return
        
    @staticmethod
    def Clone(self, res, args):
        self.src_code += f"  builder::Op {res} = {self.args_dict[args[0].name]};\n"
        return
    
    @staticmethod
    def ReduceMean(self, res, args):
        args_str = self.para_args('ReduceMean', args)
        args_str[1], args_str[2] = args_str[2], args_str[1]
            shape = self.get_shape()
        self.src_code += f'  builder::Type reducemean_shape{self.reducemean_count}({shape}, ptype);\n'
        args_str.append(f'reducemean_shape{self.reducemean_count}')
        for i in range(0, len(args_str)):
            print(args_str[i])
        tmp = args[1].copy()
        tmp.sort()
        for i in range(0, len(tmp)):
            tmp[i] = (tmp[i] + len(args[0].meta['val'].shape)) % len(args[0].meta['val'].shape)
        args_str[2] = str(tmp).replace('[', '{').replace(']', '}')
            
        args_str[2] = '{2, 3}'
        self.reducemean_count += 1
        self.src_code += f"  builder::Op {res} = builder::ReduceMean({', '.join(args_str)});\n\n"
        return
    
    @staticmethod
    def Reshape(self, res, args):
        args_str = self.args_str('Reshape', args)
        shape = slef,get_shape
        self.src_code += f'  builder::Type reshape_shape{self.reshape_count}({shape}, ptype);\n'
        args_str[1] = f'reshape_shape{self.reshape_count}'
        self.reshape_count += 1
        
        self.src_code += f"  builder::Op {res} = builder::Reshape({', '.join(args_str)});\n\n"
        return
    
    @staticmethod
    def Addmm(self, res, args):
        self.src_code += f"  builder::Op addmm{self.addmm_count} = builder::Gemm({'{' + args_str[1] + ', ' + args_str[2] + '}'});\n"
        self.src_code += f"  builder::Op {self.args_dict[name]} = builder::Add({args_str[0]}, addmm{self.addmm_count});\n"
        self.addmm_count += 1
        return
    
    @staticmethod
    def ReduceMax(self, res, args):
        args_str = self.args_str('ReduceMax', args)
        if len(args_str) ==3:
            args_str[1], args_str[2] = args_str[2], args_str[1]
        self.src_code += f"  builder::Op {res} = builder::ReduceMax({', '.join(args_str)});\n\n"
        return
    
    @staticmethod
    def ReduceSum(self, res, args):
        args_str = self.args_str('ReduceSum', args)
        if len(args_str) ==3:
            args_str[1], args_str[2] = args_str[2], args_str[1]
        self.src_code += f"  builder::Op {res} = builder::ReduceSum({', '.join(args_str)});\n\n"
        return   
    
    @staticmethod
    def Gemm(self, res, args):
        args_str = self.args_str('Gemm', args)
        self.src_code += f"  builder::Op {res} = builder::Gemm({'{' + ', '.join(args_str) + '}'});\n\n"
        return   

    @staticmethod
    def Transpose(self, res, args):
        args_str = self.para_args('Transpose', args)
        if len(args) == 1:
            args_str.append('{1, 0}')
        self.src_code += f"  builder::Op {res} = builder::Transpose({', '.join(args_str)});\n\n"
        return
    
    @staticmethod
    def Getitem(self, res, args):
        if 'max_pool2d' in args[0].name:
            self.args_dict[name] = self.args_dict[args[0].name]
            return
        
        shape = self.get_shape
        self.src_code += f'  builder::Type getitem_type{self.getitem_count}({shape}, ptype);\n\n'
        self.src_code += f"  builder::Op {self.args_dict[name]} = builder::GetTupleElement({self.args_dict[args[0].name]}, {int(self.cur_node.args[1])}, getitem_type{self.getitem_count});\n\n"
        self.getitem_count += 1
        return
    
    def conv(self, name, target, args, kwargs):
        args_str =[]
        for i in range(0, 3):
            if isinstance(args[i], type(None)):
                continue
            args_str.append(self.args_dict[args[i].name])
                
        stride = str(args[3]).replace('[','{').replace(']','}')
        
        if len(args[4]) == 1:
            row_padding, col_padding = args[4][0]
        else:
            row_padding = args[4][0]
            col_padding = args[4][1]
        
        padding = f"{'{' + str(row_padding)}, {str(row_padding)}, {str(col_padding)}, {str(col_padding) + '}'}"
        
        dilation = str(args[5]).replace('[','{').replace(']','}')
        group = args[8]
    
        self.src_code += f"  std::vector<builder::Op> {self.args_dict[name]}_inputs = {'{' + ', '.join(args_str) + '}'};\n"
        self.src_code += f'  builder::Op {self.args_dict[name]} = builder::Conv2D({self.args_dict[name]}_inputs, {group}, "NOTSET", "NCHW", {stride}, {padding}, {dilation});\n\n'
    
    def conv_grad(self, name, target, args, kwargs):
        args_str = []
        
        for i in range(0, 3):
            if isinstance(args[i], type(None)):
                continue
            else:
                args_str.append(self.args_dict[args[i].name])
        
        bias = str(args[3]).replace('[','{').replace(']','}')
        stride = str(args[4]).replace('[','{').replace(']','}')
        padding = str(args[5]).replace('[','{').replace(']','}')
        dilation = str(args[6]).replace('[','{').replace(']','}')

        self.src_code += f"  auto {self.args_dict[name]} = enflame::conv2d_grad(hlir_builder, {args_str[0]}, {args_str[1]}, {args_str[2]}, {bias}, {stride}, {padding}, {dilation});\n"
 
    def max_pool(self, name, target, args, kwargs):
        ceil_mode = 'false'
        return_indices = 'false'
        padding = '{0, 0, 0, 0}'
        dilation = '{1, 1}'
        shape = '{' + str(self.cur_node.meta['val'][0].shape).split('[')[-1].split(']')[0] + '}'
        dtype = self.cur_node.meta['val'][0].dtype
        
        self.src_code += f'  builder::Type max_pool_type{self.max_pool_count} = builder::Type({shape}, ptype);\n\n'
        print(shape)
        print(dtype)
        print(self.cur_node)
        
        if len(args) == 3:
            ksize = str(args[1]).replace('[','{').replace(']','}')
            stride = str(args[2]).replace('[','{').replace(']','}')
        else:
            ksize = str(args[1]).replace('[','{').replace(']','}')
            stride = str(args[2]).replace('[','{').replace(']','}')            

        
        self.src_code += f'  builder::Op {self.args_dict[name]} = builder::MaxPool2D({self.args_dict[args[0].name]}, {ksize}, {ceil_mode}, {return_indices}, "NOTSET", "NCHW", {stride}, {padding}, {"{" + "}"}, max_pool_type{self.max_pool_count});\n'
        
        self.max_pool_count += 1
   
    def max_pool_grad(self, name, target, args, kwargs):
        ksize = str(args[2]).replace('[','{').replace(']','}')
        strides = str(args[3]).replace('[','{').replace(']','}')
        padding = str(args[4]).replace('[','{').replace(']','}')
        
        self.src_code += f"  auto {self.args_dict[name]} = enflame::max_pool2d_grad(hlir_builder, {self.args_dict[args[0].name]}, {self.args_dict[args[1].name]}, {ksize}, {strides}, {padding});\n"

    def generate_kernel_cal(self):
        args = []
        args_str = ''
        del_args = ''
        call_str = '    kernel_cpp_0('
        for i in range(0, len(self.input_args)):
            args.append('arg' + str(i) + '_1')
            
            args_str += '    arg' + str(i) + '_1 = ' + (f"rand_strided(({str(self.input_args[i].meta['val'].shape).split('[')[-1].split(']')[0]}, ), "
                                                    f"{str(self.input_args[i].meta['val'].stride())}, "
                                                    f"device='{self.input_args[i].meta['val'].device.type}', "
                                                    f"dtype={self.input_args[i].meta['val'].dtype.__str__()})\n")
            
            del_args += '    del arg' + str(i) + '_1\n' 
            
            call_str += 'c_void_p(arg' + str(i) + '_1.data_ptr()), '
            
        if "(, )" in args_str:
            args_str = args_str.replace("(, )", "()")
        
        bufs = []
        buf_str = ''
        for i in range(0, len(self.output_args)):
            if isinstance(self.output_args[i], type(None)):
                if i != len(self.output_args) - 1:
                    call_str += ', '
                else:
                    call_str += ')\n'
                # raise ValueError("None")
                bufs.append('buf' + str(i))
                buf_str += '    buf' + str(i) + ' = ' + (f"empty_strided((, ), "
                                                    f"{'(, )'})")
                continue
            call_str += 'c_void_p(buf' + str(i) + '.data_ptr())'
            if i != len(self.output_args) - 1:
                call_str += ', '
            else:
                call_str += ')\n'
                
            bufs.append('buf' + str(i))
            
            
            
            buf_str += '    buf' + str(i) + ' = ' + (f"empty_strided(({str(self.output_args[i].meta['val'].shape).split('[')[-1].split(']')[0]}, ), "
                                                    f"{str(self.output_args[i].meta['val'].stride())}, "
                                                    f"device='{self.output_args[i].meta['val'].device.type}', "
                                                    f"dtype={self.output_args[i].meta['val'].dtype.__str__()})\r\n")
        
        if "(, )" in buf_str:
            buf_str = buf_str.replace("(, )", "()")
        
        func_call_str = f'''def call(args):
    {', '.join(args)}, = args
    args.clear()
    
{buf_str}
{call_str}
{del_args}
    return ({', '.join(bufs)})\n\n'''
    
        func_main_str = f'''if __name__ == "__main__":
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    
{args_str}
    print(call([{', '.join(args)}]))\n\n'''
    
        return func_call_str + func_main_str
   
        

def reduction_init(reduction_type, dtype):
    if reduction_type in ("sum", "any"):
        return 0
    if reduction_type in {"max", "argmax"}:
        return (
            f"-std::numeric_limits<{DTYPE_TO_CPP[dtype]}>::infinity()"
            if is_float_dtype(dtype)
            else f"std::numeric_limits<{DTYPE_TO_CPP[dtype]}>::min()"
        )
    if reduction_type in {"min", "argmin"}:
        return (
            f"std::numeric_limits<{DTYPE_TO_CPP[dtype]}>::infinity()"
            if is_float_dtype(dtype)
            else f"std::numeric_limits<{DTYPE_TO_CPP[dtype]}>::max()"
        )
    raise AssertionError(reduction_type)


def reduction_combine(reduction_type, var, next_value):
    if reduction_type == "sum":
        return f"{var} += {next_value}"
    if reduction_type == "any":
        return f"{var} = {var} || {next_value}"
    return f"{var} = std::{reduction_type}({var}, {next_value})"


def reduction_combine_vec(reduction_type, var, next_value):
    if reduction_type == "max":
        return f"{var} = at::vec::maximum({var}, {next_value})"
    elif reduction_type == "min":
        return f"{var} = at::vec::minimum({var}, {next_value})"
    elif reduction_type == "sum":
        return f"{var} += {next_value}"
    else:
        raise NotImplementedError()


index_value_name_counter = 1


def argmax_argmin_prefix(reduction_type, src_dtype, tmpvar):
    global index_value_name_counter
    struct_name = f"IndexValue_{index_value_name_counter}"
    index_value_name_counter += 1

    # A small annoyance, due to it being a little cumbersome to just throw {} into strings
    prefix = [
        f"struct {struct_name} {{size_t index; {DTYPE_TO_CPP[src_dtype]} value;}};",
        f"{struct_name} {tmpvar}{{0, {reduction_init(reduction_type, src_dtype)}}};",
    ]
    if reduction_type == "argmax":
        prefix.extend(
            [
                f"#pragma omp declare reduction(argmax : struct {struct_name} :\\",
                "    omp_out.value = omp_in.value < omp_out.value ? omp_out.value : omp_in.value,\\",
                "    omp_out.index = omp_in.value < omp_out.value ? omp_out.index : omp_in.index)\\",
                f"\tinitializer(omp_priv = {{0, {reduction_init(reduction_type, src_dtype)}}})",
            ]
        )
    elif reduction_type == "argmin":
        prefix.extend(
            [
                f"#pragma omp declare reduction(argmin : struct {struct_name} :\\",
                "    omp_out.value = omp_in.value > omp_out.value ? omp_out.value : omp_in.value,\\",
                "    omp_out.index = omp_in.value > omp_out.value ? omp_out.index : omp_in.index)\\",
                f"\tinitializer(omp_priv = {{0, {reduction_init(reduction_type, src_dtype)}}})",
            ]
        )
    return prefix


def float16_reduction_prefix(rtype):
    # TODO: This user-defined reduction uses float16 accumulation for sum. To reduce numerical
    # errors, float32 accumulation should be used instead.
    assert rtype in (
        "sum",
        "any",
    ), f"float16 user-defined reduction only supports 'sum' and 'any' but got {rtype}"
    prefix = [
        f"#pragma omp declare reduction({RTYPE_TO_CPP[rtype]}:{DTYPE_TO_CPP[torch.float16]}:"
        + f"omp_out = omp_out {RTYPE_TO_CPP[rtype]} omp_in)"
    ]
    return prefix


def parallel_num_threads():
    threads = config.cpp.threads
    if threads < 1:
        threads = torch.get_num_threads()
    return threads


@functools.lru_cache()
def cpp_prefix():
    path = Path(__file__).parent / "cpp_prefix.h"
    with path.open() as f:
        _, filename = codecache.write(
            f.read(),
            "h",
        )
    return f'#include "{filename}"'

class EnflameOverrides(OpOverrides):
    """Map element-wise ops to C++"""

    @staticmethod
    def to_dtype(x, dtype):
        assert dtype in DTYPE_TO_CPP, f"{dtype} missing from {__name__}.DTYPE_TO_CPP"
        return f"static_cast<{DTYPE_TO_CPP[dtype]}>({x})"

    @staticmethod
    def abs(x):
        # import pdb
        # pdb.set_trace()
        print("returning abs")
        return f"std::abs({x})"

    @staticmethod
    def sin(x):
        return f"std::sin({x})"

    @staticmethod
    def cos(x):
        return f"std::cos({x})"

    @staticmethod
    def exp(x):
        # return f"Sleef_expf_u10({x})"
        return f"std::exp({x})"

    @staticmethod
    def erf(x):
        return f"std::erf({x})"

    @staticmethod
    def sqrt(x):
        return f"std::sqrt({x})"

    @staticmethod
    def rsqrt(x):
        return f"1 / std::sqrt({x})"

    @staticmethod
    def log1p(x):
        return f"std::log1p({x})"

    @staticmethod
    def expm1(x):
        return f"std::expm1({x})"

    @staticmethod
    def tanh(x):
        return f"std::tanh({x})"

    @staticmethod
    def signbit(x):
        return f"std::signbit({x})"

    @staticmethod
    def pow(a, b):
        return f"std::pow({a}, {b})"

    @staticmethod
    def log(x):
        return f"std::log({x})"

    @staticmethod
    def round(x):
        return f"std::nearbyint({x})"

    @staticmethod
    def floor(x):
        return f"std::floor({x})"

    @staticmethod
    def floordiv(a, b):
        # a and b are integer type
        quot = f"{a} / {b}"
        rem = f"{a} % {b}"
        return f"(({a} < 0) != ({b} < 0) ? ({rem} != 0 ? {quot} - 1 : {quot}) : {quot})"

    @staticmethod
    def ceil(x):
        return f"std::ceil({x})"

    @staticmethod
    def trunc(x):
        return f"std::trunc({x})"

    @staticmethod
    def truncdiv(a, b):
        # a and b are integer type
        return f"{a} / {b}"

    @staticmethod
    def fmod(a, b):
        return f"std::fmod({a}, {b})"

    @staticmethod
    def isinf(x):
        return f"std::isinf({x})"

    @staticmethod
    def isnan(x):
        return f"std::isnan({x})"

    @staticmethod
    def lgamma(x):
        return f"std::lgamma({x})"

    @staticmethod
    def relu(x):
        return f"{x} * ({x}>0)"

    @staticmethod
    def minimum(a, b):
        return f"({b} != {b}) ? {b} : std::min({a}, {b})"

    @staticmethod
    def maximum(a, b):
        return f"({b} != {b}) ? {b} : std::max({a}, {b})"

    @staticmethod
    def where(a, b, c):
        return f"{a} ? {b} : {c}"

    @staticmethod
    def mod(a, b):
        return f"mod({a}, {b})"

    @staticmethod
    def constant(val, dtype):
        if val == float("inf"):
            return f"std::numeric_limits<{DTYPE_TO_CPP[dtype]}>::infinity()"
        elif val == float("-inf"):
            return f"-std::numeric_limits<{DTYPE_TO_CPP[dtype]}>::infinity()"
        elif math.isnan(val):
            return f"std::numeric_limits<{DTYPE_TO_CPP[dtype]}>::quiet_NaN()"
        elif val is True or val is False:
            return ops.to_dtype(str(val).lower(), dtype)
        return ops.to_dtype(repr(val), dtype)

    @staticmethod
    def masked(mask, body, other):
        code = BracesBuffer()
        var = V.kernel.cse.newvar()
        if other == float("-inf"):
            code.writeline(f"float {var} = -std::numeric_limits<float>::infinity();")
        elif other == float("inf"):
            code.writeline(f"float {var} = std::numeric_limits<float>::infinity();")
        elif isinstance(other, float):
            code.writeline(f"float {var} = {other};")
        else:
            code.writeline(f"auto {var} = {other!r};")
        code.writeline(f"if({mask})")
        with V.kernel.swap_buffers(code), code.indent():
            result = body()
            code.writeline(f"{var} = {result};")
        V.kernel.compute.splice(code)
        return var

    @staticmethod
    def logical_and(a, b):
        return f"{a} && {b}"

    @staticmethod
    def logical_or(a, b):
        return f"{a} || {b}"

    @staticmethod
    def rand(seed: sympy.Expr, offset: sympy.Expr, dtype):
        return f"static_cast<{DTYPE_TO_CPP[dtype]}>(normalized_rand_cpu({seed}, {offset}));"

    @staticmethod
    def randn(seed: sympy.Expr, offset: sympy.Expr, dtype):
        return f"static_cast<{DTYPE_TO_CPP[dtype]}>(randn_cpu({seed}, {offset}));"

    @staticmethod
    def sigmoid(x):
        x = ops.exp(f"-{x}")
        return f"1 / (1 + {x})"

    @staticmethod
    def sign(x):
        code = BracesBuffer()
        # auto tmp5 = tmp4 < 0 ? -1 : 1;
        left = V.kernel.cse.newvar()
        right = V.kernel.cse.newvar()
        result = V.kernel.cse.newvar()
        code.writeline(f"auto {left} = {x} > 0 ? 1 : 0;")
        code.writeline(f"auto {right} = {x} < 0 ? 1 : 0;")
        code.writeline(f"auto {result} = {left} - {right};")
        V.kernel.compute.splice(code)
        return result
