import torch
import torch._dynamo.variables.torch
import torch.fx.graph
from torch.fx.graph import (
    _Namespace,
    PythonCode,
    _custom_builtins,
    _format_target,
    magic_methods,
    inplace_methods,
    dtype_abbrs,
    _origin_type_map
)
from torch.fx.node import (
    Argument,
    Node,
    map_arg,
    _type_repr,
    _get_qualified_name
)
from typing import Any, Tuple, Dict, List
import re


def python_type_bar(self):
    raise ValueError("monkey success")
    return type(self.value)


def _gen_python_code_bar(self, nodes, root_module: str, namespace: _Namespace, *, verbose: bool = False) -> PythonCode:
    free_vars: List[str] = []
    body: List[str] = []
    globals_: Dict[str, Any] = {}
    wrapped_fns: Dict[str, None] = {}

    # Wrap string in list to pass by reference
    maybe_return_annotation: List[str] = ['']

    def add_global(name_hint: str, obj: Any):
        """Add an obj to be tracked as a global.

        We call this for names that reference objects external to the
        Graph, like functions or types.

        Returns: the global name that should be used to reference 'obj' in generated source.
        """
        # normalize the name hint to get a proper identifier
        global_name = namespace.create_name(name_hint, obj)

        if global_name in globals_:
            assert globals_[global_name] is obj
            return global_name
        globals_[global_name] = obj
        return global_name

    # Pre-fill the globals table with registered builtins.
    for name, (_, obj) in _custom_builtins.items():
        add_global(name, obj)

    def type_repr(o: Any):
        if o == ():
            # Empty tuple is used for empty tuple type annotation Tuple[()]
            return '()'

        typename = _type_repr(o)

        if hasattr(o, '__origin__'):
            # This is a generic type, e.g. typing.List[torch.Tensor]
            origin_type = _origin_type_map.get(o.__origin__, o.__origin__)
            origin_typename = add_global(_type_repr(origin_type), origin_type)

            if hasattr(o, '__args__'):
                # Assign global names for each of the inner type variables.
                args = [type_repr(arg) for arg in o.__args__]

                if len(args) == 0:
                    # Bare type, such as `typing.Tuple` with no subscript
                    # This code-path used in Python < 3.9
                    return origin_typename

                return f'{origin_typename}[{",".join(args)}]'
            else:
                # Bare type, such as `typing.Tuple` with no subscript
                # This code-path used in Python 3.9+
                return origin_typename

        # Common case: this is a regular module name like 'foo.bar.baz'
        return add_global(typename, o)

    def _format_args(args: Tuple[Argument, ...], kwargs: Dict[str, Argument]) -> str:
        def _get_repr(arg):
            # Handle NamedTuples (if it has `_fields`) via add_global.
            if isinstance(arg, tuple) and hasattr(arg, '_fields'):
                qualified_name = _get_qualified_name(type(arg))
                global_name = add_global(qualified_name, type(arg))
                return f"{global_name}{repr(tuple(arg))}"
            elif isinstance(arg, torch._ops.OpOverload):
                qualified_name = _get_qualified_name(arg)
                global_name = add_global(qualified_name, arg)
                return f"{global_name}"
            return repr(arg)
        args_s = ', '.join(_get_repr(a) for a in args)
        kwargs_s = ', '.join(f'{k} = {_get_repr(v)}' for k, v in kwargs.items())
        if args_s and kwargs_s:
            return f'{args_s}, {kwargs_s}'
        return args_s or kwargs_s

    # Run through reverse nodes and record the first instance of a use
    # of a given node. This represents the *last* use of the node in the
    # execution order of the program, which we will use to free unused
    # values
    node_to_last_use: Dict[Node, Node] = {}
    user_to_last_uses: Dict[Node, List[Node]] = {}

    def register_last_uses(n: Node, user: Node):
        if n not in node_to_last_use:
            node_to_last_use[n] = user
            user_to_last_uses.setdefault(user, []).append(n)

    for node in reversed(nodes):
        map_arg(node.args, lambda n: register_last_uses(n, node))
        map_arg(node.kwargs, lambda n: register_last_uses(n, node))

    def delete_unused_values(user: Node):
        """
        Delete values after their last use. This ensures that values that are
        not used in the remainder of the code are freed and the memory usage
        of the code is optimal.
        """
        if user.op == 'placeholder':
            return
        if user.op == 'output':
            body.append('\n')
            return
        nodes_to_delete = user_to_last_uses.get(user, [])
        if len(nodes_to_delete):
            to_delete_str = ' = '.join([repr(n) for n in nodes_to_delete] + ['None'])
            body.append(f';  {to_delete_str}\n')
        else:
            body.append('\n')

    prev_stacktrace = None

    def append_stacktrace_summary(node: Node):
        """
        Append a summary of the stacktrace to the generated code. This is
        useful for debugging.
        """
        nonlocal prev_stacktrace
        pattern = re.compile(r"^File \"(.+)\", line (\d+), in (.+)$")

        if node.op not in {'placeholder', 'output'}:
            if node.stack_trace:
                if node.stack_trace != prev_stacktrace:
                    prev_stacktrace = node.stack_trace

                    lines = node.stack_trace.strip().split('\n')
                    # stacktrace should have innermost frame last, so we
                    # iterate backwards to find the first line that starts
                    # with 'File '
                    summary_str = ""
                    for idx in range(len(lines) - 2, -1, -1):
                        line = lines[idx].strip()
                        matches = pattern.match(line)
                        if matches:
                            file = matches.group(1)
                            lineno = matches.group(2)
                            # next line should be the code
                            code = lines[idx + 1].strip()
                            summary_str = f'File: {file}:{lineno}, code: {code}'
                            break
                    body.append(f'\n# {summary_str}\n')
            elif prev_stacktrace != "":
                prev_stacktrace = ""
                body.append('\n# No stacktrace found for following nodes\n')

    def stringify_shape(shape: torch.Size) -> str:
        return f"[{', '.join(str(x) for x in shape)}]"

    def emit_node(node: Node):
        maybe_type_annotation = '' if node.type is None else f' : {type_repr(node.type)}'

        if verbose:
            # override annotation with more detailed information
            from torch._subclasses.fake_tensor import FakeTensor
            from torch.fx.experimental.proxy_tensor import py_sym_types
            from torch.fx.passes.shape_prop import TensorMetadata

            meta_val = node.meta.get('val', node.meta.get('tensor_meta', None))

            if isinstance(meta_val, FakeTensor):
                maybe_type_annotation = f': {dtype_abbrs[meta_val.dtype]}{stringify_shape(meta_val.shape)}'
            elif isinstance(meta_val, py_sym_types):
                maybe_type_annotation = f': Sym({meta_val})'
            elif isinstance(meta_val, TensorMetadata):
                maybe_type_annotation = f': {dtype_abbrs[meta_val.dtype]}{stringify_shape(meta_val.shape)}'

        if node.op == 'placeholder':
            assert isinstance(node.target, str)
            maybe_default_arg = '' if not node.args else f' = {repr(node.args[0])}'
            free_vars.append(f'{node.target}{maybe_type_annotation}{maybe_default_arg}')
            raw_name = node.target.replace('*', '')
            if raw_name != repr(node):
                body.append(f'{repr(node)} = {raw_name}\n')
            return
        elif node.op == 'call_method':
            assert isinstance(node.target, str)
            body.append(
                f'{repr(node)}{maybe_type_annotation} = {_format_target(repr(node.args[0]), node.target)}'
                f'({_format_args(node.args[1:], node.kwargs)})')
            return
        elif node.op == 'call_function':
            assert callable(node.target)
            # pretty print operators
            if getattr(node.target, "__module__", "") == '_operator' and node.target.__name__ in magic_methods:
                assert isinstance(node.args, tuple)
                body.append(f'{repr(node)}{maybe_type_annotation} = '
                            f'{magic_methods[node.target.__name__].format(*(repr(a) for a in node.args))}')
                return

            # pretty print inplace operators; required for jit.script to work properly
            # not currently supported in normal FX graphs, but generated by torchdynamo
            if getattr(node.target, "__module__", "") == '_operator' and node.target.__name__ in inplace_methods:
                body.append(f'{inplace_methods[node.target.__name__].format(*(repr(a) for a in node.args))};  '
                            f'{repr(node)}{maybe_type_annotation} = {repr(node.args[0])}')
                return

            qualified_name = _get_qualified_name(node.target)
            global_name = add_global(qualified_name, node.target)
            # special case for getattr: node.args could be 2-argument or 3-argument
            # 2-argument: attribute access; 3-argument: fall through to attrib function call with default value
            if global_name == 'getattr' and \
                    isinstance(node.args, tuple) and \
                    isinstance(node.args[1], str) and \
                    node.args[1].isidentifier() and \
                    len(node.args) == 2:
                body.append(f'{repr(node)}{maybe_type_annotation} = {_format_target(repr(node.args[0]), node.args[1])}')
                return
            body.append(f'{repr(node)}{maybe_type_annotation} = {global_name}({_format_args(node.args, node.kwargs)})')
            if node.meta.get('is_wrapped', False):
                wrapped_fns.setdefault(global_name)
            return
        elif node.op == 'call_module':
            assert isinstance(node.target, str)
            body.append(f'{repr(node)}{maybe_type_annotation} = '
                        f'{_format_target(root_module, node.target)}({_format_args(node.args, node.kwargs)})')
            return
        elif node.op == 'get_attr':
            assert isinstance(node.target, str)
            body.append(f'{repr(node)}{maybe_type_annotation} = {_format_target(root_module, node.target)}')
            return
        elif node.op == 'output':
            if node.type is not None:
                maybe_return_annotation[0] = f" -> {type_repr(node.type)}"
            body.append(self.generate_output(node.args[0]))
            return
        raise NotImplementedError(f'node: {node.op} {node.target}')

    for node in nodes:
        # NOTE: emit_node does not emit a string with newline. It depends
        # on delete_unused_values to append one
        if verbose:
            append_stacktrace_summary(node)
        emit_node(node)
        delete_unused_values(node)

    if len(body) == 0:
        # If the Graph has no non-placeholder nodes, no lines for the body
        # have been emitted. To continue to have valid Python code, emit a
        # single pass statement
        body.append('pass\n')

    if len(wrapped_fns) > 0:
        wrap_name = add_global('wrap', torch.fx.wrap)
        wrap_stmts = '\n'.join([f'{wrap_name}("{name}")' for name in wrapped_fns])
    else:
        wrap_stmts = ''

    if self._body_transformer:
        body = self._body_transformer(body)

    for name, value in self.additional_globals():
        add_global(name, value)

    prologue = self.gen_fn_def(free_vars, maybe_return_annotation[0])

    code = ''.join(body).lstrip('\n')
    code = '\n'.join('    ' + line for line in code.split('\n'))
    fn_code = f"{wrap_stmts}\n{prologue}\n{code}"
    return PythonCode(fn_code, globals_)


torch._dynamo.variables.torch.TorchVariable.python_type = python_type_bar
torch.fx.graph.CodeGen._gen_python_code = _gen_python_code_bar
