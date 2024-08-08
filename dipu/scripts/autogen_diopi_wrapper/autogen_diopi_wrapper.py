# Copyright (c) 2023, DeepLink.
import textwrap
import yaml
import re
import json
import os
from typing import Mapping, Match, Optional, Sequence
from diopi_wrapper_template import (
    diopi_wrapper_file_template_content,
    diopi_wrapper_function_template_content,
    op_no_customfallback_with_autocompare_register_template_content,
    op_no_customfallback_no_autocompare_register_template_content,
    custom_autograd_template_content,
    autocompare_template_content,
    op_with_customfallback_with_autocompare_register_template_content,
    op_with_customfallback_no_autocompare_register_template_content,
)


class CodeTemplate:
    substitution_str = r"(^[^\n\S]*)?\$([^\d\W]\w*|\{,?[^\d\W]\w*\,?})"
    substitution = re.compile(substitution_str, re.MULTILINE)

    pattern: str
    filename: str

    @staticmethod
    def from_file(filename: str) -> "CodeTemplate":
        with open(filename, "r") as f:
            return CodeTemplate(f.read(), filename)

    def __init__(self, pattern: str, filename: str = "") -> None:
        self.pattern = pattern
        self.filename = filename

    def substitute(
        self, env: Optional[Mapping[str, object]] = None, **kwargs: object
    ) -> str:
        if env is None:
            env = {}

        def lookup(v: str) -> object:
            assert env is not None
            return kwargs[v] if v in kwargs else env[v]

        def indent_lines(indent: str, v: Sequence[object]) -> str:
            return "".join(
                [indent + l + "\n" for e in v for l in str(e).splitlines()]
            ).rstrip()

        def replace(match: Match[str]) -> str:
            indent = match.group(1)
            key = match.group(2)
            comma_before = ""
            comma_after = ""
            if key[0] == "{":
                key = key[1:-1]
                if key[0] == ",":
                    comma_before = ", "
                    key = key[1:]
                if key[-1] == ",":
                    comma_after = ", "
                    key = key[:-1]
            v = lookup(key)
            if indent is not None:
                if not isinstance(v, list):
                    v = [v]
                return indent_lines(indent, v)
            elif isinstance(v, list):
                middle = ", ".join([str(x) for x in v])
                if len(v) == 0:
                    return middle
                return comma_before + middle + comma_after
            else:
                return str(v)

        return self.substitution.sub(replace, self.pattern)


def parse_function_signature(
    schema: str,
) -> tuple[str, list[tuple[str, str, str]], list[tuple[str, str]]]:
    # Note: switch to parser if rules grow complex.

    def unwarp_brackets(input: str) -> str:
        input = input.strip()
        while input.startswith("("):
            input = input.removeprefix("(").removesuffix(")").strip()
        return input

    def parse_type_name_value_tuple(input: str) -> list[tuple[str, str, str]]:
        output = []
        for pair in input.split(","):
            # for 'type name = value'
            left, _, value = pair.partition("=")
            left, _, right = left.strip().rpartition(" ")
            left = left.strip()
            right = right.strip()
            type = left if left != "" else right
            name = right if left != "" else ""
            if type != "":
                output.append((type, name, value.strip()))
        return output

    function_name, _, others = schema.partition("(")
    parameters, _, return_values = others.rpartition("->")
    parameters, _, _ = parameters.rpartition(")")
    parameters = parse_type_name_value_tuple(parameters)
    return_values = parse_type_name_value_tuple(unwarp_brackets(return_values))
    return_values = [(a, b) for a, b, _ in return_values]
    return (function_name, parameters, return_values)


def generate_cpp_function_return_values(
    parameters: list[tuple[str, str, str]], return_values: list[tuple[str, str]]
) -> list[str]:
    REFERENCE_PATTERN = re.compile(r"\w+(\([a-z]+!\))")

    output = []
    for i, (type, name) in enumerate(return_values):
        if reference := REFERENCE_PATTERN.match(type):
            tag = reference.group(1)
            target = (name for type, name, _ in parameters if tag in type)
            if found := next(target, None):
                output.append(found)

        elif re.match(r"\w+", type):
            if name == "":
                name = "out" + ("" if len(return_values) == 1 else str(i))
            output.append(name)

        elif "bool" == type:
            output.append("out")

    return output


def generate_code_to_process_symint_array(
    name_list: list[str], vector_suffix: str = "Vector", size_suffix: str = "DiopiSize"
) -> str:
    HEAD = r"""
auto to_int = [](const c10::SymInt& t) -> int64_t { return t.expect_int(); };"""
    BODY = r"""
auto {0}{1} = c10::DimVector({0}.size());
std::transform({0}.cbegin(), {0}.cend(), {0}{1}.begin(), to_int);
auto {0}{2} = ::diopiSize_t{{{0}{1}.data(), static_cast<int64_t>({0}{1}.size())}};"""

    if len(name_list) == 0:
        return ""
    return HEAD + "".join(
        BODY.format(name, vector_suffix, size_suffix) for name in name_list
    )


def generate_parameters_from_schema(schema: str) -> list[str]:
    MAPPING = [
        (re.compile(p), r)
        for p, r in [  # Order matters
            # preprocess
            (r"\*", r""),
            (r"=.+", r""),
            # process
            (r"^Tensor\?\[\]", r"const c10::List<c10::optional<at::Tensor>>&"),
            (r"^Tensor\?$", r"const c10::optional<at::Tensor>&"),
            (r"^Tensor\[\]", r"at::ArrayRef<at::Tensor>"),
            (r"^Tensor\([a-z]!\)\[\]", r"at::ArrayRef<at::Tensor>"),
            (r"^Tensor\([a-z]!\)", r"at::Tensor&"),
            (r"^Tensor$", r"const at::Tensor&"),
            (r"^SymInt\[\d*\]\?", r"at::OptionalSymIntArrayRef"),
            (r"^SymInt\[\d*\]", r"c10::SymIntArrayRef"),
            (r"^SymInt", r"c10::SymInt"),
            (r"^ScalarType", r"at::ScalarType"),
            (r"^Scalar\?", r"const c10::optional<at::Scalar>&"),
            (r"^Scalar\[\]", r"at::ArrayRef<at::Scalar>"),
            (r"^Scalar", r"const at::Scalar&"),
            (r"^Layout", r"at::Layout"),
            (r"^Generator", r"at::Generator"),
            (r"^Device", r"c10::Device"),
            (r"^str", r"c10::string_view"),
            (r"^int\[\d*\]\?", r"at::OptionalIntArrayRef"),
            (r"^int\[\d*\]", r"at::IntArrayRef"),
            (r"^int", r"int64_t"),
            (r"^float", r"double"),
            (r"^bool\[(\d+)\]", r"::std::array<bool,\1>"),
            # post-process
            (r"([\w:]+)\?", r"c10::optional<\1>"),
            (r"\(\w+!\)", r"&"),
        ]
    ]

    output = []
    _, parameters, _ = parse_function_signature(schema)
    for type, name, _ in parameters:
        for pattern, replacement in MAPPING:
            type = pattern.sub(replacement, type)
        if type != "":  # Drop '*' type
            output.append((type + " " + name).strip())
    return output


def generate_cpp_function_name(
    name: str, multiple_return_values: bool, fallback: bool
) -> str:
    MAPPING = [  # Order matters
        (r"_?\.from", ""),
        (r"_?\.to", ""),
        (r"_mode", ""),
        (r"(_foreach_\w+)\.List", r"\1"),
        (r"\.(Scalar)?(Tensor)?\w*_out", "_outf"),
        (r"\.correction", ""),
        (r"\.dim_IntList", ""),
        (r"\.dim_max", "_outf"),
        (r"\.dim_min", "_outf"),
        (r"\.dim", ""),
        (r"\.grad_input", "_outf"),
        (r"\.input", ""),
        (r"\.out\w*", "_outf"),
        (r"\.ScalarList", ""),
        (r"\.Scalar", ""),
        (r"\.self", ""),
        (r"\.Tensor_Scalar_out", "_outf"),
        (r"\.Tensor_Scalar", ""),
        (r"\.Tensor_Tensor", ""),
        (r"\.Tensor", ""),
        (r"\.values_stable", "_outf"),
        (r"\.values", "_outf"),
        (r"ctc_loss\.IntList", "ctc_loss"),
        (r"\.", "_"),
    ]

    if fallback:
        return "custom_fallback_" + create_fun_name_from_schema(name)

    function_name = get_op_name_from_schema(name)
    for pattern, replacement in MAPPING:
        function_name = re.sub(pattern, replacement, function_name)
    if function_name.endswith("_") and multiple_return_values > 0:
        function_name = function_name.removesuffix("_")
    return "at::" + function_name


def generate_cpp_function_return_values(
    parameters: list[tuple[str, str, str]], return_values: list[tuple[str, str]]
) -> list[str]:
    REFERENCE_PATTERN = re.compile(r"\w+(\([a-z]+!\))")

    output = []
    for i, (type, name) in enumerate(return_values):
        if reference := REFERENCE_PATTERN.match(type):
            label = reference.group(1)
            if found := next(
                (name for type, name, _ in parameters if label in type), None
            ):
                output.append(found)

        elif re.match(r"\w+", type):
            if name == "":
                name = "out" + ("" if len(return_values) == 1 else str(i))
            output.append(name)

        elif "bool" == type:
            output.append("out")

    return output


def generate_cpp_variable_for_return_values(
    return_values: list[tuple[str, str]], *, suffix: str = "_rv"
) -> str:
    output = [name + suffix for _, name in return_values]
    if len(output) == 0:
        return "result" + suffix
    if len(output) == 1:
        return output[0]
    return "[" + ", ".join(output) + "]"


def generate_function_call_with_cpu_tensors(
    function_name: str,
    parameters: list[tuple[str, str, str]],
    return_values_count: int,
    use_custom_fallback: bool,
    *,
    output_variable_name: str = "result_cpu",
    symint_array_suffix: str = "Vector",
) -> str:
    SYMINT_ARRAY_PATTERN = re.compile(r"SymInt\[\d?\]")

    function_name = generate_cpp_function_name(
        function_name, return_values_count > 0, use_custom_fallback
    )

    bullets = list(filter(lambda x: x != "", (name for _, name, _ in parameters)))

    symint_parameters = [name for type, name, _ in parameters if "SymInt" == type]
    symint_array_parameters = [
        name for type, name, _ in parameters if SYMINT_ARRAY_PATTERN.match(type)
    ]
    tensor_parameters = [name for type, name, _ in parameters if "Tensor" in type]
    for index, bullet in enumerate(bullets):
        if bullet in symint_parameters:
            bullets[index] += ".expect_int()"
        elif bullet in symint_array_parameters:
            bullets[index] += symint_array_suffix
        elif bullet in tensor_parameters:
            bullets[index] += "_cpu"
        elif bullet == "device":
            bullets[index] = "at::kCPU"

    output = ""
    if len(symint_array_parameters) != 0:
        output += generate_code_to_process_symint_array(symint_array_parameters) + "\n"
    if return_values_count != 0:
        output += f"auto {output_variable_name} = "
    output += function_name + "(" + ", ".join(bullets) + ");\n"
    return output


def generate_code_to_clone_to_host_tensors(
    function_name: str, parameters: list[tuple[str, str, str]]
) -> tuple[str, list[tuple[int, str]], list[tuple[int, str]]]:
    TENSOR_ARRAY_PATTERN = re.compile(r"Tensor\??\[\]")
    MUTABLE_TENSOR_PATTERN = re.compile(r"Tensor\([a-z]!\)")
    MUTABLE_TENSOR_ARRAY_PATTERN = re.compile(r"Tensor\([a-z]!\)\[\]")

    output = []
    mutable_tensor = []
    mutable_tensor_array = []

    maybe_inplace = [] if ".out" in function_name or "_out" in function_name else None
    for index, (type, name, _) in enumerate(parameters):
        if type == "Tensor":
            output.append(f"auto {name}_cpu = tensor_clone_to_host({name});")
            if maybe_inplace is not None:
                maybe_inplace.append((name, name + "_cpu"))

        elif (
            type == "Tensor?"
            or "ITensorListRef" in type
            or "TensorList" in type
            or TENSOR_ARRAY_PATTERN.match(type)
        ):
            output.append(f"auto {name}_cpu = tensor_clone_to_host({name});")

        elif MUTABLE_TENSOR_ARRAY_PATTERN.match(type):
            mutable_tensor_array.append((index, name + "_cpu"))
            output.append(f"auto {name}_cpu = tensor_clone_to_host({name});")

        elif MUTABLE_TENSOR_PATTERN.match(type):
            mutable_tensor.append((index, name + "_cpu"))
            if maybe_inplace:
                candidates = ", ".join(f"{{{a}, {b}}}" for a, b in maybe_inplace)
                output.append(
                    f"auto {name}_cpu = tensor_reference_or_clone_to_host({name}, {{{candidates}}});"
                )
            else:
                output.append(f"auto {name}_cpu = tensor_clone_to_host({name});")

    return "\n".join(output) + "\n", mutable_tensor, mutable_tensor_array


def generate_code_to_copy_tensor_from_cpu_to_device(
    parameters: list[tuple[str, str, str]],
    return_values: list[tuple[str, str]],
    mutable_tensor: list[tuple[int, str]],
    mutable_tensor_array: list[tuple[int, str]],
) -> str:
    TENSOR_PATTERN = re.compile(r"Tensor(\([a-z]!\))?")
    TENSOR_ARRAY_PATTERN = re.compile(r"Tensor\[\]")

    HEAD = "auto current_stream = dipu::getCurrentDIPUStream();\n"
    COPY_TENSOR = "tensor_copy_host_to_device({0}, {1}, current_stream);\n"
    COPY_TENSOR_ARRAY = """\
decltype(auto) {0}_vec = tensor_array_to_vector({0});
for (auto i = std::size_t{{}}; i < {0}.size(); ++i) {{
  tensor_copy_host_to_device({0}_vec[i], {1}[i], current_stream);
}}
"""

    output = HEAD
    for type, name in return_values:
        if TENSOR_ARRAY_PATTERN.match(type):
            output += COPY_TENSOR_ARRAY.format(name, name + "_rv")
        elif TENSOR_PATTERN.match(type):
            output += COPY_TENSOR.format(name, name + "_rv")
        else:
            output += f"{name} = {name}_rv;"

    for index, cpu_name in mutable_tensor:
        type, name, _ = parameters[index]
        if match := TENSOR_PATTERN.match(type):
            if tag := match.group(1):
                if any(True for type, _ in return_values if tag in type):
                    continue
        output += COPY_TENSOR.format(name, cpu_name)

    for index, cpu_name in mutable_tensor_array:
        _, name, _ = parameters[index]
        output += COPY_TENSOR_ARRAY.format(name, cpu_name)

    return output


def generate_code_fallback_to_cpu(config: dict) -> str:
    maybe_fallback = (
        config.get("enable_fallback_cpu", True)
        and config.get("enable_autocompare", True)
        and config.get("register_operator", True)
    )
    if not maybe_fallback:
        return ""

    schema = config["schema"]
    use_custom_fallback = config.get("custom_fallback", False)
    function_name, parameters, return_values = parse_function_signature(schema)
    return_value_names = generate_cpp_function_return_values(parameters, return_values)
    return_values = [(a, c) for (a, _), c in zip(return_values, return_value_names)]

    copy_to_cpu_code, mutable_tensor, mutable_tensor_array = (
        generate_code_to_clone_to_host_tensors(function_name, parameters)
    )

    function_call_code = generate_function_call_with_cpu_tensors(
        function_name,
        parameters,
        len(return_values),
        use_custom_fallback,
        output_variable_name=generate_cpp_variable_for_return_values(return_values),
    )

    copy_to_device_code = generate_code_to_copy_tensor_from_cpu_to_device(
        parameters, return_values, mutable_tensor, mutable_tensor_array
    )

    code = copy_to_cpu_code + "\n" + function_call_code + "\n" + copy_to_device_code
    return f"""if (ret == ::diopiForceFallbackToCPU) {{
{textwrap.indent(code, '  ')}
  ret = ::diopiSuccess;
}}"""


def get_fun_name_from_cppsignature(cppnature):
    return re.search(r"[a-zA-Z_:]+[\w\d:]+\(", cppnature).group().replace("(", "")


def get_op_name_from_schema(schema: str) -> str:
    name, _, _ = parse_function_signature(schema)
    name = name.strip().removeprefix("aten::")
    return name


def create_fun_name_from_schema(schema: str) -> str:
    name = get_op_name_from_schema(schema)
    name = name.replace(".", "_").lower()
    return "dipu_" + name


def create_return_code_frome_schema(schema, allow_return_ref=True):
    if re.search("\( *\)", schema[schema.find("->") :]) is not None:
        return "void "
    schema = re.sub("Tensor\([a-z]\) ", "Tensor ", schema)
    return_code = schema[schema.find("->") :].replace("->", "").strip()
    return_code = re.sub("Tensor *\[ *\] *", "std::vector<Tensor> ", return_code)
    return_code = re.sub("\([a-zA-Z]!\)", "&", return_code)
    return_code = re.sub("\([a-zA-Z]\)", "", return_code)
    return_code = re.sub("Tensor", "at::Tensor", return_code)
    return_code = re.sub("([\w_\d:&]+)[ ]+([\w\d_]+)?", R"\1", return_code)
    return_code = re.sub("\(", "std::tuple<", return_code)
    return_code = re.sub("\)", "> ", return_code)
    if allow_return_ref == False:
        return_code = return_code.replace("&", "")
    return return_code


def create_transform_input_to_cpu_code(config: dict) -> str:
    function_name, parameters, _ = parse_function_signature(config["schema"])
    code, _, _ = generate_code_to_clone_to_host_tensors(function_name, parameters)
    return code


def create_print_op_args_code(fun_config):
    args_name_list = create_args_name_list_from_schema(fun_config["schema"])
    opname = get_op_name_from_schema(fun_config["schema"])
    inputs = args_name_list.split(",") + get_function_need_alloc_args_from_schema(
        fun_config["schema"]
    )
    code = ""
    if len(inputs) < 0:
        return code
    code += "if (dumpOpArgLevel() > 1) {\n"
    for input in inputs:
        input = input.strip()
        code += f'  std::cout << "\t{opname}:\t{input}:" << dumpArg({input}) << std::endl;\n'
    code += "}"
    return code


def create_param_list_from_schema(schema: str) -> str:
    return ", ".join(generate_parameters_from_schema(schema))


def get_function_inputs_from_schema(schema):
    ins = []
    for args in generate_parameters_from_schema(schema):
        args = args.strip()
        tensor_match_result = re.search("Tensor[ ]*&+", args)
        if tensor_match_result is not None:
            in_match_result = re.search("const[ ]+[at::]*Tensor[ &]*", args)
            if in_match_result is not None:
                ins.append(args[in_match_result.span()[1] : :].strip())
        opt_tensor_match_result = re.search(
            "const[ ]+c10::optional<at::Tensor>[ &]*([a-zA-Z_0-9]+)", args
        )
        if opt_tensor_match_result is not None:
            opt_tensor = re.sub(
                "const[ ]+c10::optional<at::Tensor>[ &]*([a-zA-Z_]+)", r"\1", args
            ).strip()
            ins.append(opt_tensor + "?")
    return ins


def get_function_need_alloc_args_from_schema(schema):
    outputs = []
    param_list = schema[schema.find("->") + 2 :].strip()
    outputs += re.findall("\(?Tensor[ ]*([\w\d_]+){1}", param_list)

    no_name_args = re.findall(
        "Tensor[ ]*(?!\([a-z]!\))(?![\w\d_ ]+)(?!(\[ *\]))", param_list
    )
    no_name_args_num = len(no_name_args)
    for i in range(no_name_args_num):
        outputs.append("out" + (str(i) if no_name_args_num > 1 else ""))

    return outputs


def get_function_outputs_from_schema(schema):
    outputs = re.findall("Tensor\([a-z]!\)[ ]+([\w\d_]+){1}", schema)
    outputs += get_function_need_alloc_args_from_schema(schema)
    outputs = list(set(outputs))
    return outputs


def get_function_scalar_args_from_schema(schema):
    param_list = schema[schema.find("(") + 1 : schema.find("->")].strip()
    param_list = param_list[0 : param_list.rfind(")")]
    scalars = []
    for args in param_list.split(","):
        args = args.strip()
        scalar_match_result = re.search("[ ]?Scalar[ ]+", args)
        opt_scalar_match_result = re.search("Scalar[ ][\?]+", args)
        if scalar_match_result is not None and opt_scalar_match_result is None:
            scalar_param = args[scalar_match_result.span()[1] :].strip()
            scalar_param = re.sub("=.*,{1}", ",", scalar_param)
            scalar_param = re.sub("=.*", "", scalar_param)
            scalars.append(scalar_param.strip())
    return scalars


def get_function_optional_scalar_args_from_schema(schema):
    param_list = schema[schema.find("(") + 1 : schema.find("->")].strip()
    param_list = param_list[0 : param_list.rfind(")")]
    return re.findall("Scalar *\? +([\w\d_]+)", param_list)


def get_function_optional_generator_args_from_schema(schema):
    param_list = schema[schema.find("(") + 1 : schema.find("->")].strip()
    param_list = param_list[0 : param_list.rfind(")")]
    return re.findall("Generator *\? +([\w\d_]+)", param_list)


def get_function_int_array_args_from_schema(schema):
    int_arrays = []
    for args in generate_parameters_from_schema(schema):
        args = args.strip()
        match_result = re.search("[^Optional]SymIntArray[\w\d]*", args)
        if match_result is not None:
            int_array_param = args[match_result.span()[1] :].strip()
            int_array_param = re.sub("=.*,{1}", ",", int_array_param)
            int_array_param = re.sub("=.*", "", int_array_param)
            int_arrays.append(int_array_param.strip())
    return int_arrays


def get_function_return_param_from_schema(schema: str) -> list[str]:
    _, parameters, return_values = parse_function_signature(schema)
    return generate_cpp_function_return_values(parameters, return_values)


def create_call_diop_interface_code_from_schema(schema):
    schema = schema.replace("aten::", "").strip()
    schema = schema.replace("_.", "Inp")
    schema = schema.replace(".", "")

    outs = re.findall(",? *Tensor *\(\w+!\) *\w+", schema)[::-1]
    schema = re.sub(",? *Tensor *\(\w+!\) *\w+", "", schema)
    index = schema.find("(") + 1
    for args in outs:
        schema = schema[0:index] + args.replace(",", "") + ", " + schema[index:]

    schema = schema.replace("(", "(ctx, ", 1)
    return_index = schema.find("->")

    if return_index > 0:
        return_args = schema[return_index + 2 :].strip()
        if re.search("Tensor[ ]*\([\w]+!\)", return_args) is None:
            return_args = re.sub("Tensor[ ]*\([\w]+!\)[ ]*", "", return_args)
            return_args = re.sub("[\(\)]", "", return_args).strip()
            outs = return_args.split(",")
            retucn_code = ""
            for i in range(len(outs)):
                retucn_code += "out"
                if len(outs) > 1:
                    retucn_code += str(i)
                if i < len(outs) - 1:
                    retucn_code += ", "
            schema = re.sub("\([ ]*ctx", "(ctx, " + retucn_code, schema)
    schema = schema[0 : schema.find("->")]

    for key in [
        "Tensor[ ]*\([\w!]+\)",
        "Tensor[ ]*\?",
        "Tensor[ ]*",
        "bool",
        "float",
        "str[ ]*\?",
        "[,]? *\* *",
        "=[\w]+",
    ]:
        index = schema.find("(")
        schema = schema[0:index] + re.sub(key, "", schema[index:])

    index = schema.find("(")
    schema = schema[0:index] + re.sub("Scalar[ ]*", "&", schema[index:])

    for key in ["out", "_mode", "Tensor", "_", "[nN]{1}ative_"]:
        index = schema.find("(")
        schema = re.sub(key, "", schema[:index]) + schema[index:]

    schema = "diopi" + schema[0].upper() + schema[1:]
    schema = re.sub(" *, *", ", ", schema)
    schema = re.sub(" *, *,", ", ", schema)

    return schema


def create_cpp_signature_from_schema(schema):
    return_code = create_return_code_frome_schema(schema)
    fun_name = create_fun_name_from_schema(schema)
    param_list = create_param_list_from_schema(schema)
    cppsignature_template = CodeTemplate("$return_code $fun_name($param_list)")
    cppsignature = cppsignature_template.substitute(
        return_code=[return_code], fun_name=[fun_name], param_list=[param_list]
    )
    return cppsignature


def create_args_name_list_from_schema(schema: str) -> str:
    _, output, _ = parse_function_signature(schema)
    output = filter(lambda x: x != "", (name for _, name, _ in output))
    return ", ".join(output)


def create_call_cpp_function_code_from_schema(schema):
    code = (
        create_fun_name_from_schema(schema)
        + "("
        + create_args_name_list_from_schema(schema)
        + ");"
    )
    return code


def create_call_aten_cpu_cpp_function_code_from_config(config: dict) -> str:
    schema = config["schema"]
    function_name, parameters, return_values = parse_function_signature(schema)
    use_custom_fallback = config.get("custom_fallback", False)
    return generate_function_call_with_cpu_tensors(
        function_name, parameters, len(return_values), use_custom_fallback
    )


def create_call_dipu_cpp_function_code_from_schema(schema):
    code = create_return_code_frome_schema(schema)
    if len(get_function_return_param_from_schema(schema)) > 0:
        code += " result_device = "
    else:
        code = code.replace("void ", "")
    code += create_call_cpp_function_code_from_schema(schema).replace("; ", ";\n")
    return code.replace("; ", ";\n")


def create_result_compare_code(fun_config):
    schema = fun_config["schema"]
    op_name = get_op_name_from_schema(schema)
    return_names = get_function_return_param_from_schema(schema)
    code = ""
    separator_code = f'std::cout << "--------------------" << std::endl;\n'

    if len(return_names) == 1:
        code += separator_code
        code += f'std::cout << "autocompare:\t{op_name}\t{return_names[0]}:" << std::endl << allclose_autocompare(result_cpu, result_device) << std::endl;\n'
    elif len(return_names) > 1:
        for i in range(len(return_names)):
            code += separator_code
            code += f'std::cout << "autocompare:\t{op_name}\t{return_names[i]}:" << std::endl << allclose_autocompare(std::get<{i}>(result_cpu), std::get<{i}>(result_device)) << std::endl;\n'

    inputs = re.findall("Tensor +([\w\d_]+)", schema[: schema.find("->")])
    inputs += re.findall(
        "Tensor *\([a-z]!\) *\[ *\] +([\w\d_]+)", schema[: schema.find("->")]
    )
    for i in range(len(inputs)):
        code += separator_code
        code += f'std::cout << "autocompare:\t{op_name}\t{inputs[i]}: " << std::endl << allclose_autocompare({inputs[i]}_cpu, {inputs[i]}) << std::endl;\n'
    return code


def create_code_to_print_fun_call_info_from_schema(fun_config):
    op_name = get_op_name_from_schema(fun_config["schema"])
    diopi_func = fun_config.get("interface", "")
    diopi_func = diopi_func[0 : diopi_func.find("(")]
    debug_code = "if (dumpOpArgLevel() > 0) {\n"
    debug_code += (
        f'  printf("--%-50s %-30s \\n", "[{op_name}]:", "{diopi_func}");' + "\n"
    )
    debug_code += "}\n"
    return debug_code


def create_autograd_function_name(op_name):
    op_name = "Dipu" + op_name[0].upper() + op_name[1:]
    for patten in re.findall("[_\.][a-z]{1}", op_name):
        op_name = op_name.replace(patten, patten[1].upper())
    op_name = op_name.replace("_", "Inp")
    op_name = op_name.replace(".", "")
    return op_name + "Function"


def create_save_for_backward_code(args_name_list):
    code = ""
    for arg_name in args_name_list:
        code += f'ctx->saved_data["{arg_name}"] = {arg_name};\n'
    return code


def create_get_saved_data_code(args_name_list):
    code = ""
    for arg_name in args_name_list:
        code += f'auto {arg_name}_ = ctx->saved_data["{arg_name}"];\n'
    return code


def create_optional_scalar_process_code(arg_name):
    process_template = CodeTemplate(
        """
::diopiScalar_t ${arg_name}DiopiScalar;
const ::diopiScalar_t* ${arg_name}DiopiScalarPtr = nullptr;
if ($arg_name.has_value()) {
    ${arg_name}DiopiScalar = dipu::diopi_helper::toDiopiScalar(${arg_name}.value());
    ${arg_name}DiopiScalarPtr = &${arg_name}DiopiScalar;
}
"""
    )
    process_code = process_template.substitute(
        arg_name=[arg_name],
    )
    return process_code


def create_device_check_code(fun_config):
    code = ""
    tensors = get_function_inputs_from_schema(fun_config["schema"]) + fun_config.get(
        "ins", []
    )
    tensors += get_function_outputs_from_schema(fun_config["schema"]) + fun_config.get(
        "outs", []
    )
    tensors = set(tensors)
    exclude_tensors = fun_config.get("no_device_check_args", [])
    for args in exclude_tensors:
        tensors.discard(args)
    op_name = get_op_name_from_schema(fun_config["schema"])
    if len(tensors) > 0:
        code += "if (checkTensorDevice()) {\n"

    for args in set(tensors):
        if not args.endswith("?"):
            code += f'  TORCH_CHECK(({args}.defined() == false) || ({args}.device().type() == dipu::DIPU_DEVICE_TYPE || ignore_device_check({args})), __FILE__, ":", __LINE__, ": {op_name}: {args} should be on dipu");\n'
        else:
            args = args[0:-1]
            code += f'  TORCH_CHECK(({args}.has_value() == false) || ({args}.value().defined() == false) || ({args}.value().device().type() == dipu::DIPU_DEVICE_TYPE || ignore_device_check({args})), __FILE__, ":", __LINE__, "{op_name}: {args} should be on dipu");\n'

    if len(tensors) > 0:
        code += "}"

    return code


def create_device_guard_code(fun_config):
    code = ""
    if fun_config.get("generate_device_guard", True) in ["False", False]:
        return code

    tensors = re.findall("Tensor\(\w!\) +[\w\d_]+", fun_config["schema"]) + re.findall(
        "Tensor +[\w\d_]+", fun_config["schema"]
    )
    arg = fun_config.get("device_guard_arg", None)
    if len(tensors) > 0 or arg is not None:
        if arg is not None:
            tensor = arg
        else:
            tensor = tensors[0].split(" ")[1]
        code += f"c10::OptionalDeviceGuard guard(at::device_of({tensor}));"
    else:
        try:
            device_args = re.findall("Device. [\w\d_]+", fun_config["schema"])[0].split(
                " "
            )
            if device_args[0].endswith("?"):
                code += f"c10::OptionalDeviceGuard guard({device_args[1]});"
            else:
                code += (
                    f"c10::OptionalDeviceGuard guard(at::device_of({device_args[1]}));"
                )
        except:
            pass

    return code


def create_optional_generator_process_code(arg_name):
    process_template = CodeTemplate(
        """
::diopiGeneratorHandle_t ${arg_name}DiopiGenerator = (${arg_name}.has_value() && ${arg_name}.value().defined()) ? toDiopiGeneratorHandle(${arg_name}) : toDiopiGeneratorHandle(getDefaultDIPUGenerator());
"""
    )
    process_code = process_template.substitute(
        arg_name=[arg_name],
    )
    return process_code


file_template = CodeTemplate(diopi_wrapper_file_template_content)

fun_template = CodeTemplate(diopi_wrapper_function_template_content)

op_no_customfallback_with_autocompare_register_template = CodeTemplate(
    op_no_customfallback_with_autocompare_register_template_content
)

op_no_customfallback_no_autocompare_register_template = CodeTemplate(
    op_no_customfallback_no_autocompare_register_template_content
)

op_with_customfallback_with_autocompare_register_template = CodeTemplate(
    op_with_customfallback_with_autocompare_register_template_content
)

op_with_customfallback_no_autocompare_register_template = CodeTemplate(
    op_with_customfallback_no_autocompare_register_template_content
)

custom_autograd_template = CodeTemplate(custom_autograd_template_content)

autocompare_template = CodeTemplate(autocompare_template_content)


def functions_code_gen(fun_config):
    if "interface" in fun_config:
        diopi_fun_call_code = fun_config["interface"] + ";"
    else:
        diopi_interface = create_call_diop_interface_code_from_schema(
            fun_config["schema"]
        )
        diopi_fun_call_code = diopi_interface + ";"

    input_process_code = ""
    diopi_tensor_suffix = "DiopiTensorHandle"

    for input in set(
        get_function_inputs_from_schema(fun_config["schema"])
        + fun_config.get("ins", [])
    ):
        if input.strip().endswith("?"):
            input = input.replace("?", "")
            input_process_code += f"\n::diopiConstTensorHandle_t {input}{diopi_tensor_suffix} = nullptr;\n"
            input_process_code += (
                f"if ({input}.has_value() && {input}.value().defined())" + "{\n"
            )
            input_process_code += f"  {input}{diopi_tensor_suffix} = dipu::diopi_helper::toDiopiTensorHandle({input}.value());\n"
            input_process_code += "}\n"
        else:
            input_process_code += f"::diopiConstTensorHandle_t {input}{diopi_tensor_suffix} = dipu::diopi_helper::toDiopiTensorHandle({input});\n"

        diopi_fun_call_code = re.sub(
            input.strip() + "([,\) ]{1})",
            f"{input.strip()}{diopi_tensor_suffix}" + r"\1",
            diopi_fun_call_code,
        )

    diopi_size_suffix = "DiopiSize"
    for size_attr in fun_config.get("size_attr", []):
        input_process_code += f"::diopiSize_t {size_attr}DiopiSize = dipu::diopi_helper::toDiopiSize({size_attr});\n"
        diopi_fun_call_code = re.sub(
            size_attr.strip() + "([,\) ]{1})",
            f"{size_attr.strip()}{diopi_size_suffix}" + r"\1",
            diopi_fun_call_code,
        )

    output_process_code = ""
    for output in set(
        get_function_outputs_from_schema(fun_config["schema"])
        + fun_config.get("outs", [])
    ):
        output_process_code += f"::diopiTensorHandle_t {output}{diopi_tensor_suffix} = dipu::diopi_helper::toDiopiTensorHandle({output});\n"
        diopi_fun_call_code = re.sub(
            "([\(,& ]{1})" + output.strip() + "([,\) ]{1})",
            r"\1" + f"{output.strip()}{diopi_tensor_suffix}" + r"\2",
            diopi_fun_call_code,
        )

    attrs_process_code = ""

    diopi_scalar_suffix = "DiopiScalar"
    for scalar_param in get_function_scalar_args_from_schema(fun_config["schema"]):
        attrs_process_code += f"::diopiScalar_t {scalar_param}{diopi_scalar_suffix} = dipu::diopi_helper::toDiopiScalar({scalar_param});\n"
        diopi_fun_call_code = re.sub(
            "([,\(]) *&? *" + scalar_param + "([,\)])",
            R"\1" + f"&{scalar_param}{diopi_scalar_suffix}" + R"\2",
            diopi_fun_call_code,
        )

    for scalar_param in get_function_optional_scalar_args_from_schema(
        fun_config["schema"]
    ):
        attrs_process_code += create_optional_scalar_process_code(scalar_param)
        diopi_fun_call_code = re.sub(
            "([,\(] *&? *)" + scalar_param.strip() + "( *[,\)])",
            R"\1" + f"{scalar_param}DiopiScalarPtr" + R"\2",
            diopi_fun_call_code,
        )

    for generator_param in get_function_optional_generator_args_from_schema(
        fun_config["schema"]
    ):
        attrs_process_code += create_optional_generator_process_code(generator_param)
        diopi_fun_call_code = re.sub(
            "([,\(] *&? *)" + generator_param.strip() + "( *[,\)])",
            R"\1" + f"{generator_param}DiopiGenerator" + R"\2",
            diopi_fun_call_code,
        )

    int_array_list = get_function_int_array_args_from_schema(fun_config["schema"])
    attrs_process_code += generate_code_to_process_symint_array(int_array_list)
    for int_array_param in int_array_list:
        diopi_fun_call_code = re.sub(
            "([,\(] *&? *)" + int_array_param.strip() + "( *[,\)])",
            R"\1" + f"{int_array_param}DiopiSize" + R"\2",
            diopi_fun_call_code,
        )

    if fun_config.get("print_func_call_info", False) == True:
        fun_config["custom_code_at_the_beginning"] = (
            create_code_to_print_fun_call_info_from_schema(fun_config)
            + fun_config.get("custom_code_at_the_beginning", "")
        )

    if fun_config.get("print_op_args", False) == True:
        fun_config["custom_code_before_call_diopi"] = fun_config.get(
            "custom_code_before_call_diopi", ""
        ) + create_print_op_args_code(fun_config)

    if fun_config.get("use_diopi_adapter", False) == True:
        diopi_fun_call_code = "diopiadaptor::" + diopi_fun_call_code
    else:
        diopi_fun_call_code = "::" + diopi_fun_call_code

    if fun_config.get("dummy_call_diopi", False) in [True, "True"]:
        diopi_fun_call_code = (
            f"::diopiSuccess;/*dummy_call_diopi: {diopi_fun_call_code}*/"
        )

    return_code = ""
    return_param = get_function_return_param_from_schema(fun_config["schema"])
    if len(return_param) == 0:
        return_code = "return;\n"
    elif len(return_param) == 1:
        return_code = f"return {return_param[0]};\n"
    else:
        params = ""
        for i in range(len(return_param)):
            params += return_param[i]
            if i < len(return_param) - 1:
                params += ", "
        return_code = f"return std::tie({params});"

    custom_code_at_the_beginning = fun_config.get(
        "custom_code_at_the_beginning", fun_config.get("custom_code", "")
    )
    # strip all whitespace and divide code to different lines.
    custom_code_at_the_beginning = re.sub(";\s*$", ";\n", custom_code_at_the_beginning)

    interface_name = re.sub(R".*::(.*?)\(.*", R"\1", diopi_fun_call_code)
    fbody = fun_template.substitute(
        comment=[fun_config["schema"]],
        cppsignautre=[create_cpp_signature_from_schema(fun_config["schema"])],
        custom_code_at_the_beginning=[custom_code_at_the_beginning],
        device_guard_code=[create_device_guard_code(fun_config)],
        input_process_code=[input_process_code],
        attrs_process_code=[attrs_process_code],
        output_process_code=[output_process_code],
        custom_code_before_call_diopi=[
            fun_config.get("custom_code_before_call_diopi", "").replace("; ", ";\n")
        ],
        device_check_code=[create_device_check_code(fun_config)],
        diopi_fun_call_code=[diopi_fun_call_code],
        force_fallback_code=[generate_code_fallback_to_cpu(fun_config)],
        custom_code_before_return=[
            fun_config.get("custom_code_before_return", "").replace("; ", ";\n")
        ],
        return_code=[return_code],
        interface_name=[interface_name],
    )
    diopi_interface = fun_config.get(
        "interface", create_call_diop_interface_code_from_schema(fun_config["schema"])
    )

    fun_name = create_fun_name_from_schema(fun_config["schema"])
    raw_fun_name = fun_name

    if fun_config.get("autograd", False) == True:
        wrapper_fun_name = fun_name + "_wrapper"
        custom_autograd_function_code = custom_autograd_template.substitute(
            autograd_function_name=[
                create_autograd_function_name(
                    get_op_name_from_schema(fun_config["schema"])
                )
            ],
            cppsignautre=[
                create_cpp_signature_from_schema(fun_config["schema"]).replace(
                    fun_name, wrapper_fun_name
                )
            ],
            return_code=[
                create_return_code_frome_schema(
                    fun_config["schema"], allow_return_ref=False
                )
            ],
            save_for_backward_code=[
                create_save_for_backward_code(fun_config.get("saved_data", []))
            ],
            param_list=[create_param_list_from_schema(fun_config["schema"])],
            arg_name_list=[create_args_name_list_from_schema(fun_config["schema"])],
            call_forward_impl_code=[
                create_call_cpp_function_code_from_schema(
                    fun_config.get("forward_schema", fun_config["schema"])
                ).replace("; ", ";\n")
            ],
            forward_process_code=[
                fun_config.get("forward_process_code", "").replace("; ", ";\n")
            ],
            load_saved_data_code=[
                create_get_saved_data_code(fun_config.get("saved_data", []))
            ],
            cal_grad_code=[
                fun_config.get("cal_grad_code", "").replace("; ", ";\n")
                + "/*"
                + fun_config.get("backward_schema", "")
                + "*/"
            ],
            call_backward_impl_code=[
                (
                    (
                        "auto result = "
                        + create_call_cpp_function_code_from_schema(
                            fun_config["backward_schema"]
                        ).replace("; ", ";\n")
                    )
                    if "backward_schema" in fun_config
                    else ""
                )
            ],
            backward_return_code=[
                fun_config.get("backward_return_code", "").replace("; ", ";\n")
            ],
            wrappter_custom_return=[
                fun_config.get("wrappter_custom_return", "return result;")
            ],
        )
        fbody += custom_autograd_function_code
        fun_name = wrapper_fun_name

    if fun_config.get("enable_autocompare", True) and fun_config.get(
        "register_operator", True
    ):
        auto_compare_fun_name = fun_name + "_autocompare"
        autocompare_code = autocompare_template.substitute(
            cppsignautre=[
                create_cpp_signature_from_schema(fun_config["schema"]).replace(
                    raw_fun_name, auto_compare_fun_name
                )
            ],
            transform_input_to_cpu_code=[
                create_transform_input_to_cpu_code(fun_config)
            ],
            execute_op_on_cpu_code=[
                create_call_aten_cpu_cpp_function_code_from_config(fun_config)
            ],
            comment=[fun_config["schema"]],
            execute_op_on_device_code=[
                create_call_dipu_cpp_function_code_from_schema(
                    fun_config["schema"]
                ).replace(raw_fun_name, fun_name)
            ],
            transform_result_to_cpu_code=[],
            result_compare_code=[
                create_result_compare_code(fun_config)
                + (
                    "\nreturn result_device;\n"
                    if len(get_function_return_param_from_schema(fun_config["schema"]))
                    > 0
                    else ""
                )
            ],
        )
        fbody += autocompare_code

    # generate the OP_register code
    # case 1: custom_fallback=False and autocompare not disabled
    register_body = ""
    if fun_config.get("custom_fallback", False) in ["False", False] and fun_config.get(
        "enable_autocompare", True
    ):
        register_body = (
            op_no_customfallback_with_autocompare_register_template.substitute(
                register_name=[get_op_name_from_schema(fun_config["schema"])],
                aten_fun_name=["dipu::native::" + fun_name],
                diopi_fun_name=[
                    get_fun_name_from_cppsignature(diopi_interface).replace(
                        "diopi", "::diopi"
                    )
                ],
            )
        )

    # case2: custom_fallback=False and autocompare=disabled
    elif fun_config.get("custom_fallback", False) in [
        "False",
        False,
    ] and not fun_config.get("enable_autocompare", True):
        register_body = (
            op_no_customfallback_no_autocompare_register_template.substitute(
                register_name=[get_op_name_from_schema(fun_config["schema"])],
                aten_fun_name=["dipu::native::" + fun_name],
                diopi_fun_name=[
                    get_fun_name_from_cppsignature(diopi_interface).replace(
                        "diopi", "::diopi"
                    )
                ],
            )
        )
    # case3: custom_fallback=True and autocompare not disabled
    elif fun_config.get("custom_fallback", False) in ["True", True] and fun_config.get(
        "enable_autocompare", True
    ):
        register_body = (
            op_with_customfallback_with_autocompare_register_template.substitute(
                register_name=[get_op_name_from_schema(fun_config["schema"])],
                aten_fun_name=["dipu::native::" + fun_name],
                diopi_fun_name=[
                    get_fun_name_from_cppsignature(diopi_interface).replace(
                        "diopi", "::diopi"
                    )
                ],
                force_fallback=[
                    (
                        "false"
                        if fun_config.get("force_fallback", False) in [False, "False"]
                        else "true"
                    )
                ],
                fallbackFunc=["dipu::native::" + "custom_fallback_" + fun_name],
            )
        )
    # case4: custom_fallback=True and autocompare disabled
    elif fun_config.get("custom_fallback", False) in [
        "True",
        True,
    ] and not fun_config.get("enable_autocompare", True):
        register_body = (
            op_with_customfallback_no_autocompare_register_template.substitute(
                register_name=[get_op_name_from_schema(fun_config["schema"])],
                aten_fun_name=["dipu::native::" + fun_name],
                diopi_fun_name=[
                    get_fun_name_from_cppsignature(diopi_interface).replace(
                        "diopi", "::diopi"
                    )
                ],
                force_fallback=[
                    (
                        "false"
                        if fun_config.get("force_fallback", False) in [False, "False"]
                        else "true"
                    )
                ],
                fallbackFunc=["dipu::native::" + "custom_fallback_" + fun_name],
            )
        )

    return fbody, register_body


def boolean_string(s):
    if s.lower() in ["true", "on"]:
        return True
    elif s.lower() in ["false", "off"]:
        return False
    else:
        raise ValueError("Not a valid boolean string.")


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="autogen diopi wrapper code")
    parser.add_argument(
        "--config",
        type=str,
        default="diopi_functions.yaml",
        help="path to functions config file",
    )
    parser.add_argument(
        "--convert_config",
        type=str,
        dest="convert_config",
        default="",
        help="path to the convert_config.yaml",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="AutoGenedKernels.cpp",
        help="path to functions config file",
    )
    parser.add_argument(
        "--dummy_call_diopi",
        default=False,
        type=boolean_string,
        help="whether acctually call diopi interface",
    )
    parser.add_argument(
        "--use_diopi_adapter",
        default=True,
        type=boolean_string,
        help="whether use diopi adapter",
    )
    parser.add_argument(
        "--generate_device_guard",
        default=True,
        type=boolean_string,
        help="whether generate device guard code",
    )
    parser.add_argument(
        "--diopi_adapter_header",
        type=str,
        default="diopi_adapters.hpp",
        help="path to diopi adapter file",
    )
    parser.add_argument(
        "--print_func_call_info",
        default=False,
        type=boolean_string,
        help="whether generate code that prints function call information",
    )
    parser.add_argument(
        "--print_op_args",
        default=False,
        type=boolean_string,
        help="whether generate code that prints op args",
    )
    parser.add_argument(
        "--fun_config_dict",
        type=json.loads,
        default=dict(),
        help="fun config for all ops",
    )  # --fun_config_dict '{"register_operator": "false", "dummy_call_diopi":"True"}'

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    with open(args.config) as diopi_functions_file:
        file_data = diopi_functions_file.read()
        funcs_config = yaml.load(file_data, Loader=yaml.FullLoader)

    from op_memory_format_converter import OpMemoryFormatConverter

    memory_format_converter = OpMemoryFormatConverter(args.convert_config)

    functions_code = ""
    op_register_code = ""
    header_include_code = ""

    if args.use_diopi_adapter == True:
        if os.path.exists(args.diopi_adapter_header) == False:
            print(f"{args.diopi_adapter_header} not exists")
            args.use_diopi_adapter = False
        else:
            header_include_code += (
                f'#include "{os.path.abspath(args.diopi_adapter_header)}"'
            )

    autograd_op_register_code = ""

    for fun_config in funcs_config:
        merged_fun_config = dict(args.fun_config_dict)
        merged_fun_config.update(vars(args))
        merged_fun_config.update(fun_config)
        # filter for those device specific op.
        if "device" in merged_fun_config:
            current_device = merged_fun_config.get("current_device", "")
            if current_device not in (
                merged_fun_config["device"]
                + [
                    "all",
                ]
            ):
                create_for_this_device = "all" in merged_fun_config["device"]
                for device in merged_fun_config["device"]:
                    if ("-" + device) == current_device:
                        create_for_this_device = False
                        break
                if create_for_this_device == False:
                    continue
            if ("-" + current_device) in (merged_fun_config["device"]):
                continue

        # filter torch version
        supported_torch_ver_list = merged_fun_config.get("torch_ver", None)
        cur_torch_ver = merged_fun_config.get("current_torch_ver", None)

        if supported_torch_ver_list != None:
            exclude_torch_ver_list = []
            include_torch_ver_list = []
            all_include = False
            for supported_torch_ver in supported_torch_ver_list:
                if supported_torch_ver.startswith("-"):
                    exclude_torch_ver_list.append(supported_torch_ver[1:])
                elif supported_torch_ver == "all":
                    all_include = True
                else:
                    include_torch_ver_list.append(supported_torch_ver)

            if (cur_torch_ver in exclude_torch_ver_list) or (
                all_include == False and (cur_torch_ver not in include_torch_ver_list)
            ):
                continue

        fun_code, register_code = functions_code_gen(merged_fun_config)

        # The class object memory_format_converter will replace the prefered memory format placeholder to the prefered memory format based on the device's convert_config.yaml
        fun_code = memory_format_converter.convert(fun_code, fun_config)

        functions_code += fun_code
        if merged_fun_config.get("register_operator", True):
            if merged_fun_config.get("autograd", False) == True:
                autograd_op_register_code += register_code
            op_register_code += register_code

    autogened_file = file_template.substitute(
        functions_code=[functions_code],
        header_include_code=[header_include_code],
        op_register_code=[op_register_code],
        autograd_op_register_code=[autograd_op_register_code],
    )
    autogened_file = re.sub(R"\n{3,}", R"\n\n", autogened_file)
    autogened_file = re.sub("[ ]*,[ ]*", ", ", autogened_file)
    with open(args.out, "w") as cpp_file:
        cpp_file.write(autogened_file)

    print(
        f"Successfully generate {args.out} according to the configuration file {args.config}"
    )


if __name__ == "__main__":
    main()
