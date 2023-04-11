import yaml
import re
from typing import Mapping, Match, Optional, Sequence
from diopi_wrapper_template import diopi_wrapper_file_template_content, diopi_wrapper_function_template_content, op_registe_template_content

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


def get_fun_name_from_cppsignature(cppnature):
    return re.search(r'[a-zA-Z_:]+[\w\d:]+\(' , cppnature).group().replace('(', '')


def get_op_name_from_schema(schema):
    op_name = schema[0:schema.find('(')]
    op_name = re.sub('aten::', '', op_name)
    return op_name

def create_fun_name_from_schema(schema):
    schema = schema.strip()
    op_name = schema[0:schema.find('(')]
    op_name = op_name.replace('.','_')
    op_name = "dipu_" + re.sub('aten::', '', op_name)
    return op_name

def create_return_code_frome_schema(schema):
    return_code = schema[schema.find('->'):].replace('->', '').strip()
    return_code = re.sub('\([a-zA-Z]!\)', '&' , return_code)
    return_code = re.sub('Tensor', 'at::Tensor' , return_code)
    return_code = re.sub('\(', 'std::tuple<', return_code)
    return_code = re.sub('\)', '> ' ,return_code)
    return return_code

def create_param_list_from_schema(schema):
    param_list = schema[schema.find('(') + 1 : schema.find('->')].strip()
    param_list = param_list[0:param_list.rfind(')')]
    param_list = re.sub('[ ]*\([a-zA-Z]!\)', '&' , param_list)
    param_list = re.sub('str\?', 'c10::optional<c10::string_view>' , param_list)
    param_list = re.sub('([a-zA-Z0-9]+)\?', r'c10::optional<\1>&', param_list)
    param_list = re.sub('Tensor ', 'const Tensor& ' , param_list)
    param_list = re.sub('Scalar ', 'const Scalar& ' , param_list)
    param_list = re.sub('Tensor', 'at::Tensor' , param_list)
    param_list = re.sub('Scalar', 'at::Scalar' , param_list)
    param_list = re.sub('\*[ ,]+', '', param_list)
    param_list = re.sub('=.+,', ',', param_list)
    param_list = re.sub('=.+', '', param_list)
    return param_list

def get_function_inputs_outputs_from_schema(schema):
    param_list = create_param_list_from_schema(schema)
    ins = []
    outs = []
    for args in param_list.split(','):
        args = args.strip()
        tensor_match_result = re.search('Tensor[ ]*&+', args)
        if tensor_match_result is not None:
            in_match_result = re.search('const[ ]+[at::]*Tensor[ &]*', args)
            if in_match_result is not None:
                ins.append(args[in_match_result.span()[1]::].strip())
            else:
                outs.append(args[tensor_match_result.span()[1]::].strip())
    return ins, outs

def get_function_scalar_args_from_schema(schema):
    param_list = create_param_list_from_schema(schema)
    scalars = []
    for args in param_list.split(','):
        args = args.strip()
        scalar_match_result = re.search('Scalar[ &]*', args)
        if scalar_match_result is not None:
            scalar_param = args[scalar_match_result.span()[1]:].strip()
            scalar_param = re.sub('=.*,{1}', ',', scalar_param)
            scalar_param = re.sub('=.*', '', scalar_param)
            scalars.append(scalar_param.strip())
    return scalars

def get_function_return_from_schema(schema):
    param_list = schema[schema.find('(') + 1 : schema.find('->')].strip()
    param_list = param_list[0:param_list.rfind(')')]
    param_list = re.sub('\*[ ,]+', '', param_list)
    return_param = []
    for args in param_list.split(','):
        args = args.strip()
        match_result = re.search('Tensor\([a-zA-Z!]+\)', args)
        if match_result is not None:
            return_param.append(args[10:].strip())
    return return_param


def create_cpp_signature_from_schema(schema):
    return_code = create_return_code_frome_schema(schema)
    fun_name = create_fun_name_from_schema(schema)
    param_list = create_param_list_from_schema(schema)
    cppsignature_template = CodeTemplate("$return_code $fun_name($param_list)")
    cppsignature = cppsignature_template.substitute(
        return_code=[return_code],
        fun_name=[fun_name],
        param_list=[param_list]
    )
    return cppsignature


file_template = CodeTemplate(diopi_wrapper_file_template_content)

fun_template = CodeTemplate(diopi_wrapper_function_template_content)

op_registe_template = CodeTemplate(op_registe_template_content)

def functions_code_gen(fun_config):
    diopi_fun_call_code = fun_config['interface'] + ";"
    input_process_code = ""
    for input in get_function_inputs_outputs_from_schema(fun_config['schema'])[0]:
    #for input in fun_config['ins']:
        input_process_code += f"::diopiConstTensorHandle_t {input}_diopiHandle = dipu::diopi_helper::toDiopiTensorHandle({input});\n"
        diopi_fun_call_code = diopi_fun_call_code.replace(input, f"{input}_diopiHandle")

    output_process_code = ""
    #for output in fun_config['outs']:
    for output in get_function_inputs_outputs_from_schema(fun_config['schema'])[1]:
        output_process_code += f"::diopiTensorHandle_t {output}_diopiHandle = dipu::diopi_helper::toDiopiTensorHandle({output});\n"
        diopi_fun_call_code = diopi_fun_call_code.replace(output, f"{output}_diopiHandle")

    attrs_process_code = ""
    for scalar_param in get_function_scalar_args_from_schema(fun_config['schema']):
        attrs_process_code += f"::diopiScalar_t {scalar_param}_diopiScalar = dipu::diopi_helper::toDiopiScalar({scalar_param});\n";
        diopi_fun_call_code = diopi_fun_call_code.replace(scalar_param, f"{scalar_param}_diopiScalar")

    return_code = ""
    return_param = get_function_return_from_schema(fun_config['schema'])
    if len(return_param) == 1:
        return_code = f"return {return_param[0]};\n"
    else:
        params = ''
        for i in len(return_param):
            params += return_param[i]
            if i < len(return_param) - 1:
                params += ', '
        return_code = f"std::make_tuple({params});"

    fbody = fun_template.substitute(
            comment=[fun_config['schema']],
            cppsignautre=[create_cpp_signature_from_schema(fun_config['schema'])],
            custom_code=[fun_config.get('custom_code', '')],
            input_process_code=[input_process_code],
            output_process_code=[output_process_code],
            diopi_fun_call_code=[diopi_fun_call_code],
            attrs_process_code=[attrs_process_code],
            return_code=[return_code],
    )
    registe_body = op_registe_template.substitute(
            register_name=[get_op_name_from_schema(fun_config['schema'])],
            aten_fun_name=['dipu::native::' + create_fun_name_from_schema(fun_config['schema'])],
            diopi_fun_name=[get_fun_name_from_cppsignature(fun_config['interface']).replace('diopi', '::diopi')],
    )
    return fbody, registe_body


def main():
    with open('diopi_functions.yaml') as diopi_functions_file:
        file_data = diopi_functions_file.read()
        funcs_config = yaml.load(file_data, Loader=yaml.FullLoader)


    functions_code = ''
    op_registe_code = ''

    for fun_config in funcs_config:
        fun_code, register_code = functions_code_gen(fun_config)
        functions_code += fun_code
        op_registe_code += register_code

    autogened_file = file_template.substitute(
        functions_code=[functions_code],
        op_registe_code=[op_registe_code]
    )
    autogened_file = re.sub('\n\n\n+', '\n', autogened_file)
    with open('../AutoGenedKernels.cpp', 'w') as cpp_file:
        cpp_file.write(autogened_file)



if __name__ == "__main__":
    main()
