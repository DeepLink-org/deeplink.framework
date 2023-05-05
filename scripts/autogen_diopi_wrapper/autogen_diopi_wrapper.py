import yaml
import re
import json
from collections import OrderedDict
from typing import Mapping, Match, Optional, Sequence
from diopi_wrapper_template import diopi_wrapper_file_template_content, diopi_wrapper_function_template_content, op_register_template_content

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
    op_name = op_name.lower()
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
    args_type_map = OrderedDict({
        '[ ]*\([a-zA-Z]!\)' : '&',
        'str\?' : 'c10::optional<c10::string_view>',
        'ScalarType[ ]*\?' : 'c10::optional<at::ScalarType>',
        'Generator ?\?' : 'c10::optional<at::Generator>' ,
        'Layout ?\?' : 'c10::optional<at::Layout>' ,
        'Tensor ?\?' : 'const c10::optional<Tensor>&' ,
        '([\(, ]*)int ([\w\d_]+)' : R'\1int64_t \2',
        '([\(, ]*)float ([\w\d_]+)' : R'\1double \2',
        '([a-zA-Z0-9]+)\?' : r'c10::optional<\1>&',
        'Tensor *\[ *\]' : 'at::ArrayRef<Tensor>' ,
        'Tensor ' : 'const Tensor& ' ,
        '([, /(])Scalar ' : R'\1const at::Scalar& ' ,
        'Tensor' : 'at::Tensor' ,
        '([, \(]+)int\[\d\]\?' : R'\1at::OptionalIntArrayRef',
        'SymInt\[\d+\]' : 'c10::SymIntArrayRef' ,
        'int *\[ *\d+\ *]' : 'at::IntArrayRef' ,
        'bool\[(\d+)\]' : R'::std::array<bool,\1>' ,
        '\*[ ,]+' : '',
        '=[ ]*\w+[\d ]?' : '',
    })
    for pattern, cpp_type in args_type_map.items():
        param_list = re.sub(str(pattern), str(cpp_type), param_list)
    return param_list


def get_function_inputs_from_schema(schema):
    param_list = create_param_list_from_schema(schema)
    ins = []
    for args in param_list.split(','):
        args = args.strip()
        tensor_match_result = re.search('Tensor[ ]*&+', args)
        if tensor_match_result is not None:
            in_match_result = re.search('const[ ]+[at::]*Tensor[ &]*', args)
            if in_match_result is not None:
                ins.append(args[in_match_result.span()[1]::].strip())
        opt_tensor_match_result = re.search('const[ ]+c10::optional<at::Tensor>[ &]*([a-zA-Z_0-9]+)', args)
        if opt_tensor_match_result is not None:
            opt_tensor = re.sub('const[ ]+c10::optional<at::Tensor>[ &]*([a-zA-Z_]+)', r'\1', args).strip()
            ins.append(opt_tensor + '?')
    return ins


def get_function_outputs_from_schema(schema):
    param_list = create_param_list_from_schema(schema)
    outs = []
    for args in param_list.split(','):
        args = args.strip()
        tensor_match_result = re.search('Tensor[ ]*&+', args)
        if tensor_match_result is not None:
            in_match_result = re.search('const[ ]+[at::]*Tensor[ &]*', args)
            if in_match_result is None:
                outs.append(args[tensor_match_result.span()[1]::].strip())
    if len(outs) <= 0:
        return_param = schema[schema.find('->'):].replace('->', '').strip()
        return_param = return_param.replace('(', '')
        return_param = return_param.replace(')', '')
        params = return_param.split(',')
        if len(params) == 1 and params[0].strip() == "Tensor":
            if params[0].strip() == "Tensor":
                outs.append(f"out")
        elif len(params) > 1:
            for i in range(len(params)):
                if params[i].strip() == "Tensor":
                    outs.append(f"out{i}")

    return outs


def get_function_scalar_args_from_schema(schema):
    param_list = schema[schema.find('(') + 1 : schema.find('->')].strip()
    param_list = param_list[0:param_list.rfind(')')]
    scalars = []
    for args in param_list.split(','):
        args = args.strip()
        scalar_match_result = re.search('[ ]?Scalar[ ]+', args)
        opt_scalar_match_result = re.search('Scalar[ ][\?]+', args)
        if scalar_match_result is not None and opt_scalar_match_result is None:
            scalar_param = args[scalar_match_result.span()[1]:].strip()
            scalar_param = re.sub('=.*,{1}', ',', scalar_param)
            scalar_param = re.sub('=.*', '', scalar_param)
            scalars.append(scalar_param.strip())
    return scalars


def get_function_int_array_args_from_schema(schema):
    param_list = create_param_list_from_schema(schema)
    int_arrays = []
    for args in param_list.split(','):
        args = args.strip()
        match_result = re.search('[\w\d:]*SymIntArray[\w\d]*', args)
        if match_result is not None:
            int_array_param = args[match_result.span()[1]:].strip()
            int_array_param = re.sub('=.*,{1}', ',', int_array_param)
            int_array_param = re.sub('=.*', '', int_array_param)
            int_arrays.append(int_array_param.strip())
    return int_arrays


def get_function_return_param_from_schema(schema):
    return_schema= schema[schema.find('->' ) + 2:].strip()
    params = []
    return_params = return_schema.split(',')
    for i in range(len(return_params)):
        args = return_params[i]
        inplace_match = re.search('Tensor\([a-zA-Z]+!\)', args)
        pure_out_match = re.search('Tensor', args)
        if inplace_match is None and pure_out_match is not None:
            if len(return_params) > 1:
                params.append(f"out{i}")
            else:
                params.append("out")
        elif inplace_match is not None:
            arg_label = re.sub('.*(\(.*\))', r'\1',inplace_match.group())
            index = schema.find(arg_label) + len(arg_label)
            param = re.search("[a-zA-Z0-9_::]+", schema[index:]).group()
            params.append(param)

    return params


def create_call_diop_interface_code_from_schema(schema):
    schema = schema.replace('aten::', '').strip()
    schema = schema.replace('_.', 'Inp')
    schema = schema.replace('.', '')

    outs = re.findall(",? *Tensor *\(\w+!\) *\w+", schema)[::-1]
    schema = re.sub(",? *Tensor *\(\w+!\) *\w+", '', schema)
    index = schema.find('(') + 1
    for args in outs:
        schema = schema[0:index] + args.replace(',', '') + ', ' + schema[index:]

    schema = schema.replace('(', '(ctx, ', 1)
    return_index = schema.find('->')

    if return_index > 0:
        return_args = schema[return_index + 2 :].strip()
        if re.search('Tensor[ ]*\([\w]+!\)', return_args) is None:
            return_args = re.sub('Tensor[ ]*\([\w]+!\)[ ]*', '', return_args)
            return_args = re.sub('[\(\)]', '', return_args).strip()
            outs = return_args.split(',')
            retucn_code = ''
            for i in range(len(outs)):
                retucn_code += 'out'
                if len(outs) > 1:
                    retucn_code += str(i)
                if i < len(outs) - 1:
                    retucn_code += ', '
            schema = re.sub('\([ ]*ctx', '(ctx, ' + retucn_code, schema)
    schema = schema[0 : schema.find('->')]

    for key in ['Tensor[ ]*\([\w!]+\)', 'Tensor[ ]*\?', 'Tensor[ ]*', 'bool', 'float', 'str[ ]*\?', '[,]? *\* *', '=[\w]+']:
        index = schema.find('(')
        schema = schema[0:index] +  re.sub(key , '', schema[index:])

    index = schema.find('(')
    schema = schema[0:index] +  re.sub('Scalar[ ]*' , '&', schema[index:])

    for key in ['out', '_mode', 'Tensor', '_', '[nN]{1}ative_']:
        index = schema.find('(')
        schema = re.sub(key , '', schema[:index]) + schema[index:]

    schema = 'diopi' + schema[0].upper() + schema[1:]
    schema = re.sub(' *, *', ', ', schema)
    schema = re.sub(' *, *,', ', ', schema)

    return schema


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


def create_code_to_print_fun_call_info_from_schema(schema):
    op_name = get_op_name_from_schema(schema)
    debug_code = f'printf("[%s:%s:%d]:%s\\n",__FILE__,__FUNCTION__,__LINE__,"{op_name}");' + '\n'
    return debug_code

def create_int_array_process_code(int_array_list):
    if len(int_array_list) <= 0:
        return ''
    code = R"auto symIntToInt = [](const c10::SymInt& t)-> int64_t {return t.expect_int();};" + '\n'
    for int_array in int_array_list:
        code += f"std::vector<int64_t> {int_array}Vector({int_array}.size());\n"
        code += f"std::transform({int_array}.cbegin(), {int_array}.cend(), {int_array}Vector.begin(), symIntToInt);\n"
        code += f"::diopiSize_t {int_array}DiopiSize({int_array}Vector.data(), {int_array}Vector.size());\n"
    return code;



file_template = CodeTemplate(diopi_wrapper_file_template_content)

fun_template = CodeTemplate(diopi_wrapper_function_template_content)

op_register_template = CodeTemplate(op_register_template_content)


def functions_code_gen(fun_config):
    if 'interface' in fun_config:
        diopi_fun_call_code = fun_config['interface'] + ";"
    else:
        diopi_interface = create_call_diop_interface_code_from_schema(fun_config['schema'])
        diopi_fun_call_code = diopi_interface + ';'

    input_process_code = ""
    diopi_tensor_suffix = 'DiopiTensorHandle'

    for input in set(get_function_inputs_from_schema(fun_config['schema']) + fun_config.get('ins', [])):
        if input.strip().endswith('?'):
            input = input.replace('?', '')
            input_process_code += f"\n::diopiConstTensorHandle_t {input}{diopi_tensor_suffix} = nullptr;\n"
            input_process_code += f"if ({input}.has_value() && {input}.value().defined()) {input}{diopi_tensor_suffix} = dipu::diopi_helper::toDiopiTensorHandle({input}.value());\n\n"

        else:
            input_process_code += f"::diopiConstTensorHandle_t {input}{diopi_tensor_suffix} = dipu::diopi_helper::toDiopiTensorHandle({input});\n"

        diopi_fun_call_code = re.sub(input.strip() + '([,\) ]{1})', f"{input.strip()}{diopi_tensor_suffix}" + r'\1', diopi_fun_call_code)


    output_process_code = ""
    for output in set(get_function_outputs_from_schema(fun_config['schema']) + fun_config.get('outs', [])):
        output_process_code += f"::diopiTensorHandle_t {output}{diopi_tensor_suffix} = dipu::diopi_helper::toDiopiTensorHandle({output});\n"
        diopi_fun_call_code = re.sub(output.strip() + '([,\) ]{1})', f"{output.strip()}{diopi_tensor_suffix}" + r'\1', diopi_fun_call_code)

    attrs_process_code = ""

    diopi_scalar_suffix = 'DiopiScalar'
    for scalar_param in get_function_scalar_args_from_schema(fun_config['schema']):
        attrs_process_code += f"::diopiScalar_t {scalar_param}{diopi_scalar_suffix} = dipu::diopi_helper::toDiopiScalar({scalar_param});\n";
        diopi_fun_call_code = re.sub('&?[ ]*' + scalar_param.strip(), f"&{scalar_param}{diopi_scalar_suffix}", diopi_fun_call_code)


    int_array_list = get_function_int_array_args_from_schema(fun_config['schema'])
    attrs_process_code += create_int_array_process_code(int_array_list)
    for int_array_param in int_array_list:
        diopi_fun_call_code = re.sub(int_array_param.strip(), f"{int_array_param}DiopiSize", diopi_fun_call_code)


    if fun_config.get('print_func_call_info', False) == True:
        fun_config['custom_code_at_the_beginning'] = create_code_to_print_fun_call_info_from_schema(fun_config['schema']) + fun_config.get('custom_code_at_the_beginning', '')

    if fun_config.get('dummy_call_diopi', False) == True:
        diopi_fun_call_code = f"::diopiSuccess;/*dummy_call_diopi: {diopi_fun_call_code}*/"

    return_code = ""
    return_param = get_function_return_param_from_schema(fun_config['schema'])
    if len(return_param) == 0:
        return_code = "return;\n"
    elif len(return_param) == 1:
        return_code = f"return {return_param[0]};\n"
    else:
        params = ''
        for i in range(len(return_param)):
            params += return_param[i]
            if i < len(return_param) - 1:
                params += ', '
        return_code = f"return std::tie({params});"

    fbody = fun_template.substitute(
            comment=[fun_config['schema']],
            cppsignautre=[create_cpp_signature_from_schema(fun_config['schema'])],
            custom_code_at_the_beginning=[fun_config.get('custom_code_at_the_beginning', fun_config.get('custom_code', '')).replace('; ', ';\n')],
            input_process_code=[input_process_code],
            attrs_process_code=[attrs_process_code],
            output_process_code=[output_process_code],
            custom_code_before_call_diopi = [fun_config.get('custom_code_before_call_diopi', '').replace('; ', ';\n')],
            diopi_fun_call_code=[diopi_fun_call_code],
            custom_code_before_return=[fun_config.get('custom_code_before_return', '').replace('; ', ';\n')],
            return_code=[return_code],
    )
    diopi_interface = fun_config.get('interface', create_call_diop_interface_code_from_schema(fun_config['schema']))
    register_body = op_register_template.substitute(
            register_name=[get_op_name_from_schema(fun_config['schema'])],
            aten_fun_name=['dipu::native::' + create_fun_name_from_schema(fun_config['schema'])],
            diopi_fun_name=[get_fun_name_from_cppsignature(diopi_interface).replace('diopi', '::diopi')],
    )
    return fbody, register_body

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def parase_args():
    import argparse
    parser = argparse.ArgumentParser(description='autogen diopi wrapper code')
    parser.add_argument('--config', type=str, default = 'diopi_functions.yaml', help='path to functions config file')
    parser.add_argument('--out', type=str, default = 'AutoGenedKernels.cpp', help='path to functions config file')
    parser.add_argument('--dummy_call_diopi', default=False, type=boolean_string, help='whether acctually call diopi interface')
    parser.add_argument('--print_func_call_info', default=False, type=boolean_string, help='whether generate code that prints function call information')
    parser.add_argument('--fun_config_dict', type=json.loads, default = dict(), help='fun config for all ops') # --fun_config_dict '{"debug":"True"}'

    args = parser.parse_args()
    return args

def main():
    args = parase_args()
    with open(args.config) as diopi_functions_file:
        file_data = diopi_functions_file.read()
        funcs_config = yaml.load(file_data, Loader=yaml.FullLoader)


    functions_code = ''
    op_register_code = ''

    for fun_config in funcs_config:
        mergeed_fun_config = dict(args.fun_config_dict)
        mergeed_fun_config.update(vars(args))
        mergeed_fun_config.update(fun_config)
        fun_code, register_code = functions_code_gen(mergeed_fun_config)
        functions_code += fun_code
        if fun_config.get('register_op', True) == True:
            op_register_code += register_code

    autogened_file = file_template.substitute(
        functions_code=[functions_code],
        op_register_code=[op_register_code]
    )
    autogened_file = re.sub(R'\n{3,}', R'\n\n', autogened_file)
    autogened_file = re.sub('[ ]*,[ ]*', ', ', autogened_file)
    with open(args.out, 'w') as cpp_file:
        cpp_file.write(autogened_file)

    print(f"Successfully generate {args.out} according to the configuration file {args.config}")


if __name__ == "__main__":
    main()
