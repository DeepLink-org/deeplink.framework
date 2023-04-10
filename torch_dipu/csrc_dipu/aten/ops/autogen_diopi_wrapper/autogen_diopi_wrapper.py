import yaml
import re
from typing import Mapping, Match, Optional, Sequence

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



file_template = CodeTemplate(
"""
// autogened file
#include <ATen/Tensor.h>

#include "csrc_dipu/aten/DIPUATenFunctions.h"
#include "csrc_dipu/aten/RegisterDIPU.hpp"
#include "csrc_dipu/diopirt/diopirt_impl.h"

namespace dipu::native {

using at::Tensor;
using at::Scalar;

using namespace dipu::diopi_helper;

$functions_code

}  // namespace dipu::native

namespace at {

TORCH_LIBRARY_IMPL(aten, DIPU_DEVICE_TYPE_MACRO, m) {
    $op_registe_code
}

}  // namespace at

"""
)

fun_template = CodeTemplate(
"""
$cppsignautre {
    ::diopiContext context(dipu::getCurrentDIPUStream().rawstream());
    auto ctx = &context;
    $custom_code
    $input_process_code

    $output_process_code

    $attrs_process_code

    ::diopiError_t ret = $diopi_fun_call_code
    TORCH_CHECK(ret == ::diopiSuccess, __FILE__, ":", __LINE__,"'$diopi_fun_call_code' error, error code is ", ret, "error message is ", diopiGetLastErrorString());

    $return_code
}
"""
)

op_registe_template = CodeTemplate(
"""
DIOPI_ATEN_FUNC("$register_name", $diopi_fun_name, $aten_fun_name);
"""
)

def functions_code_gen(fun_config):
    diopi_fun_call_code = fun_config['interface'] + ";"
    input_process_code = ""
    for input in fun_config['ins']:
        input_process_code += f"::diopiConstTensorHandle_t {input}_diopiHandle = dipu::diopi_helper::toDiopiTensorHandle({input});\n"
        diopi_fun_call_code = diopi_fun_call_code.replace(input, f"{input}_diopiHandle")

    output_process_code = ""
    for output in fun_config['outs']:
        output_process_code += f"::diopiTensorHandle_t {output}_diopiHandle = dipu::diopi_helper::toDiopiTensorHandle({output});\n"
        diopi_fun_call_code = diopi_fun_call_code.replace(output, f"{output}_diopiHandle")

    attrs_process_code = ""
    if 'attrs' in fun_config:
        for attr_name, attr_type in fun_config['attrs'].items():
            if attr_type.lower() == 'scalar':
                attrs_process_code += f"::diopiScalar_t {attr_name}_diopiScalar = dipu::diopi_helper::toDiopiScalar({attr_name});\n";
                diopi_fun_call_code = diopi_fun_call_code.replace(attr_name, f"{attr_name}_diopiScalar")
            else:
                print(f"do not support {attr_type} now!")


    return_code = ""
    if len(fun_config['return']) == 1:
        return_code = f"return {fun_config['return'][0]};\n"

    fbody = fun_template.substitute(
            cppsignautre=[fun_config['cppsignature']],
            custom_code=[fun_config.get('custom_code', '')],
            input_process_code=[input_process_code],
            output_process_code=[output_process_code],
            diopi_fun_call_code=[diopi_fun_call_code],
            attrs_process_code=[attrs_process_code],
            return_code=[return_code],
    )
    registe_body = op_registe_template.substitute(
            register_name=[fun_config['name']],
            aten_fun_name=['dipu::native::' + get_fun_name_from_cppsignature(fun_config['cppsignature'])],
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

    with open('../AutoGenedKernels.cpp', 'w') as cpp_file:
        cpp_file.write(autogened_file)



if __name__ == "__main__":
    main()
