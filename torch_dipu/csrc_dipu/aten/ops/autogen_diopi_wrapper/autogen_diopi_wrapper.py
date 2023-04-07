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



code_template = CodeTemplate(
    """
    // autogened file
    #include <ATen/Tensor.h>

    #include "csrc_dipu/aten/DIPUATenFunctions.h"
    #include "csrc_dipu/diopirt/diopirt_impl.h"

    namespace dipu::native {

    using at::Tensor;
    using at::Scalar;

    $cppsignautre {
        ::diopiContext context(dipu::getCurrentDIPUStream().rawstream());
        auto ctx = &context;

        $input_process_code

        $output_process_code

        $attrs_process_code

        ::diopiError_t ret = $diopi_fun_call_code
        TORCH_CHECK(ret == ::diopiSuccess, __FILE__, ":", __LINE__,"$diopi_fun_call_code error, error code is ", ret, "error message is ", diopiGetLastErrorString());

        $return_code
    }

    }  // namespace dipu::native
    """
    )

def code_gen(fun_config):
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



    fbody = code_template.substitute(
            cppsignautre=[fun_config['cppsignature']],
            input_process_code=[input_process_code],
            output_process_code=[output_process_code],
            diopi_fun_call_code=[diopi_fun_call_code],
            attrs_process_code=[attrs_process_code],
            return_code=[return_code],
            )
    return fbody


def main():
    with open('diopi_functions.yaml') as diopi_functions_file:
        file_data = diopi_functions_file.read()
        funcs_config = yaml.load(file_data, Loader=yaml.FullLoader)

    for fun_config in funcs_config:
        code_str = code_gen(fun_config)
        with open('../'+ fun_config['name'] + '.cpp', 'w') as cpp_file:
            cpp_file.write(code_str)



if __name__ == "__main__":
    main()
