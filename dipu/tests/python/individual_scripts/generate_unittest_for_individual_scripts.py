# Copyright (c) 2023, DeepLink.
import glob
import os


def generate_unittest_for_individual_scripts():
    main_sketch = """
# Copyright (c) 2023, DeepLink.
import io
import os
import unittest
from torch_dipu.testing._internal.stdout_redirector import stdout_redirector


class TestIndividualScripts(unittest.TestCase):
    def _test_individual_script(self, script_path: str):
        captured = io.BytesIO()
        with stdout_redirector(captured):
            retcode = os.system(f"python {script_path}")
        self.assertEqual(
            retcode,
            0,
            msg=f"\\n\\n*** CAPTURED STDOUT ***\\n{captured.getvalue().decode()}"
        )

${CASES}
if __name__ == "__main__":
    unittest.main()
"""

    case_sketch = """
    def ${CASE_NAME}(self):
        self._test_individual_script("${SCRIPT_PATH}")
"""

    cwd = os.path.dirname(os.path.abspath(__file__))
    cases_code = ""
    for script_path in glob.glob(f"{cwd}/test_*.py"):
        script_name = os.path.basename(script_path)
        case_name = script_name[:-3]
        case_code = (
            case_sketch.replace("${CASE_NAME}", case_name)
            .replace("${SCRIPT_PATH}", script_path)
            .strip("\n\r")
        )
        cases_code += f"{case_code}\n\n"
    main_code = main_sketch.replace("${CASES}", cases_code).strip("\n\r")
    print(main_code)


if __name__ == "__main__":
    generate_unittest_for_individual_scripts()
