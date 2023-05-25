// Copyright (c) 2023, DeepLink.
#include <torch/csrc/python_headers.h>
#include <torch/csrc/Device.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/THP.h>
#include <torch/csrc/autograd/utils/wrap_outputs.h>
#include <torch/csrc/copy_utils.h>

#include <c10/util/intrusive_ptr.h>

#include <torch/csrc/Storage.h>
#include <torch/csrc/StorageMethods.h>

#include <ATen/ATen.h>
#include <torch/csrc/utils/pycfunction_helpers.h>
#include <torch/csrc/utils/python_arg_parser.h>
#include <torch/csrc/utils/python_numbers.h>


namespace dipu {

static PyObject* THPStorage_copy_(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS

  at::Storage self_ = torch::createStorage(self);

  static torch::PythonArgParser parser({
      "copy_(Storage src, bool? non_blocking=None)",
  });
  torch::ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  at::Storage src = torch::createStorage(parsed_args.args[0]);
  bool non_blocking = r.toBoolOptional(1).value_or(false);

  TORCH_CHECK(self_.nbytes() == src.nbytes(), "size does not match");

  storage_copy(self_, src, non_blocking);

  Py_INCREF(self);
  return self;

  END_HANDLE_TH_ERRORS
}
} // end ns dipu