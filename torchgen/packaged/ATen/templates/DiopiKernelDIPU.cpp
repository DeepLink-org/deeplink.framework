// an external backend might generate file within its code tree
// and check all the source files within the tree with clang-format.
// so, disable it since the backend might have a different config.
// clang-format off

// ${generated_comment}

#include <ATen/Tensor.h>

#include "csrc_dipu/aten/DIPUATenFunctions.h"
#include "csrc_dipu/diopirt/diopirt_impl.h"

using dipu::diopi_helper::toDiopiTensorHandle;
using dipu::diopi_helper::toDiopiScalar;

${namespace_prologue}

${kernel_defn} {
  ${arguments_convert}
  ${infer}
  ${adaptor}

  ${diopi_invoke}
}

${namespace_epilogue}