// Copyright (c) 2023, DeepLink.

#include <unordered_map>

#include <c10/util/Exception.h>

#include <csrc_dipu/aten/ops/DIPUCopy.hpp>
#include <csrc_dipu/common.h>
#include <csrc_dipu/diopirt/diopirt_impl.h>
#include <csrc_dipu/runtime/core/DIPUStream.h>

namespace dipu {

// NOTICE: diopicamb copy has many restriction ^#^
// this code based on diopi cnnl_helper.cpp gCnnlCastDataTypeMapping and
// only cover commonly used cases, leaving diopi developer to optimize.
namespace {
struct HashCnnlCastDType {
  const static int64_t HashMagic = 0x9e3779b9;
  const static int64_t ShiftLeft = 6;
  const static int64_t ShiftRight = 2;
  size_t operator()(const std::vector<diopiDtype_t>& vec) const {
    size_t ret = 0;
    for (auto it : vec) {
      ret = (ret ^ static_cast<size_t>(it)) + HashMagic + (ret << ShiftLeft) +
            (ret >> ShiftRight);
    }
    return ret;
  }
};

const std::unordered_set<std::vector<diopiDtype_t>, HashCnnlCastDType>
    cnnlCastDataTypeMapping{
        {{diopi_dtype_bool, diopi_dtype_int32}},
        {{diopi_dtype_bool, diopi_dtype_float16}},
        {{diopi_dtype_bool, diopi_dtype_float32}},

        {{diopi_dtype_int8, diopi_dtype_int16}},
        {{diopi_dtype_int8, diopi_dtype_int32}},
        {{diopi_dtype_int8, diopi_dtype_float16}},
        {{diopi_dtype_int8, diopi_dtype_float32}},

        {{diopi_dtype_uint8, diopi_dtype_int32}},
        {{diopi_dtype_uint8, diopi_dtype_int64}},
        {{diopi_dtype_uint8, diopi_dtype_float16}},
        {{diopi_dtype_uint8, diopi_dtype_float32}},

        {{diopi_dtype_int16, diopi_dtype_int32}},
        {{diopi_dtype_int16, diopi_dtype_float16}},
        {{diopi_dtype_int16, diopi_dtype_float32}},
        // no uint16 cast

        {{diopi_dtype_int32, diopi_dtype_bool}},
        {{diopi_dtype_int32, diopi_dtype_int8}},
        {{diopi_dtype_int32, diopi_dtype_int16}},
        {{diopi_dtype_int32, diopi_dtype_int64}},
        {{diopi_dtype_int32, diopi_dtype_float16}},
        {{diopi_dtype_int32, diopi_dtype_float32}},

        {{diopi_dtype_uint32, diopi_dtype_int64}},
        {{diopi_dtype_uint32, diopi_dtype_uint64}},

        {{diopi_dtype_int64, diopi_dtype_int32}},
        {{diopi_dtype_int64, diopi_dtype_uint32}},
        {{diopi_dtype_int64, diopi_dtype_float16}},
        {{diopi_dtype_int64, diopi_dtype_float32}},

        {{diopi_dtype_uint64, diopi_dtype_uint32}},

        // CNNL_CAST_HALF_TO_FLOAT_INF = 129, /*!< Converts half to float for
        // amp training. */
        {{diopi_dtype_float16, diopi_dtype_bool}},
        {{diopi_dtype_float16, diopi_dtype_int8}},
        {{diopi_dtype_float16, diopi_dtype_uint8}},
        {{diopi_dtype_float16, diopi_dtype_int16}},
        {{diopi_dtype_float16, diopi_dtype_int32}},
        {{diopi_dtype_float16, diopi_dtype_int64}},
        {{diopi_dtype_float16, diopi_dtype_float32}},

        // CNNL_CAST_FLOAT_TO_HALF_IEEE754 = 219, /*!< Converts float to half
        // for ieee754. */
        {{diopi_dtype_float32, diopi_dtype_bool}},
        {{diopi_dtype_float32, diopi_dtype_int8}},
        {{diopi_dtype_float32, diopi_dtype_uint8}},
        {{diopi_dtype_float32, diopi_dtype_int16}},
        {{diopi_dtype_float32, diopi_dtype_int32}},
        {{diopi_dtype_float32, diopi_dtype_int64}},
        {{diopi_dtype_float32, diopi_dtype_float16}},
        {{diopi_dtype_float32, diopi_dtype_float64}},

        {{diopi_dtype_float64, diopi_dtype_float32}},
    };
}  // namespace

class CambCopyInplace : public DIPUCopyInpOnDIOPI {
 public:
  CambCopyInplace() = default;
  ~CambCopyInplace() override = default;

  void copyNodirectOnDevice(at::Tensor& dst, const at::Tensor& src,
                            bool non_blocking, CopyParamsInfo& info) override {
    diopiDtype_t dstDtype = dipu::diopi_helper::toDiopiDtype(dst.scalar_type());
    diopiDtype_t srcDtype = dipu::diopi_helper::toDiopiDtype(src.scalar_type());
    // diopiCopy(cnnlTranspose_v2) cannot handle float64 stride based
    // copy, failed test: TestTorchDeviceType.test_memory_format_to
    bool noSupportedTranspose = !dst.is_contiguous() && !src.is_contiguous() &&
                                dstDtype == diopi_dtype_float64;
    bool noSupportedDtype = dst.is_complex() || src.is_complex();
    // cnnl only handle limited cast type.
    bool noSupportedCast = !info.sameDtype_ &&
                           cnnlCastDataTypeMapping.find({srcDtype, dstDtype}) ==
                               cnnlCastDataTypeMapping.end();

    if (noSupportedTranspose || noSupportedDtype || noSupportedCast) {
      doCpuRelayCopy(dst, src, info.curStream_, non_blocking);
    } else {
      DIPUCopyInplace::copyNodirectOnDevice(dst, src, non_blocking, info);
    }
  }
};

static CambCopyInplace camb_copy_inplace;  // NOLINT
const static int32_t camb_init = []() {
  setDipuCopyInstance(&camb_copy_inplace);
  return 1;
}();

}  // namespace dipu
