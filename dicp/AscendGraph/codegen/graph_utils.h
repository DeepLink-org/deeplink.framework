#ifndef DAVINCI_GRAPH_UTILS_H
#define DAVINCI_GRAPH_UTILS_H
#include <vector>
#include <string>
#include <map>
#include <iostream>
#include <numeric>
#include <functional>

#include "graph.h"
#include "types.h"
#include "tensor.h"
#include "ge_error_codes.h"
#include "ge_api_types.h"
#include "ge_api.h"
#include "all_ops.h"
#include "ascend_string.h"
#include "ge_ir_build.h"
#include "gnode.h"

#define FAILED -1
#define SUCCESS 0

using namespace ge;

template<typename T>
int64_t getVecSize(const std::vector<T>& v) {
    int64_t size = std::accumulate(v.begin(), v.end(),
                          1, std::multiplies<int64_t>());  
    return size;
}

void setTensorData(Tensor& tensor, uint8_t* src_data, uint64_t data_size, const std::string& debug_name = "") {
    auto status = tensor.SetData(reinterpret_cast<uint8_t*>(src_data), data_size);
    if (status != ge::GRAPH_SUCCESS) {
        std::cout << "Set " << debug_name << " tensor data failed!" << std::endl;
    }
}

ge::Tensor genTensor(const std::vector<int64_t>& tensor_shape, ge::Format format, ge::DataType data_type) {
  TensorDesc desc (ge::Shape(tensor_shape), format, data_type);
  Tensor result(desc);
  return result;
}

class AclgraphBuilder {
public:
  explicit AclgraphBuilder() {
    // 1. system init
    std::string kSocVersion = "Ascend910ProB";
    std::map<AscendString, AscendString> global_options = {
        {AscendString(ge::ir_option::SOC_VERSION), AscendString(kSocVersion.c_str())},
        {AscendString(ge::ir_option::PRECISION_MODE), "allow_fp32_to_fp16"},
    };
    auto status = aclgrphBuildInitialize(global_options);
    if (status != GRAPH_SUCCESS) {
      std::cout << "aclgrphBuildInitialize failed!" << std::endl;
    } else {
      std::cout << "aclgrphBuildInitialize success!" << std::endl;
    }
  }

  void saveGraph(const std::string& path, const Graph& graph) {
    ModelBufferData model;
    std::map<AscendString, AscendString> options;

    auto status = aclgrphBuildModel(graph, options, model);
    if (status == GRAPH_SUCCESS) {
        std::cout << "Build Model SUCCESS!" << std::endl;
    }
    else {
        std::cout << "Build Model Failed! " << status << std::endl;
        return;
    }

    // 4. Save Ir Model
    status = aclgrphSaveModel(path.c_str(), model);
    if (status == GRAPH_SUCCESS) {
        std::cout << "Save Offline Model SUCCESS!" << std::endl;
    }
    else {
        std::cout << "Save Offline Model Failed! " << status << std::endl;
    }
  }

  ~AclgraphBuilder() {
    aclgrphBuildFinalize();
    std::cout << "aclgrphBuildFinalize success!" << std::endl;
  }
};

#endif //DAVINCI_GRAPH_UTILS_H
