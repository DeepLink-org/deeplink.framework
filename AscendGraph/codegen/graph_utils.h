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

class AscendManager {
public:
  explicit AscendManager(uint32_t graph_id): graph_id(graph_id), sess(nullptr) {
    std::map<AscendString, AscendString> config = {
                                    {"ge.exec.deviceId", "0"},
                                    {"ge.socVersion", "Ascend910ProB"},
                                    {"ge.graphRunMode", "1"},
                                    {"ge.exec.precision_mode", "allow_fp32_to_fp16"}};
    Status ret = ge::GEInitialize(config);
    if (ret != SUCCESS) {
      std::cout<<"Initialize ge failed."<<std::endl;
    }
    std::cout<<"Initialize ge success."<<std::endl;

    std::map<AscendString, AscendString> option;
    sess = new ge::Session(option);
    if (sess == nullptr) {
        std::cout << "Create session failed." << std::endl;
    }
    std::cout<<"Create session success."<<std::endl;
  }

  ~AscendManager() {
    std::cout << "############ this is in before delete sess!" << std::endl;
    //delete sess;
    std::cout << "############ this is in after delete sess!" << std::endl;
    Status ret = ge::GEFinalize();
    if (ret != SUCCESS) {
        std::cout<<"Finalize ge failed."<<std::endl;
    }
    std::cout<<"Finalize ge success."<<std::endl;
  }

  ge::Session* session() {
    return sess;
  }

private:
  uint32_t graph_id;
  ge::Session* sess;
};

class AclgraphBuilder {
public:
  explicit AclgraphBuilder() {
    // 1. system init
    std::string kSocVersion = "Ascend910ProB";
    std::map<AscendString, AscendString> global_options = {
        {AscendString(ge::ir_option::SOC_VERSION), AscendString(kSocVersion.c_str())}  ,
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
