#ifndef DICP_ASCEND_GE_GRAPH_H
#define DICP_ASCEND_GE_GRAPH_H
#include <cctype>
#include <fstream>
#include <functional>
#include <half.hpp>
#include <iostream>
#include <json.hpp>
#include <map>
#include <numeric>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "acl/acl.h"
#include "all_ops.h"
#include "ascend_string.h"
#include "ge_api.h"
#include "ge_api_types.h"
#include "ge_error_codes.h"
#include "ge_ir_build.h"
#include "gnode.h"
#include "graph.h"
#include "tensor.h"
#include "types.h"

#include "graph_utils.h"

using json = nlohmann::json;
using namespace ge;

class GEGraph {
 public:
  explicit GEGraph(int graph_id, Graph& graph, const std::string& graph_key)
      : graph_id_(graph_id), graph_(std::move(graph)), graph_key_(graph_key)  {}

  size_t const_memory_size() {
    size_t size;
    auto status = spec_->GetConstMemorySize(size);
    if (status != GRAPH_SUCCESS) {
      std::cout << "GetConstMemorySize failed!" << std::endl;
    }
    return size;
  }

  size_t feature_memory_size() {
    size_t size;
    auto status = spec_->GetFeatureMemorySize(size);
    if (status != GRAPH_SUCCESS) {
      std::cout << "GetFeatureMemorySize failed!" << std::endl;
    }
    return size;
  }

  void prepare_output() {
    std::vector<ge::Shape> shapes;
    std::vector<ge::DataType> dtypes;
    auto status = spec_->GetOutputShapes(shapes);
    if (status != GRAPH_SUCCESS) {
      std::cout << "GetOutputShapes failed!" << std::endl;
    }
    status = spec_->GetOutputDtypes(dtypes);
    if (status != GRAPH_SUCCESS) {
      std::cout << "GetOutputDtypes failed!" << std::endl;
    }

    for (unsigned long i = 0; i < shapes.size(); ++i) {
      TensorDesc cur_desc(shapes[i], FORMAT_ND, dtypes[i]);
      Tensor cur_tensor(cur_desc);
      outputs.emplace_back(cur_tensor);
    }
  }

  void prapare_input_output_tensordesc() {
    for (const auto& i : inputs) {
      inputs_desc.emplace_back(i.GetTensorDesc());
    }
    for (const auto& i : outputs) {
      outputs_desc.emplace_back(i.GetTensorDesc());
    }
  }

  std::vector<std::vector<int64_t>> get_input_shapes() {
    std::vector<std::vector<int64_t>> res;
    for (const auto& i : inputs) {
      auto desc = i.GetTensorDesc();
      auto dims = desc.GetShape().GetDims();
      res.emplace_back(desc.GetShape().GetDims());
    }
    return res;
  }

  std::vector<int> get_input_dtypes() {
    std::vector<int> res;
    for (const auto& i : inputs) {
      auto data_type = i.GetTensorDesc().GetDataType();
      res.emplace_back(static_cast<int>(data_type));
    }
    return res;
  }

  std::vector<std::vector<int64_t>> get_output_shapes() {
    std::vector<std::vector<int64_t>> res;
    for (const auto& i : outputs) {
      auto desc = i.GetTensorDesc();
      auto dims = desc.GetShape().GetDims();
      res.emplace_back(desc.GetShape().GetDims());
    }
    return res;
  }

  std::vector<int> get_output_dtypes() {
    std::vector<int> res;
    for (const auto& i : outputs) {
      auto data_type = i.GetTensorDesc().GetDataType();
      res.emplace_back(static_cast<int>(data_type));
    }
    return res;
  }

  void set_input_output_data(void* inputs_data[], void* outputs_data[],
                             int64_t inputs_data_size[],
                             int64_t outputs_data_size[]) {
    const static ge::Tensor::DeleteFunc kDoNothing = [](uint8_t* data) {};
    for (unsigned long i = 0; i < inputs.size(); ++i) {
      auto status = inputs[i].ResetData(
          static_cast<uint8_t*>(inputs_data[i]),
          static_cast<size_t>(inputs_data_size[i]), kDoNothing);
      if (status != ge::GRAPH_SUCCESS) {
        std::cout << "Set input " << i << " tensor data failed!" << std::endl;
      }
    }
    for (unsigned long i = 0; i < outputs.size(); ++i) {
      auto status = outputs[i].ResetData(
          static_cast<uint8_t*>(outputs_data[i]),
          static_cast<size_t>(outputs_data_size[i]), kDoNothing);
      if (status != ge::GRAPH_SUCCESS) {
        std::cout << "Set input " << i << " tensor data failed!" << std::endl;
      }
    }
  }

  int graph_id_;
  Graph graph_;
  std::string graph_key_;
  std::shared_ptr<CompiledGraphSummary> spec_;
  std::vector<Tensor> inputs;
  std::vector<Tensor> outputs;
  std::vector<TensorDesc> inputs_desc;
  std::vector<TensorDesc> outputs_desc;
};

#endif  // DICP_ASCEND_GE_GRAPH_H