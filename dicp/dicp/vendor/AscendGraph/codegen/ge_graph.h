#ifndef DICP_ASCEND_GE_GRAPH_H
#define DICP_ASCEND_GE_GRAPH_H
#include <algorithm>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <vector>

#include "ge_api.h"
#include "graph_utils.h"

using namespace ge;

class GEGraph {
 public:
  explicit GEGraph(int graph_id, std::string graph_key, Graph& graph,
                   std::shared_ptr<CompiledGraphSummary> spec,
                   std::vector<Tensor>& input_tensors)
      : graph_id_(graph_id),
        graph_(std::move(graph)),
        graph_key_(std::move(graph_key)),
        spec_(std::move(spec)),
        inputs(std::move(input_tensors)),
        is_static(spec_->IsStatic()) {
    if (is_static) {
      prepare_static_output_tensors();
      prepare_input_output_tensordesc();
    }
  }

  size_t const_memory_size() {
    size_t size;
    CALL_FUNC(spec_->GetConstMemorySize(size));
    return size;
  }

  size_t feature_memory_size() {
    size_t size;
    CALL_FUNC(spec_->GetFeatureMemorySize(size));
    return size;
  }

  size_t fixed_feature_memory_size() {
    size_t size;
    CALL_FUNC(spec_->GetFixedFeatureMemorySize(size));
    return size;
  }

  int get_graph_id() const { return graph_id_; }

  std::vector<Tensor>& get_inputs() { return inputs; }

  std::vector<Tensor>& get_outputs() { return outputs; }

  void prepare_static_output_tensors() {
    std::vector<ge::Shape> shapes;
    std::vector<ge::DataType> dtypes;
    CALL_FUNC(spec_->GetOutputShapes(shapes));
    CALL_FUNC(spec_->GetOutputDtypes(dtypes));

    for (size_t i = 0; i < shapes.size(); ++i) {
      outputs.emplace_back(ge::TensorDesc(shapes[i], ge::FORMAT_ND, dtypes[i]));
    }
  }

  void prepare_input_output_tensordesc() {
    inputs_desc.reserve(inputs.size());
    outputs_desc.reserve(outputs.size());
    auto get_tensor_desc = [](const Tensor& tensor) {
      return tensor.GetTensorDesc();
    };
    std::transform(inputs.begin(), inputs.end(),
                   std::back_inserter(inputs_desc), get_tensor_desc);
    std::transform(outputs.begin(), outputs.end(),
                   std::back_inserter(outputs_desc), get_tensor_desc);
  }

  void assemble_inputs(const std::vector<ge::Shape>& shapes,
                       const std::vector<ge::DataType>& dtypes,
                       const std::vector<ge::Format>& formats) {
    inputs.clear();
    CHECK(shapes.size() == dtypes.size() && shapes.size() == formats.size());
    auto size = shapes.size();
    for (auto i = 0; i < size; ++i) {
      inputs.emplace_back(ge::TensorDesc(shapes[i], formats[i], dtypes[i]));
      inputs[i].SetPlacement(ge::Placement::kPlacementDevice);
    }
  }

  void assemble_outputs(const std::vector<ge::Shape>& shapes,
                        const std::vector<ge::DataType>& dtypes,
                        const std::vector<ge::Format>& formats) {
    outputs.clear();
    CHECK(shapes.size() == dtypes.size() && shapes.size() == formats.size());
    auto size = shapes.size();
    for (auto i = 0; i < size; ++i) {
      outputs.emplace_back(ge::TensorDesc(shapes[i], formats[i], dtypes[i]));
      outputs[i].SetPlacement(ge::Placement::kPlacementDevice);
    }
  }

  void update_inputs(const std::vector<ge::Shape>& shapes) {
    CHECK(inputs.size() == shapes.size());
    auto size = shapes.size();
    for (auto i = 0; i < size; ++i) {
      auto desc = inputs[i].GetTensorDesc();
      desc.SetShape(shapes[i]);
      CALL_FUNC(inputs[i].SetTensorDesc(desc));
    }
  }

  void update_outputs(const std::vector<ge::Shape>& shapes) {
    CHECK(outputs.size() == shapes.size());
    auto size = shapes.size();
    for (auto i = 0; i < size; ++i) {
      auto desc = outputs[i].GetTensorDesc();
      desc.SetShape(shapes[i]);
      CALL_FUNC(outputs[i].SetTensorDesc(desc));
    }
  }

  std::vector<std::vector<int64_t>> get_input_shapes() {
    return get_shapes(inputs);
  }

  std::vector<int> get_input_dtypes() { return get_dtypes(inputs); }

  std::vector<std::vector<int64_t>> get_output_shapes() {
    return get_shapes(outputs);
  }

  std::vector<int> get_output_dtypes() { return get_dtypes(outputs); }

  void set_input_output_data(void* inputs_data[], void* outputs_data[],
                             int64_t inputs_data_size[],
                             int64_t outputs_data_size[]) {
    const static ge::Tensor::DeleteFunc kDoNothing = [](uint8_t* data) {};
    for (unsigned long i = 0; i < inputs.size(); ++i) {
      CALL_FUNC(inputs[i].ResetData(static_cast<uint8_t*>(inputs_data[i]),
                                    static_cast<size_t>(inputs_data_size[i]),
                                    kDoNothing));
    }
    for (unsigned long i = 0; i < outputs.size(); ++i) {
      CALL_FUNC(outputs[i].ResetData(static_cast<uint8_t*>(outputs_data[i]),
                                     static_cast<size_t>(outputs_data_size[i]),
                                     kDoNothing));
    }
  }

 private:
  std::vector<std::vector<int64_t>> get_shapes(
      const std::vector<Tensor>& tensors) const {
    std::vector<std::vector<int64_t>> shapes;
    shapes.reserve(tensors.size());
    for (const auto& tensor : tensors) {
      shapes.emplace_back(tensor.GetTensorDesc().GetShape().GetDims());
    }
    return shapes;
  }

  std::vector<int> get_dtypes(const std::vector<Tensor>& tensors) const {
    std::vector<int> dtypes;
    dtypes.reserve(tensors.size());
    for (const auto& tensor : tensors) {
      dtypes.emplace_back(
          static_cast<int>(tensor.GetTensorDesc().GetDataType()));
    }
    return dtypes;
  }

  int graph_id_;
  Graph graph_;
  std::string graph_key_;
  std::shared_ptr<CompiledGraphSummary> spec_;
  std::vector<Tensor> inputs;
  std::vector<Tensor> outputs;
  std::vector<TensorDesc> inputs_desc;
  std::vector<TensorDesc> outputs_desc;
  bool is_static;
};

#endif  // DICP_ASCEND_GE_GRAPH_H
