#ifndef DAVINCI_GRAPH_UTILS_H
#define DAVINCI_GRAPH_UTILS_H
#include <cctype>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <numeric>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <json.hpp>

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

#define FAILED -1
#define SUCCESS 0

using json = nlohmann::json;
using namespace ge;

static std::unordered_set<std::string> op_with_dynamic_inputs_outputs = {
    "ConcatD",
    "IdentityN",
    "Pack",
    "SplitD"};

void check_op(
    std::unordered_map<std::string, ge::Operator>& op_map,
    const std::string& op_name) {
  if (op_map.count(op_name) > 0) {
    throw std::runtime_error("op_name duplicated!");
  }
}

void setTensorData(
    Tensor& tensor,
    uint8_t* src_data,
    uint64_t data_size,
    const std::string& debug_name = "") {
  auto status = tensor.SetData(reinterpret_cast<uint8_t*>(src_data), data_size);
  if (status != ge::GRAPH_SUCCESS) {
    std::cout << "Set " << debug_name << " tensor data failed!" << std::endl;
  }
}

ge::Tensor genTensor(
    const std::vector<int64_t>& tensor_shape,
    ge::Format format,
    ge::DataType data_type) {
  TensorDesc desc(ge::Shape(tensor_shape), format, data_type);
  Tensor result(desc);
  return result;
}

template <typename T>
ge::Tensor genTensorWithData(
    const std::vector<int64_t>& tensor_shape,
    ge::Format format,
    ge::DataType data_type,
    std::vector<T> value) {
  TensorDesc desc(ge::Shape(tensor_shape), format, data_type);
  Tensor result(desc);
  setTensorData(
      result,
      reinterpret_cast<uint8_t*>(value.data()),
      value.size() * sizeof(T),
      "genTensorWithData");
  return result;
}

ge::Operator genInput(
    const std::string op_name,
    const std::vector<int64_t> shape,
    ge::Format format,
    ge::DataType data_type,
    int index = -1) {
  TensorDesc tensor_desc_data_op =
      TensorDesc(ge::Shape(shape), format, data_type);
  auto op = op::Data(op_name.c_str());
  op.update_input_desc_x(tensor_desc_data_op);
  op.update_output_desc_y(tensor_desc_data_op);
  if (index > -1) {
    op.set_attr_index(index);
  }
  return op;
}

class AclgraphBuilder {
 public:
  explicit AclgraphBuilder() {
    // 1. system init
    std::string kSocVersion = "Ascend910ProB";
    std::map<AscendString, AscendString> global_options = {
        {AscendString(ge::ir_option::SOC_VERSION),
         AscendString(kSocVersion.c_str())},
        {AscendString(ge::ir_option::PRECISION_MODE), "allow_fp32_to_fp16"},
    };
    auto status = aclgrphBuildInitialize(global_options);
    if (status != GRAPH_SUCCESS) {
      std::cout << "aclgrphBuildInitialize failed!" << std::endl;
    } else {
      std::cout << "aclgrphBuildInitialize success!" << std::endl;
    }
  }

  void saveGraph(
      const std::string& path,
      const Graph& graph,
      std::map<AscendString, AscendString>& options) {
    ModelBufferData model;

    auto status = aclgrphBuildModel(graph, options, model);
    if (status == GRAPH_SUCCESS) {
      std::cout << "Build Model SUCCESS!" << std::endl;
    } else {
      std::cout << "Build Model Failed! " << status << std::endl;
      return;
    }

    // 4. Save Ir Model
    status = aclgrphSaveModel(path.c_str(), model);
    if (status == GRAPH_SUCCESS) {
      std::cout << "Save Offline Model SUCCESS!" << std::endl;
    } else {
      std::cout << "Save Offline Model Failed! " << status << std::endl;
    }
  }

  ~AclgraphBuilder() {
    aclgrphBuildFinalize();
    std::cout << "aclgrphBuildFinalize success!" << std::endl;
  }
};

ge::Format get_ascend_format(const std::string& format) {
  static std::unordered_map<std::string, ge::Format> format_map = {
      {"NCHW", FORMAT_NCHW},
      {"NHWC", FORMAT_NHWC},
      {"ND", FORMAT_ND},
      {"FRACTAL_NZ", FORMAT_FRACTAL_NZ},
  };
  if (format_map.count(format) > 0) {
    return format_map[format];
  }
  throw std::runtime_error("invalid ascend foramt!");
}

ge::DataType get_ascend_datatype(const std::string& data_type) {
  static std::unordered_map<std::string, ge::DataType> datatype_map = {
      {"FLOAT", ge::DataType::DT_FLOAT},
      {"FLOAT16", ge::DataType::DT_FLOAT16},
      {"INT32", ge::DataType::DT_INT32},
      {"INT64", ge::DataType::DT_INT64},
      {"BOOL", ge::DataType::DT_BOOL},
  };
  if (datatype_map.count(data_type) > 0) {
    return datatype_map[data_type];
  }
  throw std::runtime_error("invalid ascend data type!");
}

template <typename T>
T genDynamicOp(const std::string& op_name) {
  return T(op_name.c_str());
}

template <typename T>
void parseDynamicInput(
    std::unordered_map<std::string, ge::Operator>& op_map,
    T& op,
    const json& node) {
  if (node.contains("dynamic_inputs")) {
    for (const auto& i : node["dynamic_inputs"]) {
      auto num = i["num"].get<unsigned int>();
      auto name = i["name"].get<std::string>();
      if (name == "x") {
        op.create_dynamic_input_x(num);
        for (const auto& item : i["value"]) {
          auto index = item["index"].get<uint32_t>();
          auto value = op_map[item["value"].get<std::string>()];
          if (item.contains("edge")) {
            op.set_dynamic_input_x(
                index, value, item["edge"].get<std::string>().c_str());
          } else {
            op.set_dynamic_input_x(index, value);
          }
        }
      } else {
        throw std::runtime_error("invalid dynamic input name");
      }
    }
  }
}

template <typename T>
void parseDynamicOutput(T& op, const json& node) {
  if (node.contains("dynamic_outputs")) {
    for (const auto& o : node["dynamic_outputs"]) {
      auto name = o["name"].get<std::string>();
      auto num = o["num"].get<uint32_t>();
      if (name == "y") {
        op.create_dynamic_output_y(num);
      } else {
        throw std::runtime_error("invalid dynamic output name");
      }
    }
  }
}

ge::Operator genDynamicOperator(
    std::unordered_map<std::string, ge::Operator>& op_map,
    const json& node) {
  auto op_type = node["op_type"].get<std::string>();
  auto op_name = node["op_name"].get<std::string>();
  if (op_type == "ConcatD") {
    auto op = genDynamicOp<op::ConcatD>(op_name);
    parseDynamicInput<op::ConcatD>(op_map, op, node);
    return op;
  } else if (op_type == "IdentityN") {
    auto op = genDynamicOp<op::IdentityN>(op_name);
    parseDynamicInput<op::IdentityN>(op_map, op, node);
    parseDynamicOutput(op, node);
    return op;
  } else if (op_type == "Pack") {
    auto op = genDynamicOp<op::Pack>(op_name);
    parseDynamicInput<op::Pack>(op_map, op, node);
    return op;
  } else if (op_type == "SplitD") {
    auto op = genDynamicOp<op::SplitD>(op_name);
    parseDynamicOutput(op, node);
    return op;
  }
  throw std::runtime_error("invalid dynamic opeartor!");
}

void parseCommonNode(
    std::unordered_map<std::string, ge::Operator>& op_map,
    ge::Operator& op,
    const json& node) {
  if (node.contains("inputs")) {
    for (const auto& i : node["inputs"]) {
      auto name = i["name"].get<std::string>().c_str();
      auto value = op_map[i["value"].get<std::string>()];
      if (i.contains("index")) {
        op.SetInput(name, value, i["index"].get<int>());
      } else if (i.contains("update_desc")) {
        auto desc = i["update_desc"];
        auto format = desc["format"].get<std::string>();
        auto data_type = desc["data_type"].get<std::string>();
        auto shape = desc["shape"].get<std::vector<int64_t>>();
        TensorDesc tensor_desc = TensorDesc(
            ge::Shape(shape),
            get_ascend_format(format),
            get_ascend_datatype(data_type));
        auto output_name = desc["output_name"].get<std::string>();
        op_map[i["value"].get<std::string>()].UpdateOutputDesc(
                      output_name.c_str(), tensor_desc);
        op.SetInput(name, value);
      } else {
        op.SetInput(name, value);
      }
    }
  }
  if (node.contains("attrs")) {
    for (const auto& attr : node["attrs"]) {
      auto attr_name = attr["name"].get<std::string>();
      auto value_type = attr["value_type"];
      if (value_type == "str") {
        op.SetAttr(attr_name, attr["value"].get<std::string>());
      } else if (value_type == "list_int") {
        auto value = attr["value"].get<std::vector<int64_t>>();
        op.SetAttr(attr_name.c_str(), value);
      } else if (value_type == "list_float") {
        auto value = attr["value"].get<std::vector<float>>();
        op.SetAttr(attr_name.c_str(), value);
      } else if (value_type == "float") {
        auto value = attr["value"].get<float>();
        op.SetAttr(attr_name.c_str(), value);
      } else if (value_type == "int") {
        auto value = attr["value"].get<int>();
        op.SetAttr(attr_name.c_str(), value);
      } else if (value_type == "bool") {
        auto value = attr["value"].get<bool>();
        op.SetAttr(attr_name.c_str(), value);
      } else if (value_type == "int64") {
        auto value = attr["value"].get<int64_t>();
        op.SetAttr(attr_name.c_str(), value);
      } else if (value_type == "tensor") {
        auto cpp_data_type = attr["tensor_cpp_data_type"].get<std::string>();
        auto data_type =
            get_ascend_datatype(attr["tensor_data_type"].get<std::string>());
        auto format =
            get_ascend_format(attr["tensor_format"].get<std::string>());
        auto tensor_dims = attr["tensor_dims"];
        auto dims = tensor_dims.get<std::vector<int64_t>>();
        if (cpp_data_type == "FLOAT") {
          auto value = attr["tensor_value"].get<std::vector<float>>();
          auto tensor =
              genTensorWithData<float>(dims, format, data_type, value);
          op.SetAttr(attr_name.c_str(), tensor);
        } else if (cpp_data_type == "INT32") {
          auto value = attr["tensor_value"].get<std::vector<int>>();
          auto tensor = genTensorWithData<int>(dims, format, data_type, value);
          op.SetAttr(attr_name.c_str(), tensor);
        } else if (cpp_data_type == "INT64") {
          auto value = attr["tensor_value"].get<std::vector<int64_t>>();
          auto tensor =
              genTensorWithData<int64_t>(dims, format, data_type, value);
          op.SetAttr(attr_name.c_str(), tensor);
        } else {
          throw std::runtime_error("invalid cpp data type!");
        }
      } else {
        throw std::runtime_error("invalid attr value type!");
      }
    }
  }
}

void buildGraph(Graph& graph, const json& graph_json) {
  std::unordered_map<std::string, ge::Operator> op_map;
  json data_nodes = graph_json["data_nodes"];
  for (const auto& node : graph_json["data_nodes"]) {
    auto node_name = node["op_name"].get<std::string>();
    auto format = get_ascend_format(node["format"].get<std::string>());
    auto data_type = get_ascend_datatype(node["data_type"].get<std::string>());
    auto index = node["index"].get<int>();
    auto dims = node["dims"].get<std::vector<int64_t>>();
    check_op(op_map, node_name);
    op_map[node_name] = genInput(node_name, dims, format, data_type, index);
    graph.AddOp(op_map[node_name]);
  }
  for (const auto& node : graph_json["common_nodes"]) {
    auto node_name = node["op_name"].get<std::string>();
    auto op_type = node["op_type"].get<std::string>();

    check_op(op_map, node_name);
    if (op_with_dynamic_inputs_outputs.count(op_type) > 0) {
      op_map[node_name] = genDynamicOperator(op_map, node);
    } else {
      op_map[node_name] = ge::OperatorFactory::CreateOperator(
          node_name.c_str(), op_type.c_str());
    }
    parseCommonNode(op_map, op_map[node_name], node);
    graph.AddOp(op_map[node_name]);
  }
  std::vector<ge::Operator> graph_inputs;
  std::vector<ge::Operator> graph_outputs;
  for (const auto& i : graph_json["input_names"]) {
    graph_inputs.push_back(op_map[i.get<std::string>()]);
  }
  for (const auto& i : graph_json["output_names"]) {
    graph_outputs.push_back(op_map[i.get<std::string>()]);
  }
  graph.SetInputs(graph_inputs).SetOutputs(graph_outputs);
}

#endif // DAVINCI_GRAPH_UTILS_H
