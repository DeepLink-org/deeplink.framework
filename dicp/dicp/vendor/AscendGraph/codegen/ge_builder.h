#ifndef DICP_ASCEND_GE_BUILDER_H
#define DICP_ASCEND_GE_BUILDER_H
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
#include "graph_utils.h"
#include "tensor.h"
#include "types.h"

#define FAILED -1
#define SUCCESS 0

using namespace ge;
using json = nlohmann::json;
using OperatorMap = std::unordered_map<std::string, ge::Operator>;

static std::unordered_set<std::string> op_with_dynamic_inputs_outputs = {
    "ConcatD", "IdentityN", "Pack", "SplitD", "SplitVD", "IncreFlashAttention"};

void check_op(std::unordered_map<std::string, ge::Operator>& op_map,
              const std::string& op_name) {
  if (op_map.count(op_name) > 0) {
    throw std::runtime_error("op_name duplicated: " + op_name);
  }
}

void setTensorData(Tensor& tensor, uint8_t* src_data, uint64_t data_size,
                   const std::string& debug_name = "") {
  auto status = tensor.SetData(reinterpret_cast<uint8_t*>(src_data), data_size);
  if (status != ge::GRAPH_SUCCESS) {
    std::cout << "Set " << debug_name << " tensor data failed!" << std::endl;
  }
}

ge::Tensor genTensor(const std::vector<int64_t>& tensor_shape,
                     ge::Format format, ge::DataType data_type) {
  TensorDesc desc(ge::Shape(tensor_shape), format, data_type);
  Tensor result(desc);
  return result;
}

template <typename T>
ge::Tensor genTensorWithData(const std::vector<int64_t>& tensor_shape,
                             ge::Format format, ge::DataType data_type,
                             std::vector<T> value) {
  TensorDesc desc(ge::Shape(tensor_shape), format, data_type);
  Tensor result(desc);
  setTensorData(result, reinterpret_cast<uint8_t*>(value.data()),
                value.size() * sizeof(T), "genTensorWithData");
  return result;
}

ge::Operator genInput(const std::string op_name,
                      const std::vector<int64_t> shape, ge::Format format,
                      ge::DataType data_type, int index = -1) {
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

class GEGraphBuilder {
 public:
  explicit GEGraphBuilder(const std::string& fusion_switch_file,
                          const std::string& ge_builder_config_file)
      : _fusion_switch_file(fusion_switch_file),
        _ge_builder_config_file(ge_builder_config_file) {
    // 1. system init
    std::map<AscendString, AscendString> global_options;

    auto kSocVersion = aclrtGetSocName();
    global_options[ge::ir_option::SOC_VERSION] = kSocVersion;
    global_options[ge::ir_option::FUSION_SWITCH_FILE] =
        _fusion_switch_file.c_str();

    auto raw_conf = parse_json_to_map(_ge_builder_config_file);
    for (const auto& item : raw_conf) {
      global_options[item.first.c_str()] = item.second.c_str();
    }
    CALL_FUNC(aclgrphBuildInitialize(global_options));
  }

  void saveGraph(const std::string& path, const Graph& graph,
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

  ~GEGraphBuilder() {
    aclgrphBuildFinalize();
    std::cout << "aclgrphBuildFinalize success!" << std::endl;
  }

 private:
  std::string _fusion_switch_file;
  std::string _ge_builder_config_file;
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
  std::string error_msg = "invalid ascend foramt! format: " + format;
  throw std::runtime_error(error_msg);
}

ge::DataType get_ascend_datatype(const std::string& data_type) {
  static std::unordered_map<std::string, ge::DataType> datatype_map = {
      {"FLOAT", ge::DataType::DT_FLOAT}, {"FLOAT16", ge::DataType::DT_FLOAT16},
      {"INT32", ge::DataType::DT_INT32}, {"INT64", ge::DataType::DT_INT64},
      {"BOOL", ge::DataType::DT_BOOL},   {"UINT8", ge::DataType::DT_UINT8},
      {"BF16", ge::DataType::DT_BF16},
  };
  if (datatype_map.count(data_type) > 0) {
    return datatype_map[data_type];
  }
  std::string error_msg = "invalid ascend data type! data type: " + data_type;
  throw std::runtime_error(error_msg);
}

template <typename T>
T genDynamicOp(const std::string& op_name) {
  return T(op_name.c_str());
}

template <typename T>
void parseDynamicInput(std::unordered_map<std::string, ge::Operator>& op_map,
                       T& op, const json& node) {
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
            op.set_dynamic_input_x(index, value,
                                   item["edge"].get<std::string>().c_str());
          } else {
            op.set_dynamic_input_x(index, value);
          }
        }
      } else {
        throw std::runtime_error("invalid dynamic input name: " + name);
      }
    }
  }
}

template <>
void parseDynamicInput(std::unordered_map<std::string, ge::Operator>& op_map,
                       op::IncreFlashAttention& op, const json& node) {
  if (node.contains("dynamic_inputs")) {
    int kv_inputs_num = 0;
    for (const auto& i : node["dynamic_inputs"]) {
      auto num = i["num"].get<unsigned int>();
      auto name = i["name"].get<std::string>();
      if (name == "key") {
        kv_inputs_num = static_cast<int>(num);
        op.create_dynamic_input_byindex_key(num, 1);
        for (const auto& item : i["value"]) {
          auto index = item["index"].get<uint32_t>();
          auto value = op_map[item["value"].get<std::string>()];
          op.set_dynamic_input_key(index, value);
        }
      } else if (name == "value") {
        if (kv_inputs_num == 0 && static_cast<int>(num) == kv_inputs_num) {
          throw std::runtime_error(
              "need first set dynamic key input for IncreFlashAttention Op"
              "and kv_inputs_num == num !!");
        }
        op.create_dynamic_input_byindex_value(num, 1 + num);
        for (const auto& item : i["value"]) {
          auto index = item["index"].get<uint32_t>();
          auto value = op_map[item["value"].get<std::string>()];
          op.set_dynamic_input_value(index, value);
        }
      } else {
        throw std::runtime_error("invalid dynamic input name: " + name);
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
        throw std::runtime_error("invalid dynamic output name: " + name);
      }
    }
  }
}

ge::Operator genDynamicOperator(OperatorMap& op_map, const json& node) {
  auto op_type = node["op_type"].get<std::string>();
  auto op_name = node["op_name"].get<std::string>();

  using OpHandler = std::function<ge::Operator(const std::string&, OperatorMap&,
                                               const json&)>;
  static const std::unordered_map<std::string, OpHandler> handlers = {
      {"ConcatD",
       [](const std::string& op_name, OperatorMap& op_map, const json& node) {
         auto op = genDynamicOp<op::ConcatD>(op_name);
         parseDynamicInput<op::ConcatD>(op_map, op, node);
         return op;
       }},
      {"IdentityN",
       [](const std::string& op_name, OperatorMap& op_map, const json& node) {
         auto op = genDynamicOp<op::IdentityN>(op_name);
         parseDynamicInput<op::IdentityN>(op_map, op, node);
         parseDynamicOutput(op, node);
         return op;
       }},
      {"Pack",
       [](const std::string& op_name, OperatorMap& op_map, const json& node) {
         auto op = genDynamicOp<op::Pack>(op_name);
         parseDynamicInput<op::Pack>(op_map, op, node);
         return op;
       }},
      {"IncreFlashAttention",
       [](const std::string& op_name, OperatorMap& op_map, const json& node) {
         auto op = genDynamicOp<op::IncreFlashAttention>(op_name);
         parseDynamicInput<op::IncreFlashAttention>(op_map, op, node);
         return op;
       }},
      {"SplitD",
       [](const std::string& op_name, OperatorMap& op_map, const json& node) {
         auto op = genDynamicOp<op::SplitD>(op_name);
         parseDynamicOutput(op, node);
         return op;
       }},
      {"SplitVD",
       [](const std::string& op_name, OperatorMap& op_map, const json& node) {
         auto op = genDynamicOp<op::SplitVD>(op_name);
         parseDynamicOutput(op, node);
         return op;
       }}};

  auto it = handlers.find(op_type);
  if (it != handlers.end()) {
    return it->second(op_name, op_map, node);
  } else {
    throw std::runtime_error("invalid dynamic operator: " + op_type);
  }
}

template <typename T>
T getValue(const json& node, const std::string& key) {
  try {
    return node.at(key).get<T>();
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    std::cerr << "JSON Node: " << node.dump(4) << std::endl;
    throw std::runtime_error("getValue failed!");
  }
}

TensorDesc getTensorDescFromJson(const json& desc) {
  auto format = getValue<std::string>(desc, "format");
  auto data_type = getValue<std::string>(desc, "data_type");
  auto shape = getValue<std::vector<int64_t>>(desc, "shape");
  TensorDesc tensor_desc(ge::Shape(shape), get_ascend_format(format),
                         get_ascend_datatype(data_type));
  return tensor_desc;
}

void parseInputs(std::unordered_map<std::string, ge::Operator>& op_map,
                 ge::Operator& op, const json& inputs) {
  for (const auto& i : inputs) {
    auto name = getValue<std::string>(i, "name").c_str();
    auto value = op_map[getValue<std::string>(i, "value")];

    if (i.contains("index")) {
      op.SetInput(name, value, getValue<int>(i, "index"));
    } else if (i.contains("update_desc")) {
      auto desc = i["update_desc"];
      auto tensor_desc = getTensorDescFromJson(desc);
      auto output_name = getValue<std::string>(desc, "output_name");
      if (output_name != "none") {
        op_map[getValue<std::string>(i, "value")].UpdateOutputDesc(
            output_name.c_str(), tensor_desc);
      } else {
        op.UpdateInputDesc(name, tensor_desc);
      }
      op.SetInput(name, value);
    } else {
      op.SetInput(name, value);
    }
  }
}

void parseOutputs(ge::Operator& op, const json& outputs) {
  for (const auto& i : outputs) {
    auto name = getValue<std::string>(i, "output_name").c_str();
    auto tensor_desc = getTensorDescFromJson(i["update_desc"]);
    op.UpdateOutputDesc(name, tensor_desc);
  }
}

template <typename T>
void setTensorAttrHelper(ge::Operator& op, const std::string& attr_name,
                         const json& attr, const std::vector<int64_t>& dims,
                         const ge::Format& format,
                         const ge::DataType& data_type) {
  auto value = getValue<std::vector<T>>(attr, "tensor_value");
  auto tensor = genTensorWithData<T>(dims, format, data_type, value);
  op.SetAttr(attr_name.c_str(), tensor);
}

void setTensorAttr(ge::Operator& op, const std::string& attr_name,
                   const json& attr) {
  auto cpp_data_type = getValue<std::string>(attr, "tensor_cpp_data_type");
  auto data_type =
      get_ascend_datatype(getValue<std::string>(attr, "tensor_data_type"));
  auto format = get_ascend_format(getValue<std::string>(attr, "tensor_format"));
  auto dims = getValue<std::vector<int64_t>>(attr, "tensor_dims");

  if (cpp_data_type == "FLOAT") {
    setTensorAttrHelper<float>(op, attr_name, attr, dims, format, data_type);
  } else if (cpp_data_type == "FLOAT16") {
    auto values = getValue<std::vector<float>>(attr, "tensor_value");
    std::vector<half_float::half> half_values(values.begin(), values.end());
    auto tensor = genTensorWithData<half_float::half>(dims, format, data_type,
                                                      half_values);
    op.SetAttr(attr_name.c_str(), tensor);
  } else if (cpp_data_type == "INT32") {
    setTensorAttrHelper<int>(op, attr_name, attr, dims, format, data_type);
  } else if (cpp_data_type == "INT64") {
    setTensorAttrHelper<int64_t>(op, attr_name, attr, dims, format, data_type);
  } else {
    throw std::runtime_error("invalid cpp data type: " + cpp_data_type);
  }
}

void parseAttrs(ge::Operator& op, const json& attrs) {
  using AttrHandler =
      std::function<void(ge::Operator&, const std::string&, const json&)>;
  static const std::unordered_map<std::string, AttrHandler> handlers = {
      {"str",
       [](ge::Operator& op, const std::string& name, const json& attr) {
         op.SetAttr(name.c_str(), getValue<std::string>(attr, "value"));
       }},
      {"dtype_str",
       [](ge::Operator& op, const std::string& name, const json& attr) {
         auto value = getValue<std::string>(attr, "value");
         op.SetAttr(name.c_str(), get_ascend_datatype(value));
       }},
      {"list_int",
       [](ge::Operator& op, const std::string& name, const json& attr) {
         op.SetAttr(name.c_str(),
                    getValue<std::vector<int64_t>>(attr, "value"));
       }},
      {"list_float",
       [](ge::Operator& op, const std::string& name, const json& attr) {
         op.SetAttr(name.c_str(), getValue<std::vector<float>>(attr, "value"));
       }},
      {"float",
       [](ge::Operator& op, const std::string& name, const json& attr) {
         op.SetAttr(name.c_str(), getValue<float>(attr, "value"));
       }},
      {"int",
       [](ge::Operator& op, const std::string& name, const json& attr) {
         op.SetAttr(name.c_str(), getValue<int>(attr, "value"));
       }},
      {"bool",
       [](ge::Operator& op, const std::string& name, const json& attr) {
         op.SetAttr(name.c_str(), getValue<bool>(attr, "value"));
       }},
      {"int64_t",
       [](ge::Operator& op, const std::string& name, const json& attr) {
         op.SetAttr(name.c_str(), getValue<int64_t>(attr, "value"));
       }},
      {"tensor", [](ge::Operator& op, const std::string& name,
                    const json& attr) { setTensorAttr(op, name, attr); }}};

  for (const auto& attr : attrs) {
    auto attr_name = getValue<std::string>(attr, "name");
    auto value_type = getValue<std::string>(attr, "value_type");

    auto it = handlers.find(value_type);
    if (it != handlers.end()) {
      it->second(op, attr_name, attr);
    } else {
      throw std::runtime_error("Invalid attr value type: " + value_type);
    }
  }
}

void parseCommonNode(std::unordered_map<std::string, ge::Operator>& op_map,
                     ge::Operator& op, const json& node) {
  if (node.contains("inputs")) {
    parseInputs(op_map, op, node["inputs"]);
  }
  if (node.contains("outputs")) {
    parseOutputs(op, node["outputs"]);
  }
  if (node.contains("attrs")) {
    parseAttrs(op, node["attrs"]);
  }
}

void buildGraph(Graph& graph, const json& graph_json,
                std::vector<Tensor>& input_tensors) {
  std::unordered_map<std::string, ge::Operator> op_map;
  json data_nodes = graph_json["data_nodes"];
  for (const auto& node : graph_json["data_nodes"]) {
    auto node_name = getValue<std::string>(node, "op_name");
    auto format = get_ascend_format(getValue<std::string>(node, "format"));
    auto data_type =
        get_ascend_datatype(getValue<std::string>(node, "data_type"));
    auto index = getValue<int>(node, "index");
    auto dims = getValue<std::vector<int64_t>>(node, "dims");
    check_op(op_map, node_name);
    op_map[node_name] = genInput(node_name, dims, format, data_type, index);
    graph.AddOp(op_map[node_name]);

    // add tensor to inputs
    TensorDesc cur_desc(ge::Shape(dims), format, data_type);
    Tensor cur_tensor(cur_desc);
    input_tensors.emplace_back(cur_tensor);
  }
  for (const auto& node : graph_json["common_nodes"]) {
    auto node_name = getValue<std::string>(node, "op_name");
    auto op_type = getValue<std::string>(node, "op_type");

    check_op(op_map, node_name);
    if (op_with_dynamic_inputs_outputs.count(op_type) > 0) {
      op_map[node_name] = genDynamicOperator(op_map, node);
    } else {
      op_map[node_name] = ge::OperatorFactory::CreateOperator(node_name.c_str(),
                                                              op_type.c_str());
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

#endif  // DICP_ASCEND_GE_BUILDER_H
