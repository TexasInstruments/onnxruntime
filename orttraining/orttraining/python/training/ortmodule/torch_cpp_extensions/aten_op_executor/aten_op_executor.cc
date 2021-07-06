// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <torch/torch.h>
#include <ATen/DLConvertor.h>
#include <unordered_map>
#include <vector>

template <typename T>
c10::IValue ToIValue(const DLManagedTensor* dlpack, bool is_optional) {
  TORCH_INTERNAL_ASSERT((dlpack->dl_tensor.ndim == 0 && dlpack->dl_tensor.shape == nullptr) ||
                        (dlpack->dl_tensor.ndim == 1 && dlpack->dl_tensor.shape[0] == 1));
  T value = *reinterpret_cast<const T*>(dlpack->dl_tensor.data);
  return is_optional ? c10::IValue(c10::optional<T>(value)) : c10::IValue(value);
}

template <typename T>
c10::IValue ToListIValue(const DLManagedTensor* dlpack, bool is_optional) {
  TORCH_INTERNAL_ASSERT(dlpack->dl_tensor.ndim == 1);
  const T* p_data = reinterpret_cast<const T*>(dlpack->dl_tensor.data);
  c10::List<T> list_value;
  for (size_t i = 0; i < dlpack->dl_tensor.shape[0]; i++) {
    list_value.emplace_back(p_data[i]);
  }
  return is_optional ? c10::IValue(c10::optional<c10::List<T>>(list_value)) : c10::IValue(list_value);
}

struct ATenOperator {
  std::shared_ptr<torch::jit::Operator> op;
  size_t argument_size;
  std::vector<c10::TypeKind> elem_kinds;
  std::vector<bool> is_list_arguments;
  std::vector<bool> is_optional_arguments;
  std::vector<c10::optional<c10::IValue>> default_values;
  size_t return_size;

  c10::IValue ToIValueArgument(const DLManagedTensor* dlpack, size_t index) const {
    TORCH_INTERNAL_ASSERT(index < argument_size);
    bool is_optional = is_optional_arguments[index];
    TORCH_INTERNAL_ASSERT(dlpack || is_optional || default_values[index]);
    if (!dlpack) {
      if (is_optional) {
        return c10::IValue(c10::nullopt);
      }

      return *default_values[index];
    }

    bool is_list = is_list_arguments[index];
    c10::IValue i_value;
    switch (elem_kinds[index]) {
      case c10::TypeKind::TensorType: {
        at::Tensor tensor = at::fromDLPack(dlpack);
        i_value = is_optional ? c10::IValue(c10::optional<at::Tensor>(tensor)) : c10::IValue(tensor);
      } break;
      case c10::TypeKind::IntType: {
        i_value = is_list ? ToListIValue<int64_t>(dlpack, is_optional) : ToIValue<int64_t>(dlpack, is_optional);
      } break;
      case c10::TypeKind::FloatType: {
        i_value = is_list ? ToListIValue<float>(dlpack, is_optional) : ToIValue<float>(dlpack, is_optional);
      } break;
      case c10::TypeKind::BoolType: {
        i_value = is_list ? ToListIValue<bool>(dlpack, is_optional) : ToIValue<bool>(dlpack, is_optional);
      } break;
      default:
        TORCH_INTERNAL_ASSERT(false);
    }

    return i_value;
  }
};

class ATenOperatorCache {
 public:
  static ATenOperatorCache& Instance() {
    static ATenOperatorCache instance;
    return instance;
  }

  const ATenOperator& GetOperator(const std::string& op_name) {
    if (ops_.find(op_name) == ops_.end()) {
      // Some op name can get multiple ops with different overload names,
      // we are using the one with empty overload name.
      c10::OperatorName full_name(op_name, "");
      auto op = torch::jit::findOperatorFor(full_name);
      TORCH_INTERNAL_ASSERT(op);
      ATenOperator aten_op;
      aten_op.op = op;
      const auto& schema = aten_op.op->schema();
      aten_op.argument_size = schema.arguments().size();
      for (const auto& argument : schema.arguments()) {
        c10::TypePtr type = argument.type();
        c10::TypeKind elem_type = type->kind();
        bool is_optional = elem_type == c10::TypeKind::OptionalType;
        bool is_list = elem_type == c10::TypeKind::ListType;
        if (is_optional) {
          type = reinterpret_cast<c10::OptionalType*>(type.get())->getElementType();
          elem_type = type->kind();
          is_list = elem_type == c10::TypeKind::ListType;
        }
        if (is_list) {
          elem_type = reinterpret_cast<c10::ListType*>(type.get())->getElementType()->kind();
        }
        TORCH_INTERNAL_ASSERT(elem_type != c10::TypeKind::TensorType || !is_list);
        aten_op.elem_kinds.emplace_back(elem_type);
        aten_op.is_list_arguments.emplace_back(is_list);
        aten_op.is_optional_arguments.emplace_back(is_optional);
        aten_op.default_values.emplace_back(argument.default_value());
      }
      aten_op.return_size = schema.returns().size();
      for (const auto& ret : schema.returns()) {
        TORCH_INTERNAL_ASSERT(ret.type()->kind() == c10::TypeKind::TensorType);
      }
      ops_[op_name] = aten_op;
    }
    return ops_.at(op_name);
  }

 private:
  ATenOperatorCache() = default;
  std::unordered_map<std::string, ATenOperator> ops_;
};

bool IsTensorArgument(const char* op_name, size_t index) {
  const auto& aten_op = ATenOperatorCache::Instance().GetOperator(op_name);
  TORCH_INTERNAL_ASSERT(index < aten_op.argument_size);
  return aten_op.elem_kinds[index] == c10::TypeKind::TensorType;
}

std::vector<DLManagedTensor*> ExecuteATenOperator(const char* op_name, const std::vector<DLManagedTensor*>& dlpacks) {
  const auto& aten_op = ATenOperatorCache::Instance().GetOperator(op_name);
  TORCH_INTERNAL_ASSERT(dlpacks.size() == aten_op.argument_size);
  std::vector<c10::IValue> arguments;
  for (size_t i = 0; i < dlpacks.size(); i++) {
    arguments.emplace_back(aten_op.ToIValueArgument(dlpacks[i], i));
  }

  torch::jit::Stack stack;
  for (size_t i = 0; i < arguments.size(); i++) {
    torch::jit::push(stack, arguments[i]);
  }

  aten_op.op->getOperation()(&stack);
  std::vector<DLManagedTensor*> result;
  for (const auto& ret : torch::jit::pop(stack, aten_op.return_size)) {
    const auto& tensor = ret.toTensor();
    result.emplace_back(at::toDLPack(tensor.is_contiguous() ? tensor : tensor.contiguous()));
  }

  return result;
}

size_t is_tensor_argument_address() { return reinterpret_cast<size_t>(&IsTensorArgument); }
size_t execute_aten_operator_address() { return reinterpret_cast<size_t>(&ExecuteATenOperator); }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("is_tensor_argument_address", &is_tensor_argument_address, "Address of tensor argument check.");
  m.def("execute_aten_operator_address", &execute_aten_operator_address, "Address of Aten operator executor");
}
