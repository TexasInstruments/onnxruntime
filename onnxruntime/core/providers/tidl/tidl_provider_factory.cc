// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.


#include "core/providers/tidl/tidl_provider_factory.h"
#include "tidl_execution_provider.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/inference_session.h"
#include "core/session/ort_apis.h"
#include <string.h>
#include <float.h>

using namespace onnxruntime;

namespace onnxruntime {

struct TidlProviderFactory : IExecutionProviderFactory {
  TidlProviderFactory(const std::string& type, const TIDLProviderOptions& options_tidl_onnx_vec)
      : options_tidl_onnx_vec_(options_tidl_onnx_vec), type_(type) {}
  ~TidlProviderFactory() override {}

  std::unique_ptr<IExecutionProvider> CreateProvider() override;

  private:
  TIDLProviderOptions options_tidl_onnx_vec_;
  std::string type_;
};

std::unique_ptr<IExecutionProvider> TidlProviderFactory::CreateProvider() {
  TidlExecutionProviderInfo info(type_, options_tidl_onnx_vec_);
  return std::make_unique<TidlExecutionProvider>(info);
}

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Tidl(const std::string &type, const TIDLProviderOptions& options_tidl_onnx_vec) {
  return std::make_shared<onnxruntime::TidlProviderFactory>(type, options_tidl_onnx_vec);
}

}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsInitialize_Tidl, _Inout_ c_api_tidl_options* options) {
  if (options == nullptr)
  {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Invalid argument: options is null");
  }

  // Initialize the options structure
  memset(options, 0, sizeof(c_api_tidl_options));
  options->count = 0;

  return nullptr;
}

ORT_API_STATUS_IMPL(OrtSessionOptionsSet_Tidl, _Inout_ c_api_tidl_options* options, _In_ const char* key, _In_ const char* value) {
  if (options == nullptr || key == nullptr || value == nullptr)
  {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Invalid arguments: options, key, or value is null");
  }

  // Check if the key already exists, if so update it
  for (int i = 0; i < options->count; i++)
  {
    if (strcmp(options->option[i].key, key) == 0)
    {
      strncpy(options->option[i].value, value, TIDL_MAX_STRING_LENGTH - 1);
      options->option[i].value[TIDL_MAX_STRING_LENGTH - 1] = '\0';
      return nullptr;
    }
  }

  // If key doesn't exist, add it if there's space
  if (options->count < TIDL_MAX_OPTIONS)
  {
    strncpy(options->option[options->count].key, key, TIDL_MAX_STRING_LENGTH - 1);
    options->option[options->count].key[TIDL_MAX_STRING_LENGTH - 1] = '\0';

    strncpy(options->option[options->count].value, value, TIDL_MAX_STRING_LENGTH - 1);
    options->option[options->count].value[TIDL_MAX_STRING_LENGTH - 1] = '\0';

    options->count++;
    return nullptr;
  }

  return OrtApis::CreateStatus(ORT_FAIL, "Maximum number of options exceeded");
}

ORT_API_STATUS_IMPL(OrtSessionOptionsGet_Tidl, _In_ const c_api_tidl_options* options, _In_ const char* key,
                    _Out_writes_bytes_all_(value_len) char* value, _In_ size_t value_len) {
  if (options == nullptr || key == nullptr || value == nullptr)
  {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Invalid arguments: options, key, or value is null");
  }

  for (int i = 0; i < options->count; i++)
  {
    if (strcmp(options->option[i].key, key) == 0)
    {
      strncpy(value, options->option[i].value, value_len - 1);
      value[value_len - 1] = '\0';
      return nullptr;
    }
  }

  return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Option not found");
}

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_Tidl, _In_ OrtSessionOptions* options, _In_ const c_api_tidl_options* tidl_options) {
  if (options == nullptr || tidl_options == nullptr)
  {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Invalid arguments: options or tidl_options is null");
  }

  TIDLProviderOptions options_tidl_onnx_vec;

  for (int i = 0; i < tidl_options->count; i++)
  {
    options_tidl_onnx_vec.push_back(std::make_pair(
      std::string(tidl_options->option[i].key),
      std::string(tidl_options->option[i].value)
    ));
  }

  options->provider_factories.push_back(onnxruntime::CreateExecutionProviderFactory_Tidl("", options_tidl_onnx_vec));
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtSessionGetTIBenchmarkData_Tidl, _In_ OrtSession* session, _Out_ c_api_tidl_benchmark_data* benchmark_data) {
  int32_t i = 0;
  std::vector<std::pair<std::string, uint64_t>> data = {};
  auto inference_session = reinterpret_cast<::onnxruntime::InferenceSession*>(session);
  data = inference_session->get_TI_benchmark_data();

  benchmark_data->run_start = data[i++].second;
  benchmark_data->run_end = data[i++].second;
  benchmark_data->ddr_read_start = data[i++].second;
  benchmark_data->ddr_read_end = data[i++].second;
  benchmark_data->ddr_write_start = data[i++].second;
  benchmark_data->ddr_write_end = data[i++].second;

  benchmark_data->num_subgraph_data = 0;

  while (i < data.size())
  {
    benchmark_data->copy_in_start[benchmark_data->num_subgraph_data] = data[i++].second;
    benchmark_data->copy_in_end[benchmark_data->num_subgraph_data] = data[i++].second;
    benchmark_data->proc_start[benchmark_data->num_subgraph_data] = data[i++].second;
    benchmark_data->proc_end[benchmark_data->num_subgraph_data] = data[i++].second;
    benchmark_data->copy_out_start[benchmark_data->num_subgraph_data] = data[i++].second;
    benchmark_data->copy_out_end[benchmark_data->num_subgraph_data] = data[i++].second;
    benchmark_data->num_subgraph_data++;
  }

  return nullptr;
}

ORT_API_STATUS_IMPL(OrtSessionDisableIOValidation_Tidl, _In_ OrtSession* session) {

  auto inference_session = reinterpret_cast<::onnxruntime::InferenceSession*>(session);
  inference_session->disableValidateInputs();
  inference_session->disableValidateOutputs();
  return nullptr;
}
