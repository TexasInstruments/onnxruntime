// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.


#include "core/providers/tidl/tidl_provider_factory.h"
#include "tidl_execution_provider.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/inference_session.h"

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

ORT_API_STATUS_IMPL(OrtSessionsOptionsSetDefault_Tidl, _In_ c_api_tidl_options * options_tidl_onnx) {

  memset(options_tidl_onnx, 0, sizeof(c_api_tidl_options));
  options_tidl_onnx->debug_level = 0;
  options_tidl_onnx->priority = 0;
  options_tidl_onnx->max_pre_empt_delay = FLT_MAX;
  options_tidl_onnx->core_number = 1;
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_Tidl, _In_ OrtSessionOptions* options, c_api_tidl_options * options_tidl_onnx) {
  TIDLProviderOptions options_tidl_onnx_vec;

  options_tidl_onnx_vec.push_back(std::make_pair("debug_level", std::to_string(options_tidl_onnx->debug_level)));
  options_tidl_onnx_vec.push_back(std::make_pair("priority", std::to_string(options_tidl_onnx->priority)));
  options_tidl_onnx_vec.push_back(std::make_pair("max_pre_empt_delay", std::to_string(options_tidl_onnx->max_pre_empt_delay)));
  options_tidl_onnx_vec.push_back(std::make_pair("artifacts_folder", std::string(options_tidl_onnx->artifacts_folder)));
  options_tidl_onnx_vec.push_back(std::make_pair("core_number", std::to_string(options_tidl_onnx->core_number)));

  options->provider_factories.push_back(onnxruntime::CreateExecutionProviderFactory_Tidl("", options_tidl_onnx_vec));
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtSessionGetTIBenchmarkData_Tidl, _In_ OrtSession* session, _Out_ c_api_tidl_benchmark_data * benchmark_data) {

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
