// Copyright 2019 JD.com Inc. JD AI

#include "core/providers/tidl/tidl_provider_factory.h"
#include "tidl_execution_provider.h"
#include "core/session/abi_session_options_impl.h"

using namespace onnxruntime;

namespace onnxruntime {

struct TidlProviderFactory : IExecutionProviderFactory {
  TidlProviderFactory(std::vector<std::string> options_tidl_onnx_vec)
      : options_tidl_onnx_vec_(options_tidl_onnx_vec) {}
  ~TidlProviderFactory() override {}

  std::unique_ptr<IExecutionProvider> CreateProvider() override;

  private:
  std::vector<std::string> options_tidl_onnx_vec_;
};

std::unique_ptr<IExecutionProvider> TidlProviderFactory::CreateProvider() {
  //return onnxruntime::make_unique<TidlExecutionProvider>();
  TidlExecutionProviderInfo info(options_tidl_onnx_vec_);
  return onnxruntime::make_unique<TidlExecutionProvider>(info);
}

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Tidl(std::vector<std::string> options_tidl_onnx_vec) {
  return std::make_shared<onnxruntime::TidlProviderFactory>(options_tidl_onnx_vec);
}
}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_Tidl, _In_ OrtSessionOptions* options, c_api_tidl_options * options_tidl_onnx) {
  std::vector<std::string> options_tidl_onnx_vec;
  
  options_tidl_onnx_vec.push_back("import");
  std::string import(options_tidl_onnx->import);
  options_tidl_onnx_vec.push_back(import);
  
  options_tidl_onnx_vec.push_back("debug_level");
  options_tidl_onnx_vec.push_back(std::to_string(options_tidl_onnx->debug_level));

  options_tidl_onnx_vec.push_back("tidl_tensor_bits");
  options_tidl_onnx_vec.push_back(std::to_string(options_tidl_onnx->tidl_tensor_bits));

  options_tidl_onnx_vec.push_back("tidl_tools_path");
  std::string tidl_tools_path(options_tidl_onnx->tidl_tools_path);
  options_tidl_onnx_vec.push_back(tidl_tools_path);
  
  options_tidl_onnx_vec.push_back("artifacts_folder");
  std::string artifacts_folder(options_tidl_onnx->artifacts_folder);
  options_tidl_onnx_vec.push_back(artifacts_folder);

  options->provider_factories.push_back(onnxruntime::CreateExecutionProviderFactory_Tidl(options_tidl_onnx_vec));
  return nullptr;
}
