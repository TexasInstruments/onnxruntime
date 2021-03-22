// Copyright 2019 JD.com Inc. JD AI

#pragma once

#include <vector>
#include <map>
#include <sstream>
#include <cstring>
#include <algorithm>
#include <dlfcn.h>
#include <cmath>
#include <float.h>
#include <errno.h>
#include <sys/stat.h>
#include <dirent.h>

#include "core/framework/execution_provider.h"
#include "core/graph/onnx_protobuf.h"

#include "tidl_onnxRtImport_EP.h"
#include "onnxrt_EP.h"


#define TIDL_STRING_SIZE        ((int32_t) 512)
#define TIDL_MAX_ALG_IN_BUFS    ((int32_t) 32)
#define TIDL_MAX_ALG_OUT_BUFS   ((int32_t) 32)
//#define MAX_FILE_PATH (512)
//NCI, NCO, PRED_CIRC, DWS_CONV_STRIDE
#define DEFAULT_COMPILE_CONSTRAINT_NC_FLAGS (0x1 | 0x40 | 0x200 | 0x400)

using TIDLProviderOptions = std::vector<std::pair<std::string,std::string>>;

namespace onnxruntime {

typedef  struct 
{
    void *lib;
    decltype(&::TIDL_getSupportedNodes) TIDL_getSupportedNodes;
    decltype(&::TIDL_populateOptions) TIDL_populateOptions;
    decltype(&::TIDL_createStateFunc) TIDL_createStateFunc;
    decltype(&::TIDL_computeImportFunc) TIDL_computeImportFunc;
    decltype(&::TIDL_computeInvokeFunc) TIDL_computeInvokeFunc;
    decltype(&::TIDL_isInputConst) TIDL_isInputConst;
    decltype(&::TIDL_getOutputShape) TIDL_getOutputShape;
    decltype(&::TIDLEP_getDdrStats) TIDLEP_getDdrStats;
} tidl_ops;

// Information needed to construct TIDL execution providers.
struct TidlExecutionProviderInfo {
  TIDLProviderOptions options_tidl_onnx_vec;

  explicit TidlExecutionProviderInfo(const TIDLProviderOptions& in_options_tidl_onnx_vec)
      : options_tidl_onnx_vec(in_options_tidl_onnx_vec) {}
  TidlExecutionProviderInfo() = default;
};

class TidlExecutionProvider : public IExecutionProvider {
 public:
  TidlExecutionProvider(const TidlExecutionProviderInfo& info);
  virtual ~TidlExecutionProvider();

  std::vector<std::unique_ptr<ComputeCapability>>
  GetCapability(const onnxruntime::GraphViewer& graph,
                const std::vector<const KernelRegistry*>& /*kernel_registries*/) const override;
  common::Status Compile(const std::vector<onnxruntime::Node*>& fused_nodes,
                         std::vector<NodeComputeInfo>& node_compute_funcs) override;
  
  int32_t GetCustomMemStats(uint64_t * read, uint64_t * write) const; 

 private:
  std::unordered_map<std::string, std::string*> model_protos_;
  tidl_ops * tidl_ops_ = new tidl_ops;
  int32_t is_import_;
  
};
}  // namespace onnxruntime
