// Copyright 2019 JD.com Inc. JD AI

#include "onnxruntime_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
  char import[512];
  int debug_level;
  int tidl_tensor_bits;
  char tidl_tools_path[512];
  char artifacts_folder[512];
} c_api_tidl_options;

ORT_API_STATUS(OrtSessionOptionsAppendExecutionProvider_Tidl, _In_ OrtSessionOptions* options, c_api_tidl_options * tidl_options);

#ifdef __cplusplus
}
#endif


