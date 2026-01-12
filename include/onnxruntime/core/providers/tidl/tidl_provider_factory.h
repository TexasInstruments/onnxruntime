// Copyright 2019 JD.com Inc. JD AI

#include "onnxruntime_c_api.h"

#define TIDL_MAX_STRING_LENGTH (512)
#define TIDL_MAX_SUBGRAPH_DATA (128)

#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
  int debug_level;
  char artifacts_folder[TIDL_MAX_STRING_LENGTH];
  int priority;
  float max_pre_empt_delay;
  int core_number;  // C7x core number to be used for inference
} c_api_tidl_options;

typedef struct
{
  uint64_t run_start;
  uint64_t run_end;
  uint64_t ddr_read_start;
  uint64_t ddr_read_end;
  uint64_t ddr_write_start;
  uint64_t ddr_write_end;
  uint64_t copy_in_start[TIDL_MAX_SUBGRAPH_DATA];
  uint64_t copy_in_end[TIDL_MAX_SUBGRAPH_DATA];
  uint64_t proc_start[TIDL_MAX_SUBGRAPH_DATA];
  uint64_t proc_end[TIDL_MAX_SUBGRAPH_DATA];
  uint64_t copy_out_start[TIDL_MAX_SUBGRAPH_DATA];
  uint64_t copy_out_end[TIDL_MAX_SUBGRAPH_DATA];
  uint32_t num_subgraph_data;
} c_api_tidl_benchmark_data;

ORT_API_STATUS(OrtSessionsOptionsSetDefault_Tidl, _In_ c_api_tidl_options * tidl_options);
ORT_API_STATUS(OrtSessionOptionsAppendExecutionProvider_Tidl, _In_ OrtSessionOptions* options, c_api_tidl_options * tidl_options);
ORT_API_STATUS_IMPL(OrtSessionGetTIBenchmarkData_Tidl, _In_ OrtSession* session, _Out_ c_api_tidl_benchmark_data * benchmark_data);
ORT_API_STATUS_IMPL(OrtSessionDisableIOValidation_Tidl, _In_ OrtSession* session);

#ifdef __cplusplus
}
#endif
