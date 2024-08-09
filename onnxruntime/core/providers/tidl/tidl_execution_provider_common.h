// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#ifndef TIDL_ONNX_RT_EP_COMMON_H
#define TIDL_ONNX_RT_EP_COMMON_H 1

#include <stdint.h>
#include <stdarg.h>
#include <string.h>
#include <stdlib.h>

using std::string;
using std::vector;

#define TIDL_STRING_SIZE        ((int32_t) 512)
#define TIDL_MAX_ALG_IN_BUFS    ((int32_t) 32)
#define TIDL_MAX_ALG_OUT_BUFS   ((int32_t) 128)
#define TIDL_MAX_DIM ((int32_t)6U)

typedef struct {
  int32_t numNetInData;
  int32_t numNetOutData;
  int32_t tensorShape[TIDL_MAX_ALG_IN_BUFS][TIDL_MAX_DIM];
  int8_t  inDataNames[TIDL_MAX_ALG_IN_BUFS][TIDL_STRING_SIZE];
  int8_t  outDataNames[TIDL_MAX_ALG_OUT_BUFS][TIDL_STRING_SIZE];
  void *  inputTensorData[TIDL_MAX_ALG_IN_BUFS];
  void *  outputTensorData[TIDL_MAX_ALG_IN_BUFS];
  int64_t inputTensorElementType[TIDL_MAX_ALG_IN_BUFS];
  int64_t outputTensorElementType[TIDL_MAX_ALG_IN_BUFS];
} onnxRtParams_t;

typedef struct
{
  void * rtHandle;
  void * rtInList;
  void * rtOutList;
  void * stats;
  void * netPtr;
  void * ioBufDesc;
} tidlRtParamsOnnx;

typedef struct
{
  void* string_buf;
  int32_t serialNumber;
  int32_t currFrameIdx_;
  void *subGraphPtr_;
  char subGraphName_[TIDL_MAX_ALG_OUT_BUFS * TIDL_STRING_SIZE];
  int32_t inputIdx[TIDL_MAX_ALG_IN_BUFS];
  int32_t numInputs;
  int32_t numOutputs;
  tidlRtParamsOnnx tidlRtParams;
  onnxRtParams_t onnxRtParams;
}OnnxTIDLSubGraphParams;

extern "C"
{
  bool TIDL_populateOptions(std::vector<std::pair<std::string,std::string>> interface_options);
  int32_t TIDL_getSupportedNodesImport(std::string& data, std::string ortVersion, int32_t opsetVersion, std::vector<std::vector<int>> &nodeGroups);
  int32_t TIDL_getSupportedNodesInfer(std::vector<std::vector<int>> &nodeGroups);
  int32_t TIDL_createStateImportFunc(OnnxTIDLSubGraphParams * state_subgraph, std::string * string_buf, const std::string node_name);
  int32_t TIDL_createStateInferFunc(OnnxTIDLSubGraphParams * state_subGraph, const std::string node_name);
  int32_t TIDL_computeImportFunc(OnnxTIDLSubGraphParams * state_subGraph, std::string * string_buf, int32_t opSetVersion);
  int32_t TIDL_computeInvokeFunc(OnnxTIDLSubGraphParams * state_subGraph);
  int32_t TIDL_releaseRtFunc(OnnxTIDLSubGraphParams * state_subGraph);
  int32_t TIDL_getOutputShape(void * ioBufDescVPtr, int8_t onnxName[], std::vector<int64_t> &shape);
  int32_t TIDLEP_getDdrStats(uint64_t * read, uint64_t * write);
  int32_t TIDLEP_getSubGraphStats(OnnxTIDLSubGraphParams * state_subGraph, char **node_name, void **node_data);
  bool TIDLEP_checkCompatibility(const char *ortVersion);

}


#endif
