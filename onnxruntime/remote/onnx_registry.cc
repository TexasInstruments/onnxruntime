#include <op_registry.h>
#include <packet.h>

#include "onnx_messages.h"

REGISTER_MSG(onnx_initialize_cpu_session);
REGISTER_MSG(onnx_initialize_tidl_session);
REGISTER_MSG(onnx_run_session);
REGISTER_MSG(onnx_session_from_file);
REGISTER_MSG(onnx_session_from_buffer);
REGISTER_MSG(onnx_destroy_session);
REGISTER_MSG(onnx_get_TI_benchmark_data_session);
