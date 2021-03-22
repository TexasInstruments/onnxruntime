// Copyright 2019 JD.com Inc. JD AI

#include "tidl_execution_provider.h"
#include "core/framework/allocatormgr.h"
#include "core/framework/compute_capability.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/inference_session.h"
#include "core/graph/model.h"
#include "float.h"
#include "math.h"

#include <string>


//#define TIDL_IMPORT_ONNX

namespace onnxruntime {

constexpr const char* TIDL = "Tidl";
constexpr const char* TIDL_CPU = "TidlCpu";


TidlExecutionProvider::TidlExecutionProvider(const TidlExecutionProviderInfo& info)
    : IExecutionProvider{onnxruntime::kTidlExecutionProvider} {
  AllocatorCreationInfo default_memory_info{
      [](int) {
        return onnxruntime::make_unique<CPUAllocator>(OrtMemoryInfo(TIDL, OrtAllocatorType::OrtDeviceAllocator));
      },
      0};

   AllocatorCreationInfo cpu_memory_info{
      [](int) {
        return onnxruntime::make_unique<CPUAllocator>(
            OrtMemoryInfo(TIDL_CPU, OrtAllocatorType::OrtDeviceAllocator, OrtDevice(), 0, OrtMemTypeCPUOutput));
      }};

  InsertAllocator(CreateAllocator(default_memory_info));
  InsertAllocator(CreateAllocator(cpu_memory_info));  
  TIDLProviderOptions interface_options = info.options_tidl_onnx_vec;
  
  for(auto option : interface_options)
  {
    auto key = option.first;
    auto value = option.second;
    if (!strcmp("import", key.c_str()))
    {
      std::map<std::string, int> valid_import_vals {{"yes", 1}, {"no", 0}};
      if(valid_import_vals.find(value.c_str()) == valid_import_vals.end())
      {
          printf("ERROR : unsupported import value \n");
      }
      is_import_ = valid_import_vals[value];
    }
  }

  if(is_import_)
  {
    tidl_ops_->lib = dlopen("libtidl_model_import_onnx.so", RTLD_NOW | RTLD_GLOBAL);
    if(! tidl_ops_->lib)
    {
      printf("Error -   %s \n", dlerror());
    }
    assert(tidl_ops_->lib);
    tidl_ops_->TIDL_getSupportedNodes = reinterpret_cast<decltype(tidl_ops_->TIDL_getSupportedNodes)>(dlsym(tidl_ops_->lib, "TIDL_getSupportedNodes"));
    tidl_ops_->TIDL_populateOptions = reinterpret_cast<decltype(tidl_ops_->TIDL_populateOptions)>(dlsym(tidl_ops_->lib, "TIDL_populateOptions"));
    tidl_ops_->TIDL_createStateFunc = reinterpret_cast<decltype(tidl_ops_->TIDL_createStateFunc)>(dlsym(tidl_ops_->lib, "TIDL_createStateFunc"));
    tidl_ops_->TIDL_computeImportFunc = reinterpret_cast<decltype(tidl_ops_->TIDL_computeImportFunc)>(dlsym(tidl_ops_->lib, "TIDL_computeImportFunc"));
    tidl_ops_->TIDL_computeInvokeFunc = reinterpret_cast<decltype(tidl_ops_->TIDL_computeInvokeFunc)>(dlsym(tidl_ops_->lib, "TIDL_computeInvokeFunc"));
    tidl_ops_->TIDL_isInputConst = reinterpret_cast<decltype(tidl_ops_->TIDL_isInputConst)>(dlsym(tidl_ops_->lib, "TIDL_isInputConst"));
    tidl_ops_->TIDL_getOutputShape = reinterpret_cast<decltype(tidl_ops_->TIDL_getOutputShape)>(dlsym(tidl_ops_->lib, "TIDL_getOutputShape"));
    tidl_ops_->TIDLEP_getDdrStats = reinterpret_cast<decltype(tidl_ops_->TIDLEP_getDdrStats)>(dlsym(tidl_ops_->lib, "TIDLEP_getDdrStats"));
  }
  else
  {
    tidl_ops_->lib = dlopen("libtidl_onnxrt_EP.so.1.0", RTLD_NOW | RTLD_GLOBAL);
    if(! tidl_ops_->lib)
    {
      printf("Error -   %s \n", dlerror());
    }
    printf("libtidl_onnxrt_EP loaded %p \n", tidl_ops_->lib);
    assert(tidl_ops_->lib);

    tidl_ops_->TIDL_getSupportedNodes = reinterpret_cast<decltype(tidl_ops_->TIDL_getSupportedNodes)>(dlsym(tidl_ops_->lib, "TIDL_getSupportedNodes"));
    tidl_ops_->TIDL_populateOptions = reinterpret_cast<decltype(tidl_ops_->TIDL_populateOptions)>(dlsym(tidl_ops_->lib, "TIDL_populateOptions"));
    tidl_ops_->TIDL_createStateFunc = reinterpret_cast<decltype(tidl_ops_->TIDL_createStateFunc)>(dlsym(tidl_ops_->lib, "TIDL_createStateFunc"));
    tidl_ops_->TIDL_computeInvokeFunc = reinterpret_cast<decltype(tidl_ops_->TIDL_computeInvokeFunc)>(dlsym(tidl_ops_->lib, "TIDL_computeInvokeFunc"));
    tidl_ops_->TIDL_isInputConst = reinterpret_cast<decltype(tidl_ops_->TIDL_isInputConst)>(dlsym(tidl_ops_->lib, "TIDL_isInputConst"));
    tidl_ops_->TIDL_getOutputShape = reinterpret_cast<decltype(tidl_ops_->TIDL_getOutputShape)>(dlsym(tidl_ops_->lib, "TIDL_getOutputShape"));
    tidl_ops_->TIDLEP_getDdrStats = reinterpret_cast<decltype(tidl_ops_->TIDLEP_getDdrStats)>(dlsym(tidl_ops_->lib, "TIDLEP_getDdrStats"));
  }
  bool status = false;
  status = tidl_ops_->TIDL_populateOptions(interface_options);
  // TODO : how to pass error if status is false?
}

TidlExecutionProvider::~TidlExecutionProvider() {
  //TODO : Add delete here
}


int32_t TidlExecutionProvider::GetCustomMemStats(uint64_t * read, uint64_t * write) const {

  return (tidl_ops_->TIDLEP_getDdrStats(read, write));
}

std::vector<std::unique_ptr<ComputeCapability>>
TidlExecutionProvider::GetCapability(const onnxruntime::GraphViewer& graph,
                                      const std::vector<const KernelRegistry*>& /*kernel_registries*/) const {
  // This method is based on that of TRT EP
  // Construct modelproto from graph
  //OnnxTIDLSubGraphParams *state_subGraph = (OnnxTIDLSubGraphParams*)malloc(sizeof(OnnxTIDLSubGraphParams));
   // Dump model Proto to file to pass it to pyxir
  auto logger = *GetLogger();

  const Graph& node_graph = graph.GetGraph();
  const std::string& name_ = node_graph.Name();
  onnxruntime::Model model{name_, true, ModelMetaData{}, PathString{},
                           IOnnxRuntimeOpSchemaRegistryList{},
                           node_graph.DomainToVersionMap(),
                           std::vector<ONNX_NAMESPACE::FunctionProto>(),
                           logger};

  ONNX_NAMESPACE::ModelProto model_proto = model.ToProto();
  model_proto.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);

  *(model_proto.mutable_graph()) = node_graph.ToGraphProto();

  GraphProto onnxGraph = model_proto.graph();
  
  std::string string_buf;
  string_buf = model_proto.SerializeAsString();

  const auto supported_nodes_vector = tidl_ops_->TIDL_getSupportedNodes(string_buf);

  onnxruntime::Graph& graph_build = model.MainGraph();
  const std::vector<NodeIndex>& node_index = graph.GetNodesInTopologicalOrder();
  std::set<NodeArg*> all_node_inputs;
  for (const auto& node : graph.Nodes()) {
    std::vector<onnxruntime::NodeArg*> inputs, outputs;
    for (auto input : node.InputDefs()) {
      auto& n_input = graph_build.GetOrCreateNodeArg(input->Name(), input->TypeAsProto());
      inputs.push_back(&n_input);
      all_node_inputs.insert(&n_input);
    }
    for (auto output : node.OutputDefs()) {
      auto& n_output = graph_build.GetOrCreateNodeArg(output->Name(), output->TypeAsProto());
      outputs.push_back(&n_output);
    }
    graph_build.AddNode(node.Name(), node.OpType(), node.Description(), inputs, outputs, &node.GetAttributes(), node.Domain());
  }
  const auto graph_outputs = graph.GetOutputs();
  //Add initializer to graph
  const auto& init_tensors = graph.GetAllInitializedTensors();
  for (const auto& tensor : init_tensors) {
    graph_build.AddInitializedTensor(*(tensor.second));
  }

  ORT_ENFORCE(graph_build.Resolve().IsOK());

  std::unique_ptr<IndexedSubGraph> sub_graph = onnxruntime::make_unique<IndexedSubGraph>();

  // Find inputs, initializers and outputs for each supported subgraph
  std::vector<std::unique_ptr<ComputeCapability>> result;

  int counter = 0;

  for (const auto& group : supported_nodes_vector) {
    if (!group.empty()) {
      std::unordered_set<size_t> node_set;
      node_set.reserve(group.size());
      for (const auto& index : group) {
        node_set.insert(node_index[index]);
      }
      std::unique_ptr<IndexedSubGraph> sub_graph = onnxruntime::make_unique<IndexedSubGraph>();
      // Find inputs and outputs of the subgraph
      std::unordered_map<const NodeArg*, int> fused_inputs, fused_outputs, fused_outputs_to_add;
      std::unordered_set<const NodeArg*> erased;
      int input_order = 0;
      int output_order = 0;

      for (const auto& index : group) {
        sub_graph->nodes.push_back(node_index[index]);
        const auto& node = graph.GetNode(node_index[index]);

        for (const auto& input : node->InputDefs()) {
          const auto& it = fused_outputs.find(input);

          if (it != fused_outputs.end()) {
            fused_outputs.erase(it);
            erased.insert(input);
          }
          //only when input is neither in output list nor erased list, add the input to input list
          else if (erased.find(input) == erased.end()) {
            fused_inputs[input] = input_order++;
          }
        }

        // For output searching, there is a special case:
        // If node's OutputEdges are more than its outputs, meaning certain output is used more than once,
        // if the output is connected to nodes that don't belong to the subgraph, the output need to be added
        // to the output list
        if (node->GetOutputEdgesCount() > node->OutputDefs().size()) {
          for (auto it = node->OutputEdgesBegin(), end = node->OutputEdgesEnd(); it != end; ++it) {
            const auto& node_idx = it->GetNode().Index();
            const auto& output = (it->GetNode()).InputDefs()[it->GetDstArgIndex()];

            if (node_set.find(node_idx) != node_set.end()) {
              const auto& iter = fused_inputs.find(output);

              if (iter != fused_inputs.end()) {
                fused_inputs.erase(iter);
                erased.insert(output);
              } else if (erased.find(output) == erased.end()) {
                fused_outputs[output] = output_order++;
              }
            } else {
              fused_outputs_to_add[output] = output_order++;
            }
          }
        } else {
          for (const auto& output : node->OutputDefs()) {
            const auto& it = fused_inputs.find(output);

            if (it != fused_inputs.end()) {
              fused_inputs.erase(it);
              erased.insert(output);
            }
            // only when output is neither in input list nor erased list, add the output to output list
            else if (erased.find(output) == erased.end()) {
              fused_outputs[output] = output_order++;
            }
          }
        }
      }

      fused_outputs.insert(fused_outputs_to_add.begin(), fused_outputs_to_add.end());

      // Sort inputs and outputs by the order they were added
      std::multimap<int, const NodeArg*> inputs, outputs;

      for (auto it = fused_inputs.begin(), end = fused_inputs.end(); it != end; ++it) {
        inputs.insert(std::pair<int, const NodeArg*>(it->second, it->first));
      }

      for (auto it = fused_outputs.begin(), end = fused_outputs.end(); it != end; ++it) {
        for (const auto& x : all_node_inputs) {
          if (x->Name() == it->first->Name()) {
            outputs.insert(std::pair<int, const NodeArg*>(it->second, it->first));
            break;
          }
        }
        if (std::find(graph_outputs.begin(), graph_outputs.end(), it->first) != graph_outputs.end()) {
          outputs.insert(std::pair<int, const NodeArg*>(it->second, it->first));
        }
      }

      // Assign inputs and outputs to subgraph's meta_def
      auto meta_def = onnxruntime::make_unique<::onnxruntime::IndexedSubGraph::MetaDef>();
      meta_def->name = "TIDL_" + std::to_string(counter++);
      meta_def->domain = kMSDomain;

      for (const auto& input : inputs) {
        meta_def->inputs.push_back(input.second->Name());
      }

      for (const auto& output : outputs) {
        meta_def->outputs.push_back(output.second->Name());
      }

      meta_def->since_version = 1;
      sub_graph->SetMetaDef(std::move(meta_def));

      result.push_back(onnxruntime::make_unique<ComputeCapability>(std::move(sub_graph)));
    }
  }
  return result;
}
void populateOnnxRtInputParams(Ort::CustomOpApi ort, OrtKernelContext * context, onnxRtParams_t * onnxRtParams, 
                          tidl_ops * tidl_ops, OnnxTIDLSubGraphParams * state_subGraph)
{
  int32_t i, currInIdx = 0; 

  // populate input params  
  for (i = 0; i < state_subGraph->numInputs; i++) 
  {    
    const OrtValue* input_tensor = ort.KernelContext_GetInput(context, state_subGraph->inputIdx[i]);
    OrtTensorTypeAndShapeInfo* input_tensor_info = ort.GetTensorTypeAndShape(input_tensor);
    int64_t inTensorElementType = ort.GetTensorElementType(input_tensor_info);
    const auto& tensor_shape = ort.GetTensorShape(input_tensor_info);
    ort.ReleaseTensorTypeAndShapeInfo(input_tensor_info);

    void * input;
    if (inTensorElementType == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8)
    {
      input = const_cast<uint8_t*>(ort.GetTensorData<uint8_t>(input_tensor));
    }
    else if (inTensorElementType == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32)
    {
      input = const_cast<int32_t*>(ort.GetTensorData<int32_t>(input_tensor));
    }
    else if (inTensorElementType == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)
    {
      input = const_cast<int64_t*>(ort.GetTensorData<int64_t>(input_tensor));
    }
    else if (inTensorElementType == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
    {
      input = const_cast<float*>(ort.GetTensorData<float>(input_tensor));
    }
    else
    {
      printf("ERROR : Unsupported input_tensor element type %d \n", inTensorElementType);
    }
    onnxRtParams->inputTensorData[currInIdx] = (void *)input; 
    onnxRtParams->inputTensorElementType[currInIdx] = inTensorElementType;
    onnxRtParams->tensorShape[currInIdx][3] = tensor_shape[3];
    onnxRtParams->tensorShape[currInIdx][2] = tensor_shape[2];
    onnxRtParams->tensorShape[currInIdx][1] = tensor_shape[1];
    onnxRtParams->tensorShape[currInIdx][0] = tensor_shape[0];
    currInIdx++;
  }
  onnxRtParams->numNetInData = state_subGraph->numInputs;
  onnxRtParams->numNetOutData = state_subGraph->numOutputs;
}

void populateOnnxRtOutputParams(Ort::CustomOpApi ort, OrtKernelContext * context, onnxRtParams_t * onnxRtParams,  
                          tidl_ops * tidl_ops, OnnxTIDLSubGraphParams * state_subGraph)
{
  //populate output params
  for (int j = 0; j < onnxRtParams->numNetOutData; j++) 
  {
    std::vector<int64_t> nchw_shape = tidl_ops->TIDL_getOutputShape(&state_subGraph->ioBuffDesc, j);
    auto* output_tensor = ort.KernelContext_GetOutput(context, j, nchw_shape.data(), nchw_shape.size());
    OrtTensorTypeAndShapeInfo* output_info = ort.GetTensorTypeAndShape(output_tensor);
    int64_t outTensorElementType = ort.GetTensorElementType(output_info);
    ort.ReleaseTensorTypeAndShapeInfo(output_info);
    //printf("Invoke : outTensorElementType = %d, numchOut = %d, outHeight = %d, outWidth = %d \n", outTensorElementType, ioBufDescPtr->outNumChannels[j], ioBufDescPtr->outHeight[j], ioBufDescPtr->outWidth[j]);
    
    void * output;
    if (outTensorElementType == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8)
    {
      output = ort.GetTensorMutableData<uint8_t>(output_tensor);
    }
    else if (outTensorElementType == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32)
    {
      output = ort.GetTensorMutableData<int32_t>(output_tensor);
    }
    else if (outTensorElementType == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)
    {
      output = ort.GetTensorMutableData<int64_t>(output_tensor);
    }
    else if (outTensorElementType == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
    {
      output = ort.GetTensorMutableData<float>(output_tensor);
    }
    else
    {
      printf("ERROR : Unsupported output tensor element type %d \n", outTensorElementType);
    }
    onnxRtParams->outputTensorData[j] = (void *)output; 
    onnxRtParams->outputTensorElementType[j] = outTensorElementType;
  }
}

common::Status TidlExecutionProvider::Compile(const std::vector<onnxruntime::Node*>& fused_nodes,
                                               std::vector<NodeComputeInfo>& node_compute_funcs) {
   for (const auto* fused_node : fused_nodes) {
    // Reconstruct graph proto from fused node's function body
    const auto* func_body = fused_node->GetFunctionBody();
    if (!func_body) {
      return common::Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "Function body is empty");
    }
    const Graph& graph_body = func_body->Body();
    onnxruntime::Model model(graph_body.Name(), true, ModelMetaData(), PathString(),
                             IOnnxRuntimeOpSchemaRegistryList(), graph_body.DomainToVersionMap(),
                             std::vector<ONNX_NAMESPACE::FunctionProto>(), *GetLogger());
    ONNX_NAMESPACE::ModelProto* model_proto = new ONNX_NAMESPACE::ModelProto();
    *model_proto = model.ToProto();
    *(model_proto->mutable_graph()) = graph_body.ToGraphProto();
    model_proto->set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);

    std::string * string_buf = new std::string();
    *string_buf = model_proto->SerializeAsString();


    model_protos_.emplace(fused_node->Name(), string_buf);

    NodeComputeInfo compute_info;
    
    compute_info.create_state_func = [&](ComputeContext* context, FunctionState* state) 
    {
      OnnxTIDLSubGraphParams *state_subGraph = (OnnxTIDLSubGraphParams*)malloc(sizeof(OnnxTIDLSubGraphParams));
      std::string * string_buf = model_protos_[context->node_name];

      tidl_ops_->TIDL_createStateFunc(state_subGraph, string_buf, context->node_name);

      *state = state_subGraph;

      return 0;      
    };

    compute_info.release_state_func = [](FunctionState state) 
    {
      free(state);
      //ORT_UNUSED_PARAMETER(state);
    };

    compute_info.compute_func = [&](FunctionState state, const OrtCustomOpApi* api, OrtKernelContext* context) 
    {

      OnnxTIDLSubGraphParams *state_subGraph = reinterpret_cast<OnnxTIDLSubGraphParams*>(state);
      Ort::CustomOpApi ort{*api};
      onnxRtParams_t onnxRtParams;

      populateOnnxRtInputParams(ort, context, &onnxRtParams, tidl_ops_, state_subGraph);
      if(is_import_)
      {
        std::string * string_buf = reinterpret_cast<std::string *>(state_subGraph->string_buf);
        tidl_ops_->TIDL_computeImportFunc(state_subGraph, &onnxRtParams, string_buf);
      }
      populateOnnxRtOutputParams(ort, context, &onnxRtParams, tidl_ops_, state_subGraph);
      tidl_ops_->TIDL_computeInvokeFunc(state_subGraph, &onnxRtParams);

      return Status::OK();

    };
 
    compute_info.custom_func = [&](FunctionState state , char **node_name, void **node_data) 
    {
      OnnxTIDLSubGraphParams *state_subGraph = reinterpret_cast<OnnxTIDLSubGraphParams*>(state);

      std::vector<uint64_t> *v = new std::vector<uint64_t>();

      v->push_back(uint64_t(state_subGraph->stats->cpIn_time_start));
      v->push_back(uint64_t(state_subGraph->stats->cpIn_time_end));
      v->push_back(uint64_t(state_subGraph->stats->proc_time_start));
      v->push_back(uint64_t(state_subGraph->stats->proc_time_end));
      v->push_back(uint64_t(state_subGraph->stats->cpOut_time_start));
      v->push_back(uint64_t(state_subGraph->stats->cpOut_time_end));
      *node_data = static_cast<void *>(v);
      *node_name = const_cast<char *>(state_subGraph->subGraphName_);
      return (Status::OK());
    };

    node_compute_funcs.push_back(compute_info);
  };
  return Status::OK();
}
}  // namespace onnxruntime
