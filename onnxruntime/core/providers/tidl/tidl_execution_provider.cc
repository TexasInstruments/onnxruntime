// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.


#include "tidl_execution_provider.h"
#include "core/framework/compute_capability.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/inference_session.h"
#include "core/graph/model.h"
#include "core/graph/graph.h"
#include "float.h"
#include "math.h"
#include "core/graph/function.cc"

#include <string>
#include <unordered_set>


//#define TIDL_IMPORT_ONNX

namespace onnxruntime {

constexpr const char* TIDL = "Tidl";
constexpr const char* TIDL_CPU = "TidlCpu";

TidlExecutionProvider::TidlExecutionProvider(const TidlExecutionProviderInfo& info)
    : IExecutionProvider{onnxruntime::kTidlExecutionProvider} {
  TIDLProviderOptions interface_options = info.options_tidl_onnx_vec;
  bool status = true;

  is_import_ = (info.type == "TIDLCompilationProvider");

  if(is_import_)
  {
    std::string tidl_tools_path;
    for (auto _ : info.options_tidl_onnx_vec)
    {
      auto key = _.first;
      auto value = _.second;
      if(key == "tidl_tools_path")
      {
        tidl_tools_path = value;
      }
    }

    tidl_ops_->lib = dlopen((tidl_tools_path + "/tidl_model_import_onnx.so").c_str(), RTLD_NOW | RTLD_GLOBAL);
    if(!tidl_ops_->lib)
    {
      printf("Error -   %s \n", dlerror());
      status = false;
    }
    ORT_ENFORCE(status == true, "Could not open tidl_model_import_onnx.so");
  }
  else
  {
    tidl_ops_->lib = dlopen("libtidl_onnxrt_EP.so", RTLD_NOW | RTLD_GLOBAL);
    if(!tidl_ops_->lib)
    {
      printf("Error -   %s \n", dlerror());
    }
    ORT_ENFORCE(status == true, "Could not open libtidl_onnxrt_EP.so");
    printf("libtidl_onnxrt_EP loaded %p \n", tidl_ops_->lib);
  }

  tidl_ops_->TIDL_getSupportedNodesImport = reinterpret_cast<decltype(tidl_ops_->TIDL_getSupportedNodesImport)>(dlsym(tidl_ops_->lib, "TIDL_getSupportedNodesImport"));
  tidl_ops_->TIDL_getSupportedNodesInfer = reinterpret_cast<decltype(tidl_ops_->TIDL_getSupportedNodesInfer)>(dlsym(tidl_ops_->lib, "TIDL_getSupportedNodesInfer"));
  tidl_ops_->TIDL_populateOptions = reinterpret_cast<decltype(tidl_ops_->TIDL_populateOptions)>(dlsym(tidl_ops_->lib, "TIDL_populateOptions"));
  tidl_ops_->TIDL_createStateImportFunc = reinterpret_cast<decltype(tidl_ops_->TIDL_createStateImportFunc)>(dlsym(tidl_ops_->lib, "TIDL_createStateImportFunc"));
  tidl_ops_->TIDL_createStateInferFunc = reinterpret_cast<decltype(tidl_ops_->TIDL_createStateInferFunc)>(dlsym(tidl_ops_->lib, "TIDL_createStateInferFunc"));
  tidl_ops_->TIDL_computeImportFunc = reinterpret_cast<decltype(tidl_ops_->TIDL_computeImportFunc)>(dlsym(tidl_ops_->lib, "TIDL_computeImportFunc"));
  tidl_ops_->TIDL_computeInvokeFunc = reinterpret_cast<decltype(tidl_ops_->TIDL_computeInvokeFunc)>(dlsym(tidl_ops_->lib, "TIDL_computeInvokeFunc"));
  tidl_ops_->TIDL_releaseRtFunc = reinterpret_cast<decltype(tidl_ops_->TIDL_releaseRtFunc)>(dlsym(tidl_ops_->lib, "TIDL_releaseRtFunc"));
  tidl_ops_->TIDL_getOutputShape = reinterpret_cast<decltype(tidl_ops_->TIDL_getOutputShape)>(dlsym(tidl_ops_->lib, "TIDL_getOutputShape"));
  tidl_ops_->TIDLEP_getDdrStats = reinterpret_cast<decltype(tidl_ops_->TIDLEP_getDdrStats)>(dlsym(tidl_ops_->lib, "TIDLEP_getDdrStats"));
  tidl_ops_->TIDLEP_getSubGraphStats = reinterpret_cast<decltype(tidl_ops_->TIDLEP_getSubGraphStats)>(dlsym(tidl_ops_->lib, "TIDLEP_getSubGraphStats"));
  tidl_ops_->TIDLEP_checkCompatibility = reinterpret_cast<decltype(tidl_ops_->TIDLEP_checkCompatibility)>(dlsym(tidl_ops_->lib, "TIDLEP_checkCompatibility"));
  if (tidl_ops_->TIDL_populateOptions == nullptr || tidl_ops_->TIDLEP_checkCompatibility == nullptr || tidl_ops_->TIDL_computeInvokeFunc == nullptr || tidl_ops_->TIDL_releaseRtFunc == nullptr || tidl_ops_->TIDL_getOutputShape == nullptr)
  {
    status = false;
  }

  if(is_import_)
	{
    if (tidl_ops_->TIDL_getSupportedNodesImport == nullptr || tidl_ops_->TIDL_createStateImportFunc == nullptr || tidl_ops_->TIDL_computeImportFunc == nullptr)
    {
      status = false;
    }
	}
  else
  {
    if (tidl_ops_->TIDL_getSupportedNodesInfer == nullptr || tidl_ops_->TIDL_createStateInferFunc == nullptr)
    {
      status = false;
    }
  }

  ORT_ENFORCE(status == true, "Could not load function from share object file");

  status = tidl_ops_->TIDLEP_checkCompatibility(OrtGetApiBase()->GetVersionString());
  ORT_ENFORCE(status == true, "Version compatibility check failed.");

  status = tidl_ops_->TIDL_populateOptions(interface_options);
  ORT_ENFORCE(status == true);
}

TidlExecutionProvider::~TidlExecutionProvider() {
  //TODO : Add delete here
}

std::vector<AllocatorPtr> TidlExecutionProvider::CreatePreferredAllocators() {

  AllocatorCreationInfo default_memory_info{
      [](int) {
        return std::make_unique<CPUAllocator>(OrtMemoryInfo(TIDL, OrtAllocatorType::OrtDeviceAllocator));
      },
      0};

  AllocatorCreationInfo cpu_memory_info{
      [](int) {
        return std::make_unique<CPUAllocator>(
            OrtMemoryInfo(TIDL_CPU, OrtAllocatorType::OrtDeviceAllocator, OrtDevice(), OrtMemTypeCPUOutput));
      }};

  return std::vector<AllocatorPtr>{CreateAllocator(default_memory_info), CreateAllocator(cpu_memory_info)};
}


int32_t TidlExecutionProvider::GetCustomMemStats(uint64_t * read, uint64_t * write) const {

  return (tidl_ops_->TIDLEP_getDdrStats(read, write));
}

std::vector<std::unique_ptr<ComputeCapability>>
TidlExecutionProvider::GetCapability(const onnxruntime::GraphViewer& graph,
                                     const IKernelLookup& /*kernel_lookup*/,
                                     const GraphOptimizerRegistry& /* graph_optimizer_registry */,
                                     onnxruntime::IResourceAccountant* /* resource_accountant */) const {
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

  onnx::GraphProto onnxGraph = model_proto.graph();

  std::string string_buf;
  string_buf = model_proto.SerializeAsString();

  int32_t status = 0;
  std::vector<std::vector<int>> supported_nodes_vector;
  if(is_import_)
  {
    status = tidl_ops_->TIDL_getSupportedNodesImport(string_buf, OrtGetApiBase()->GetVersionString(), node_graph.DomainToVersionMap().at(kOnnxDomain), supported_nodes_vector);
    ORT_ENFORCE(status == 0, "Could not get supported nodes for compilation.");
  }
  else
  {
    status = tidl_ops_->TIDL_getSupportedNodesInfer(supported_nodes_vector);
    ORT_ENFORCE(status == 0, "Could not get supported nodes for inference.");
  }

  onnxruntime::Graph& graph_build = model.MainGraph();
  const std::vector<NodeIndex>& node_index = graph.GetNodesInTopologicalOrder();

  //Mapping Supported nodes to topological order using names from node's output
  std::vector<std::vector<int>> to_supported_nodes_vector;
  std::string output_no, output_to;

  for(int i = 0; i < supported_nodes_vector.size(); i++)
  {
    std::vector<int> group_nodes;
    for(int j = 0; j < supported_nodes_vector[i].size(); j++)
    {
      output_no = onnxGraph.node(supported_nodes_vector[i][j]).output(0);
      for(int k=0; k< node_index.size(); k++)
      {
        const auto& node = graph.GetNode(node_index[k]);
        const auto& output_defs = node->OutputDefs();
        output_to = output_defs[0]->Name();
        if((strcmp(output_no.c_str(), output_to.c_str())==0))
        {
          group_nodes.push_back(k);
          break;
        }
      }
    }
    to_supported_nodes_vector.push_back(group_nodes);
  }

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
  std::unordered_set<const NodeArg*> graph_outputs_set(graph_outputs.cbegin(), graph_outputs.cend());
  //Add initializer to graph
  const auto& init_tensors = graph.GetAllInitializedTensors();
  for (const auto& tensor : init_tensors) {
    graph_build.AddInitializedTensor(*(tensor.second));
  }

  ORT_ENFORCE(graph_build.Resolve().IsOK());

  std::unique_ptr<IndexedSubGraph> sub_graph = std::make_unique<IndexedSubGraph>(); //Changed onnxruntime::make_unique to std::make_unique

  // Find inputs, initializers and outputs for each supported subgraph
  std::vector<std::unique_ptr<ComputeCapability>> result;

  int counter = 0;

  for (const auto& group : to_supported_nodes_vector) {
    if (!group.empty()) {
      std::unordered_set<size_t> node_set;
      node_set.reserve(group.size());
      for (const auto& index : group) {
        node_set.insert(node_index[index]);
      }
      std::unique_ptr<IndexedSubGraph> sub_graph = std::make_unique<IndexedSubGraph>(); //Changed onnxruntime::make_unique to std::make_unique
      // Find inputs and outputs of the subgraph
      std::unordered_map<const NodeArg*, int> fused_inputs, fused_outputs, fused_outputs_to_add, overall_graph_output_to_add;
      std::unordered_set<const NodeArg*> erased;
      int input_order = 0;
      int output_order = 0;
      for (const auto& index : group) {
        sub_graph->nodes.push_back(node_index[index]);
        const auto& node = graph.GetNode(node_index[index]);

        for (const auto& input : node->InputDefs()) {
          if(!input->Exists()) /* Empty name indicates optional non-existent input argument, do not add to meta def input list */
          {
            continue;
          }
          const auto& it = fused_outputs.find(input);

          if (it != fused_outputs.end()) {
            // Adding graph overall outputs which are also input for other nodes
            if(graph_outputs_set.count(input)!= 0){
              overall_graph_output_to_add[input] = it->second;
            }
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
            if(!output->Exists())
            {
              continue;
            }
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

      for (auto it = overall_graph_output_to_add.begin(), end = overall_graph_output_to_add.end(); it != end; ++it) {
        outputs.insert(std::pair<int, const NodeArg*>(it->second, it->first));
      }

      // Assign inputs and outputs to subgraph's meta_def
      auto meta_def = std::make_unique<::onnxruntime::IndexedSubGraph::MetaDef>(); //Changed onnxruntime::make_unique to std::make_unique
      // auto meta_def = ::onnxruntime::IndexedSubGraph_MetaDef::Create();
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
      result.push_back(std::make_unique<ComputeCapability>(std::move(sub_graph))); //Changed onnxruntime::make_unique to std::make_unique
    }
  }

  return result;
}
int32_t populateOnnxRtInputParams(const OrtApi *ort, OrtKernelContext *context,
                                  tidl_ops *tidl_ops, OnnxTIDLSubGraphParams *state_subGraph)
{
  int32_t status = 0;
  int32_t i, currInIdx = 0;
  onnxRtParams_t * onnxRtParams = &state_subGraph->onnxRtParams;
  for (i = 0; i < state_subGraph->numInputs; i++)
  {
    const OrtValue *input_tensor = nullptr;
    OrtTensorTypeAndShapeInfo *input_tensor_info = NULL;
    ONNXTensorElementDataType inTensorElementType;
    size_t inputNumDims = 0;

    Ort::ThrowOnError(ort->KernelContext_GetInput(context, state_subGraph->inputIdx[i], &input_tensor));
    Ort::ThrowOnError(ort->GetTensorTypeAndShape(input_tensor, &input_tensor_info));
    Ort::ThrowOnError(ort->GetTensorElementType(input_tensor_info, &inTensorElementType));
    Ort::ThrowOnError(ort->GetDimensionsCount(input_tensor_info, &inputNumDims));

    int64_t* inputDims = (int64_t*) malloc(sizeof(int64_t) * inputNumDims);
    Ort::ThrowOnError(ort->GetDimensions(input_tensor_info, inputDims, inputNumDims ));

    ort->ReleaseTensorTypeAndShapeInfo(input_tensor_info);

    // Check supported input type
    switch (inTensorElementType)
    {
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
        // allowed
        break;

      default:
        printf("TIDL_EP: Unsupported input tensor element type: %d \n",inTensorElementType);
        return -1;
        break;
    }

    void *input;
    Ort::ThrowOnError(ort->GetTensorMutableData(const_cast<OrtValue*>(input_tensor), (void **)&input));

    if (inputNumDims > TIDL_MAX_DIM)
    {
      printf("TIDL_EP: Only tensors up to 6 dimensions are supported");
      return -1;
    }

    /*
     * If input to the network has less than 4 dimensions, populate the
     * lower index dimensions correctly and put dimension as 1 for the remaining
     * higher indices
     */
    for(int i = 0; i < inputNumDims; i++)
    {
      if ((TIDL_MAX_DIM - inputNumDims + i) >= 0)
      {
        onnxRtParams->tensorShape[currInIdx][TIDL_MAX_DIM-inputNumDims + i] = inputDims[i];
      }
    }
    for(int i = 0; i < (TIDL_MAX_DIM - inputNumDims); i++)
    {
      onnxRtParams->tensorShape[currInIdx][i] = 1;
    }

    onnxRtParams->inputTensorData[currInIdx] = (void *)input;
    onnxRtParams->inputTensorElementType[currInIdx] = inTensorElementType;

    currInIdx++;

    free(inputDims);
    inputDims = NULL;
  }
  onnxRtParams->numNetInData = state_subGraph->numInputs;
  onnxRtParams->numNetOutData = state_subGraph->numOutputs;

  return status;
}

int32_t populateOnnxRtOutputParams(const OrtApi *ort, OrtKernelContext * context, tidl_ops * tidl_ops, OnnxTIDLSubGraphParams * state_subGraph)
{
  int32_t status = 0;
  onnxRtParams_t * onnxRtParams = &state_subGraph->onnxRtParams;

  // Populate output params
  for (int j = 0; j < onnxRtParams->numNetOutData; j++)
  {
    std::vector<int64_t> nchw_shape{};
    status = tidl_ops->TIDL_getOutputShape(state_subGraph->tidlRtParams.ioBufDesc, onnxRtParams->outDataNames[j], nchw_shape);
    if(status != 0)
      return status;

    OrtValue *output_tensor = NULL;
    OrtTensorTypeAndShapeInfo *output_tesnor_info = NULL;
    ONNXTensorElementDataType outTensorElementType;

    Ort::ThrowOnError(ort->KernelContext_GetOutput(context, j, nchw_shape.data(), nchw_shape.size(), &output_tensor));
    Ort::ThrowOnError(ort->GetTensorTypeAndShape(output_tensor, &output_tesnor_info));
    Ort::ThrowOnError(ort->GetTensorElementType(output_tesnor_info, &outTensorElementType));

    ort->ReleaseTensorTypeAndShapeInfo(output_tesnor_info);

    // Check supported output type
    switch (outTensorElementType)
    {
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
        // allowed
        break;

      default:
        printf("TIDL_EP: Unsupported output tensor element type: %d \n",outTensorElementType);
        return -1;
        break;
    }

    void *output;
    Ort::ThrowOnError(ort->GetTensorMutableData(output_tensor, (void **)&output));

    onnxRtParams->outputTensorData[j] = (void *)output;
    onnxRtParams->outputTensorElementType[j] = outTensorElementType;
  }
  return status;
}
Status TidlExecutionProvider::Compile(const std::vector<FusedNodeAndGraph>& fused_nodes_and_graphs, // !!AL!! Changed from onnxruntime::Node* fused_nodes to <FusedNodeAndGraph> fused_nodes_and_graphs
                                               std::vector<NodeComputeInfo>& node_compute_funcs) {
  for (auto& fused_node_graph : fused_nodes_and_graphs) {
    std::string *string_buf = new std::string();
    if (is_import_)
    {
      const GraphViewer& graph_body_viewer = fused_node_graph.filtered_graph;
      const Node& fused_node = fused_node_graph.fused_node;
      const Graph& graph = graph_body_viewer.GetGraph();
      Graph& graph_body1 = const_cast<Graph&>(graph);
      const IndexedSubGraph* indexed_sub_graph = graph_body_viewer.GetFilterInfo(); //changed from func_body->Body() to graph_body_viewer.GetGraph()
      const auto func_body = FunctionImpl(graph_body1, *indexed_sub_graph);
      const Graph& graph_body = func_body.Body();
      onnxruntime::Model model(graph_body.Name(), true, ModelMetaData(), PathString(),
                              IOnnxRuntimeOpSchemaRegistryList(), graph.DomainToVersionMap(),
                              std::vector<ONNX_NAMESPACE::FunctionProto>(), *GetLogger());
      ONNX_NAMESPACE::ModelProto* model_proto = new ONNX_NAMESPACE::ModelProto();
      *model_proto = model.ToProto();
      *(model_proto->mutable_graph()) = graph_body.ToGraphProto();
      model_proto->set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);

      *string_buf = model_proto->SerializeAsString();

      model_protos_.emplace(fused_node.Name(), string_buf);
      model_opset_version_ = graph.DomainToVersionMap().at(kOnnxDomain);
    }

    NodeComputeInfo compute_info;
    subgraph_serial_number_ = 0;

    compute_info.create_state_func = [&](ComputeContext* context, FunctionState* state)
    {
      int32_t status = 0;
      OnnxTIDLSubGraphParams *state_subGraph = (OnnxTIDLSubGraphParams*)malloc(sizeof(OnnxTIDLSubGraphParams));

      state_subGraph->serialNumber = subgraph_serial_number_;

      if(is_import_)
      {
        std::string * string_buf = model_protos_[context->node_name];
        status = tidl_ops_->TIDL_createStateImportFunc(state_subGraph, string_buf, context->node_name);
      }
      else
      {
        status = tidl_ops_->TIDL_createStateInferFunc(state_subGraph, context->node_name);
      }

      subgraph_serial_number_++;

      *state = state_subGraph;
      return status;
    };

    compute_info.release_state_func = [&](FunctionState state)
    {
      int32_t status = 0;
      OnnxTIDLSubGraphParams *state_subGraph = reinterpret_cast<OnnxTIDLSubGraphParams*>(state);
      status = tidl_ops_->TIDL_releaseRtFunc(state_subGraph);
      free(state);
    };

    compute_info.compute_func = [&](FunctionState state, const OrtApi* api, OrtKernelContext* context)
    {///* !!AL!! Changes OrtCustomApi* to OrtApi* */
      int32_t status = 0;
      OnnxTIDLSubGraphParams *state_subGraph = reinterpret_cast<OnnxTIDLSubGraphParams*>(state);
      status = populateOnnxRtInputParams(api, context, tidl_ops_, state_subGraph);
      if(status != 0)
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Populate OnnxRT Input Params Failed.");
      if(is_import_)
      {
        std::string * string_buf = reinterpret_cast<std::string *>(state_subGraph->string_buf);
        status = tidl_ops_->TIDL_computeImportFunc(state_subGraph, string_buf, model_opset_version_);
        if(status != 0)
          return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "TIDL Compute Import Failed.");
      }
      status = populateOnnxRtOutputParams(api, context, tidl_ops_, state_subGraph);
      if(status != 0)
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Populate OnnxRT Output Params Failed.");
      status = tidl_ops_->TIDL_computeInvokeFunc(state_subGraph);
      if(status != 0)
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "TIDL Compute Invoke Failed.");
      return Status::OK();
    };

    compute_info.custom_func = [&](FunctionState state , char **node_name, void **node_data)
    {
      OnnxTIDLSubGraphParams *state_subGraph = reinterpret_cast<OnnxTIDLSubGraphParams*>(state);

      tidl_ops_->TIDLEP_getSubGraphStats(state_subGraph, node_name, node_data);
      return (Status::OK());
    };

    node_compute_funcs.push_back(compute_info);
  };
  return Status::OK();
}
}  // namespace onnxruntime
