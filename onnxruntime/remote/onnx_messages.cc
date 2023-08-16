#include <set>
#include "core/session/inference_session.h"
#include "core/session/environment.h"
#include "core/framework/allocator.h"
#include "core/common/logging/logging.h"
#include "core/common/logging/severity.h"
#include "core/common/logging/sinks/clog_sink.h"

#include "core/providers/cpu/cpu_provider_factory_creator.h"
#include "core/providers/cpu/cpu_execution_provider.h"
using TIDLProviderOptions = std::vector<std::pair<std::string,std::string>>;
namespace onnxruntime {
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Tidl(const std::string& provider_type, const TIDLProviderOptions& options);
}

#include "onnx_messages.h"

using TIIETensor = std::tuple<std::string,std::vector<uint8_t>,std::string,std::vector<int64_t>>;

onnxruntime::AllocatorPtr& GetAllocator() {
  static onnxruntime::AllocatorPtr alloc = std::make_shared<onnxruntime::TAllocator>();
  return alloc;
}

onnxruntime::MLDataType FromString(const std::string& type_name) {
  static std::map<std::string, onnxruntime::MLDataType> type_map{
      {"bool", onnxruntime::DataTypeImpl::GetType<bool>()},
      {"float", onnxruntime::DataTypeImpl::GetType<float>()},
      {"float16", onnxruntime::DataTypeImpl::GetType<onnxruntime::MLFloat16>()},
      {"double", onnxruntime::DataTypeImpl::GetType<double>()},
      {"int8", onnxruntime::DataTypeImpl::GetType<int8_t>()},
      {"uint8", onnxruntime::DataTypeImpl::GetType<uint8_t>()},
      {"int16", onnxruntime::DataTypeImpl::GetType<int16_t>()},
      {"uint16", onnxruntime::DataTypeImpl::GetType<uint16_t>()},
      {"int32", onnxruntime::DataTypeImpl::GetType<int32_t>()},
      {"uint32", onnxruntime::DataTypeImpl::GetType<uint32_t>()},
      {"int64", onnxruntime::DataTypeImpl::GetType<int64_t>()},
      {"uint64", onnxruntime::DataTypeImpl::GetType<uint64_t>()},
      {"string", onnxruntime::DataTypeImpl::GetType<std::string>()},
  };
  const auto it = type_map.find(type_name);
  if (it == type_map.end())
    ERROR("No corresponding Numpy type for Tensor Type.");
  return it->second;
}


template <class T>
class idr {
public:
    int add(T item);
    T remove(int id);
    T find(int id);
private:
    std::set<int> m_allocated_ids;
    std::set<int> m_free_ids;
    std::map<int, T> m_models;
};

template <class T>
int idr<T>::add(T item)
{
    int id;

    if(!m_free_ids.empty()) {
        std::set<int>::iterator it = m_free_ids.begin();
        id = *it;
        m_free_ids.erase(it);
    } else if(!m_allocated_ids.empty()) {
        std::set<int>::reverse_iterator it = m_allocated_ids.rbegin();
        id = *it + 1;
    } else
        id = 0;

    if(m_models.find(id) != m_models.end() ||
            m_allocated_ids.find(id) != m_allocated_ids.end() ||
            m_free_ids.find(id) != m_free_ids.end()) {
        ERROR("Exception: idr allocation failure for id = %d", id);
    }

    m_allocated_ids.insert(id);
    m_models[id] = item;
    return id;
}

template <class T>
T idr<T>::remove(int id)
{
    T item;

    if(m_models.find(id) == m_models.end() ||
            m_allocated_ids.find(id) == m_allocated_ids.end() ||
            m_free_ids.find(id) != m_free_ids.end()) {
        ERROR("Exception: failed to remove id = %d, not allocated", id);
    }

    item = m_models[id];
    m_models.erase(m_models.find(id));
    m_allocated_ids.erase(m_allocated_ids.find(id));
    m_free_ids.insert(id);

    return item;
}

template <class T>
T idr<T>::find(int id)
{
    T item;

    if(m_models.find(id) == m_models.end() ||
            m_allocated_ids.find(id) == m_allocated_ids.end() ||
            m_free_ids.find(id) != m_free_ids.end()) {
        ERROR("Exception: could not find id = %d", id);
    }

    item = m_models[id];

    return item;
}

idr<onnxruntime::InferenceSession *> model_idr;

onnxruntime::Environment& GetEnv() {
    onnxruntime::Status status;
    static std::unique_ptr<onnxruntime::Environment> session_env;
    static bool initialized = false;
    static std::string logger_id;
    if (initialized) {
      return *session_env;
    }
    status = onnxruntime::Environment::Create(onnxruntime::make_unique<onnxruntime::logging::LoggingManager>(
                                                  std::unique_ptr<onnxruntime::logging::ISink>{new onnxruntime::logging::CLogSink{}},
                                                  onnxruntime::logging::Severity::kWARNING, false, onnxruntime::logging::LoggingManager::InstanceType::Default,
                                                  &logger_id),
                                              session_env);
    if (!status.IsOK())
        ERROR("%s", status.ToString().c_str());

    initialized = true;
    return *session_env;
}

DECLARE_MSG(ONNX_SECTION + __COUNTER__, onnx_initialize_cpu_session);
    onnxruntime::Status status;
    auto sess = model_idr.find(m_id);
    auto func = onnxruntime::CreateExecutionProviderFactory_CPU(sess->GetSessionOptions().enable_cpu_mem_arena)->CreateProvider();

    status = sess->RegisterExecutionProvider(std::move(func));
    if (!status.IsOK())
        ERROR("%s", status.ToString().c_str());

    status = sess->Initialize();
    if (!status.IsOK())
        ERROR("%s", status.ToString().c_str());

    p->m_status = 0;
    return p;
}

DECLARE_MSG(ONNX_SECTION + __COUNTER__, onnx_initialize_tidl_session);
    onnxruntime::Status status;
    auto sess = model_idr.find(m_id);
    auto func_cpu = onnxruntime::CreateExecutionProviderFactory_CPU(sess->GetSessionOptions().enable_cpu_mem_arena)->CreateProvider();
    auto func_tidl = onnxruntime::CreateExecutionProviderFactory_Tidl(onnxruntime::kTidlExecutionProvider, m_options)->CreateProvider();

    status = sess->RegisterExecutionProvider(std::move(func_tidl));
    if (!status.IsOK())
        ERROR("%s", status.ToString().c_str());

    status = sess->RegisterExecutionProvider(std::move(func_cpu));
    if (!status.IsOK())
        ERROR("%s", status.ToString().c_str());

    status = sess->Initialize();
    if (!status.IsOK())
        ERROR("%s", status.ToString().c_str());

    p->m_status = 0;
    return p;
}

DECLARE_MSG(ONNX_SECTION + __COUNTER__, onnx_run_session);
    onnxruntime::NameMLValMap feeds;
    onnxruntime::Status status;
    auto sess = model_idr.find(m_id);

    for(auto _ : m_in_data) {
        OrtValue ml_value;

        auto in_name = std::get<0>(_);
        auto in_data = std::get<1>(_);
        onnxruntime::MLDataType in_type = FromString(std::get<2>(_));
        onnxruntime::TensorShape in_shape(std::get<3>(_));
        
        auto p_tensor = onnxruntime::make_unique<onnxruntime::Tensor>(in_type, in_shape, GetAllocator());
        memcpy(p_tensor->MutableDataRaw(in_type), in_data.data(), in_data.size());
        auto ml_tensor = onnxruntime::DataTypeImpl::GetType<onnxruntime::Tensor>();
        ml_value.Init(p_tensor.release(), ml_tensor, ml_tensor->GetDeleteFunc());
        
        feeds.insert(std::make_pair(in_name, ml_value));
    }
    std::vector<OrtValue> fetches;
    status = sess->Run(feeds, m_out_names, &fetches);
    if(!status.IsOK())
        ERROR("%s", status.ToString().c_str());

    p->m_status = 0;
    for(auto _ : fetches) {
        if(!_.IsTensor())
            ERROR("output is not Tensor");
        const onnxruntime::Tensor &t = _.Get<onnxruntime::Tensor>();
        const uint8_t *data = static_cast<const uint8_t *>(t.DataRaw());
        size_t len;
        if (!onnxruntime::IAllocator::CalcMemSizeForArray(t.DataType()->Size(), t.Shape().Size(), &len))
            throw std::runtime_error("length overflow");

        TIIETensor this_feed = std::make_tuple(
                "", std::vector<uint8_t>(data, data + len),
                onnxruntime::DataTypeImpl::ToString(t.DataType()),
                t.Shape().GetDims());
        p->m_out_data.push_back(this_feed);
    }
    return p;
}

DECLARE_MSG(ONNX_SECTION + __COUNTER__, onnx_session_from_file);
    onnxruntime::Status status;
    onnxruntime::SessionOptions so;

    onnxruntime::InferenceSession *sess = new onnxruntime::InferenceSession(so, GetEnv());
    status = sess->Load(std::string(BASE) + m_path);
    if (!status.IsOK())
        ERROR("%s", status.ToString().c_str());

    p->m_status = 0;
    p->m_id = model_idr.add(sess);

    return p;
}

DECLARE_MSG(ONNX_SECTION + __COUNTER__, onnx_session_from_buffer);
    onnxruntime::Status status;
    onnxruntime::SessionOptions so;

    onnxruntime::InferenceSession *sess = new onnxruntime::InferenceSession(so, GetEnv());
    status = sess->Load(m_buffer.data(), m_buffer.size());
    if (!status.IsOK())
        ERROR("%s", status.ToString().c_str());

    p->m_status = 0;
    p->m_id = model_idr.add(sess);

    return p;
}

DECLARE_MSG(ONNX_SECTION + __COUNTER__, onnx_destroy_session);
    auto sess = model_idr.remove(m_id);
    delete sess;

    p->m_status = 0;
    return p;
}

DECLARE_MSG(ONNX_SECTION + __COUNTER__, onnx_get_TI_benchmark_data_session);
    auto sess = model_idr.find(m_id);

    p->m_benchmark_data = sess->get_TI_benchmark_data();

    p->m_status = 0;
    return p;
}

/*
 * onnx_initialize_tidl_session request packet has a complex structure
 * e.g., vector<string> which I could not figure out how to automate
 * serialization / deserialization
 *
 * the yaml parser lets these classes have user-defined overrides
 * to read_from() / write_to() if __SKIP_SERIALIZE__ is added to the
 * yaml list
 */
void onnx_initialize_tidl_session_req::write_to(std::ostream& output) const {
    packet::write<uint32_t>(output, m_id);
    packet::write<uint32_t>(output, m_options.size());
    for(auto it : m_options) {
        packet::write_string(output, it.first);
        packet::write_string(output, it.second);
    }
}
void onnx_initialize_tidl_session_req::read_from(std::istream& input) {
    m_id = packet::read<uint32_t>(input);
    m_options = std::vector<std::pair<std::string, std::string>>(packet::read<uint32_t>(input));
    for(auto& it : m_options) {
        it.first = packet::read_string(input);
        it.second = packet::read_string(input);
    }
}

/*
 * onnx_run_session request / response packets have a complex structure
 * e.g., vector<string> which I could not figure out how to automate
 * serialization / deserialization
 *
 * the yaml parser lets these classes have user-defined overrides
 * to read_from() / write_to() if __SKIP_SERIALIZE__ is added to the
 * yaml list
 */
void onnx_run_session_req::write_to(std::ostream& output) const {
    packet::write<uint32_t>(output, m_id);
    packet::write<uint32_t>(output, m_in_data.size());
    for(auto it : m_in_data) {
        packet::write_string(output, std::get<0>(it));
        packet::write_vector<uint8_t>(output, std::get<1>(it));
        packet::write_string(output, std::get<2>(it));
        packet::write_vector<int64_t>(output, std::get<3>(it));
    }
    packet::write<uint32_t>(output, m_out_names.size());
    for(auto it : m_out_names)
        packet::write_string(output, it);
}
void onnx_run_session_resp::write_to(std::ostream& output) const {
    packet::write<int32_t>(output, m_status);
    packet::write<uint32_t>(output, m_out_data.size());
    for(auto it : m_out_data) {
        packet::write_string(output, std::get<0>(it));
        packet::write_vector<uint8_t>(output, std::get<1>(it));
        packet::write_string(output, std::get<2>(it));
        packet::write_vector<int64_t>(output, std::get<3>(it));
    }
}
void onnx_run_session_req::read_from(std::istream& input) {
    m_id = packet::read<int32_t>(input);
    m_in_data = std::vector<std::tuple<std::string,std::vector<uint8_t>,std::string,std::vector<int64_t>>>(packet::read<uint32_t>(input));
    for(auto& it : m_in_data) {
        auto name = packet::read_string(input);
        auto data = packet::read_vector<uint8_t>(input); 
        auto type = packet::read_string(input); 
        auto shape = packet::read_vector<int64_t>(input); 
        it = std::make_tuple(name, data, type, shape);
    }
    m_out_names = std::vector<std::string>(packet::read<uint32_t>(input));
    for(auto& it : m_out_names)
        it = packet::read_string(input);
}
void onnx_run_session_resp::read_from(std::istream& input) {
    m_status = packet::read<int32_t>(input);
    m_out_data = std::vector<std::tuple<std::string,std::vector<uint8_t>,std::string,std::vector<int64_t>>>(packet::read<uint32_t>(input));
    for(auto& it : m_out_data) {
        auto name = packet::read_string(input);
        auto data = packet::read_vector<uint8_t>(input); 
        auto type = packet::read_string(input); 
        auto shape = packet::read_vector<int64_t>(input); 
        it = std::make_tuple(name, data, type, shape);
    }
}

/*
 * onnx_get_ti_benchmark_data_session response packet has a complex structure
 * e.g., vector<string> which I could not figure out how to automate
 * serialization / deserialization
 *
 * the yaml parser lets these classes have user-defined overrides
 * to read_from() / write_to() if __SKIP_SERIALIZE__ is added to the
 * yaml list
 */
void onnx_get_TI_benchmark_data_session_resp::write_to(std::ostream& output) const {
    packet::write<int32_t>(output, m_status);
    packet::write<uint32_t>(output, m_benchmark_data.size());
    for(auto it : m_benchmark_data) {
        packet::write_string(output, it.first);
        packet::write<uint64_t>(output, it.second);
    }
}
void onnx_get_TI_benchmark_data_session_resp::read_from(std::istream& input) {
    m_status = packet::read<int32_t>(input);
    m_benchmark_data = std::vector<std::pair<std::string, uint64_t>>(packet::read<uint32_t>(input));
    for(auto& it : m_benchmark_data) {
        it.first = packet::read_string(input);
        it.second = packet::read<uint64_t>(input);
    }
}
