/*
 * Auto-generated header file. Do not modify.
 * Make changes to /home/a0133185/ti/GIT_C7x_MMA_TIDL/c7x-mma-tidl/ti_dl/release/build_cloud/test/onnxruntime/onnxruntime/remote/onnx_messages.yaml and run
 *
 * # /home/a0133185/ti/GIT_C7x_MMA_TIDL/c7x-mma-tidl/ti_dl/release/build_cloud/test/ti-inference-engine/parse_messages.py /home/a0133185/ti/GIT_C7x_MMA_TIDL/c7x-mma-tidl/ti_dl/release/build_cloud/test/onnxruntime/onnxruntime/remote/onnx_messages.yaml
 *
 * and save the output to /home/a0133185/ti/GIT_C7x_MMA_TIDL/c7x-mma-tidl/ti_dl/release/build_cloud/test/onnxruntime/onnxruntime/remote/onnx_messages.h
 */

#ifndef __ONNX_MESSAGES_H__
#define __ONNX_MESSAGES_H__

#include <memstreambuf.h>
#include <op_registry.h>
#include <packet.h>

class onnx_initialize_cpu_session_resp : public packet {
public:
    ~onnx_initialize_cpu_session_resp() {};
    explicit onnx_initialize_cpu_session_resp() {};
    explicit onnx_initialize_cpu_session_resp(const int32_t& status) :
        m_status(status) {}
    const int32_t& status() { return m_status; }
protected:
    virtual void write_to(std::ostream& output) const override {
        packet::write<int32_t>(output, m_status);
    }
    virtual void read_from(std::istream& input) override {
        m_status = packet::read<int32_t>(input);
    }
private:
    int32_t m_status;
    friend class onnx_initialize_cpu_session_req;
};
class onnx_initialize_cpu_session_req : public packet {
public:
    ~onnx_initialize_cpu_session_req() {};
    explicit onnx_initialize_cpu_session_req() {};
    explicit onnx_initialize_cpu_session_req(const uint32_t& id) :
        m_id(id) {}
    const uint32_t& id() { return m_id; }
    virtual std::unique_ptr<packet> handle() override;
    static const uint32_t transport_id;
protected:
    virtual void write_to(std::ostream& output) const override {
        packet::write<uint32_t>(output, m_id);
    }
    virtual void read_from(std::istream& input) override {
        m_id = packet::read<uint32_t>(input);
    }
private:
    uint32_t m_id;
};

class onnx_initialize_tidl_session_resp : public packet {
public:
    ~onnx_initialize_tidl_session_resp() {};
    explicit onnx_initialize_tidl_session_resp() {};
    explicit onnx_initialize_tidl_session_resp(const int32_t& status) :
        m_status(status) {}
    const int32_t& status() { return m_status; }
protected:
    virtual void write_to(std::ostream& output) const override {
        packet::write<int32_t>(output, m_status);
    }
    virtual void read_from(std::istream& input) override {
        m_status = packet::read<int32_t>(input);
    }
private:
    int32_t m_status;
    friend class onnx_initialize_tidl_session_req;
};
class onnx_initialize_tidl_session_req : public packet {
public:
    ~onnx_initialize_tidl_session_req() {};
    explicit onnx_initialize_tidl_session_req() {};
    explicit onnx_initialize_tidl_session_req(const uint32_t& id, const std::vector<std::pair<std::string, std::string>>& options) :
        m_id(id), 
        m_options(options) {}
    const uint32_t& id() { return m_id; }
    const std::vector<std::pair<std::string, std::string>>& options() { return m_options; }
    virtual std::unique_ptr<packet> handle() override;
    static const uint32_t transport_id;
protected:
    virtual void write_to(std::ostream& output) const override;
    virtual void read_from(std::istream& input) override;
private:
    uint32_t m_id;
    std::vector<std::pair<std::string, std::string>> m_options;
};

class onnx_run_session_resp : public packet {
public:
    ~onnx_run_session_resp() {};
    explicit onnx_run_session_resp() {};
    explicit onnx_run_session_resp(const int32_t& status, const std::vector<std::tuple<std::string,std::vector<uint8_t>,std::string,std::vector<int64_t>>>& out_data) :
        m_status(status), 
        m_out_data(out_data) {}
    const int32_t& status() { return m_status; }
    const std::vector<std::tuple<std::string,std::vector<uint8_t>,std::string,std::vector<int64_t>>>& out_data() { return m_out_data; }
protected:
    virtual void write_to(std::ostream& output) const override;
    virtual void read_from(std::istream& input) override;
private:
    int32_t m_status;
    std::vector<std::tuple<std::string,std::vector<uint8_t>,std::string,std::vector<int64_t>>> m_out_data;
    friend class onnx_run_session_req;
};
class onnx_run_session_req : public packet {
public:
    ~onnx_run_session_req() {};
    explicit onnx_run_session_req() {};
    explicit onnx_run_session_req(const uint32_t& id, const std::vector<std::tuple<std::string,std::vector<uint8_t>,std::string,std::vector<int64_t>>>& in_data, const std::vector<std::string>& out_names) :
        m_id(id), 
        m_in_data(in_data), 
        m_out_names(out_names) {}
    const uint32_t& id() { return m_id; }
    const std::vector<std::tuple<std::string,std::vector<uint8_t>,std::string,std::vector<int64_t>>>& in_data() { return m_in_data; }
    const std::vector<std::string>& out_names() { return m_out_names; }
    virtual std::unique_ptr<packet> handle() override;
    static const uint32_t transport_id;
protected:
    virtual void write_to(std::ostream& output) const override;
    virtual void read_from(std::istream& input) override;
private:
    uint32_t m_id;
    std::vector<std::tuple<std::string,std::vector<uint8_t>,std::string,std::vector<int64_t>>> m_in_data;
    std::vector<std::string> m_out_names;
};

class onnx_session_from_file_resp : public packet {
public:
    ~onnx_session_from_file_resp() {};
    explicit onnx_session_from_file_resp() {};
    explicit onnx_session_from_file_resp(const int32_t& status, const uint32_t& id) :
        m_status(status), 
        m_id(id) {}
    const int32_t& status() { return m_status; }
    const uint32_t& id() { return m_id; }
protected:
    virtual void write_to(std::ostream& output) const override {
        packet::write<int32_t>(output, m_status);
        packet::write<uint32_t>(output, m_id);
    }
    virtual void read_from(std::istream& input) override {
        m_status = packet::read<int32_t>(input);
        m_id = packet::read<uint32_t>(input);
    }
private:
    int32_t m_status;
    uint32_t m_id;
    friend class onnx_session_from_file_req;
};
class onnx_session_from_file_req : public packet {
public:
    ~onnx_session_from_file_req() {};
    explicit onnx_session_from_file_req() {};
    explicit onnx_session_from_file_req(const std::string& path) :
        m_path(path) {}
    const std::string& path() { return m_path; }
    virtual std::unique_ptr<packet> handle() override;
    static const uint32_t transport_id;
protected:
    virtual void write_to(std::ostream& output) const override {
        packet::write_string(output, m_path);
    }
    virtual void read_from(std::istream& input) override {
        m_path = packet::read_string(input);
    }
private:
    std::string m_path;
};

class onnx_session_from_buffer_resp : public packet {
public:
    ~onnx_session_from_buffer_resp() {};
    explicit onnx_session_from_buffer_resp() {};
    explicit onnx_session_from_buffer_resp(const int32_t& status, const uint32_t& id) :
        m_status(status), 
        m_id(id) {}
    const int32_t& status() { return m_status; }
    const uint32_t& id() { return m_id; }
protected:
    virtual void write_to(std::ostream& output) const override {
        packet::write<int32_t>(output, m_status);
        packet::write<uint32_t>(output, m_id);
    }
    virtual void read_from(std::istream& input) override {
        m_status = packet::read<int32_t>(input);
        m_id = packet::read<uint32_t>(input);
    }
private:
    int32_t m_status;
    uint32_t m_id;
    friend class onnx_session_from_buffer_req;
};
class onnx_session_from_buffer_req : public packet {
public:
    ~onnx_session_from_buffer_req() {};
    explicit onnx_session_from_buffer_req() {};
    explicit onnx_session_from_buffer_req(const std::vector<uint8_t>& buffer) :
        m_buffer(buffer) {}
    const std::vector<uint8_t>& buffer() { return m_buffer; }
    virtual std::unique_ptr<packet> handle() override;
    static const uint32_t transport_id;
protected:
    virtual void write_to(std::ostream& output) const override {
        packet::write_vector<uint8_t>(output, m_buffer);
    }
    virtual void read_from(std::istream& input) override {
        m_buffer = packet::read_vector<uint8_t>(input);
    }
private:
    std::vector<uint8_t> m_buffer;
};

class onnx_destroy_session_resp : public packet {
public:
    ~onnx_destroy_session_resp() {};
    explicit onnx_destroy_session_resp() {};
    explicit onnx_destroy_session_resp(const int32_t& status) :
        m_status(status) {}
    const int32_t& status() { return m_status; }
protected:
    virtual void write_to(std::ostream& output) const override {
        packet::write<int32_t>(output, m_status);
    }
    virtual void read_from(std::istream& input) override {
        m_status = packet::read<int32_t>(input);
    }
private:
    int32_t m_status;
    friend class onnx_destroy_session_req;
};
class onnx_destroy_session_req : public packet {
public:
    ~onnx_destroy_session_req() {};
    explicit onnx_destroy_session_req() {};
    explicit onnx_destroy_session_req(const uint32_t& id) :
        m_id(id) {}
    const uint32_t& id() { return m_id; }
    virtual std::unique_ptr<packet> handle() override;
    static const uint32_t transport_id;
protected:
    virtual void write_to(std::ostream& output) const override {
        packet::write<uint32_t>(output, m_id);
    }
    virtual void read_from(std::istream& input) override {
        m_id = packet::read<uint32_t>(input);
    }
private:
    uint32_t m_id;
};

class onnx_get_TI_benchmark_data_session_resp : public packet {
public:
    ~onnx_get_TI_benchmark_data_session_resp() {};
    explicit onnx_get_TI_benchmark_data_session_resp() {};
    explicit onnx_get_TI_benchmark_data_session_resp(const int32_t& status, const std::vector<std::pair<std::string, uint64_t>>& benchmark_data) :
        m_status(status), 
        m_benchmark_data(benchmark_data) {}
    const int32_t& status() { return m_status; }
    const std::vector<std::pair<std::string, uint64_t>>& benchmark_data() { return m_benchmark_data; }
protected:
    virtual void write_to(std::ostream& output) const override;
    virtual void read_from(std::istream& input) override;
private:
    int32_t m_status;
    std::vector<std::pair<std::string, uint64_t>> m_benchmark_data;
    friend class onnx_get_TI_benchmark_data_session_req;
};
class onnx_get_TI_benchmark_data_session_req : public packet {
public:
    ~onnx_get_TI_benchmark_data_session_req() {};
    explicit onnx_get_TI_benchmark_data_session_req() {};
    explicit onnx_get_TI_benchmark_data_session_req(const uint32_t& id) :
        m_id(id) {}
    const uint32_t& id() { return m_id; }
    virtual std::unique_ptr<packet> handle() override;
    static const uint32_t transport_id;
protected:
    virtual void write_to(std::ostream& output) const override {
        packet::write<uint32_t>(output, m_id);
    }
    virtual void read_from(std::istream& input) override {
        m_id = packet::read<uint32_t>(input);
    }
private:
    uint32_t m_id;
};


#endif
