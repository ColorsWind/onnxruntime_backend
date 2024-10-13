#pragma once

#include <triton/backend/backend_input_collector.h>
#include <triton/backend/backend_memory.h>
#include <triton/backend/backend_model_instance.h>
#include <triton/backend/backend_model.h>

#include <onnxruntime_utils.h>

#include <onnxruntime_c_api.h>
#include <variant>

namespace triton::backend::onnxruntime {
//
// ModelState
//
// State associated with a model that is using this backend. An object
// of this class is created and associated with each
// TRITONBACKEND_Model.
//
class ModelState : public BackendModel {
 public:
  static TRITONSERVER_Error* Create(
      TRITONBACKEND_Model* triton_model, ModelState** state);
  virtual ~ModelState() = default;

  // Load an ONNX model using 'artifact_name' as the name for the ONNX
  // file/directory. If 'instance_group_kind' is not
  // TRITONSERVER_INSTANCEGROUPKIND_AUTO then use it and
  // 'instance_group_device_id' to initialize the appropriate
  // execution providers. Return in 'model_path' the full path to the
  // onnx file, return in 'session' and 'allocator' the ORT session
  // and allocator.
  TRITONSERVER_Error* LoadModel(
      const std::string& artifact_name,
      const TRITONSERVER_InstanceGroupKind instance_group_kind,
      const int32_t instance_group_device_id, std::string* model_path,
      OrtSession** session, OrtAllocator** default_allocator,
      cudaStream_t stream);

  const std::map<std::string, std::pair<int64_t, int64_t>>& ModelOutputs()
  {
    return model_outputs_;
  }

 private:
  ModelState(TRITONBACKEND_Model* triton_model);
  TRITONSERVER_Error* AutoCompleteConfig();
  TRITONSERVER_Error* AutoCompleteMaxBatch(
      const OnnxTensorInfoMap& input_tensor_infos,
      const OnnxTensorInfoMap& output_tensor_infos);
  TRITONSERVER_Error* AutoCompleteIO(
      const char* key, const OnnxTensorInfoMap& io_infos);
  std::vector<std::pair<std::string, std::variant<std::string, void*>>> cuda_options_str;
  // Session options used when creating a ORT session.
  std::unique_ptr<OrtSessionOptions, SessionOptionsDeleter> session_options_;

  // model_outputs is a map that contains unique outputs that the model must
  // provide. In the model configuration, the output in the state configuration
  // can have intersection with the outputs section of the model. If an output
  // is specified both in the output section and state section, it indicates
  // that the backend must return the output state to the client too.
  std::map<std::string, std::pair<int64_t, int64_t>> model_outputs_;
};

//
// ModelInstanceState
//
// State associated with a model instance. An object of this class is
// created and associated with each TRITONBACKEND_ModelInstance.
//
class ModelInstanceState : public BackendModelInstance {
 public:
  static TRITONSERVER_Error* Create(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance,
      ModelInstanceState** state);
  virtual ~ModelInstanceState();

  // Get the state of the model that corresponds to this instance.
  ModelState* StateForModel() const { return model_state_; }

  // Execute...
  void ProcessRequests(
      TRITONBACKEND_Request** requests, const uint32_t request_count);

 private:
  ModelInstanceState(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance);
  void ReleaseOrtRunResources();
  TRITONSERVER_Error* ValidateBooleanSequenceControl(
      triton::common::TritonJson::Value& sequence_batching,
      const std::string& control_kind, bool required, bool* have_control);
  TRITONSERVER_Error* ValidateTypedSequenceControl(
      triton::common::TritonJson::Value& sequence_batching,
      const std::string& control_kind, bool required, bool* have_control);
  TRITONSERVER_Error* ValidateInputs(const size_t expected_input_cnt);
  TRITONSERVER_Error* ValidateOutputs();
  TRITONSERVER_Error* OrtRun(
      std::vector<TRITONBACKEND_Response*>* responses,
      const uint32_t response_count);
  TRITONSERVER_Error* SetInputTensors(
      size_t total_batch_size, TRITONBACKEND_Request** requests,
      const uint32_t request_count,
      std::vector<TRITONBACKEND_Response*>* responses,
      BackendInputCollector* collector, std::vector<const char*>* input_names,
      bool* cuda_copy);
  TRITONSERVER_Error* SetStringInputTensor(
      TRITONBACKEND_Request** requests, const uint32_t request_count,
      std::vector<TRITONBACKEND_Response*>* responses, const char* input_name,
      std::vector<const char*>* string_ptrs, bool* cuda_copy);
  void SetStringInputBuffer(
      const std::string& name, const std::vector<size_t>& expected_byte_sizes,
      const std::vector<size_t>& expected_element_cnts,
      std::vector<TRITONBACKEND_Response*>* responses, char* input_buffer,
      std::vector<const char*>* string_ptrs);
  void FillStringData(std::vector<const char*>* string_ptrs, size_t cnt);
  TRITONSERVER_Error* ReadOutputTensors(
      size_t total_batch_size, TRITONBACKEND_Request** requests,
      const uint32_t request_count,
      std::vector<TRITONBACKEND_Response*>* responses);

  TRITONSERVER_Error* ReadOutputTensor(
      std::vector<int64_t>& batchn_shape, TRITONSERVER_DataType& dtype,
      OrtValue* output_tensor, void** output_buffer,
      std::vector<std::vector<char>>& string_buffers,
      std::vector<size_t>& offsets);
  bool SetStringOutputBuffer(
      const std::string& name, const char* content, const size_t* offsets,
      std::vector<int64_t>* batchn_shape, TRITONBACKEND_Request** requests,
      const uint32_t request_count,
      std::vector<TRITONBACKEND_Response*>* responses);
  bool SetStringStateBuffer(
      const std::string& name, const char* content, const size_t* offsets,
      std::vector<int64_t>* batchn_shape, TRITONBACKEND_Request** requests,
      const uint32_t request_count,
      std::vector<TRITONBACKEND_Response*>* responses);
  bool SetStringBuffer(
      const std::string& name, const char* content, const size_t* offsets,
      std::vector<int64_t>* batchn_shape, TRITONBACKEND_Request** requests,
      const uint32_t request_count,
      std::vector<TRITONBACKEND_Response*>* responses, bool state);

  ModelState* model_state_;

  // The full path to the ONNX model file.
  std::string model_path_;

  // Onnx Runtime variables that are used across runs on this
  // instance.
  OrtSession* session_;
  OrtAllocator* default_allocator_;
  OrtMemoryInfo* cuda_allocator_info_;
  const OrtMemoryInfo* cpu_allocator_info_;
  OrtIoBinding* io_binding_;
  OrtRunOptions* runOptions_;
  // map of output name -> bound mem type and id
  std::unordered_map<std::string, std::pair<TRITONSERVER_MemoryType, int64_t>>
      output_device_info_;
  // map of output name -> tensor info
  OnnxTensorInfoMap output_tensor_infos_;

  // map of input name -> tensor info
  OnnxTensorInfoMap input_tensor_infos_;

  // A map from scalar output tensors to the dimension specified in model config
  std::unordered_map<std::string, std::vector<int64_t>> scalar_outputs_;

  // Onnx Runtime variables that will be reset and used for every run
  // on this instance.
  std::vector<OrtValue*> input_tensors_;
  std::vector<OrtValue*> output_tensors_;
  OrtValue** output_buffer_;
  std::vector<BackendMemory*> input_tensor_memories_;
};

} 