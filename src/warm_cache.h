#pragma once

#include <algorithm>
#include <cstddef>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <atomic>
#include <onnxruntime_loader.h>
#include <onnxruntime_utils.h>
#include <triton/backend/backend_common.h>
#include <triton/backend/backend_model.h>
#include <triton/core/tritonserver.h>
#include <triton/core/tritonbackend.h>
#include <onnxruntime.h>

#include <onnxruntime_c_api.h>

namespace triton::backend::onnxruntime {

static const constexpr bool SKIP_WARM_CACHE = false;

class CacheModelInstanceState {
 private:
  std::mutex s_mutex_;

  ModelState *arg0_model_state_;
  TRITONBACKEND_ModelInstance* arg1_triton_model_instance;
  std::atomic<size_t> hotness{0};
  ModelInstanceState *mayeb_state_{nullptr};
  
  CacheModelInstanceState(ModelState *arg0_model_state, TRITONBACKEND_ModelInstance* arg1_triton_model_instance)
    : arg0_model_state_(arg0_model_state), arg1_triton_model_instance(arg1_triton_model_instance) {
  }

  ~CacheModelInstanceState() {
    if (mayeb_state_) {
      LOG_MESSAGE(TRITONSERVER_LOG_ERROR, std::string("not delete maybe_state_: " + arg0_model_state_->Name()).c_str());
    }
  }

  static std::recursive_mutex g_mutex;
  static std::unordered_map<std::string, CacheModelInstanceState*> sessions;
  static size_t loaded_model_num;
  static const constexpr size_t MAX_LOADED_MODEL_NUM = 4;

  static std::vector<std::pair<size_t, CacheModelInstanceState*>> GetSessionsHotness() {
    std::vector<std::pair<size_t, CacheModelInstanceState*>> ret;
    ret.reserve(sessions.size());
    for (auto &session : sessions) {
      ret.emplace_back(session.second->GetHotness(), session.second);
    }
    return ret;
  }

 public:
  const std::string& Name() {
    return arg0_model_state_->Name();
  }
  std::unique_lock<std::mutex> ReserveMutex() {
    if constexpr (SKIP_WARM_CACHE) { return {}; }
    LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string("[WarmCache] ReserveMutex: ") + arg0_model_state_->Name() + ".").c_str());
    std::unique_lock g_lock{g_mutex};
    std::unique_lock s_lock{s_mutex_};
    if (mayeb_state_ != nullptr) {
      LOG_MESSAGE(TRITONSERVER_LOG_INFO, 
      (std::string("[WarmCache] ReserveMutex: ") + Name() + ":  still alive.").c_str());
      return s_lock;
    }
    if (loaded_model_num >= MAX_LOADED_MODEL_NUM) {
      LOG_MESSAGE(TRITONSERVER_LOG_INFO, 
      (std::string("[WarmCache] ReserveMutex: ") + Name() + ": dead, no room.").c_str());
      auto sesions_hotness = GetSessionsHotness();
      LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string("[WarmCache] Hotness ") + std::to_string(sesions_hotness.size()) + ".").c_str());
      std::sort(sesions_hotness.begin(), sesions_hotness.end(), 
      [](auto &a, auto &b) { return a.first < b.first; });
      for (auto try_lock : {true, false}) {
        for (auto &&[hotness, t_state] : sesions_hotness) {
          if (t_state == this) { continue; }
          std::unique_lock<std::mutex> t_lock;
          if (try_lock) { 
            t_lock = std::unique_lock{t_state->s_mutex_, std::try_to_lock};
            if (!t_lock.owns_lock()) { continue; }
          } else { 
            t_lock = std::unique_lock{t_state->s_mutex_};
          }
          if (t_state->mayeb_state_ != nullptr) {
             LOG_MESSAGE(TRITONSERVER_LOG_INFO, 
            (std::string("[WarmCache] ReserveMutex: ") + Name() + ": evict " + t_state->Name() + ".").c_str());
            delete t_state->mayeb_state_;
            t_state->mayeb_state_ = nullptr;
            loaded_model_num--;
            break;
          }
        }
      }
      if (loaded_model_num >= MAX_LOADED_MODEL_NUM) {
        LOG_MESSAGE(TRITONSERVER_LOG_ERROR, (std::string("fail to release model: ") + std::to_string(loaded_model_num)).c_str());
      }
    } else {
      LOG_MESSAGE(TRITONSERVER_LOG_INFO, 
      (std::string("[WarmCache] ReserveMutex: ") + Name() + ": dead, has room.").c_str());
    }
    loaded_model_num++;
    auto err = ModelInstanceState::Create(arg0_model_state_, arg1_triton_model_instance, &mayeb_state_);
    if (err != nullptr) {
      LOG_MESSAGE(TRITONSERVER_LOG_INFO, 
      (std::string("Fail to create instance state: ") 
            + arg0_model_state_->Name()
            + " : " 
            + std::string(TRITONSERVER_ErrorMessage(err)) 
            + ".").c_str());
    }
    return s_lock;
  }
  ModelInstanceState *State(std::unique_lock<std::mutex> &lock) { return mayeb_state_; }

  void IncHotness() {
    hotness.fetch_add(1, std::memory_order_relaxed);
  }

  size_t GetHotness() {
    return hotness.load(std::memory_order_relaxed);
  }

  static TRITONSERVER_Error* Create(
    ModelState *arg0_model_state, TRITONBACKEND_ModelInstance* arg1_triton_model_instance, CacheModelInstanceState** state) {
    LOG_MESSAGE(TRITONSERVER_LOG_INFO, 
    (std::string("[WarmCache] Create cache item: ") + arg0_model_state->Name() + ".").c_str());
    *state = new CacheModelInstanceState(arg0_model_state, arg1_triton_model_instance);
    {
      std::unique_lock lock{g_mutex};
      if (sessions.find(arg0_model_state->Name()) != sessions.cend()) {
        LOG_MESSAGE(TRITONSERVER_LOG_ERROR, (std::string("Duplicate key: " + arg0_model_state->Name() + ".").c_str()));
        throw std::runtime_error("Fail to Create.");
      }
      sessions[arg0_model_state->Name()] = *state;
    }

    return nullptr;
  }

  static TRITONSERVER_Error* Delete(CacheModelInstanceState* state) {
    LOG_MESSAGE(TRITONSERVER_LOG_INFO, 
    (std::string("[WarmCache] Release cache item: ") + state->Name() + ".").c_str());
    {
      std::unique_lock g_lock{g_mutex};
      std::unique_lock s_lock{state->s_mutex_};
      if (state->mayeb_state_ != nullptr) {
        delete state->mayeb_state_;
        state->mayeb_state_ = nullptr;
        loaded_model_num--;
      }
      sessions.erase(state->Name());
    }
    delete state;
    return nullptr;
  }

};
}  // namespace triton::backend::onnxruntime