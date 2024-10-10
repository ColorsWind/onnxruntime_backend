#pragma once

#include <algorithm>
#include <mutex>
#include <string>
#include <unordered_map>
#include <atomic>
#include "onnxruntime_loader.h"
#include "triton/backend/backend_common.h"
#include <triton/core/tritonserver.h>

#include <onnxruntime_c_api.h>

namespace triton::backend::onnxruntime {

class CacheOrtSession {
 private:
  std::mutex mutex_;

  const bool is_path_;
  const std::string &model_;
  const OrtSessionOptions *session_options_;
  OrtSession *session_;
  std::atomic<size_t> hotness;
  
  CacheOrtSession(const bool is_path, const std::string &model, const OrtSessionOptions *session_options)
    : is_path_(is_path), model_(model), session_options_(session_options), session_(nullptr), hotness(0) {
  }

  static std::mutex g_mutex;
  static std::unordered_map<std::string, CacheOrtSession*> sessions;
  static size_t loaded_model_num;
  static const constexpr size_t MAX_LOADED_MODEL_NUM = 4;

  static std::vector<std::pair<size_t, CacheOrtSession*>> GetSessionsHotness() {
    std::vector<std::pair<size_t, CacheOrtSession*>> ret;
    ret.reserve(sessions.size());
    for (auto &session : sessions) {
      ret.emplace_back(session.second->GetHotness(), session.second);
    }
    return ret;
  }

 public:
  std::unique_lock<std::mutex> ReserveMutex() {
    std::unique_lock g_lock{g_mutex};
    std::unique_lock lock{mutex_};
    if (session_ != nullptr) {
      return lock;
    }
    if (loaded_model_num >= MAX_LOADED_MODEL_NUM) {
      auto sesions_hotness = GetSessionsHotness();
      std::sort(sesions_hotness.begin(), sesions_hotness.end(), 
      [](auto &a, auto &b) { return a.first < b.first; });
      for (auto try_lock : {true, false}) {
        for (auto &&[hotness, session] : sesions_hotness) {
          if (session == this) { continue; }
          std::unique_lock<std::mutex> other_lock;
          if (try_lock) { 
            other_lock = std::unique_lock{session->mutex_, std::try_to_lock};
            if (!other_lock.owns_lock()) { continue; }
          } else { 
            other_lock = std::unique_lock{session->mutex_};
          }
          if (session->session_ != nullptr) {
            OnnxLoader::UnloadSession(session->session_);
            session->session_ = nullptr;
            loaded_model_num--;
            break;
          }
        }
      }
      if (loaded_model_num >= MAX_LOADED_MODEL_NUM) {
        LOG_MESSAGE(TRITONSERVER_LOG_ERROR, "fail to release model");
      }
    }
    loaded_model_num++;
    OnnxLoader::LoadSession(is_path_, model_, session_options_, &this->session_);
    return lock;
  }
  OrtSession *Session(std::unique_lock<std::mutex> &lock) { return session_; }

  void IncHotness() {
    hotness.fetch_add(1, std::memory_order_relaxed);
  }

  size_t GetHotness() {
    return hotness.load(std::memory_order_relaxed);
  }

  static TRITONSERVER_Error* LoadSession(
    const bool is_path, const std::string& model,
    const OrtSessionOptions* session_options, CacheOrtSession** session) {
    *session = new CacheOrtSession(is_path, model, session_options);
    return nullptr;
  }

  static TRITONSERVER_Error* UnloadSession(CacheOrtSession* session) {
    std::unique_lock lock{session->mutex_};
    if (session->session_ != nullptr) {
      OnnxLoader::UnloadSession(session->session_);
    }
    delete session;
    return nullptr;
  }

};
}  // namespace triton::backend::onnxruntime