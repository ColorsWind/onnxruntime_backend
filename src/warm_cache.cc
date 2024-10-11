#include <warm_cache.h>
#include <mutex>


namespace triton::backend::onnxruntime {

std::recursive_mutex CacheModelInstanceState::g_mutex;
std::unordered_map<std::string, CacheModelInstanceState*> CacheModelInstanceState::sessions;
size_t CacheModelInstanceState::loaded_model_num = 0;

}