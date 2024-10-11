#include <warm_cache.h>
#include <mutex>


namespace triton::backend::onnxruntime {

std::recursive_mutex CacheOrtSession::g_mutex;
std::unordered_map<std::string, CacheOrtSession*> CacheOrtSession::sessions;
size_t CacheOrtSession::loaded_model_num = 0;

}