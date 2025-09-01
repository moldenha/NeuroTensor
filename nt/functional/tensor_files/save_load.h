#ifndef NT_FUNCTIONAL_TENSOR_FILES_SAVE_LOAD_H__
#define NT_FUNCTIONAL_TENSOR_FILES_SAVE_LOAD_H__

#include "../../Tensor.h"
#include <string>

namespace nt {
namespace functional {
NEUROTENSOR_API Tensor load(std::string);
NEUROTENSOR_API void save(Tensor, std::string);
NEUROTENSOR_API void save(const Tensor &, const char *);
NEUROTENSOR_API Tensor load(const char *);
} // namespace functional
} // namespace nt

#endif
