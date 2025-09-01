#ifndef NT_LINALG_EYE_H__
#define NT_LINALG_EYE_H__

#include "../../Tensor.h"
#include <functional>
#include <optional>
#include <tuple>
#include <variant>
#include <vector>

namespace nt {
namespace linalg {
NEUROTENSOR_API Tensor eye(int64_t n, int64_t b=0, DType dtype = DType::Float32); //make the identity matrix, b means batches
NEUROTENSOR_API Tensor eye_like(const Tensor&);
} // namespace linalg
} // namespace nt

#endif //_NT_LINALG_EYE_H_
