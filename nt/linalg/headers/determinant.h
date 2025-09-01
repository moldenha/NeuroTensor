#ifndef NT_LINALG_DETERMINANT_H__
#define NT_LINALG_DETERMINANT_H__

#include "../../Tensor.h"
#include <functional>
#include <optional>
#include <tuple>
#include <variant>
#include <vector>

namespace nt {
namespace linalg {
NEUROTENSOR_API Tensor determinant(const Tensor&);
} // namespace linalg
} // namespace nt

#endif //_NT_LINALG_DETERMINANT_H_
