#ifndef NT_LINALG_ADJUGATE_H__
#define NT_LINALG_ADJUGATE_H__

#include "../../Tensor.h"
#include <functional>
#include <optional>
#include <tuple>
#include <variant>
#include <vector>

namespace nt {
namespace linalg {
NEUROTENSOR_API Tensor adjugate(const Tensor&);
} // namespace linalg
} // namespace nt

#endif //_NT_LINALG_ADJUGATE_H_
