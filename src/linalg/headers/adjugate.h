#ifndef _NT_LINALG_ADJUGATE_H_
#define _NT_LINALG_ADJUGATE_H_

#include "../../Tensor.h"
#include <functional>
#include <optional>
#include <tuple>
#include <variant>
#include <vector>

namespace nt {
namespace linalg {
Tensor adjugate(const Tensor&);
} // namespace linalg
} // namespace nt

#endif //_NT_LINALG_ADJUGATE_H_
