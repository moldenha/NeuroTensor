#ifndef _NT_LINALG_DETERMINANT_H_
#define _NT_LINALG_DETERMINANT_H_

#include "../../Tensor.h"
#include <functional>
#include <optional>
#include <tuple>
#include <variant>
#include <vector>

namespace nt {
namespace linalg {
Tensor determinant(const Tensor&);
} // namespace linalg
} // namespace nt

#endif //_NT_LINALG_DETERMINANT_H_
