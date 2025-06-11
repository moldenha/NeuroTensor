#ifndef _NT_LINALG_NORM_H_
#define _NT_LINALG_NORM_H_

#include "../../Tensor.h"
#include "../../utils/optional_list.h"
#include <functional>
#include <optional>
#include <tuple>
#include <variant>
#include <vector>

namespace nt {
namespace linalg {
Tensor norm(const Tensor& A, std::variant<std::nullptr_t, std::string, int64_t> ord = nullptr, utils::optional_list dim = nullptr, bool keepdim = false);
} // namespace linalg
} // namespace nt

#endif //_NT_LINALG_NORM_H_
