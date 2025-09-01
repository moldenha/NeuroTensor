#ifndef NT_LINALG_NORM_H__
#define NT_LINALG_NORM_H__

#include "../../Tensor.h"
#include "../../utils/optional_list.h"
#include <functional>
#include <optional>
#include <tuple>
#include <variant>
#include <vector>

namespace nt {
namespace linalg {
NEUROTENSOR_API Tensor norm(const Tensor& A, std::variant<std::nullptr_t, std::string, int64_t> ord = nullptr, utils::optional_list dim = nullptr, bool keepdim = false);
} // namespace linalg
} // namespace nt

#endif //_NT_LINALG_NORM_H_
