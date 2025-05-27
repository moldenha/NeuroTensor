// these are functions specifically designed for min-max type functions
// this includes clamp, relu, etc
#ifndef __NT_FUNCTIONAL_TENSOR_FILES_MIN_MAX_H__
#define __NT_FUNCTIONAL_TENSOR_FILES_MIN_MAX_H__

#include "../../Tensor.h"
#include "../../dtype/Scalar.h"

#include <optional>
#include <vector>

namespace nt {
namespace functional {

Tensor clamp(const Tensor &x, std::optional<Scalar> min = std::nullopt,
             std::optional<Scalar> max = std::nullopt);
Tensor& clamp_(Tensor &x, std::optional<Scalar> min = std::nullopt,
             std::optional<Scalar> max = std::nullopt);

Scalar min(std::vector<Scalar>);
Scalar max(std::vector<Scalar>);
Tensor min(std::vector<Tensor>);
Tensor max(std::vector<Tensor>);
Tensor min(std::vector<Tensor>, Scalar);
Tensor max(std::vector<Tensor>, Scalar);
result_types::max<Tensor, Tensor> max(const Tensor&, utils::optional_list list = nullptr);
result_types::max<Tensor, Tensor> min(const Tensor&, utils::optional_list list = nullptr);
Tensor& max_indices(const Tensor& tensor, Tensor& indices, utils::optional_list list = nullptr);
Tensor& min_indices(const Tensor& tensor, Tensor& indices, utils::optional_list list = nullptr);
Tensor max_indices(const Tensor& tensor, utils::optional_list list = nullptr);
Tensor min_indices(const Tensor& tensor, utils::optional_list list = nullptr);

//need to implement the Tensor::min and Tensor::max here as well
Tensor argmin(Tensor);
Tensor argmax(Tensor);
Tensor argmin(Tensor, int64_t dim, bool keepdims=false);
Tensor argmax(Tensor, int64_t dim, bool keepdims=false);

} // namespace functional
} // namespace nt

#include "min_max.hpp"

#endif
