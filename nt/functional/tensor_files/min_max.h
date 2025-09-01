// these are functions specifically designed for min-max type functions
// this includes clamp, relu, etc
#ifndef NT_FUNCTIONAL_TENSOR_FILES_MIN_MAX_H__
#define NT_FUNCTIONAL_TENSOR_FILES_MIN_MAX_H__

#include "../../Tensor.h"
#include "../../dtype/Scalar.h"

#include <optional>
#include <vector>

namespace nt {
namespace functional {

NEUROTENSOR_API Tensor clamp(const Tensor &x, std::optional<Scalar> min = std::nullopt,
             std::optional<Scalar> max = std::nullopt);
NEUROTENSOR_API Tensor& clamp_(Tensor &x, std::optional<Scalar> min = std::nullopt,
             std::optional<Scalar> max = std::nullopt);

NEUROTENSOR_API Scalar minimum(std::vector<Scalar>);
NEUROTENSOR_API Scalar maximum(std::vector<Scalar>);
NEUROTENSOR_API Tensor minimum(std::vector<Tensor>);
NEUROTENSOR_API Tensor maximum(std::vector<Tensor>);
NEUROTENSOR_API Tensor minimum(std::vector<Tensor>, Scalar);
NEUROTENSOR_API Tensor maximum(std::vector<Tensor>, Scalar);
NEUROTENSOR_API result_types::max<Tensor, Tensor> max(const Tensor&, utils::optional_list list = nullptr, bool keepdim = false);
NEUROTENSOR_API result_types::max<Tensor, Tensor> min(const Tensor&, utils::optional_list list = nullptr, bool keepdim = false);
NEUROTENSOR_API Tensor& max_indices(const Tensor& tensor, Tensor& indices, utils::optional_list list = nullptr);
NEUROTENSOR_API Tensor& min_indices(const Tensor& tensor, Tensor& indices, utils::optional_list list = nullptr);
NEUROTENSOR_API Tensor max_indices(const Tensor& tensor, utils::optional_list list = nullptr);
NEUROTENSOR_API Tensor min_indices(const Tensor& tensor, utils::optional_list list = nullptr);

//need to implement the Tensor::min and Tensor::max here as well
NEUROTENSOR_API Tensor argmin(Tensor);
NEUROTENSOR_API Tensor argmax(Tensor);
NEUROTENSOR_API Tensor argmin(Tensor, int64_t dim, bool keepdims=false);
NEUROTENSOR_API Tensor argmax(Tensor, int64_t dim, bool keepdims=false);

} // namespace functional
} // namespace nt

// #include "min_max.hpp"

#endif
