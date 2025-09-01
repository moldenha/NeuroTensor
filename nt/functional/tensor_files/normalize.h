#ifndef NT_FUNCTIONAL_TENSOR_FILES_NORMALIZE_H__
#define NT_FUNCTIONAL_TENSOR_FILES_NORMALIZE_H__

#include "../../Tensor.h"
#include "../../dtype/Scalar.h"

namespace nt {
namespace functional {

NEUROTENSOR_API Tensor& xavier_uniform_(Tensor& tensor);
NEUROTENSOR_API Tensor var(const Tensor &, utils::optional_list dim = nullptr,
           int64_t correction = 1,
           bool keepdim = false); // delta degrees of freedom (0 for population
                                  // variance, 1 for sample variance).
NEUROTENSOR_API Tensor dvar(const Tensor &dx, const Tensor &x,
            utils::optional_list dim = nullptr,
            int64_t correction = 1); // derivative of the var function with
                                     // respect to xi element of the the tensor

} // namespace functional
} // namespace nt

#endif
