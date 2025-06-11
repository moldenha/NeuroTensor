#ifndef __NT_FUNCTIONAL_TENSOR_FILES_NORMALIZE_H__
#define __NT_FUNCTIONAL_TENSOR_FILES_NORMALIZE_H__

#include "../../Tensor.h"
#include "../../dtype/Scalar.h"

namespace nt {
namespace functional {

void xavier_uniform_(Tensor& tensor);
Tensor var(const Tensor &, utils::optional_list dim = nullptr,
           int64_t correction = 1,
           bool keepdim = false); // delta degrees of freedom (0 for population
                                  // variance, 1 for sample variance).
Tensor dvar(const Tensor &dx, const Tensor &x,
            utils::optional_list dim = nullptr,
            int64_t correction = 1); // derivative of the var function with
                                     // respect to xi element of the the tensor

} // namespace functional
} // namespace nt

#endif
