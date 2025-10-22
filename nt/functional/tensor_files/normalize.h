#ifndef NT_FUNCTIONAL_TENSOR_FILES_NORMALIZE_H__
#define NT_FUNCTIONAL_TENSOR_FILES_NORMALIZE_H__

#include "../../Tensor.h"
#include "../../dtype/Scalar.h"
#include "../../utils/optional_tensor.h"

namespace nt::functional {


NEUROTENSOR_API Tensor& xavier_uniform_(Tensor& tensor);
NEUROTENSOR_API Tensor var(const Tensor &, utils::optional_list dim = nullptr,
           int64_t correction = 1,
           bool keepdim = false); // delta degrees of freedom (0 for population
                                  // variance, 1 for sample variance).
NEUROTENSOR_API Tensor dvar(const Tensor &dx, const Tensor &x,
            utils::optional_list dim = nullptr,
            int64_t correction = 1); // derivative of the var function with
                                     // respect to xi element of the the tensor

namespace no_grad {
NEUROTENSOR_API Tensor& batch_norm_(Tensor& x, Tensor running_mean, Tensor running_var, 
                                    utils::optional_tensor weight = nullptr, utils::optional_tensor bias = nullptr,
                                    bool training = false, Scalar momentum = 0.1, Scalar eps = 1e-05,
                                    intrusive_ptr<tensor_holder> stored_means = nullptr, intrusive_ptr<tensor_holder> = nullptr);

NEUROTENSOR_API Tensor batch_norm(const Tensor& x, Tensor running_mean, Tensor running_var, 
                                    utils::optional_tensor weight = nullptr, utils::optional_tensor bias = nullptr,
                                    bool training = false, Scalar momentum = 0.1, Scalar eps = 1e-05,
                                    intrusive_ptr<tensor_holder> stored_means = nullptr, intrusive_ptr<tensor_holder> stored_inv = nullptr);

NEUROTENSOR_API Tensor batch_norm_backward(const Tensor& grad, const Tensor& input,
                                           const Tensor& stored_means, const Tensor& stored_inv,
                                           Tensor original_weight = Tensor::Null(), Tensor original_bias = Tensor::Null(),
                                           Scalar momentum = 0.1, Scalar eps = 1e-05);

NEUROTENSOR_API Tensor& group_norm_(Tensor& input, int64_t num_groups, 
                    utils::optional_tensor weight = nullptr, utils::optional_tensor bias = nullptr,
                    Scalar eps = 1e-05, intrusive_ptr<tensor_holder> stored_means = nullptr, intrusive_ptr<tensor_holder> stored_inv = nullptr);

NEUROTENSOR_API Tensor group_norm(const Tensor& input, int64_t num_groups, 
                    utils::optional_tensor weight = nullptr, utils::optional_tensor bias = nullptr,
                    Scalar eps = 1e-05, intrusive_ptr<tensor_holder> stored_means = nullptr, intrusive_ptr<tensor_holder> stored_inv = nullptr);

NEUROTENSOR_API Tensor group_norm_backward(const Tensor& grad, const Tensor& input, int64_t num_groups,
                                           const Tensor& stored_means, const Tensor& stored_inv,
                                           Tensor original_weight = Tensor::Null(), Tensor original_bias = Tensor::Null(), Scalar eps = 1e-05);

} // namespace no_grad
} // namespace nt::functional

#endif
