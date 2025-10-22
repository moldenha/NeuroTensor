#ifndef NT_FUNCTIONAL_CPU_NORMALIZE_H__
#define NT_FUNCTIONAL_CPU_NORMALIZE_H__

#include "../../dtype/ArrayVoid.h"
#include "../../dtype/Scalar.h"
#include "../../refs/SizeRef.h"

namespace nt::functional::cpu {

NEUROTENSOR_API void xavier_uniform_(ArrayVoid& output, double bound);

NEUROTENSOR_API void _batch_norm_(ArrayVoid& input, ArrayVoid& running_mean, ArrayVoid& running_var,
                                  const ArrayVoid& weight, const ArrayVoid& bias, bool training,
                                  Scalar momentum, Scalar eps, int64_t N, int64_t C, int64_t HW);

NEUROTENSOR_API void _batch_norm_(ArrayVoid& input, ArrayVoid& running_mean, ArrayVoid& running_var,
                                  const ArrayVoid& weight, const ArrayVoid& bias, bool training,
                                  Scalar momentum, Scalar eps, int64_t N, int64_t C, int64_t HW,
                                  ArrayVoid& mean_vals, ArrayVoid& inv_vals);

NEUROTENSOR_API void _batch_norm_backward_weight_bias_(ArrayVoid& grad_weight, ArrayVoid& grad_bias, const ArrayVoid& grad_output,
                                       const ArrayVoid& input, 
                                       Scalar _momentum, Scalar _eps, int64_t N, int64_t C, int64_t HW,
                                        const ArrayVoid& stored_mean_vals, const ArrayVoid& stored_inv_vals);
                                 
NEUROTENSOR_API void _batch_norm_backward_input_(ArrayVoid& grad_input, const ArrayVoid& grad_output,
                                       const ArrayVoid& input,
                                       const ArrayVoid& weight, Scalar _momentum, Scalar _eps, 
                                       int64_t N, int64_t C, int64_t HW,
                                       const ArrayVoid& stored_mean_vals, const ArrayVoid& stored_inv_vals);

NEUROTENSOR_API void _group_norm_(ArrayVoid& input, const ArrayVoid& weight, const ArrayVoid& bias,
                                  int64_t N, int64_t C, int64_t HW,
                                  int64_t num_groups, Scalar _eps, ArrayVoid& mean_vals, ArrayVoid& inv_vals);

NEUROTENSOR_API void _group_norm_(ArrayVoid& input, const ArrayVoid& weight, const ArrayVoid& bias,
                                  int64_t N, int64_t C, int64_t HW,
                                  int64_t num_groups, Scalar _eps);

NEUROTENSOR_API void _group_norm_backward_weight_bias_(ArrayVoid& grad_weight, ArrayVoid& grad_bias, const ArrayVoid& grad_output,
                                                       const ArrayVoid& input, int64_t N, int64_t C, int64_t HW,
                                                       int64_t num_groups, const ArrayVoid& mean_vals, const ArrayVoid& inv_vals);

NEUROTENSOR_API void _group_norm_backward_input_(ArrayVoid& grad_input, const ArrayVoid& grad_output, const ArrayVoid& input,
                                                 const ArrayVoid& weight, int64_t N, int64_t C, int64_t HW,
                                                 int64_t num_groups, Scalar _eps, const ArrayVoid& mean_vals, const ArrayVoid& inv_vals);

}

#endif  
