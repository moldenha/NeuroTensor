#include "normalize.h"
#include "../../dtype/ArrayVoid.h"
#include "../../dtype/ArrayVoid_NTensor.hpp"
#include "../../dtype/DType_enum.h"
#include "../../dtype/Scalar.h"
#include "../../refs/SizeRef.h"
#include <random>
#include "rand.h"

namespace nt::functional::cpu {



// Forward: instance norm
void _instance_norm_(
    ArrayVoid& input,
    const ArrayVoid& weight,
    const ArrayVoid& bias,
    int64_t N, int64_t C, int64_t HW,
    Scalar _eps, Scalar _momentum,
    bool use_input_stats,
    ArrayVoid& running_mean,
    ArrayVoid& running_var);

// Forward: instance norm and store mean/inv (mean_vals and inv_vals must be size N*C)

void _instance_norm_(
    ArrayVoid& input,
    const ArrayVoid& weight,
    const ArrayVoid& bias,
    int64_t N, int64_t C, int64_t HW,
    Scalar _eps, Scalar _momentum,
    bool use_input_stats,
    ArrayVoid& running_mean,
    ArrayVoid& running_var,
    ArrayVoid& mean_vals,
    ArrayVoid& inv_vals);


// Backward for weight and bias
void _instance_norm_backward_weight_bias_(
    ArrayVoid& grad_weight,
    ArrayVoid& grad_bias,
    const ArrayVoid& grad_output,
    const ArrayVoid& input,
    int64_t N, int64_t C, int64_t HW,
    const ArrayVoid& mean_vals,
    const ArrayVoid& inv_vals);

// Backward for input
void _instance_norm_backward_input_(
    ArrayVoid& grad_input,
    const ArrayVoid& grad_output,
    const ArrayVoid& input,
    const ArrayVoid& weight,
    int64_t N, int64_t C, int64_t HW,
    Scalar _eps,
    const ArrayVoid& mean_vals,
    const ArrayVoid& inv_vals);


} // namespace nt::functional::cpu::
