#include "normalize.h"
#include "../../dtype/ArrayVoid.h"
#include "../../dtype/ArrayVoid_NTensor.hpp"
#include "../../dtype/DType_enum.h"
#include "../../dtype/Scalar.h"
#include "../../refs/SizeRef.h"
#include <random>
#include "rand.h"

namespace nt::functional::cpu {


template<typename iter, typename T>
void calc_stats_use_input_update(
    T& mean, T& var, iter x_ptr,
    const int64_t& inner_size, const int64_t& HW, const int64_t& C,
    const int64_t& N, const int64_t& c, const int64_t& n,
    T* run_mean_ptr, T* run_var_ptr, T momentum)
{
    mean = 0;
    for (int64_t hw = 0; hw < HW; ++hw)
        mean += x_ptr[((n * C + c) * HW) + hw];
    mean /= inner_size;

    var = 0;
    for (int64_t hw = 0; hw < HW; ++hw) {
        T diff = x_ptr[((n * C + c) * HW) + hw] - mean;
        var += diff * diff;
    }
    var /= inner_size;

    // --- Update running stats ---
    run_mean_ptr[c] = (1 - momentum) * run_mean_ptr[c] + momentum * mean;
    run_var_ptr[c]  = (1 - momentum) * run_var_ptr[c] + momentum * var;
}

template<typename iter, typename T>
void calc_stats_use_input(
    T& mean, T& var, iter x_ptr,
    const int64_t& inner_size, const int64_t& HW, const int64_t& C,
    const int64_t& N, const int64_t& c, const int64_t& n,
    T*, T*, T)
{
    mean = 0;
    for (int64_t hw = 0; hw < HW; ++hw)
        mean += x_ptr[((n * C + c) * HW) + hw];
    mean /= inner_size;

    var = 0;
    for (int64_t hw = 0; hw < HW; ++hw) {
        T diff = x_ptr[((n * C + c) * HW) + hw] - mean;
        var += diff * diff;
    }
    var /= inner_size;
}

template<typename iter, typename T>
void calc_stats_no_use_input(
    T& mean, T& var, iter, 
    const int64_t&, const int64_t&, const int64_t&, const int64_t&, const int64_t&, const int64_t&,
    T* run_mean_ptr, T* run_var_ptr, T)
{
    mean = run_mean_ptr[c];
    var = run_var_ptr[c];
}

// Forward: instance norm
void _instance_norm_(
    ArrayVoid& input,
    const ArrayVoid& weight,
    const ArrayVoid& bias,
    int64_t N, int64_t C, int64_t HW,
    Scalar _eps, Scalar _momentum,
    bool use_input_stats,
    ArrayVoid& running_mean,
    ArrayVoid& running_var)
{
    if (!weight.is_contiguous())
        throw std::logic_error("weights for cpu::_instance_norm_ must be contiguous");
    if (!bias.is_contiguous())
        throw std::logic_error("bias for cpu::_instance_norm_ must be contiguous");

    if (!running_mean.is_null() && (!running_mean.is_contiguous() || running_mean.Size() != C))
        throw std::logic_error("Running mean must be contiguous and size == C");
    if (!running_var.is_null() && (!running_var.is_contiguous() || running_var.Size() != C))
        throw std::logic_error("Running var must be contiguous and size == C");
    if (!use_input_stats && (running_mean.is_null() || running_var.is_null()))
        throw std::logic_error("If not using input stats, running mean and var must be defined");

    input.execute_function<WRAP_DTYPES<NumberTypesL>>([&](auto x_ptr, auto x_end) {
        using scalar_t = utils::IteratorBaseType_t<decltype(x_ptr)>;

        const scalar_t* w_ptr = reinterpret_cast<const scalar_t*>(weight.data_ptr());
        const scalar_t* b_ptr = reinterpret_cast<const scalar_t*>(bias.data_ptr());
        scalar_t* run_mean_ptr = reinterpret_cast<scalar_t*>(running_mean.data_ptr());
        scalar_t* run_var_ptr  = reinterpret_cast<scalar_t*>(running_var.data_ptr());
        scalar_t eps = _eps.to<scalar_t>();
        scalar_t momentum = _momentum.to<scalar_t>();

        using func_t = void(*)(scalar_t&, scalar_t&, decltype(x_ptr),
                               const int64_t&, const int64_t&, const int64_t&, const int64_t&, const int64_t&, const int64_t&,
                               scalar_t*, scalar_t*, scalar_t);
        func_t func = nullptr;

        if (use_input_stats) {
            if (run_mean_ptr && run_var_ptr)
                func = &calc_stats_use_input_update<decltype(x_ptr), scalar_t>;
            else
                func = &calc_stats_use_input<decltype(x_ptr), scalar_t>;
        } else {
            func = &calc_stats_no_use_input<decltype(x_ptr), scalar_t>;
        }

        int64_t inner_size = HW;

        for (int64_t n = 0; n < N; ++n) {
            for (int64_t c = 0; c < C; ++c) {
                scalar_t mean = 0, var = 0;
                func(mean, var, x_ptr, inner_size, HW, C, N, c, n, run_mean_ptr, run_var_ptr, momentum);

                scalar_t inv_std = 1.0 / std::sqrt(var + eps);
                scalar_t gamma = w_ptr[c];
                scalar_t beta  = b_ptr[c];

                for (int64_t hw = 0; hw < HW; ++hw) {
                    int64_t idx = ((n * C + c) * HW) + hw;
                    scalar_t norm = (x_ptr[idx] - mean) * inv_std;
                    x_ptr[idx] = norm * gamma + beta;
                }
            }
        }
    });
}

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
    ArrayVoid& inv_vals)

{
    if (!weight.is_contiguous())
        throw std::logic_error("weights for cpu::_instance_norm_ must be contiguous");
    if (!bias.is_contiguous())
        throw std::logic_error("bias for cpu::_instance_norm_ must be contiguous");
    if (!running_mean.is_null() && (!running_mean.is_contiguous() || running_mean.Size() != C))
        throw std::logic_error("Running mean must be contiguous and size == C");
    if (!running_var.is_null() && (!running_var.is_contiguous() || running_var.Size() != C))
        throw std::logic_error("Running var must be contiguous and size == C");
    if (!use_input_stats && (running_mean.is_null() || running_var.is_null()))
        throw std::logic_error("If not using input stats, running mean and var must be defined");
    if (!mean_vals.is_contiguous() || !(mean_vals.Size() == N * C))
        throw std::logic_error("mean_vals for cpu::_instance_norm_ must be contiguous and size == N*C");
    if (!inv_vals.is_contiguous() || !(inv_vals.Size() == N * C))
        throw std::logic_error("inv_vals for cpu::_instance_norm_ must be contiguous and size == N*C");

    input.execute_function<WRAP_DTYPES<NumberTypesL>>([&](auto x_ptr, auto x_end) {
        using scalar_t = utils::IteratorBaseType_t<decltype(x_ptr)>;

        const scalar_t* w_ptr = reinterpret_cast<const scalar_t*>(weight.data_ptr());
        const scalar_t* b_ptr = reinterpret_cast<const scalar_t*>(bias.data_ptr());
        scalar_t* run_mean_ptr = reinterpret_cast<scalar_t*>(running_mean.data_ptr());
        scalar_t* run_var_ptr  = reinterpret_cast<scalar_t*>(running_var.data_ptr());
        scalar_t eps = _eps.to<scalar_t>();
        scalar_t momentum = _momentum.to<scalar_t>();

        using func_t = void(*)(scalar_t&, scalar_t&, decltype(x_ptr),
                               const int64_t&, const int64_t&, const int64_t&, const int64_t&, const int64_t&, const int64_t&,
                               scalar_t*, scalar_t*, scalar_t);
        func_t func = nullptr;

        if (use_input_stats) {
            if (run_mean_ptr && run_var_ptr)
                func = &calc_stats_use_input_update<decltype(x_ptr), scalar_t>;
            else
                func = &calc_stats_use_input<decltype(x_ptr), scalar_t>;
        } else {
            func = &calc_stats_no_use_input<decltype(x_ptr), scalar_t>;
        }

        int64_t inner_size = HW;

        scalar_t* mean_ptr = reinterpret_cast<scalar_t*>(mean_vals.data_ptr());
        scalar_t* inv_ptr  = reinterpret_cast<scalar_t*>(inv_vals.data_ptr());

        for (int64_t n = 0; n < N; ++n) {
            for (int64_t c = 0; c < C; ++c) {
                scalar_t mean = 0, var = 0;
                func(mean, var, x_ptr, inner_size, HW, C, N, c, n, run_mean_ptr, run_var_ptr, momentum);

                scalar_t inv_std = 1.0 / std::sqrt(var + eps);

                mean_ptr[(n * C) + c] = mean;
                inv_ptr[(n * C) + c] = inv_std;

                scalar_t gamma = w_ptr[c];
                scalar_t beta  = b_ptr[c];

                for (int64_t hw = 0; hw < HW; ++hw) {
                    int64_t idx = ((n * C + c) * HW) + hw;
                    scalar_t norm = (x_ptr[idx] - mean) * inv_std;
                    x_ptr[idx] = norm * gamma + beta;
                }
            }
        }
    });
}


// Backward for weight and bias
void _instance_norm_backward_weight_bias_(
    ArrayVoid& grad_weight,
    ArrayVoid& grad_bias,
    const ArrayVoid& grad_output,
    const ArrayVoid& input,
    int64_t N, int64_t C, int64_t HW,
    const ArrayVoid& mean_vals,
    const ArrayVoid& inv_vals)
{
    if (!grad_weight.is_contiguous() || !grad_bias.is_contiguous())
        throw std::logic_error("grads for cpu::_instance_norm_backward_weight_bias_ must be contiguous");
    if (!mean_vals.is_contiguous() || !(mean_vals.Size() == N * C))
        throw std::logic_error("mean_vals for cpu::_instance_norm_backward_weight_bias_ must be contiguous and size == N*C");
    if (!inv_vals.is_contiguous() || !(inv_vals.Size() == N * C))
        throw std::logic_error("inv_vals for cpu::_instance_norm_backward_weight_bias_ must be contiguous and size == N*C");

    grad_output.cexecute_function<WRAP_DTYPES<NumberTypesL>>([&](auto dy_ptr, auto dy_end, auto x_ptr){
        using scalar_t = utils::IteratorBaseType_t<decltype(dy_ptr)>;

        scalar_t* gw_ptr = reinterpret_cast<scalar_t*>(grad_weight.data_ptr());
        scalar_t* gb_ptr = reinterpret_cast<scalar_t*>(grad_bias.data_ptr());
        const scalar_t* mean_ptr = reinterpret_cast<const scalar_t*>(mean_vals.data_ptr());
        const scalar_t* inv_ptr  = reinterpret_cast<const scalar_t*>(inv_vals.data_ptr());

        for (int64_t c = 0; c < C; ++c) {
            scalar_t gw = 0, gb = 0;
            for (int64_t n = 0; n < N; ++n) {
                scalar_t mean = mean_ptr[(n * C) + c];
                scalar_t inv_std = inv_ptr[(n * C) + c];
                for (int64_t hw = 0; hw < HW; ++hw) {
                    int64_t idx = ((n * C + c) * HW) + hw;
                    scalar_t xmu = x_ptr[idx] - mean;
                    scalar_t dy  = dy_ptr[idx];
                    gw += dy * xmu * inv_std;
                    gb += dy;
                }
            }
            gw_ptr[c] += gw;
            gb_ptr[c] += gb;
        }
    }, input);
}

// Backward for input
void _instance_norm_backward_input_(
    ArrayVoid& grad_input,
    const ArrayVoid& grad_output,
    const ArrayVoid& input,
    const ArrayVoid& weight,
    int64_t N, int64_t C, int64_t HW,
    Scalar _eps,
    const ArrayVoid& mean_vals,
    const ArrayVoid& inv_vals)
{
    if (!grad_input.is_contiguous())
        throw std::logic_error("grad_input for cpu::_instance_norm_backward_input_ must be contiguous");
    if (!mean_vals.is_contiguous() || !(mean_vals.Size() == N * C))
        throw std::logic_error("mean_vals for cpu::_instance_norm_backward_input_ must be contiguous and size == N*C");
    if (!inv_vals.is_contiguous() || !(inv_vals.Size() == N * C))
        throw std::logic_error("inv_vals for cpu::_instance_norm_backward_input_ must be contiguous and size == N*C");

    grad_output.cexecute_function<WRAP_DTYPES<NumberTypesL>>([&](auto dy_ptr, auto dy_end, auto x_ptr){
        using scalar_t = utils::IteratorBaseType_t<decltype(dy_ptr)>;

        scalar_t* dx_ptr = reinterpret_cast<scalar_t*>(grad_input.data_ptr());
        const scalar_t* w_ptr = reinterpret_cast<const scalar_t*>(weight.data_ptr());
        const scalar_t* mean_ptr = reinterpret_cast<const scalar_t*>(mean_vals.data_ptr());
        const scalar_t* inv_ptr  = reinterpret_cast<const scalar_t*>(inv_vals.data_ptr());

        int64_t m = HW; // normalization size per instance (per N,C)

        for (int64_t n = 0; n < N; ++n) {
            for (int64_t c = 0; c < C; ++c) {
                int64_t base_idx = (n * C + c) * HW;
                scalar_t mean = mean_ptr[(n * C) + c];
                scalar_t inv_std = inv_ptr[(n * C) + c];
                scalar_t gamma = w_ptr[c];

                // compute mean(dy) and mean(dy * (x - mean)) over HW
                scalar_t mean_dy = 0;
                scalar_t mean_dy_xmu = 0;
                for (int64_t hw = 0; hw < HW; ++hw) {
                    int64_t idx = base_idx + hw;
                    scalar_t dy = dy_ptr[idx];
                    scalar_t xmu = x_ptr[idx] - mean;
                    scalar_t dy_gamma = dy * gamma;
                    mean_dy += dy_gamma;
                    mean_dy_xmu += dy_gamma * xmu;
                }

                mean_dy *= (1.0 / m);
                mean_dy_xmu *= (1.0 / m);

                // compute dx for each spatial location
                scalar_t inv_std_sq = inv_std * inv_std;
                for (int64_t hw = 0; hw < HW; ++hw) {
                    int64_t idx = base_idx + hw;
                    scalar_t dy = dy_ptr[idx];
                    scalar_t xmu = x_ptr[idx] - mean;
                    scalar_t dy_gamma = dy * gamma;

                    // formula: dx = gamma * inv_std * (dy - mean_dy - xmu * inv_std^2 * mean_dy_xmu)
                    scalar_t dx = (inv_std) * (dy_gamma - mean_dy - xmu * inv_std_sq * mean_dy_xmu);
                    dx_ptr[idx] = dx;
                }
            }
        }
    }, input);
}


} // namespace nt::functional::cpu::
