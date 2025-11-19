#include "normalize.h"
#include "../../dtype/ArrayVoid.h"
#include "../../dtype/ArrayVoid_NTensor.hpp"
#include "../../dtype/DType_enum.h"
#include "../../dtype/Scalar.h"
#include "../../refs/SizeRef.h"
#include "../../types/math.h"
#include <random>
#include "rand.h"

namespace nt::functional::cpu {


void _group_norm_(
    ArrayVoid& input,
    const ArrayVoid& weight,
    const ArrayVoid& bias,
    int64_t N, int64_t C, int64_t HW,
    int64_t num_groups,
    Scalar _eps,
    ArrayVoid& mean_vals,
    ArrayVoid& inv_vals)
{
    if (!weight.is_contiguous())
        throw std::logic_error("weights for cpu::_group_norm_ must be contiguous");
    if (!bias.is_contiguous())
        throw std::logic_error("bias for cpu::_group_norm_ must be contiguous");
    if (!mean_vals.is_contiguous() || !(mean_vals.Size() == num_groups * N))
        throw std::logic_error("mean_vals for cpu::_group_norm_ must be contiguous and size == num_groups");
    if (!inv_vals.is_contiguous() || !(inv_vals.Size() == num_groups * N))
        throw std::logic_error("inv_vals for cpu::_group_norm_ must be contiguous and size == num_groups");

    input.execute_function<WRAP_DTYPES<NumberTypesL>>([&](auto x_ptr, auto x_end){
        using scalar_t = utils::IteratorBaseType_t<decltype(x_ptr)>;

        const scalar_t* w_ptr = reinterpret_cast<const scalar_t*>(weight.data_ptr());
        const scalar_t* b_ptr = reinterpret_cast<const scalar_t*>(bias.data_ptr());
        scalar_t* mean_ptr = reinterpret_cast<scalar_t*>(mean_vals.data_ptr());
        scalar_t* inv_ptr  = reinterpret_cast<scalar_t*>(inv_vals.data_ptr());
        scalar_t eps = _eps.to<scalar_t>();

        int64_t C_per_G = C / num_groups;
        int64_t inner_size = C_per_G * HW;

        for (int64_t n = 0; n < N; ++n) {
            for (int64_t g = 0; g < num_groups; ++g) {
                // compute mean and var for this group
                scalar_t mean = 0, var = 0;
                int64_t base_c = g * C_per_G;

                // compute mean
                for (int64_t c = 0; c < C_per_G; ++c)
                    for (int64_t hw = 0; hw < HW; ++hw)
                        mean += x_ptr[((n * C + base_c + c) * HW) + hw];
                mean /= inner_size;

                // compute var
                for (int64_t c = 0; c < C_per_G; ++c){
                    for (int64_t hw = 0; hw < HW; ++hw) {
                        scalar_t diff = x_ptr[((n * C + base_c + c) * HW) + hw] - mean;
                        var += diff * diff;
                    }
                }
                var /= inner_size;

                scalar_t inv_std = 1.0 / ::nt::sqrt(var + eps);
                mean_ptr[(n * num_groups) + g] = mean;
                inv_ptr[(n * num_groups) + g] = inv_std;

                // normalize and apply affine
                for (int64_t c = 0; c < C_per_G; ++c) {
                    scalar_t gamma = w_ptr[base_c + c];
                    scalar_t beta  = b_ptr[base_c + c];
                    for (int64_t hw = 0; hw < HW; ++hw) {
                        int64_t idx = ((n * C + base_c + c) * HW) + hw;
                        scalar_t norm = (x_ptr[idx] - mean) * inv_std;
                        x_ptr[idx] = norm * gamma + beta;
                    }
                }
            }
        }
    });
}

void _group_norm_(
    ArrayVoid& input,
    const ArrayVoid& weight,
    const ArrayVoid& bias,
    int64_t N, int64_t C, int64_t HW,
    int64_t num_groups,
    Scalar _eps)
{
    if (!weight.is_contiguous())
        throw std::logic_error("weights for cpu::_group_norm_ must be contiguous");
    if (!bias.is_contiguous())
        throw std::logic_error("bias for cpu::_group_norm_ must be contiguous");
    
    input.execute_function<WRAP_DTYPES<NumberTypesL>>([&](auto x_ptr, auto x_end){
        using scalar_t = utils::IteratorBaseType_t<decltype(x_ptr)>;

        const scalar_t* w_ptr = reinterpret_cast<const scalar_t*>(weight.data_ptr());
        const scalar_t* b_ptr = reinterpret_cast<const scalar_t*>(bias.data_ptr());
        scalar_t eps = _eps.to<scalar_t>();

        int64_t C_per_G = C / num_groups;
        int64_t inner_size = C_per_G * HW;

        for (int64_t n = 0; n < N; ++n) {
            for (int64_t g = 0; g < num_groups; ++g) {
                // compute mean and var for this group
                scalar_t mean = 0, var = 0;
                int64_t base_c = g * C_per_G;

                // compute mean
                for (int64_t c = 0; c < C_per_G; ++c)
                    for (int64_t hw = 0; hw < HW; ++hw)
                        mean += x_ptr[((n * C + base_c + c) * HW) + hw];
                mean /= inner_size;

                // compute var
                for (int64_t c = 0; c < C_per_G; ++c){
                    for (int64_t hw = 0; hw < HW; ++hw) {
                        scalar_t diff = x_ptr[((n * C + base_c + c) * HW) + hw] - mean;
                        var += diff * diff;
                    }
                }
                var /= inner_size;

                scalar_t inv_std = 1.0 / ::nt::sqrt(var + eps);

                // normalize and apply affine
                for (int64_t c = 0; c < C_per_G; ++c) {
                    scalar_t gamma = w_ptr[base_c + c];
                    scalar_t beta  = b_ptr[base_c + c];
                    for (int64_t hw = 0; hw < HW; ++hw) {
                        int64_t idx = ((n * C + base_c + c) * HW) + hw;
                        scalar_t norm = (x_ptr[idx] - mean) * inv_std;
                        x_ptr[idx] = norm * gamma + beta;
                    }
                }
            }
        }
    });
}

void _group_norm_backward_weight_bias_(
    ArrayVoid& grad_weight,
    ArrayVoid& grad_bias,
    const ArrayVoid& grad_output,
    const ArrayVoid& input,
    int64_t N, int64_t C, int64_t HW,
    int64_t num_groups,
    const ArrayVoid& mean_vals,
    const ArrayVoid& inv_vals)
{
    if (!grad_weight.is_contiguous() || !grad_bias.is_contiguous())
        throw std::logic_error("grads for cpu::_group_norm_backward_weight_bias_ must be contiguous");

    grad_output.cexecute_function<WRAP_DTYPES<NumberTypesL>>([&](auto dy_ptr, auto dy_end, auto x_ptr){
        using scalar_t = utils::IteratorBaseType_t<decltype(dy_ptr)>;

        scalar_t* gw_ptr = reinterpret_cast<scalar_t*>(grad_weight.data_ptr());
        scalar_t* gb_ptr = reinterpret_cast<scalar_t*>(grad_bias.data_ptr());
        const scalar_t* mean_ptr = reinterpret_cast<const scalar_t*>(mean_vals.data_ptr());
        const scalar_t* inv_ptr  = reinterpret_cast<const scalar_t*>(inv_vals.data_ptr());

        int64_t C_per_G = C / num_groups;

        for (int64_t g = 0; g < num_groups; ++g) {

            int64_t base_c = g * C_per_G;

            for (int64_t c = 0; c < C_per_G; ++c) {
                scalar_t gw = 0, gb = 0;
                for (int64_t n = 0; n < N; ++n){
                    scalar_t mean = mean_ptr[(n * num_groups) + g];
                    scalar_t inv_std = inv_ptr[(n * num_groups) + g];

                    for (int64_t hw = 0; hw < HW; ++hw) {
                        int64_t idx = ((n * C + base_c + c) * HW) + hw;
                        scalar_t xmu = x_ptr[idx] - mean;
                        scalar_t dy  = dy_ptr[idx];

                        gw += dy * xmu * inv_std;
                        gb += dy;
                    }
                }
                gw_ptr[base_c + c] += gw;
                gb_ptr[base_c + c] += gb;
            }
        }
    }, input);

}



void _group_norm_backward_input_(
    ArrayVoid& grad_input,
    const ArrayVoid& grad_output,
    const ArrayVoid& input,
    const ArrayVoid& weight,
    int64_t N, int64_t C, int64_t HW,
    int64_t num_groups, Scalar _eps,
    const ArrayVoid& mean_vals,
    const ArrayVoid& inv_vals)
{
    if (!grad_input.is_contiguous())
        throw std::logic_error("grad_input for cpu::_group_norm_backward_input_ must be contiguous");

    grad_output.cexecute_function<WRAP_DTYPES<NumberTypesL>>([&](auto dy_ptr, auto dy_end, auto x_ptr){
        using scalar_t = utils::IteratorBaseType_t<decltype(dy_ptr)>;

        scalar_t* dx_ptr = reinterpret_cast<scalar_t*>(grad_input.data_ptr());
        const scalar_t* w_ptr = reinterpret_cast<const scalar_t*>(weight.data_ptr());
        const scalar_t* mean_ptr = reinterpret_cast<const scalar_t*>(mean_vals.data_ptr());
        const scalar_t* inv_ptr  = reinterpret_cast<const scalar_t*>(inv_vals.data_ptr());

        int64_t C_per_G = C / num_groups;
        int64_t inner_size = C_per_G * HW;

        for (int64_t n = 0; n < N; ++n) {
            for (int64_t g = 0; g < num_groups; ++g) {
                int64_t base_c = g * C_per_G;
                scalar_t mean = mean_ptr[(n * num_groups) + g];
                scalar_t inv_std = inv_ptr[(n * num_groups) + g];

                scalar_t sum_dy = 0;
                scalar_t sum_dy_xmu = 0;

                // Precompute sums for group
                for (int64_t c = 0; c < C_per_G; ++c) {
                    scalar_t gamma = w_ptr[base_c + c];
                    for (int64_t hw = 0; hw < HW; ++hw) {
                        int64_t idx = ((n * C + base_c + c) * HW) + hw;
                        scalar_t dy = dy_ptr[idx] * gamma;
                        scalar_t xmu = x_ptr[idx] - mean;
                        sum_dy += dy;
                        sum_dy_xmu += dy * xmu;
                    }
                }

                // Compute gradient for each element
                scalar_t coeff1 = inv_std / inner_size;
                scalar_t coeff2 = sum_dy_xmu * inv_std * inv_std * inv_std / inner_size;

                for (int64_t c = 0; c < C_per_G; ++c) {
                    scalar_t gamma = w_ptr[base_c + c];
                    for (int64_t hw = 0; hw < HW; ++hw) {
                        int64_t idx = ((n * C + base_c + c) * HW) + hw;
                        scalar_t dy = dy_ptr[idx] * gamma;
                        scalar_t xmu = x_ptr[idx] - mean;

                        dx_ptr[idx] = coeff1 * inner_size * dy 
                                    - coeff1 * sum_dy 
                                    - coeff2 * xmu;
                    }
                }
            }
        }
    }, input);
}



} // namespace nt::functional::cpu::
