#include "normalize.h"
#include "../../dtype/ArrayVoid.h"
#include "../../dtype/ArrayVoid_NTensor.hpp"
#include "../../dtype/DType_enum.h"
#include "../../dtype/Scalar.h"
#include "../../refs/SizeRef.h"
#include <random>
#include "rand.h"

namespace nt::functional::cpu {

void xavier_uniform_(ArrayVoid &output, double bound) {
    rand_(output, Scalar(bound), Scalar(-bound)); //uniform real distribution
}

void _batch_norm_(ArrayVoid& input, ArrayVoid& running_mean, ArrayVoid& running_var,
                                  const ArrayVoid& weight, const ArrayVoid& bias, bool training,
                                  Scalar _momentum, Scalar _eps, int64_t N, int64_t C, int64_t HW){
    if(!weight.is_contiguous()){
        throw std::logic_error("weights for cpu::_batch_norm_ must be contiguous");
    }
    if(!bias.is_contiguous()){
        throw std::logic_error("bias for cpu::_batch_norm_ must be contiguous");
    }
    input.execute_function<WRAP_DTYPES<NumberTypesL>>(
        [&](auto x_ptr, auto x_end, auto mean_ptr){
            using scalar_t = utils::IteratorBaseType_t<decltype(x_ptr)>;
            running_var.execute_function<WRAP_DTYPES<DTypeEnum<DTypeFuncs::type_to_dtype<scalar_t>>>>(
            [&](auto var_ptr, auto var_end){
                int64_t total_size = N * C * HW;
                const scalar_t* w_ptr  = reinterpret_cast<const scalar_t*>(weight.data_ptr());
                const scalar_t* b_ptr  = reinterpret_cast<const scalar_t*>(bias.data_ptr());
                scalar_t eps = _eps.to<scalar_t>();
                scalar_t momentum = _momentum.to<scalar_t>();
                if(training){
                    for(int64_t c = 0; c < C; ++c){
                        scalar_t mean = 0;
                        scalar_t var = 0;
                        scalar_t inv_m = 1.0 / scalar_t(N * HW);
                        // Here starts training only
                        for(int64_t n = 0; n < N; ++n)
                            for(int64_t hw = 0; hw < HW; ++hw)
                                mean += x_ptr[((n * C + c) * HW) + hw];
                        mean *= inv_m;
                        for(int64_t n = 0; n < N; ++n){
                            for(int64_t hw = 0; hw < HW; ++hw){
                                scalar_t diff = x_ptr[((n * C + c) * HW) + hw] - mean;
                                var += diff * diff;
                            }
                        }
                        var *= inv_m;
                        // update running stats
                        mean_ptr[c] = momentum * mean + (1 - momentum) * mean_ptr[c];
                        var_ptr[c] = momentum * var + (1 - momentum) * var_ptr[c];
                        // Here ends training only

                        scalar_t inv_std = 1.0 / std::sqrt(var + eps);
                        scalar_t gamma = w_ptr[c];
                        scalar_t beta = b_ptr[c];
                        for(int64_t n = 0; n < N; ++n){
                            for(int64_t hw = 0; hw < HW; ++hw){
                                int64_t idx = ((n * C + c) * HW) + hw;
                                scalar_t norm = (x_ptr[idx] - mean) * inv_std;
                                x_ptr[idx] = norm * gamma + beta;
                            }
                        }
                    }
                }else{
                    for(int64_t c = 0; c < C; ++c){
                        scalar_t mean = mean_ptr[c];
                        scalar_t var = var_ptr[c];
                        scalar_t inv_m = 1.0 / (N * HW);

                        scalar_t inv_std = 1.0 / std::sqrt(var + eps);
                        scalar_t gamma = w_ptr[c];
                        scalar_t beta = b_ptr[c];
                        for(int64_t n = 0; n < N; ++n){
                            for(int64_t hw = 0; hw < HW; ++hw){
                                int64_t idx = ((n * C + c) * HW) + hw;
                                scalar_t norm = (x_ptr[idx] - mean) * inv_std;
                                x_ptr[idx] = norm * gamma + beta;
                            }
                        }
                    }
                }
            });
        }, running_mean);
}


void _batch_norm_(ArrayVoid& input, ArrayVoid& running_mean, ArrayVoid& running_var,
                                  const ArrayVoid& weight, const ArrayVoid& bias, bool training,
                                  Scalar _momentum, Scalar _eps, int64_t N, int64_t C, int64_t HW,
                                  ArrayVoid& mean_vals, ArrayVoid& inv_vals){
    if(!weight.is_contiguous()){
        throw std::logic_error("weights for cpu::_batch_norm_ must be contiguous");
    }
    if(!bias.is_contiguous()){
        throw std::logic_error("bias for cpu::_batch_norm_ must be contiguous");
    }
    if(!inv_vals.is_contiguous() || !(inv_vals.Size() == C) || !(inv_vals.dtype() == input.dtype())){
        throw std::logic_error("if storing inv_vals, need to have them be contiguous, numel C, and same dtype as input");
    }
    if(!mean_vals.is_contiguous() || !(mean_vals.Size() == C) || !(mean_vals.dtype() == input.dtype())){
        throw std::logic_error("if storing mean_vals, need to have them be contiguous, numel C, and same dtype as input");
    }
    input.execute_function<WRAP_DTYPES<NumberTypesL>>(
        [&](auto x_ptr, auto x_end, auto mean_ptr){
            using scalar_t = utils::IteratorBaseType_t<decltype(x_ptr)>;
            running_var.execute_function<WRAP_DTYPES<DTypeEnum<DTypeFuncs::type_to_dtype<scalar_t>>>>(
            [&](auto var_ptr, auto var_end){
                int64_t total_size = N * C * HW;
                const scalar_t* w_ptr  = reinterpret_cast<const scalar_t*>(weight.data_ptr());
                const scalar_t* b_ptr  = reinterpret_cast<const scalar_t*>(bias.data_ptr());
                scalar_t* storing_mean_vals = reinterpret_cast<scalar_t*>(mean_vals.data_ptr());
                scalar_t* storing_inv_vals = reinterpret_cast<scalar_t*>(inv_vals.data_ptr());
                scalar_t eps = _eps.to<scalar_t>();
                scalar_t momentum = _momentum.to<scalar_t>();
                if(training){
                    for(int64_t c = 0; c < C; ++c){
                        scalar_t mean = 0;
                        scalar_t var = 0;
                        scalar_t inv_m = 1.0 / scalar_t(N * HW);
                        // Here starts training only
                        for(int64_t n = 0; n < N; ++n)
                            for(int64_t hw = 0; hw < HW; ++hw)
                                mean += x_ptr[((n * C + c) * HW) + hw];
                        mean *= inv_m;
                        for(int64_t n = 0; n < N; ++n){
                            for(int64_t hw = 0; hw < HW; ++hw){
                                scalar_t diff = x_ptr[((n * C + c) * HW) + hw] - mean;
                                var += diff * diff;
                            }
                        }
                        var *= inv_m;
                        // update running stats
                        mean_ptr[c] = (1 - momentum) * mean_ptr[c] + momentum * mean;
                        var_ptr[c] =  (1 - momentum) * var_ptr[c]  + momentum * var;
                        // Here ends training only
                        scalar_t inv_std = 1.0 / std::sqrt(var + eps);
                        scalar_t gamma = w_ptr[c];
                        scalar_t beta = b_ptr[c];
                        storing_mean_vals[c] = mean;
                        storing_inv_vals[c] = inv_std;
                        for(int64_t n = 0; n < N; ++n){
                            for(int64_t hw = 0; hw < HW; ++hw){
                                int64_t idx = ((n * C + c) * HW) + hw;
                                scalar_t norm = (x_ptr[idx] - mean) * inv_std;
                                x_ptr[idx] = norm * gamma + beta;
                            }
                        }
                    }
                }else{
                    for(int64_t c = 0; c < C; ++c){
                        scalar_t mean = mean_ptr[c];
                        scalar_t var = var_ptr[c];
                        scalar_t inv_m = 1.0 / (N * HW);

                        scalar_t inv_std = 1.0 / std::sqrt(var + eps);
                        scalar_t gamma = w_ptr[c];
                        scalar_t beta = b_ptr[c];
                        storing_mean_vals[c] = mean;
                        storing_inv_vals[c] = inv_std;
                        for(int64_t n = 0; n < N; ++n){
                            for(int64_t hw = 0; hw < HW; ++hw){
                                int64_t idx = ((n * C + c) * HW) + hw;
                                scalar_t norm = (x_ptr[idx] - mean) * inv_std;
                                x_ptr[idx] = norm * gamma + beta;
                            }
                        }
                    }
                }
            });
        }, running_mean);
}

void _batch_norm_backward_weight_bias_(ArrayVoid& grad_weight, ArrayVoid& grad_bias, const ArrayVoid& grad_output,
                                       const ArrayVoid& input, 
                                       Scalar _momentum, Scalar _eps, int64_t N, int64_t C, int64_t HW,
                                       const ArrayVoid& stored_mean_vals, const ArrayVoid& stored_inv_vals){
    if(!grad_weight.is_contiguous()){
        throw std::logic_error("grad_weights for cpu::_batch_norm_backward_weight_bias_ must be contiguous");
    }
    if(!grad_bias.is_contiguous()){
        throw std::logic_error("grad_bias for cpu::_batch_norm_backward_weight_bias_ must be contiguous");
    }
    if(!stored_mean_vals.is_contiguous()){
        throw std::logic_error("stored_mean_vals for cpu::_batch_norm_backward_weight_bias_ must be contiguous");
    }
    if(!stored_inv_vals.is_contiguous()){
        throw std::logic_error("stored_inv_vals for cpu::_batch_norm_backward_weight_bias_ must be contiguous");
    }
    grad_output.cexecute_function<WRAP_DTYPES<NumberTypesL>>(
        [&](auto dy_ptr, auto dy_end, auto x_ptr){
            using scalar_t = utils::IteratorBaseType_t<decltype(dy_ptr)>;
            int64_t total_size = N * C * HW;
            scalar_t* gw_ptr = reinterpret_cast<scalar_t*>(grad_weight.data_ptr());
            scalar_t* gb_ptr = reinterpret_cast<scalar_t*>(grad_bias.data_ptr());
            const scalar_t* stored_mean_ptr = reinterpret_cast<const scalar_t*>(stored_mean_vals.data_ptr());
            const scalar_t* stored_inv_ptr = reinterpret_cast<const scalar_t*>(stored_inv_vals.data_ptr());
            scalar_t eps = _eps.to<scalar_t>();
            scalar_t momentum = _momentum.to<scalar_t>();
            int64_t stride_c = HW;
            for (int64_t c = 0; c < C; ++c) {
                // scalar_t mean_val = mean_ptr[c];
                // scalar_t inv_std = 1.0 / std::sqrt(var_ptr[c] + eps);
                const scalar_t& mean_val = stored_mean_ptr[c];
                const scalar_t& inv_std = stored_inv_ptr[c];
                scalar_t gw = 0, gb = 0;
                for (int64_t n = 0; n < N; ++n){
                    for (int64_t hw = 0; hw < HW; ++hw) {
                        int64_t idx = ((n * C + c) * HW) + hw;
                        scalar_t xmu = x_ptr[idx] - mean_val;
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


void _batch_norm_backward_input_(ArrayVoid& grad_input, const ArrayVoid& grad_output,
                                       const ArrayVoid& input,
                                       const ArrayVoid& weight, Scalar _momentum, Scalar _eps, 
                                       int64_t N, int64_t C, int64_t HW,
                                       const ArrayVoid& stored_mean_vals, const ArrayVoid& stored_inv_vals){
    if(!weight.is_contiguous()){
        throw std::logic_error("weights for cpu::_batch_norm_backward_input_ must be contiguous");
    }
    if(!grad_input.is_contiguous()){
        throw std::logic_error("grad_input for cpu::_batch_norm_backward_input_ must be contiguous");
    }
    if(!stored_mean_vals.is_contiguous()){
        throw std::logic_error("stored_mean_vals for cpu::_batch_norm_backward_weight_bias_ must be contiguous");
    }
    if(!stored_inv_vals.is_contiguous()){
        throw std::logic_error("stored_inv_vals for cpu::_batch_norm_backward_weight_bias_ must be contiguous");
    }
    grad_output.cexecute_function<WRAP_DTYPES<NumberTypesL>>(
        [&](auto dy_ptr, auto dy_end, auto x_ptr){
            using scalar_t = utils::IteratorBaseType_t<decltype(dy_ptr)>;
            int64_t total_size = N * C * HW;
            scalar_t* g_ptr = reinterpret_cast<scalar_t*>(grad_input.data_ptr());
            const scalar_t* w_ptr = reinterpret_cast<const scalar_t*>(weight.data_ptr());
            const scalar_t* stored_mean_ptr = reinterpret_cast<const scalar_t*>(stored_mean_vals.data_ptr());
            const scalar_t* stored_inv_ptr = reinterpret_cast<const scalar_t*>(stored_inv_vals.data_ptr());
            scalar_t eps = _eps.to<scalar_t>();
            scalar_t momentum = _momentum.to<scalar_t>();
            scalar_t inv_NHW = 1.0 / (N * HW);
            int64_t stride_c = HW;

            for (int64_t c = 0; c < C; ++c) {
                // const scalar_t& mean_val = mean_ptr[c];
                // const scalar_t& var_val = var_ptr[c];
                const scalar_t& gamma = w_ptr[c];

                // scalar_t inv_std = 1.0 / std::sqrt(var_val + eps);

                const scalar_t& mean_val = stored_mean_ptr[c];
                const scalar_t& inv_std = stored_inv_ptr[c];

                // compute mean(dy) and mean(dy * (x - mean))
                scalar_t mean_dy = 0, mean_dy_xmu = 0;

                for (int64_t n = 0; n < N; ++n){
                    for (int64_t hw = 0; hw < HW; ++hw) {
                        int64_t idx = ((n * C + c) * HW) + hw;
                        scalar_t xmu = x_ptr[idx] - mean_val;
                        scalar_t dy  = dy_ptr[idx];
                        mean_dy     += dy;
                        mean_dy_xmu += dy * xmu;
                    }
                }

                mean_dy     *= inv_NHW;
                mean_dy_xmu *= inv_NHW;

                // gradient wrt input
                for (int64_t n = 0; n < N; ++n){
                    for (int64_t hw = 0; hw < HW; ++hw) {
                        int64_t idx = ((n * C + c) * HW) + hw;
                        scalar_t xmu = x_ptr[idx] - mean_val;
                        scalar_t dy  = dy_ptr[idx];

                        scalar_t dx = (gamma * inv_std) *
                            (dy - mean_dy - xmu * inv_std * inv_std * mean_dy_xmu);
                        g_ptr[idx] = dx;
                    }
                }
            }

        }, input);
}


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
    if (!mean_vals.is_contiguous() || !(mean_vals.Size() == num_groups))
        throw std::logic_error("mean_vals for cpu::_group_norm_ must be contiguous and size == num_groups");
    if (!inv_vals.is_contiguous() || !(inv_vals.Size() == num_groups))
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

                scalar_t inv_std = 1.0 / std::sqrt(var + eps);
                mean_ptr[g] = mean;
                inv_ptr[g] = inv_std;

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

                scalar_t inv_std = 1.0 / std::sqrt(var + eps);

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
            scalar_t mean = mean_ptr[g];
            scalar_t inv_std = inv_ptr[g];
            int64_t base_c = g * C_per_G;

            for (int64_t c = 0; c < C_per_G; ++c) {
                scalar_t gw = 0, gb = 0;
                for (int64_t n = 0; n < N; ++n){
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
    int64_t num_groups,
    Scalar _eps,
    const ArrayVoid& mean_vals,
    const ArrayVoid& inv_vals)
{
    if (!grad_input.is_contiguous())
        throw std::logic_error("grad_input must be contiguous");

    grad_output.cexecute_function<WRAP_DTYPES<NumberTypesL>>([&](auto dy_ptr, auto dy_end, auto x_ptr){
        using scalar_t = utils::IteratorBaseType_t<decltype(dy_ptr)>;

        scalar_t* g_ptr = reinterpret_cast<scalar_t*>(grad_input.data_ptr());
        const scalar_t* w_ptr = reinterpret_cast<const scalar_t*>(weight.data_ptr());
        const scalar_t* mean_ptr = reinterpret_cast<const scalar_t*>(mean_vals.data_ptr());
        const scalar_t* inv_ptr  = reinterpret_cast<const scalar_t*>(inv_vals.data_ptr());
        scalar_t eps = _eps.to<scalar_t>();

        int64_t C_per_G = C / num_groups;
        int64_t group_size = C_per_G * HW;
        scalar_t inv_group_size = 1.0 / group_size;

        for (int64_t n = 0; n < N; ++n) {
            for (int64_t g = 0; g < num_groups; ++g) {
                scalar_t mean = mean_ptr[g];
                scalar_t inv_std = inv_ptr[g];
                int64_t base_c = g * C_per_G;

                scalar_t mean_dy = 0, mean_dy_xmu = 0;

                // First pass: compute means
                for (int64_t c = 0; c < C_per_G; ++c)
                    for (int64_t hw = 0; hw < HW; ++hw) {
                        int64_t idx = ((n * C + base_c + c) * HW) + hw;
                        scalar_t dy = dy_ptr[idx];
                        scalar_t xmu = x_ptr[idx] - mean;
                        mean_dy += dy;
                        mean_dy_xmu += dy * xmu;
                    }

                mean_dy *= inv_group_size;
                mean_dy_xmu *= inv_group_size;

                // Second pass: compute grad input
                for (int64_t c = 0; c < C_per_G; ++c) {
                    scalar_t gamma = w_ptr[base_c + c];
                    for (int64_t hw = 0; hw < HW; ++hw) {
                        int64_t idx = ((n * C + base_c + c) * HW) + hw;
                        scalar_t dy = dy_ptr[idx];
                        scalar_t xmu = x_ptr[idx] - mean;
                        scalar_t dx = (gamma * inv_std) *
                            (dy - mean_dy - xmu * inv_std * inv_std * mean_dy_xmu);
                        g_ptr[idx] = dx;
                    }
                }
            }
        }
    }, input);
}




} // namespace nt::functional::cpu::
