#include "normalize.h"
#include "../../dtype/ArrayVoid.h"
#include "../../dtype/ArrayVoid_NTensor.hpp"
#include "../../dtype/DType_enum.h"
#include "../../dtype/Scalar.h"
#include "../../refs/SizeRef.h"
#include "../../math/math.h"
#include <random>
#include "rand.h"

namespace nt::functional::cpu {


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
                        mean_ptr[c] = (1 - momentum) * mean_ptr[c] + momentum * mean;
                        var_ptr[c] =  (1 - momentum) * var_ptr[c]  + momentum * var;
                        // Here ends training only

                        scalar_t inv_std = 1.0 / ::nt::math::sqrt(var + eps);
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

                        scalar_t inv_std = 1.0 / ::nt::math::sqrt(var + eps);
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
                        scalar_t inv_std = 1.0 / ::nt::math::sqrt(var + eps);
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

                        scalar_t inv_std = 1.0 / ::nt::math::sqrt(var + eps);
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
                // scalar_t inv_std = 1.0 / ::nt::math::sqrt(var_ptr[c] + eps);
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

                // scalar_t inv_std = 1.0 / ::nt::math::sqrt(var_val + eps);

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


} // namespace nt::functional::cpu::
