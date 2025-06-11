#ifndef __NT_FUNCTIONAL_TENSOR_FILES_POOLING_POOL_UTILITIES_HPP__
#define __NT_FUNCTIONAL_TENSOR_FILES_POOLING_POOL_UTILITIES_HPP__


#include "../exceptions.hpp"
#include "../padding.h"
#include "../fill.h"
#include "../../../Tensor.h"

namespace nt{
namespace functional{

inline int64_t find_pooling_size(const int64_t& input_size, const int64_t& kernel_size, const int64_t& stride, const int64_t& padding) noexcept {
    return (input_size + 2 * padding - kernel_size) / stride + 1;
}

inline int64_t find_pooling_size_ceil(const int64_t& input_size, const int64_t& kernel_size, const int64_t& stride, const int64_t& padding) noexcept {
    return (input_size + 2 * padding - kernel_size + (stride-1)) / stride + 1;
}

inline void check_pool_args(const int64_t& input_size, const int64_t& kernel_size, const int64_t& stride, const int64_t& padding){
    utils::throw_exception(kernel_size <= input_size,
                           "Error: kernel size ($) can not be larger than singleton ($) at corresponding dimension",
                           kernel_size, input_size);
    utils::throw_exception(padding >= 0,
                               "Error: padding ($) cannot be less than 0", padding);
    utils::throw_exception(padding <= (kernel_size / 2),
                           "Error: padding ($) should be at most half of kernel_size ($)",
                           padding, kernel_size);
    utils::throw_exception(stride >= 1,"Expected stride to be greater than or equal to 1 got $", stride);
}

inline void check_pool_args(const Tensor& input, int64_t dim, const int64_t& kernel_size, const int64_t& stride, const int64_t& padding){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(input);
    dim = (dim < 0) ? dim + input.dims() : dim;
    utils::throw_exception(input.dims() > dim && dim >= 0,
                           "Specified pool expects a tensor with dimensionality at least $ got shape $", dim, input.shape());
    check_pool_args(input.shape()[dim], kernel_size, stride, padding);
}

inline int64_t find_pooling_kernel_size(const int64_t& output_size, const int64_t& input_size, const int64_t& stride, const int64_t& padding) noexcept {

    return -((output_size-1) * stride - (2*padding) - input_size);
}

inline void find_adaptive(int64_t output_size, int64_t input_size, int64_t& kernel_size, int64_t& stride, int64_t& padding){
    stride = input_size / output_size;
    padding = 0;
    kernel_size = find_pooling_kernel_size(output_size, input_size, stride, padding);
}


inline void assert_dilation(const int64_t& dilation){
    utils::throw_exception(dilation >= 1,
                           "Dilation must be greater than or equal to 1 got $", dilation);
}

inline void assert_dilation(const utils::my_tuple& dilation){
    utils::throw_exception(dilation[0] >= 1 && dilation[1] >= 1,
                           "Dilation must be greater than or equal to 1 got $", dilation);
}

inline void assert_dilation(const utils::my_n_tuple<3>& dilation){
    utils::throw_exception(dilation[0] >= 1 && dilation[1] >= 1 && dilation[2] >= 1,
                           "Dilation must be greater than or equal to 1 got $", dilation);
}


}} //functional::
#endif
