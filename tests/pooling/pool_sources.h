#ifndef _NT_TESTS_POOL_SOURCES_H_
#define _NT_TESTS_POOL_SOURCES_H_

#include <nt/Tensor.h>


inline int64_t find_pooling_size(const int64_t& input_size, const int64_t& kernel_size, const int64_t& stride, const int64_t& padding) noexcept {
    return (input_size + 2 * padding - kernel_size) / stride + 1;
}

inline int64_t find_pooling_size_ceil(const int64_t& input_size, const int64_t& kernel_size, const int64_t& stride, const int64_t& padding) noexcept {
    return (input_size + 2 * padding - kernel_size + (stride-1)) / stride + 1;
}

inline void check_pool_args(const int64_t& input_size, const int64_t& kernel_size, const int64_t& stride, const int64_t& padding){
    nt::utils::throw_exception(kernel_size <= input_size,
                           "Error: kernel size ($) can not be larger than singleton ($) at corresponding dimension",
                           kernel_size, input_size);
    nt::utils::throw_exception(padding >= 0,
                               "Error: padding ($) cannot be less than 0", padding);
    nt::utils::throw_exception(padding <= (kernel_size / 2),
                           "Error: padding ($) should be at most half of kernel_size ($)",
                           padding, kernel_size);
    nt::utils::throw_exception(stride >= 1,"Expected stride to be greater than or equal to 1 got $", stride);
}

inline void check_pool_args(const nt::Tensor& input, int64_t dim, const int64_t& kernel_size, const int64_t& stride, const int64_t& padding){
    dim = (dim < 0) ? dim + input.dims() : dim;
    nt::utils::throw_exception(input.dims() > dim && dim >= 0,
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
    nt::utils::throw_exception(dilation >= 1,
                           "Dilation must be greater than or equal to 1 got $", dilation);
}

inline void assert_dilation(const nt::utils::my_tuple& dilation){
    nt::utils::throw_exception(dilation[0] >= 1 && dilation[1] >= 1,
                           "Dilation must be greater than or equal to 1 got $", dilation);
}

inline void assert_dilation(const nt::utils::my_n_tuple<3>& dilation){
    nt::utils::throw_exception(dilation[0] >= 1 && dilation[1] >= 1 && dilation[2] >= 1,
                           "Dilation must be greater than or equal to 1 got $", dilation);
}



inline nt::Tensor unpad(const nt::Tensor& t, std::vector<nt::Tensor::size_value_t> vec){
    using namespace nt;
    std::vector<my_range> ranges(t.dims(), my_range(0, -1));
    utils::throw_exception((vec.size()/2) <= ranges.size(),
                           "Cannot unpad greater than the dimensions of the tensor");
    auto in_shape = t.shape();
    for(int64_t i = 0; i < ranges.size(); ++i){
        ranges[i].end = in_shape[i];
    }
    auto begin = vec.crbegin();
    auto end = vec.crend();
    auto range_begin = ranges.rbegin();
    int64_t index = -1;
    for(;begin != end; ++begin, ++range_begin, --index){
        range_begin->end = in_shape[index]-*begin;
        ++begin;
        range_begin->begin = *begin;
    }
    return t[std::move(ranges)].contiguous();
}

#endif
