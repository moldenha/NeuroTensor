#include "../exceptions.hpp"
#include "../../../Tensor.h"

#include "../compare.h"
#include "../combine.h"
#include "../index.h"
#include "../fill.h"
#include "../../cpu/fractional_pooling.h"

#include <vector>
#include <variant>
#include <tuple>
#include <cmath>

namespace nt{
namespace functional{

inline std::vector<int64_t> getSlidingWindowSizesOutput(int64_t kernel_size, int64_t input_size, int64_t output_size) {
    std::vector<int64_t> window_sizes(output_size, 0);
    double stride = static_cast<double>(input_size - kernel_size) / (output_size - 1);
    
    int64_t at = 0;

    for (int64_t i = 0; i < output_size; ++i) {
        window_sizes[i] = static_cast<int64_t>(std::round(kernel_size + i * stride)) - at;
        at += window_sizes[i];
    }
    
    return window_sizes;
}

// Function 2: Sliding window sizes based on ratio
inline std::vector<int64_t> getSlidingWindowSizesRatio(int64_t kernel_size, int64_t input_size, double ratio) {
    // Calculate the output size based on the ratio
    int64_t output_size = static_cast<int64_t>(std::round(input_size * ratio));
    return getSlidingWindowSizesOutput(kernel_size, input_size, output_size);
}

inline void check_parameters_fractional(const Tensor& input, int64_t output_size, int64_t kernel_size, int64_t dim){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(input);
    utils::throw_exception(!input.is_null(), "Cannot perform fractional maxpooling on a null tensor");
    dim = (dim < 0) ? dim + input.dims() : dim;
    utils::throw_exception(dim >= 0 && dim < input.dims(), "Expected dimensions of input tensor $ to be greater than or equal to at least $ for fractional max pooling",
                        input.dims(), dim);
    utils::throw_exception(output_size + kernel_size - 1 <= input.shape()[dim],
                           "Error output_size ($) + kernel_size ($) - 1 <= input_shape at $ ($)", 
                           output_size, kernel_size, dim, input.shape()[dim]);

}


inline void check_parameters_fractional(const Tensor& input, double output_ratio, int64_t kernel_size, int64_t dim){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(input);
    check_parameters_fractional(input, static_cast<int64_t>(input.shape()[dim] * output_ratio), kernel_size, dim);
}

Tensor extract_sliding_windows_max_2d(const std::vector<int64_t>& rows, const std::vector<int64_t>& cols, Tensor input){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(input);
    int64_t batches = input.dims() == 2 ? 1 : input.flatten(0, -3).shape()[0];
    Tensor out = nums(input.shape(), false, DType::Bool);
    const SizeRef& in_shape = input.shape();
    cpu::_extract_sliding_windows_max_2d(input.arr_void(), out.arr_void(),rows, cols, batches, in_shape);
    return std::move(out);
}

Tensor extract_sliding_windows_max_3d(const std::vector<int64_t>& channels, const std::vector<int64_t>& rows, const std::vector<int64_t>& cols, Tensor input){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(input);
    int64_t batches = input.dims() == 3 ? 1 : input.flatten(0, -4).shape()[0];
    Tensor out = nums(input.shape(), false, DType::Bool);
    const SizeRef& in_shape = input.shape();
    cpu::_extract_sliding_windows_max_3d(input.arr_void(), out.arr_void(),channels, rows, cols, batches, in_shape);
    return std::move(out);
}


Tensor fractional_max_pool2d(Tensor input, utils::my_tuple kernel_size, utils::my_tuple output_size = -1, 
                                 std::variant<double, std::tuple<double, double>> output_ratio = double(-1.0), bool return_indices = false){
    std::vector<int64_t> row_bounds, col_bounds;
    if(!(output_size == -1)){
        utils::throw_exception(output_ratio.index() == 0 && std::get<0>(output_ratio) == -1.0,
                               "Expected if output size is defined, then output ratio is not defined [and is equal to -1]");
        check_parameters_fractional(input, output_size[0], kernel_size[0], -2);
        check_parameters_fractional(input, output_size[1], kernel_size[1], -1);
        row_bounds = getSlidingWindowSizesOutput(kernel_size[0], input.shape()[-2], output_size[0]);
        col_bounds = getSlidingWindowSizesOutput(kernel_size[1], input.shape()[-1], output_size[1]);
    }else{
        std::tuple<double, double> output_ratio_tup;
        if(output_ratio.index() == 0){
            output_ratio_tup = std::tuple<double, double>{std::get<0>(output_ratio), std::get<0>(output_ratio)};
        }else{
            output_ratio_tup = std::get<1>(output_ratio);
        }

        utils::throw_exception(std::get<0>(output_ratio_tup) > 0.0 && std::get<0>(output_ratio_tup) < 1.0
                               && std::get<1>(output_ratio_tup) > 0.0 && std::get<1>(output_ratio_tup) < 1.0,
                               "Error, expected output ratio to be between 0 and 1 for both arguments, but got {$,$}",
                               std::get<0>(output_ratio_tup), std::get<1>(output_ratio_tup));
        output_size = utils::my_tuple(static_cast<int64_t>(std::get<0>(output_ratio_tup) * input.shape()[-2]),
                       static_cast<int64_t>(std::get<1>(output_ratio_tup) * input.shape()[-1]));
        check_parameters_fractional(input, output_size[0], kernel_size[0], -2);
        check_parameters_fractional(input, output_size[1], kernel_size[1], -1);
        row_bounds = getSlidingWindowSizesOutput(kernel_size[0], input.shape()[-2], output_size[0]);
        col_bounds = getSlidingWindowSizesOutput(kernel_size[1], input.shape()[-1], output_size[1]);
    }
    
    Tensor bools = extract_sliding_windows_max_2d(row_bounds, col_bounds, input);
    Tensor out_max = input[bools].view(input.shape().redo_index(-2, output_size[0]).redo_index(-1, output_size[1]));
    if(!return_indices) return out_max;
    Tensor indices = where(bools.flatten(-2, -1))[-1].item<Tensor>().view(out_max.shape());
    return list(out_max, indices);
}



Tensor fractional_max_pool3d(Tensor input, utils::my_n_tuple<3> kernel_size, utils::my_n_tuple<3> output_size = -1, 
                                 std::variant<double, std::tuple<double, double, double>> output_ratio = double(-1.0), bool return_indices = false){
    std::vector<int64_t> chan_bounds, row_bounds, col_bounds;
    if(!(output_size == -1)){
        utils::throw_exception(output_ratio.index() == 0 && std::get<0>(output_ratio) == -1.0,
                               "Expected if output size is defined, then output ratio is not defined [and is equal to -1]");
        check_parameters_fractional(input, output_size[0], kernel_size[0], -3);
        check_parameters_fractional(input, output_size[1], kernel_size[1], -2);
        check_parameters_fractional(input, output_size[2], kernel_size[2], -1);
        chan_bounds = getSlidingWindowSizesOutput(kernel_size[0], input.shape()[-3], output_size[0]);
        row_bounds = getSlidingWindowSizesOutput(kernel_size[1], input.shape()[-2], output_size[1]);
        col_bounds = getSlidingWindowSizesOutput(kernel_size[2], input.shape()[-1], output_size[2]);
    }else{
        std::tuple<double, double, double> output_ratio_tup;
        if(output_ratio.index() == 0){
            output_ratio_tup = std::tuple<double, double, double>{std::get<0>(output_ratio), std::get<0>(output_ratio), std::get<0>(output_ratio)};
        }else{
            output_ratio_tup = std::get<1>(output_ratio);
        }

        utils::throw_exception(std::get<0>(output_ratio_tup) > 0.0 && std::get<0>(output_ratio_tup) < 1.0
                               && std::get<1>(output_ratio_tup) > 0.0 && std::get<1>(output_ratio_tup) < 1.0
                               && std::get<2>(output_ratio_tup) > 0.0 && std::get<2>(output_ratio_tup) < 1.0,
                               "Error, expected output ratio to be between 0 and 1 for both arguments, but got {$,$,$}",
                               std::get<0>(output_ratio_tup), std::get<1>(output_ratio_tup), std::get<2>(output_ratio_tup));
        output_size = utils::my_n_tuple<3>(static_cast<int64_t>(std::get<0>(output_ratio_tup) * input.shape()[-3]),
                       static_cast<int64_t>(std::get<1>(output_ratio_tup) * input.shape()[-2]),
                       static_cast<int64_t>(std::get<2>(output_ratio_tup) * input.shape()[-1])
        );
        check_parameters_fractional(input, output_size[0], kernel_size[0], -3);
        check_parameters_fractional(input, output_size[1], kernel_size[1], -2);
        check_parameters_fractional(input, output_size[2], kernel_size[2], -1);
        chan_bounds = getSlidingWindowSizesOutput(kernel_size[0], input.shape()[-3], output_size[0]);
        row_bounds = getSlidingWindowSizesOutput(kernel_size[1], input.shape()[-2], output_size[1]);
        col_bounds = getSlidingWindowSizesOutput(kernel_size[2], input.shape()[-1], output_size[2]);
    }
    
    Tensor bools = extract_sliding_windows_max_3d(chan_bounds, row_bounds, col_bounds, input);
    Tensor out_max = input[bools].view(input.shape().redo_index(-3, output_size[0]).redo_index(-2, output_size[1]).redo_index(-1, output_size[2]));
    if(!return_indices) return out_max;
    Tensor indices = where(bools.flatten(-3, -1))[-1].item<Tensor>().view(out_max.shape());
    return list(out_max, indices);
}

}
}
