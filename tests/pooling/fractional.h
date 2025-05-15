#ifndef __NT_FUNCTIONAL_FRACTIONAL_POOLING_2D_H__
#define __NT_FUNCTIONAL_FRACTIONAL_POOLING_2D_H__

#include <nt/Tensor.h>
#include <nt/dtype/ArrayVoid.hpp>

#include <variant>
#include <tuple>
#include <vector>
#include <cmath>

std::vector<int64_t> getSlidingWindowSizesOutput(int64_t kernel_size, int64_t input_size, int64_t output_size) {
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
std::vector<int64_t> getSlidingWindowSizesRatio(int64_t kernel_size, int64_t input_size, double ratio) {
    // Calculate the output size based on the ratio
    int64_t output_size = static_cast<int64_t>(std::round(input_size * ratio));
    return getSlidingWindowSizesOutput(kernel_size, input_size, output_size);
}

void check_parameters_fractional(const nt::Tensor& input, int64_t output_size, int64_t kernel_size, int64_t dim){
    using namespace nt;
    utils::throw_exception(!input.is_null(), "Cannot perform fractional maxpooling on a null tensor");
    dim = (dim < 0) ? dim + input.dims() : dim;
    utils::throw_exception(dim >= 0 && dim < input.dims(), "Expected dimensions of input tensor $ to be greater than or equal to at least $ for fractional max pooling",
                        input.dims(), dim);
    utils::throw_exception(output_size + kernel_size - 1 <= input.shape()[dim],
                           "Error output_size ($) + kernel_size ($) - 1 <= input_shape at $ ($)", 
                           output_size, kernel_size, dim, input.shape()[dim]);

}


void check_parameters_fractional(const nt::Tensor& input, double output_ratio, int64_t kernel_size, int64_t dim){
    check_parameters_fractional(input, static_cast<int64_t>(input.shape()[dim] * output_ratio), kernel_size, dim);
}

//nt::Tensor extract_sliding_windows_2d(const std::vector<int64_t>& rows, const std::vector<int64_t>& cols, const nt::Tensor input){
//    using namespace nt;
//    int64_t batches = input.dims() == 2 ? 1 : input.flatten(0, -3).shape()[0];
//    ArrayVoid cur_vals = input.arr_void().bucket_all_indices(); //ensures this is strided
//    ArrayVoid out_vals = cur_vals.new_strides(input.numel());
//    //out vals has undefined values
//    int64_t _row = rows.size();
//    int64_t _col = cols.size();
//    const int64_t& in_cols = input.shape()[-1];
//    void** out_strides = out_vals.stride_begin();
//    void** in_strides = cur_vals.stride_begin();
//    // void** in_stride_end = cur_vals.stride_end();
//    int64_t batch_add = input.shape()[-1] * input.shape()[-2];
//    int64_t b_add = 0;
//    for(int64_t b = 0; b < batches; ++b, b_add += batch_add){
//        int64_t cur_row = 0;
//        for(int64_t r = 0; r < rows.size(); ++r){
//            int64_t cur_col = 0;
//            for(int64_t c = 0; c < cols.size(); ++c){
//                for(int64_t _r = 0; _r < rows[r]; ++_r){
//                    for(int64_t _c = 0; _c < cols[c]; ++_c, ++out_strides){
//                        *out_strides = in_strides[(b_add) + (cur_row + _r) * in_cols + (cur_col + _c)];
//                    }
//                }
//                cur_col += cols[c];
//            }
//            cur_row += rows[r];
//        }
//    }
    
//    return Tensor(out_vals, {batches, batch_add});
//}


nt::Tensor extract_sliding_windows_max_2d(const std::vector<int64_t>& rows, const std::vector<int64_t>& cols, nt::Tensor input){
    using namespace nt;
    input = input.contiguous(); // for speed reasons
    int64_t batches = input.dims() == 2 ? 1 : input.flatten(0, -3).shape()[0];
    Tensor out = functional::nums(input.shape(), false, DType::Bool);
    const SizeRef& in_shape = input.shape();
    bool* o_begin = reinterpret_cast<bool*>(out.data_ptr());
    input.arr_void().cexecute_function<WRAP_DTYPES<NumberTypesL>>(
    [&rows, &cols, o_begin, &batches, &in_shape](auto begin, auto end){
        using value_t = utils::IteratorBaseType_t<decltype(begin)>;
        int64_t _row = rows.size();
        int64_t _col = cols.size();
        const int64_t& in_cols = in_shape[-1];
        int64_t batch_add = in_shape[-1] * in_shape[-2];
        int64_t b_add = 0;
        for(int64_t b = 0; b < batches; ++b, b_add += batch_add){
            int64_t cur_row = 0;
            for(int64_t r = 0; r < rows.size(); ++r){
                int64_t cur_col = 0;
                for(int64_t c = 0; c < cols.size(); ++c){
                    value_t val = begin[(b_add) + cur_row * in_cols + cur_col];
                    bool* b_val = &o_begin[(b_add) + cur_row * in_cols + cur_col];
                    for(int64_t _r = rows[r]-1; _r >= 0; --_r){
                        for(int64_t _c = cols[c]-1; _c >= 0; --_c){
                            if(begin[(b_add) + (cur_row + _r) * in_cols + (cur_col + _c)] > val){
                                val = begin[(b_add) + (cur_row + _r) * in_cols + (cur_col + _c)];
                                b_val = &o_begin[(b_add) + (cur_row + _r) * in_cols + (cur_col + _c)];
                            }
                        }
                    }
                    *b_val = true;
                    cur_col += cols[c];
                }
                cur_row += rows[r];
            }
        }
    });
    return std::move(out);
}


nt::Tensor fractional_max_pool2d(nt::Tensor input, nt::utils::my_tuple kernel_size, nt::utils::my_tuple output_size = -1, 
                                 std::variant<double, std::tuple<double, double>> output_ratio = double(-1.0), bool return_indices = false){
    using namespace nt;
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
    Tensor indices = functional::where(bools.flatten(-2, -1))[-1].item<nt::Tensor>().view(out_max.shape());
    return functional::list(out_max, indices);
}



nt::Tensor extract_sliding_windows_max_3d(const std::vector<int64_t>& channels, const std::vector<int64_t>& rows, const std::vector<int64_t>& cols, nt::Tensor input){
    using namespace nt;
    input = input.contiguous(); // for speed reasons
    int64_t batches = input.dims() == 3 ? 1 : input.flatten(0, -4).shape()[0];
    Tensor out = functional::nums(input.shape(), false, DType::Bool);
    const SizeRef& in_shape = input.shape();
    bool* o_begin = reinterpret_cast<bool*>(out.data_ptr());
    input.arr_void().cexecute_function<WRAP_DTYPES<NumberTypesL>>(
    [&channels, &rows, &cols, o_begin, &batches, &in_shape](auto begin, auto end){
        using value_t = utils::IteratorBaseType_t<decltype(begin)>;
        int64_t _chan = channels.size();
        int64_t _row = rows.size();
        int64_t _col = cols.size();
        const int64_t& in_cols = in_shape[-1];
        const int64_t in_matrix = in_shape[-2] * in_shape[-1];
        int64_t batch_add = in_shape[-1] * in_shape[-2] * in_shape[-3];
        int64_t b_add = 0;
        for(int64_t b = 0; b < batches; ++b, b_add += batch_add){
            int64_t cur_chan = 0;
            for(int64_t d = 0; d < channels.size(); ++d){
                int64_t cur_row = 0;
                for(int64_t r = 0; r < rows.size(); ++r){
                    int64_t cur_col = 0;
                    for(int64_t c = 0; c < cols.size(); ++c){
                        value_t val = begin[(b_add) + cur_chan * in_matrix + cur_row * in_cols + cur_col];
                        bool* b_val = &o_begin[(b_add) + cur_chan * in_matrix + cur_row * in_cols + cur_col];
                        for(int64_t _d = channels[d]-1; _d >= 0; --_d){
                            for(int64_t _r = rows[r]-1; _r >= 0; --_r){
                                for(int64_t _c = cols[c]-1; _c >= 0; --_c){
                                    if(begin[(b_add) + (cur_chan + _d) * in_matrix + (cur_row + _r) * in_cols + (cur_col + _c)] > val){
                                        val = begin[(b_add) + (cur_chan + _d) * in_matrix + (cur_row + _r) * in_cols + (cur_col + _c)];
                                        b_val = &o_begin[(b_add) + (cur_chan + _d) * in_matrix + (cur_row + _r) * in_cols + (cur_col + _c)];
                                    }
                                }
                            }
                        }
                        *b_val = true;
                        cur_col += cols[c];
                    }
                    cur_row += rows[r];
                }
                cur_chan += channels[d];
            }
        }
    });
    return std::move(out);
}



nt::Tensor fractional_max_pool3d(nt::Tensor input, nt::utils::my_n_tuple<3> kernel_size, nt::utils::my_n_tuple<3> output_size = -1, 
                                 std::variant<double, std::tuple<double, double, double>> output_ratio = double(-1.0), bool return_indices = false){
    using namespace nt;
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
    Tensor indices = functional::where(bools.flatten(-3, -1))[-1].item<nt::Tensor>().view(out_max.shape());
    return functional::list(out_max, indices);
}

#endif
