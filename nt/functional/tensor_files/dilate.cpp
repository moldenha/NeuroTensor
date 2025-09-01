#include "../../Tensor.h"
#include "../../dtype/ArrayVoid.h"
#include "exceptions.hpp"
#include "../../mp/Threading.h"
#include "fill.h"

namespace nt{
namespace functional{


Tensor undilate_(const Tensor& input, Tensor::size_value_t row_dil, Tensor::size_value_t col_dil){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(input);
    using size_value_t = Tensor::size_value_t;
    if ((row_dil == 0 || row_dil == 1) && (col_dil == 0 || col_dil == 1)) {
        return input;
    }
    utils::throw_exception(input.dims() >= 2, "Expected dim size to be greater than or equal to 2 for undilation but got $", input.dims());
    utils::throw_exception(row_dil >= 1 && col_dil >= 1,
                           "Cannot dilate less than 1 but got dilations of {$, $}",
                           row_dil, col_dil);
    utils::throw_exception(!input.is_null(),
                           "Cannot undilate a null tensor");

    std::vector<size_value_t> vec = input.shape().Vec();
    vec.back() = (vec.back() + (col_dil - 1)) / col_dil;
    vec[vec.size() - 2] = (vec[vec.size() - 2] + (row_dil - 1)) / row_dil;
    SizeRef outp_shape(std::move(vec));
    
    ArrayVoid cpy1 = input.arr_void().bucket_all_indices();
    ArrayVoid cpy = cpy1.new_strides(outp_shape.multiply());
    void **my_strides = cpy1.stride_begin();
    void **outp_strides = cpy.stride_begin();


    auto& sh = input.shape();
    size_value_t original_cols = sh.back();
    size_value_t original_rows = sh[-2];
    size_value_t matrix_size = original_rows * original_cols;
    size_value_t batches = input.numel() / matrix_size;
    for(int64_t b = 0; b < batches; ++b, my_strides += matrix_size){
        void **cur_begin = my_strides;
        for(int64_t r = 0; r < original_rows; r += row_dil, cur_begin += (original_cols * row_dil)){
            void **mBegin = cur_begin;
            for(int64_t c = 0; c < original_cols; c += col_dil, mBegin += col_dil){
                *outp_strides++ = *mBegin;
            }
        }
    }

    return Tensor(std::move(cpy), std::move(outp_shape)).set_mutability(input.is_mutable());
}

Tensor undilate_(const Tensor& input, Tensor::size_value_t dil){
    return undilate_(input, 1, dil);
}

Tensor undilate_(const Tensor& input, Tensor::size_value_t chan_dil, Tensor::size_value_t row_dil, Tensor::size_value_t col_dil){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(input);
    using size_value_t = Tensor::size_value_t;
    if ((row_dil == 0 || row_dil == 1) && (col_dil == 0 || col_dil == 1) && (chan_dil == 0 || chan_dil == 1)) {
        return input;
    }

    // Check dimensionality and validate dilation values
    utils::throw_exception(input.dims() >= 3, "Expected dim size to be greater than or equal to 3 for 3D undilation but got $", input.dims());
    utils::throw_exception(row_dil >= 1 && col_dil >= 1 && chan_dil >= 1,
                           "Cannot dilate less than 1 but got dilations of {$, $, $}",
                           row_dil, col_dil, chan_dil);
    utils::throw_exception(!input.is_null(),
                           "Cannot undilate a null tensor");

    // Calculate the new shape after undilation for each dimension
    std::vector<size_value_t> vec = input.shape().Vec();
    vec[vec.size() - 3] = (vec[vec.size() - 3] + (chan_dil - 1)) / chan_dil; // Channel dimension
    vec[vec.size() - 2] = (vec[vec.size() - 2] + (row_dil - 1)) / row_dil;    // Row dimension
    vec.back() = (vec.back() + (col_dil - 1)) / col_dil;                        // Column dimension
    SizeRef outp_shape(std::move(vec));
    
    ArrayVoid cpy1 = input.arr_void().bucket_all_indices();
    ArrayVoid cpy = cpy1.new_strides(outp_shape.multiply());
    void **my_strides = cpy1.stride_begin();
    void **outp_strides = cpy.stride_begin();


    auto& sh = input.shape();
    size_value_t original_cols = sh.back();
    size_value_t original_rows = sh[-2];
    size_value_t original_channels = sh[-3];
    size_value_t matrix_size = original_rows * original_cols;
    size_value_t channel_size = matrix_size * original_channels;
    size_value_t batches = input.numel() / channel_size;

    for(int64_t b = 0; b < batches; ++b, my_strides += channel_size){
        void **cur_begin = my_strides;
        for(int64_t d = 0; d < original_channels; d += chan_dil, cur_begin += matrix_size * chan_dil){
            void **dBegin = cur_begin;
            for(int64_t r = 0; r < original_rows; r += row_dil, dBegin += (original_cols * row_dil)){
                void **mBegin = dBegin;
                for(int64_t c = 0; c < original_cols; c += col_dil, mBegin += col_dil){
                    *outp_strides++ = *mBegin;
                }
            }
        }
    }

    return Tensor(std::move(cpy), std::move(outp_shape)).set_mutability(input.is_mutable());
    
}

Tensor undilate(const Tensor& t, Tensor::size_value_t dil) {
    return undilate_(t, dil).clone();
}
Tensor undilate(const Tensor& t, Tensor::size_value_t row_dil, Tensor::size_value_t col_dil) {
    return undilate_(t, row_dil, col_dil).clone(); 
}
Tensor undilate(const Tensor& t, Tensor::size_value_t chan_dil, Tensor::size_value_t row_dil, Tensor::size_value_t col_dil) {
    return undilate_(t, chan_dil, row_dil, col_dil).clone(); 
}


Tensor dilate(const Tensor& t, Tensor::size_value_t row_dil, Tensor::size_value_t col_dil) {
    _NT_FUNCTIONAL_ALWAYS_CHECK_(t);
    using size_value_t = Tensor::size_value_t;
    if (row_dil == 0 && col_dil == 0)
        return t.contiguous();

    utils::throw_exception(t.dims() >= 2, "Expected dim size to be greater than or equal to 2 for dilation but got $", t.dims());

    std::vector<size_value_t> vec = t.shape().Vec();

    // Adjust shape for dilation (applies to the last two dimensions)
    vec[vec.size() - 1] *= col_dil; // Adjust columns
    vec[vec.size() - 1] -= (col_dil - 1);
    vec[vec.size() - 2] *= row_dil; // Adjust rows
    vec[vec.size() - 2] -= (row_dil - 1);

    Tensor outp = zeros(SizeRef(vec), t.dtype());
    undilate_(outp, row_dil, col_dil).set_(t);
    return std::move(outp);
}

Tensor dilate(const Tensor& t, Tensor::size_value_t dil){return dilate(t, 1, dil);}
Tensor dilate(const Tensor& t, Tensor::size_value_t chan_dil, Tensor::size_value_t row_dil, Tensor::size_value_t col_dil){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(t);
    using size_value_t = Tensor::size_value_t;
    if ((row_dil == 0 || row_dil == 1) && (col_dil == 0 || col_dil == 1) && (chan_dil == 0 || chan_dil == 1))
        return t.contiguous();

    utils::throw_exception(row_dil >= 1 && col_dil >= 1 && chan_dil >= 1,
                           "Cannot dilate less than 1 but got dilations of {$, $, $}",
                           chan_dil, row_dil, col_dil);

    utils::throw_exception(t.dims() >= 3, "Expected dim size to be greater than or equal to 3 for 3D dilation but got $", t.dims());

    std::vector<size_value_t> vec = t.shape().Vec();

    // Adjust shape for dilation (applies to the last two dimensions)
    vec[vec.size() - 1] *= col_dil; // Adjust columns
    vec[vec.size() - 1] -= (col_dil - 1);
    vec[vec.size() - 2] *= row_dil; // Adjust rows
    vec[vec.size() - 2] -= (row_dil - 1);
    vec[vec.size() - 3] *= chan_dil; // Adjust channels
    vec[vec.size() - 3] -= (chan_dil - 1);
    
    Tensor outp = zeros(SizeRef(vec), t.dtype());
    undilate_(outp, chan_dil, row_dil, col_dil).set_(t);
    return std::move(outp);
}


// Tensor undilate_(size_value_t dil) const {
    // return this->undilate_(dil, dil);
    // if (dil == 0) {
    //     return contiguous();
    // }

    // // Calculate the original shape before dilation
    // std::vector<size_value_t> vec = shape().Vec();
    // vec.back() = (vec.back() + (dil - 1)) / dil;
    // vec[vec.size() - 2] = (vec[vec.size() - 2] + (dil - 1)) / dil;
    // SizeRef outp_shape(std::move(vec));
    // /* Tensor outp = functional::zeros(SizeRef(vec), dtype); */

    // ArrayVoid cpy1 = _vals.bucket_all_indices();
    // ArrayVoid cpy = cpy1.new_strides(outp_shape.multiply());
    // void **my_strides = cpy1.stride_begin();
    // void **outp_strides = cpy.stride_begin();

    // size_value_t cols = shape()[-1];
    // size_value_t i_total = shape().multiply(-2);
    // size_value_t outp_cols = vec.back();
    // size_value_t outp_i_total = vec[vec.size() - 2];

    // for (size_value_t i = 0; i < numel(); ++i, ++my_strides) {
    //     // Check if the current element should be part of the original tensor
    //     if ((i % (outp_cols * dil)) % dil == 0 &&
    //         (i / (outp_cols * dil)) % dil == 0) {
    //         *outp_strides = *my_strides;
    //         ++outp_strides;
    //     }
    // }

    // return Tensor(std::move(cpy), std::move(outp_shape));
// }

/*

these is the old dilate functions:
[consider making cpu functions]

Tensor Tensor::dilate(size_value_t row_dil, size_value_t col_dil) const {
    __NT_HANDLE_NULL_TENSORS__();
    if (row_dil == 0 && col_dil == 0)
        return contiguous();

    utils::throw_exception(dims() >= 2, "Expected dim size to be greater than or equal to 2 for dilation but got $", dims());

    std::vector<size_value_t> vec = shape().Vec();

    // Adjust shape for dilation (applies to the last two dimensions)
    vec[vec.size() - 1] *= col_dil; // Adjust columns
    vec[vec.size() - 1] -= (col_dil - 1);
    vec[vec.size() - 2] *= row_dil; // Adjust rows
    vec[vec.size() - 2] -= (row_dil - 1);

    Tensor outp = functional::zeros(SizeRef(vec), dtype);

    auto sh = shape();
    auto outp_shape = outp.shape();
    size_value_t rows = sh[-2]; // Original rows
    size_value_t cols = sh[-1]; // Original columns
    size_value_t num_batches = numel() / (rows * cols); // Total number of batches (product of all dims except last two)
    size_value_t output_cols = outp.shape().back();
    size_value_t back_add = (outp.shape().back()  * (row_dil-1)) + 1;
    // size_value_t back_add = (output_cols - cols * col_dil) + col_dil;

    _vals.cexecute_function<WRAP_DTYPES<AllTypesL>>(
        [&sh, &outp_shape, &num_batches, &rows, &cols, &output_cols, &back_add, &row_dil, &col_dil](auto abegin, auto aend, void *obegin) {
            using value_t = utils::IteratorBaseType_t<decltype(abegin)>;
            size_value_t cols = sh[-1];
            size_value_t i_total = sh.multiply(-2);
            value_t *begin = reinterpret_cast<value_t *>(obegin);
            for (uint64_t i = 0; abegin != aend; ++abegin, ++i) {
                *begin = *abegin;
                if ((i + 1) % cols == 0) {
                    if ((i + 1) % i_total == 0) {
                        ++begin;
                        continue;
                    }
                    begin += back_add;
                    continue;
                }
                begin += col_dil;
            }
        },
        outp.data_ptr());
    return outp;
}

Tensor Tensor::dilate(size_value_t chan_dil, size_value_t row_dil, size_value_t col_dil) const {
    __NT_HANDLE_NULL_TENSORS__();
    if ((row_dil == 0 || row_dil == 1) && (col_dil == 0 || col_dil == 1) && (chan_dil == 0 || chan_dil == 1))
        return contiguous();

    utils::throw_exception(row_dil >= 1 && col_dil >= 1 && chan_dil >= 1,
                           "Cannot dilate less than 1 but got dilations of {$, $, $}",
                           chan_dil, row_dil, col_dil);

    utils::throw_exception(dims() >= 3, "Expected dim size to be greater than or equal to 3 for 3D dilation but got $", dims());

    std::vector<size_value_t> vec = shape().Vec();

    // Adjust shape for dilation (applies to the last two dimensions)
    vec[vec.size() - 1] *= col_dil; // Adjust columns
    vec[vec.size() - 1] -= (col_dil - 1);
    vec[vec.size() - 2] *= row_dil; // Adjust rows
    vec[vec.size() - 2] -= (row_dil - 1);
    vec[vec.size() - 3] *= chan_dil; // Adjust channels
    vec[vec.size() - 3] -= (chan_dil - 1);

    Tensor outp = functional::zeros(SizeRef(vec), dtype);

    auto sh = shape();
    auto outp_shape = outp.shape();
    size_value_t channels = sh[-3]; // Original channels
    size_value_t rows = sh[-2]; // Original rows
    size_value_t cols = sh[-1]; // Original columns

    // size_value_t num_batches = numel() / (rows * cols); // Total number of batches (product of all dims except last two)
    
    size_value_t output_cols = outp.shape().back();
    size_value_t col_back_add = 1;
    size_value_t row_back_add = (outp.shape().back()  * (row_dil-1)) + 1;
    size_value_t channel_back_add = ((outp.shape()[-1] * outp.shape()[-2]) * chan_dil - 1) + 1;
    // size_value_t back_add = (output_cols - cols * col_dil) + col_dil;

    _vals.cexecute_function<WRAP_DTYPES<AllTypesL>>(
        [&sh, &outp_shape, &channels, &rows, &cols, 
            &col_back_add, &row_back_add, &channel_back_add, 
            &chan_dil, &row_dil, &col_dil](auto abegin, auto aend, void *obegin) {
            using value_t = utils::IteratorBaseType_t<decltype(abegin)>;
            size_value_t cols = sh[-1];
            size_value_t mat_total = sh.multiply(-2);
            size_value_t batched_mat_total = sh.multiply(-3);
            value_t *begin = reinterpret_cast<value_t *>(obegin);
            for (uint64_t i = 0; abegin != aend; ++abegin, ++i) {
                *begin = *abegin;
                if ((i + 1) % cols == 0) {
                    if ((i + 1) % mat_total == 0) {
                        if((i + 1) % batched_mat_total == 0){
                            begin++;
                            continue;
                        }
                        begin += channel_back_add;
                        continue;
                    }
                    begin += row_back_add;
                    continue;
                }
                begin += col_dil;
            }
        },
        outp.data_ptr());
    return outp;
}



*/

}
}
