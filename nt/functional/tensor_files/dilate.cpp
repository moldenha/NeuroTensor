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
    // std::cout << "sizes: "<<batches << ' ' << matrix_size << ' '  << original_cols << std::endl;
    // std::cout << "originals: "<< original_rows << ' ' << original_cols << std::endl;
    for(int64_t b = 0; b < batches; ++b, my_strides += matrix_size){
        void **cur_begin = my_strides;
        for(int64_t r = 0; r < original_rows; r += row_dil, cur_begin += (original_cols * row_dil)){
            void **mBegin = cur_begin;
            for(int64_t c = 0; c < original_cols; c += col_dil, mBegin += col_dil){
                *outp_strides++ = *mBegin;
            }
        }
    }
    // std::cout << "for loops inside of 2d undilate: " << std::endl;
    // std::cout << "for(int64_t b = 0; b < " << batches << "; ++b, my_strides += " << matrix_size<<"){" << std::endl;
    // std::cout << "\tvoid **cur_begin = my_strides;"<<std::endl;
    // std::cout << "\tfor(int64_t r = 0; r < " <<  original_rows << "; r += " << row_dil<<", cur_begin += "<<original_cols * row_dil << "){"<<std::endl;
    // std::cout << "\t\tvoid **mBegin = cur_begin;"<<std::endl;
    // std::cout << "\t\tfor(int64_t c = 0; c < " <<  original_cols << "; c += " << col_dil<<", mBegin += "<<col_dil << "){"<<std::endl;
    // std::cout << "\t\t\t *outp_strides++ = *mBegin;" << std::endl;
    // std::cout << "\t\t}\n\t}\n}" << std::endl;



    // for(int64_t b = 0; b < batches; ++b, my_strides += channel_size){
    //     void **cur_begin = my_strides;
    //     for(int64_t d = 0; d < original_channels; d += chan_dil, cur_begin += matrix_size * chan_dil){
    //         void **dBegin = cur_begin;
    //         for(int64_t r = 0; r < original_rows; r += row_dil, dBegin += (original_cols * row_dil)){
    //             void **mBegin = dBegin;
    //             for(int64_t c = 0; c < original_cols; c += col_dil, mBegin += col_dil){
    //                 *outp_strides++ = *mBegin;
    //             }
    //         }
    //     }
    // }

    return Tensor(std::move(cpy), std::move(outp_shape)).set_mutability(input.is_mutable());
}

Tensor undilate_(const Tensor& input, Tensor::size_value_t col_dil){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(input);
    using size_value_t = Tensor::size_value_t;
    if ((col_dil == 0 || col_dil == 1)) {
        return input;
    }
    utils::throw_exception(input.dims() >= 1, "Expected dim size to be greater than or equal to 1 for undilation but got $", input.dims());
    utils::throw_exception(col_dil >= 1,
                           "Cannot dilate less than 1 but got dilations of {$}",
                           col_dil);
    utils::throw_exception(!input.is_null(),
                           "Cannot undilate a null tensor");

    std::vector<size_value_t> vec = input.shape().Vec();
    vec.back() = (vec.back() + (col_dil - 1)) / col_dil;
    SizeRef outp_shape(std::move(vec));
    
    ArrayVoid cpy1 = input.arr_void().bucket_all_indices();
    ArrayVoid cpy = cpy1.new_strides(outp_shape.multiply());
    void **my_strides = cpy1.stride_begin();
    void **outp_strides = cpy.stride_begin();


    auto& sh = input.shape();
    size_value_t original_cols = sh.back();
    size_value_t batches = input.numel() / original_cols;
    for(int64_t b = 0; b < batches; ++b, my_strides += original_cols){
        void **cur_begin = my_strides;
        for(int64_t c = 0; c < original_cols; c += col_dil, cur_begin += col_dil){
            *outp_strides++ = *cur_begin;
        }
    }

    return Tensor(std::move(cpy), std::move(outp_shape)).set_mutability(input.is_mutable());
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
    
    // std::cout << "originals: "<<original_channels << ' ' << original_rows << ' ' << original_cols << std::endl;
    // std::cout << "sizes: "<< batches << ' ' << channel_size << ' ' << matrix_size << ' ' << original_cols << std::endl;

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



inline void sub_n_undilate_(const std::vector<Tensor::size_value_t>& originals, const std::vector<Tensor::size_value_t>& sizes,
                       const std::vector<Tensor::size_value_t>& dilations, const size_t index,
                        void** my_strides, void**& outp_strides){
    using size_value_t = Tensor::size_value_t;
    if(index == (sizes.size()-1)){
        // handle last for loop for columns
        const size_value_t& end = originals.back();
        const size_value_t& add = dilations.back();
        for(size_value_t c = 0; c < end; c += add, my_strides += add){
            *outp_strides++ = *my_strides;
        }
        return;
    }
    const size_value_t& end = (index == 0) ? sizes[0] : originals[index-1];
    const size_value_t index_add = (index == 0) ? 1 : dilations[index-1];
    const size_value_t my_add = sizes[index+1] * index_add;
    for(size_value_t b = 0; b < end; b += index_add, my_strides += my_add){
        sub_n_undilate_(originals, sizes, dilations, index+1, my_strides, outp_strides);
    }
}

// inline void sub_n_undilate_print_(const std::vector<Tensor::size_value_t>& originals, const std::vector<Tensor::size_value_t>& sizes,
//                        const std::vector<Tensor::size_value_t>& dilations, const size_t index,
//                         void** my_strides, void**& outp_strides, std::string tabs = ""){
//     using size_value_t = Tensor::size_value_t;
//     if(index == (sizes.size()-1)){
//         // handle last for loop for columns
//         const size_value_t& end = originals.back();
//         const size_value_t& add = dilations.back();
//         std::cout << tabs << "for(int64_t c = 0; c < "<<end<<"; c += " << add << ", my_strides += "<<add<<"){"<<std::endl;
//         tabs += '\t';
//         std::cout << tabs << "*outp_strides++ = *my_strides" << std::endl;
//         // for(size_value_t c = 0; c < end; c += add, my_strides += add){
//         //     *outp_strides++ = *my_strides;
//         // }
//         return;
//     }
//     const size_value_t& end = (index == 0) ? sizes[0] : originals[index-1];
//     const size_value_t index_add = (index == 0) ? 1 : dilations[index-1];
//     const size_value_t my_add = sizes[index+1] * index_add;
//     std::cout << tabs << "for(int64_t i = 0; i < "<<end<<"; i += "<<index_add<<", my_strides += "<<my_add<<"){" << std::endl;
//     sub_n_undilate_print_(originals, sizes, dilations, index+1, my_strides, outp_strides, tabs + "\t");

//     // for(size_value_t b = 0; b < end; b += index_add, my_strides += my_add){
//     //     sub_n_undilate_(originals, sizes, dilations, index+1, my_strides, outp_strides);
//     // }
// }

Tensor undilate_(const Tensor& t, std::vector<Tensor::size_value_t> vec, bool test){
    using size_value_t = Tensor::size_value_t;
    if(vec.size() == 1 && !test){
        return undilate_(t, vec[0]);
    }
    if(vec.size() == 2 && !test){
        return undilate_(t, vec[0], vec[1]);
    }
    if(vec.size() == 3 && !test){
        return undilate_(t, vec[0], vec[1], vec[2]);
    }
    if(std::all_of(vec.cbegin(), vec.cend(), [](const auto& val){return val == 0 || val == 1;})){
        return t;
    }

    // Check dimensionality and validate dilation values
    utils::throw_exception(t.dims() >= vec.size(), "Expected dim size to be greater than or equal to $ for $D undilation but got $", 
                           vec.size(), vec.size(), 
                           t.dims());
    utils::throw_exception(std::all_of(vec.cbegin(), vec.cend(), [](const auto& val){return val >= 1;}),
                           "Cannot dilate less than 1 but got dilations of $",
                           SizeRef(vec));
    utils::throw_exception(!t.is_null(),
                           "Cannot undilate a null tensor");

    // Calculate the new shape after undilation for each dimension
    const int64_t dim = vec.size();
    // this needs to be fixed:
    SizeRef outp_shape = t.shape().redo_index(-1, (t.shape().back() + (vec.back() - 1)) / vec.back());
    for(int64_t i = 1; i < dim; ++i){
        const int64_t& shape_val = t.shape()[-1 * (i+1)];
        const int64_t& vec_val = vec[vec.size() - (i+1)];
        outp_shape = outp_shape.redo_index(-1 * (i+1), (shape_val + (vec_val - 1)) / vec_val);
    }



    ArrayVoid cpy1 = t.arr_void().bucket_all_indices();
    ArrayVoid cpy = cpy1.new_strides(outp_shape.multiply());
    void **my_strides = cpy1.stride_begin();
    void **outp_strides = cpy.stride_begin();

    
    const auto& sh = t.shape();
    // std::vector<size_value_t> originals(t.shape().end() - dim, t.shape().end());
    std::vector<size_value_t> originals = t.shape()[-1 * (dim+1) <range> -1].Vec();
    
    std::vector<size_value_t> sizes(vec.size() + 1); // batches, ... , channel_size, matrix_size, column_size
    // std::cout << originals.size() << ' ' << sizes.size() << std::endl;
    sizes.back() = originals.back();
    for(int64_t i = vec.size()-1; i > 0; --i){
        // std::cout << "sizes["<<i<<"]  = (sizes["<<i+1<<"] * originals["<<i-1<<"])"<<std::endl;
        // std::cout << sizes[i+1] << ' ' << originals[i-1] << std::endl;
        sizes[i] = (sizes[i+1] * originals[i-1]);
    }
    sizes[0] = t.numel() / sizes[1];
    // std::cout << "my shape: "<<sh << " versus output shape: "<<outp_shape<<std::endl;
    // std::cout << "originals: ";
    // for(const auto& val : originals)
    //     std::cout << val << ' ';
    // std::cout << std::endl;
    // std::cout << "sizes: ";
    // for(const auto& val : sizes)
    //     std::cout << val << ' ';
    // std::cout << std::endl;

    // size_value_t original_cols = sh.back();
    // size_value_t original_rows = sh[-2];
    // size_value_t original_channels = sh[-3];
    // size_value_t matrix_size = original_rows * original_cols;
    // size_value_t channel_size = matrix_size * original_channels;
    // size_value_t batches = input.numel() / channel_size;

    
    // sub_n_undilate_print_(originals, sizes, vec, 0, my_strides, outp_strides);
    sub_n_undilate_(originals, sizes, vec, 0, my_strides, outp_strides);
    // std::cout << "returning tensor" << std::endl;
    return Tensor(std::move(cpy), std::move(outp_shape)).set_mutability(t.is_mutable());
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
Tensor undilate(const Tensor& t, std::vector<Tensor::size_value_t> dilations) {
    return undilate_(t, std::move(dilations), false).clone(); 
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

Tensor dilate(const Tensor& t, Tensor::size_value_t col_dil){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(t);
    using size_value_t = Tensor::size_value_t;
    if (col_dil == 0)
        return t.contiguous();

    utils::throw_exception(t.dims() >= 1, "Expected dim size to be greater than or equal to 1 for dilation but got $", t.dims());

    std::vector<size_value_t> vec = t.shape().Vec();

    // Adjust shape for dilation (applies to the last two dimensions)
    vec.back() *= col_dil; // Adjust columns
    vec.back() -= (col_dil - 1);

    Tensor outp = zeros(SizeRef(vec), t.dtype());
    Tensor view = undilate_(outp, col_dil);
    view.set_(t);
    return std::move(outp);


}
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


Tensor dilate(const Tensor& t, std::vector<Tensor::size_value_t> dilations, bool test){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(t);
    using size_value_t = Tensor::size_value_t;
    if(dilations.size() == 1 && !test){
        return dilate(t, dilations[0]);
    }
    if(dilations.size() == 2 && !test){
        return dilate(t, dilations[0], dilations[1]);
    }
    if(dilations.size() == 3 && !test){
        return dilate(t, dilations[0], dilations[1], dilations[2]);
    }
    if(std::all_of(dilations.cbegin(), dilations.cend(), [](const auto& val){return val == 0 || val == 1;})){
        return t.contiguous();
    }

    utils::throw_exception(t.dims() >= dilations.size(), "Expected dim size to be greater than or equal to $ for $D undilation but got $", 
                           dilations.size(), dilations.size(), 
                           t.dims());
    utils::throw_exception(std::all_of(dilations.cbegin(), dilations.cend(), [](const auto& val){return val >= 1;}),
                           "Cannot dilate less than 1 but got dilations of $",
                           SizeRef(dilations));
    
    SizeRef n_outp_shape = t.shape().redo_index(-1, (t.shape()[-1] * dilations.back()) - (dilations.back()-1));
    for(size_t i = 1; i < dilations.size(); ++i){
        const int64_t& val = t.shape()[-1 * (i+1)];
        const int64_t& dilation_val = dilations[dilations.size() - (i+1)];
        n_outp_shape = n_outp_shape.redo_index(-1 * (i+1), (val * dilation_val) - (dilation_val - 1));
        // vec[vec.size() - (dilations.size() + i)] *= dilations[i];
        // vec[vec.size() - (dilations.size() + i)] -= (dilations[i] - 1);
    }

    Tensor outp = zeros(std::move(n_outp_shape), t.dtype());
    Tensor undilate_view = undilate_(outp, std::move(dilations), test);
    undilate_view.set_(t);
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
