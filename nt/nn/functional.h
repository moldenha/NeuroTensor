// this is a friend class to TensorGrad that holds the functional functions
// dedicated to tensorgrad this is meant to re-create the functional.h that is
// dedicated to the Tensor class

#ifndef _NT_TENSORGAD_FUNCTIONAL_H_
#define _NT_TENSORGAD_FUNCTIONAL_H_



#include "functional_class.h"

namespace nt {
namespace functional {

inline TensorGrad matmult(const TensorGrad &a, const TensorGrad &b, bool transpose_a = false, bool transpose_b = false) {
    return TensorGrad_Functional_Class::matmult(a, b, transpose_a, transpose_b);
}

inline TensorGrad matmult(const Tensor &a, const TensorGrad &b, bool transpose_a = false, bool transpose_b = false) {
    return TensorGrad_Functional_Class::matmult(a, b, transpose_a, transpose_b);
}

inline TensorGrad matmult(const TensorGrad &a, const Tensor &b, bool transpose_a = false, bool transpose_b = false) {
    return TensorGrad_Functional_Class::matmult(a, b, transpose_a, transpose_b);
}

inline TensorGrad linear(const TensorGrad& input, const TensorGrad& weight, const TensorGrad& bias, bool transpose_a = false, bool transpose_b = false){
    return TensorGrad_Functional_Class::linear(input, weight, bias, transpose_a, transpose_b);
}
inline TensorGrad linear(const Tensor& input, const TensorGrad& weight, const TensorGrad& bias, bool transpose_a = false, bool transpose_b = false){
    return TensorGrad_Functional_Class::linear(input, weight, bias, transpose_a, transpose_b);
}
inline TensorGrad linear(const TensorGrad& input, const Tensor& weight, const TensorGrad& bias, bool transpose_a = false, bool transpose_b = false){
    return TensorGrad_Functional_Class::linear(input, weight, bias, transpose_a, transpose_b);
}
inline TensorGrad linear(const TensorGrad& input, const TensorGrad& weight, const Tensor& bias, bool transpose_a = false, bool transpose_b = false){
    return TensorGrad_Functional_Class::linear(input, weight, bias, transpose_a, transpose_b);
}
inline TensorGrad linear(const TensorGrad& input, const Tensor& weight, const Tensor& bias, bool transpose_a = false, bool transpose_b = false){
    return TensorGrad_Functional_Class::linear(input, weight, bias, transpose_a, transpose_b);
}
inline TensorGrad linear(const Tensor& input, const TensorGrad& weight, const Tensor& bias, bool transpose_a = false, bool transpose_b = false){
    return TensorGrad_Functional_Class::linear(input, weight, bias, transpose_a, transpose_b);
}
inline TensorGrad linear(const Tensor& input, const Tensor& weight, const TensorGrad& bias, bool transpose_a = false, bool transpose_b = false){
    return TensorGrad_Functional_Class::linear(input, weight, bias, transpose_a, transpose_b);
}

inline TensorGrad
unfold1d(const TensorGrad &a, Tensor::size_value_t kernel_size,
         Tensor::size_value_t dilation = 1, Tensor::size_value_t padding = 0,
         Tensor::size_value_t stride = 1, bool transpose_out = true) {
    return TensorGrad_Functional_Class::unfold1d(
        a, kernel_size, dilation, padding, stride, transpose_out);
}

inline TensorGrad unfold(const TensorGrad &a, utils::my_tuple kernel_size,
                         utils::my_tuple dilation = 1,
                         utils::my_tuple padding = 0,
                         utils::my_tuple stride = 1,
                         bool transpose_out = true) {
    return TensorGrad_Functional_Class::unfold(a, kernel_size, dilation,
                                               padding, stride, transpose_out);
}

inline TensorGrad
unfold3d(const TensorGrad &a, utils::my_n_tuple<3> kernel_size,
         utils::my_n_tuple<3> dilation = 1, utils::my_n_tuple<3> padding = 0,
         utils::my_n_tuple<3> stride = 1, bool transpose_out = true) {
    return TensorGrad_Functional_Class::unfold3d(
        a, kernel_size, dilation, padding, stride, transpose_out);
}

inline TensorGrad fold(const TensorGrad &a, utils::my_tuple output_size,
                       utils::my_tuple kernel_size,
                       utils::my_tuple dilation = 1,
                       utils::my_tuple padding = 0,
                       utils::my_tuple stride = 1) {
    return TensorGrad_Functional_Class::fold(a, output_size, kernel_size,
                                             dilation, padding, stride);
}

inline TensorGrad conv1d(const TensorGrad &image, const TensorGrad &kernel,
                         int64_t stride = 1, int64_t padding = 0,
                         int64_t dilation = 1, int64_t groups = 1) {
    return TensorGrad_Functional_Class::conv1d(image, kernel, stride,
                                                   padding, dilation, groups);
}

inline TensorGrad conv1d(const Tensor &image, const TensorGrad &kernel,
                         int64_t stride = 1, int64_t padding = 0,
                         int64_t dilation = 1, int64_t groups = 1) {
    return TensorGrad_Functional_Class::conv1d(image, kernel, stride,
                                                   padding, dilation, groups);
}

inline TensorGrad conv1d(const TensorGrad &image, const Tensor &kernel,
                         int64_t stride = 1, int64_t padding = 0,
                         int64_t dilation = 1, int64_t groups = 1) {
    return TensorGrad_Functional_Class::conv1d(image, kernel, stride,
                                                   padding, dilation, groups);
}


inline TensorGrad conv2d(const TensorGrad &image, const TensorGrad &kernel,
                         utils::my_tuple stride = 1, utils::my_tuple padding = 0,
                         utils::my_tuple dilation = 1, int64_t groups = 1) {
    return TensorGrad_Functional_Class::conv2d(image, kernel, stride,
                                                   padding, dilation, groups);
}

inline TensorGrad conv2d(const Tensor &image, const TensorGrad &kernel,
                         utils::my_tuple stride = 1, utils::my_tuple padding = 0,
                         utils::my_tuple dilation = 1, int64_t groups = 1) {
    return TensorGrad_Functional_Class::conv2d(image, kernel, stride,
                                                   padding, dilation, groups);
}

inline TensorGrad conv2d(const TensorGrad &image, const Tensor &kernel,
                         utils::my_tuple stride = 1, utils::my_tuple padding = 0,
                         utils::my_tuple dilation = 1, int64_t groups = 1) {
    return TensorGrad_Functional_Class::conv2d(image, kernel, stride,
                                                   padding, dilation, groups);
}

inline TensorGrad conv3d(const TensorGrad &image, const TensorGrad &kernel,
                         utils::my_n_tuple<3> stride = 1, utils::my_n_tuple<3> padding = 0,
                         utils::my_n_tuple<3> dilation = 1, int64_t groups = 1) {
    return TensorGrad_Functional_Class::conv3d(image, kernel, stride,
                                                   padding, dilation, groups);
}

inline TensorGrad conv3d(const Tensor &image, const TensorGrad &kernel,
                         utils::my_n_tuple<3> stride = 1, utils::my_n_tuple<3> padding = 0,
                         utils::my_n_tuple<3> dilation = 1, int64_t groups = 1) {
    return TensorGrad_Functional_Class::conv3d(image, kernel, stride,
                                                   padding, dilation, groups);
}

inline TensorGrad conv3d(const TensorGrad &image, const Tensor &kernel,
                         utils::my_n_tuple<3> stride = 1, utils::my_n_tuple<3> padding = 0,
                         utils::my_n_tuple<3> dilation = 1, int64_t groups = 1) {
    return TensorGrad_Functional_Class::conv3d(image, kernel, stride,
                                                   padding, dilation, groups);
}


inline TensorGrad conv_transpose1d(const TensorGrad &image, const TensorGrad &kernel,
                         int64_t stride = 1, int64_t padding = 0,
                         int64_t output_padding = 0, int64_t dilation = 1, 
                         int64_t groups = 1) {
    return TensorGrad_Functional_Class::conv_transpose1d(image, kernel, stride,
                                                padding, output_padding, dilation, groups);
}

inline TensorGrad conv_transpose1d(const Tensor &image, const TensorGrad &kernel,
                         int64_t stride = 1, int64_t padding = 0,
                         int64_t output_padding = 0, int64_t dilation = 1, 
                         int64_t groups = 1) {
    return TensorGrad_Functional_Class::conv_transpose1d(image, kernel, stride,
                                                padding, output_padding, dilation, groups);
}

inline TensorGrad conv_transpose1d(const TensorGrad &image, const Tensor &kernel,
                         int64_t stride = 1, int64_t padding = 0,
                         int64_t output_padding = 0, int64_t dilation = 1, 
                         int64_t groups = 1) {
    return TensorGrad_Functional_Class::conv_transpose1d(image, kernel, stride,
                                                padding, output_padding, dilation, groups);
}


inline TensorGrad conv_transpose2d(const TensorGrad &image, const TensorGrad &kernel,
                         utils::my_tuple stride = 1, utils::my_tuple padding = 0,
                         utils::my_tuple output_padding = 0,utils::my_tuple dilation = 1, 
                         int64_t groups = 1) {
    return TensorGrad_Functional_Class::conv_transpose2d(image, kernel, stride,
                                                padding, output_padding, dilation, groups);
}

inline TensorGrad conv_transpose2d(const Tensor &image, const TensorGrad &kernel,
                         utils::my_tuple stride = 1, utils::my_tuple padding = 0,
                         utils::my_tuple output_padding = 0,utils::my_tuple dilation = 1, 
                         int64_t groups = 1) {
    return TensorGrad_Functional_Class::conv_transpose2d(image, kernel, stride,
                                                padding, output_padding, dilation, groups);
}

inline TensorGrad conv_transpose2d(const TensorGrad &image, const Tensor &kernel,
                         utils::my_tuple stride = 1, utils::my_tuple padding = 0,
                         utils::my_tuple output_padding = 0,utils::my_tuple dilation = 1, 
                         int64_t groups = 1) {
    return TensorGrad_Functional_Class::conv_transpose2d(image, kernel, stride,
                                                padding, output_padding, dilation, groups);
}

inline TensorGrad conv_transpose3d(const TensorGrad &image, const TensorGrad &kernel,
                         utils::my_n_tuple<3> stride = 1, utils::my_n_tuple<3> padding = 0,
                         utils::my_n_tuple<3> output_padding = 0, utils::my_n_tuple<3> dilation = 1, 
                         int64_t groups = 1) {
    return TensorGrad_Functional_Class::conv_transpose3d(image, kernel, stride,
                                                padding, output_padding, dilation, groups);
}

inline TensorGrad conv_transpose3d(const Tensor &image, const TensorGrad &kernel,
                         utils::my_n_tuple<3> stride = 1, utils::my_n_tuple<3> padding = 0,
                         utils::my_n_tuple<3> output_padding = 0, utils::my_n_tuple<3> dilation = 1, 
                         int64_t groups = 1) {
    return TensorGrad_Functional_Class::conv_transpose3d(image, kernel, stride,
                                                padding, output_padding, dilation, groups);
}

inline TensorGrad conv_transpose3d(const TensorGrad &image, const Tensor &kernel,
                         utils::my_n_tuple<3> stride = 1, utils::my_n_tuple<3> padding = 0, 
                         utils::my_n_tuple<3> output_padding = 0, utils::my_n_tuple<3> dilation = 1, 
                         int64_t groups = 1) {
    return TensorGrad_Functional_Class::conv_transpose3d(image, kernel, stride,
                                                padding, output_padding, dilation, groups);
}



inline TensorGrad sigmoid(const TensorGrad &a) {
    return TensorGrad_Functional_Class::sigmoid(a);
}

inline TensorGrad clamp(const TensorGrad &a,
                        std::optional<int64_t> min = std::nullopt,
                        std::optional<int64_t> max = std::nullopt) {
    return TensorGrad_Functional_Class::clamp(a, min, max);
}

inline TensorGrad relu(const TensorGrad &a) {
    return TensorGrad_Functional_Class::relu(a);
}

inline TensorGrad var(const TensorGrad &a, utils::optional_list dim = nullptr,
                      int64_t correction = 1, bool keepdim = false) {
    return TensorGrad_Functional_Class::var(a, dim, correction, keepdim);
}

inline TensorGrad sqrt(const TensorGrad &a) {
    return TensorGrad_Functional_Class::sqrt(a);
}

inline TensorGrad invsqrt(const TensorGrad &a) {
    return TensorGrad_Functional_Class::invsqrt(a);
}

inline TensorGrad silu(const TensorGrad &a) {
    return TensorGrad_Functional_Class::silu(a);
}
inline TensorGrad gelu(const TensorGrad &a) {
    return TensorGrad_Functional_Class::gelu(a);
}
inline TensorGrad tanh(const TensorGrad &a) {
    return TensorGrad_Functional_Class::tanh(a);
}
inline TensorGrad tan(const TensorGrad &a) {
    return TensorGrad_Functional_Class::tan(a);
}
inline TensorGrad sinh(const TensorGrad &a) {
    return TensorGrad_Functional_Class::sinh(a);
}
inline TensorGrad sin(const TensorGrad &a) {
    return TensorGrad_Functional_Class::sin(a);
}
inline TensorGrad cosh(const TensorGrad &a) {
    return TensorGrad_Functional_Class::cosh(a);
}
inline TensorGrad cos(const TensorGrad &a) {
    return TensorGrad_Functional_Class::cos(a);
}
inline TensorGrad atanh(const TensorGrad &a) {
    return TensorGrad_Functional_Class::atanh(a);
}
inline TensorGrad atan(const TensorGrad &a) {
    return TensorGrad_Functional_Class::atan(a);
}
inline TensorGrad asinh(const TensorGrad &a) {
    return TensorGrad_Functional_Class::asinh(a);
}
inline TensorGrad asin(const TensorGrad &a) {
    return TensorGrad_Functional_Class::asin(a);
}
inline TensorGrad acosh(const TensorGrad &a) {
    return TensorGrad_Functional_Class::acosh(a);
}
inline TensorGrad acos(const TensorGrad &a) {
    return TensorGrad_Functional_Class::acos(a);
}
inline TensorGrad cotanh(const TensorGrad &a) {
    return TensorGrad_Functional_Class::cotanh(a);
}
inline TensorGrad cotan(const TensorGrad &a) {
    return TensorGrad_Functional_Class::cotan(a);
}
inline TensorGrad csch(const TensorGrad &a) {
    return TensorGrad_Functional_Class::csch(a);
}
inline TensorGrad csc(const TensorGrad &a) {
    return TensorGrad_Functional_Class::csc(a);
}
inline TensorGrad sech(const TensorGrad &a) {
    return TensorGrad_Functional_Class::sech(a);
}
inline TensorGrad sec(const TensorGrad &a) {
    return TensorGrad_Functional_Class::sec(a);
}


inline TensorGrad cat(std::vector<TensorGrad> tgs, int64_t dim = 0) {
    return TensorGrad_Functional_Class::cat(std::move(tgs), dim);
}

inline TensorGrad cat(TensorGrad tg, int64_t dim = 0) {
    return TensorGrad_Functional_Class::cat(std::move(tg), dim);
}

inline TensorGrad chunk(TensorGrad input, typename Tensor::size_value_t chunks,
                        int64_t dim = 0) {
    return TensorGrad_Functional_Class::chunk(std::move(input), chunks, dim);
}


inline TensorGrad split(TensorGrad input, typename Tensor::size_value_t split_size, int64_t dim){
    return TensorGrad_Functional_Class::split(std::move(input), split_size, dim);
}


inline TensorGrad split(TensorGrad input, std::vector<typename Tensor::size_value_t> split_sections, int64_t dim){
    return TensorGrad_Functional_Class::split(std::move(input), std::move(split_sections), dim);
    
}

inline TensorGrad stack(TensorGrad input, int64_t dim){
    return TensorGrad_Functional_Class::stack(std::move(input), dim);
}

inline TensorGrad stack(std::vector<TensorGrad> input, int64_t dim){
    return TensorGrad_Functional_Class::stack(std::move(input), dim);
}


inline TensorGrad log(const TensorGrad &a) {
    return TensorGrad_Functional_Class::log(a);
}


inline TensorGrad logsumexp(const TensorGrad &a, utils::optional_list list = nullptr, bool keepdim = false) {
    return TensorGrad_Functional_Class::logsumexp(a, list, keepdim);
}

inline TensorGrad dropout(const TensorGrad &input, double p) {
    return TensorGrad_Functional_Class::dropout(input, p);
}

inline TensorGrad abs(const TensorGrad &input){
    return TensorGrad_Functional_Class::abs(input);
}

inline TensorGrad softplus(const TensorGrad &input, Scalar beta=1.0, Scalar threshold=20.0){
    return TensorGrad_Functional_Class::softplus(input, beta, threshold);
}

//by default is stable
inline TensorGrad softmax(const TensorGrad& input){
    return TensorGrad_Functional_Class::softmax(input, true);
}

inline TensorGrad softmax(const TensorGrad& input, typename SizeRef::value_type dim){
    return TensorGrad_Functional_Class::softmax(input, dim, true);
}


inline TensorGrad softmax_unstable(const TensorGrad& input){
    return TensorGrad_Functional_Class::softmax(input, false);
}

inline TensorGrad softmax_unstable(const TensorGrad& input, typename SizeRef::value_type dim){
    return TensorGrad_Functional_Class::softmax(input, dim, false);
}

inline TensorGrad gumbel_softmax(const TensorGrad& input, Scalar tau, bool hard, bool stable = true){
    return TensorGrad_Functional_Class::gumbel_softmax(input, tau, hard, stable);
}

inline TensorGrad symmetric_bilinear(const TensorGrad& input, const TensorGrad& W1, const TensorGrad& W2){
    return TensorGrad_Functional_Class::symmetric_bilinear(input, W1, W2);
}
inline TensorGrad symmetric_bilinear(const TensorGrad& input, const TensorGrad& W1, const Tensor& W2){
    return TensorGrad_Functional_Class::symmetric_bilinear(input, W1, W2);
}
inline TensorGrad symmetric_bilinear(const TensorGrad& input, const Tensor& W1, const TensorGrad& W2){
    return TensorGrad_Functional_Class::symmetric_bilinear(input, W1, W2);
}
inline TensorGrad symmetric_bilinear(const Tensor& input, const TensorGrad& W1, const TensorGrad& W2){
    return TensorGrad_Functional_Class::symmetric_bilinear(input, W1, W2);
}
inline TensorGrad symmetric_bilinear(const TensorGrad& input, const Tensor& W1, const Tensor& W2){
    return TensorGrad_Functional_Class::symmetric_bilinear(input, W1, W2);
}
inline TensorGrad symmetric_bilinear(const Tensor& input, const Tensor& W1, const TensorGrad& W2){
    return TensorGrad_Functional_Class::symmetric_bilinear(input, W1, W2);
}
inline TensorGrad symmetric_bilinear(const Tensor& input, const TensorGrad& W1, const Tensor& W2){
    return TensorGrad_Functional_Class::symmetric_bilinear(input, W1, W2);
}


//Average Pooling
inline TensorGrad avg_pool1d(TensorGrad input, int64_t kernel_size, int64_t stride = -1, int64_t padding = 0, bool ceil_mode = false, bool count_include_pad = true){
    return TensorGrad_Functional_Class::avg_pool1d(input, kernel_size, stride, padding, ceil_mode, count_include_pad); 
}
inline TensorGrad adaptive_avg_pool1d(TensorGrad x, int64_t l_out){
    return TensorGrad_Functional_Class::adaptive_avg_pool1d(x, l_out); 
}

inline TensorGrad avg_pool2d(TensorGrad input, utils::my_tuple kernel_size, utils::my_tuple stride = -1, utils::my_tuple padding = 0, 
                      bool ceil_mode = false, bool count_include_pad = true){
    return TensorGrad_Functional_Class::avg_pool2d(input, kernel_size, stride, padding, 
                    ceil_mode, count_include_pad); 
}
inline TensorGrad adaptive_avg_pool2d(TensorGrad x, utils::my_tuple out_shape){
    return TensorGrad_Functional_Class::adaptive_avg_pool2d(x, out_shape); 
}

inline TensorGrad avg_pool3d(TensorGrad input, utils::my_n_tuple<3> kernel_size, utils::my_n_tuple<3> stride = -1, utils::my_n_tuple<3> padding = 0, 
                      bool ceil_mode = false, bool count_include_pad = true){
    return TensorGrad_Functional_Class::avg_pool3d(input, kernel_size, stride, padding, 
                    ceil_mode, count_include_pad); 
}
inline TensorGrad adaptive_avg_pool3d(TensorGrad x, utils::my_n_tuple<3> out_shape){
    return TensorGrad_Functional_Class::adaptive_avg_pool3d(x, out_shape); 
}

//LP Pooling
inline TensorGrad lp_pool1d(TensorGrad input, Scalar power, int64_t kernel_size, int64_t stride = -1, bool ceil_mode = false){
    return TensorGrad_Functional_Class::lp_pool1d(input, power, kernel_size, stride, ceil_mode); 
}
inline TensorGrad adaptive_lp_pool1d(TensorGrad x, int64_t l_out, Scalar power){
    return TensorGrad_Functional_Class::adaptive_lp_pool1d(x, l_out, power); 
}

inline TensorGrad lp_pool2d(TensorGrad input, Scalar power, utils::my_tuple kernel_size, utils::my_tuple stride = -1, bool ceil_mode = false){
    return TensorGrad_Functional_Class::lp_pool2d(input, power, kernel_size, stride, ceil_mode); 
}
inline TensorGrad adaptive_lp_pool2d(TensorGrad x, utils::my_tuple out_shape, Scalar power){
    return TensorGrad_Functional_Class::adaptive_lp_pool2d(x, out_shape, power); 
}

inline TensorGrad lp_pool3d(TensorGrad input, Scalar power, utils::my_n_tuple<3> kernel_size, utils::my_n_tuple<3> stride = -1, bool ceil_mode = false){
    return TensorGrad_Functional_Class::lp_pool3d(input, power, kernel_size, stride, ceil_mode); 
}
inline TensorGrad adaptive_lp_pool3d(TensorGrad x, utils::my_n_tuple<3> out_shape, Scalar power){
    return TensorGrad_Functional_Class::adaptive_lp_pool3d(x, out_shape, power); 
}


//Max Pooling
inline TensorGrad max_pool1d(TensorGrad input, int64_t kernel_size, int64_t stride = -1, int64_t padding = 0, int64_t dilation=1, bool ceil_mode = false, bool return_indices = false){
    return TensorGrad_Functional_Class::max_pool1d(input, kernel_size, stride, padding, dilation, ceil_mode, return_indices); 
}
inline TensorGrad max_unpool1d(TensorGrad input, TensorGrad indices, int64_t kernel_size, int64_t stride = -1, int64_t padding = 0, int64_t output_size=-1){
    return TensorGrad_Functional_Class::max_unpool1d(input, indices, kernel_size, stride, padding, output_size); 
}
inline TensorGrad max_unpool1d(TensorGrad input, Tensor indices, int64_t kernel_size, int64_t stride = -1, int64_t padding = 0, int64_t output_size=-1){
    return TensorGrad_Functional_Class::max_unpool1d(input, indices, kernel_size, stride, padding, output_size); 
}
inline TensorGrad adaptive_max_pool1d(TensorGrad x, int64_t l_out, bool return_indices = false){
    return TensorGrad_Functional_Class::adaptive_max_pool1d(x, l_out, return_indices); 
}

inline TensorGrad max_pool2d(TensorGrad input, 
                      utils::my_tuple kernel_size, utils::my_tuple stride = -1, utils::my_tuple padding = 0,
                      utils::my_tuple dilation=1, bool ceil_mode = false, bool return_indices = false){
    return TensorGrad_Functional_Class::max_pool2d(input, 
                      kernel_size, stride, padding,
                      dilation, ceil_mode, return_indices);
}
inline TensorGrad max_unpool2d(TensorGrad input, TensorGrad indices, utils::my_tuple kernel_size, utils::my_tuple stride = -1, utils::my_tuple padding = 0, utils::my_tuple output_size = -1){
    return TensorGrad_Functional_Class::max_unpool2d(input, indices, kernel_size, stride, padding, output_size); 
}

inline TensorGrad max_unpool2d(TensorGrad input, Tensor indices, utils::my_tuple kernel_size, utils::my_tuple stride = -1, utils::my_tuple padding = 0, utils::my_tuple output_size = -1){
    return TensorGrad_Functional_Class::max_unpool2d(input, indices, kernel_size, stride, padding, output_size); 
}
inline TensorGrad adaptive_max_pool2d(TensorGrad x, utils::my_tuple out_shape, bool return_indices = false){
        return TensorGrad_Functional_Class::adaptive_max_pool2d(x, out_shape, return_indices); 
    }

inline TensorGrad max_pool3d(TensorGrad input, 
                      utils::my_n_tuple<3> kernel_size, utils::my_n_tuple<3> stride = -1, utils::my_n_tuple<3> padding = 0,
                      utils::my_n_tuple<3> dilation=1, bool ceil_mode = false, bool return_indices = false){
    return TensorGrad_Functional_Class::max_pool3d(input, 
                  kernel_size, stride, padding,
                  dilation, ceil_mode, return_indices); 
}
inline TensorGrad max_unpool3d(TensorGrad input, TensorGrad indices, utils::my_n_tuple<3> kernel_size, utils::my_n_tuple<3> stride = -1, utils::my_n_tuple<3> padding = 0, utils::my_n_tuple<3> output_size = -1){
    return TensorGrad_Functional_Class::max_unpool3d(input, indices, kernel_size, stride, padding, output_size); 
}
inline TensorGrad max_unpool3d(TensorGrad input, Tensor indices, utils::my_n_tuple<3> kernel_size, utils::my_n_tuple<3> stride = -1, utils::my_n_tuple<3> padding = 0, utils::my_n_tuple<3> output_size = -1){
    return TensorGrad_Functional_Class::max_unpool3d(input, indices, kernel_size, stride, padding, output_size); 
}

inline TensorGrad adaptive_max_pool3d(TensorGrad x, utils::my_n_tuple<3> out_shape, bool return_indices = false){
    return TensorGrad_Functional_Class::adaptive_max_pool3d(x, out_shape, return_indices); 
}

//Fractional Max Pooling
inline TensorGrad fractional_max_pool2d(TensorGrad input, utils::my_tuple kernel_size, utils::my_tuple output_size = -1, 
                                 std::variant<double, std::tuple<double, double>> output_ratio = double(-1.0), bool return_indices = false){
    return TensorGrad_Functional_Class::fractional_max_pool2d(input, kernel_size, output_size, 
                         output_ratio, return_indices); 
}
inline TensorGrad fractional_max_pool3d(TensorGrad input, utils::my_n_tuple<3> kernel_size, utils::my_n_tuple<3> output_size = -1, 
                                 std::variant<double, std::tuple<double, double, double>> output_ratio = double(-1.0), bool return_indices = false){
    return TensorGrad_Functional_Class::fractional_max_pool3d(input, kernel_size, output_size, 
                         output_ratio, return_indices);
}


} // namespace functional
} // namespace nt

#include "functional/functional_list.h"

#endif // _NT_TENSORGAD_FUNCTIONAL_H_
