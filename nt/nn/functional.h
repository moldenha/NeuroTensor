// this is a friend class to TensorGrad that holds the functional functions
// dedicated to tensorgrad this is meant to re-create the functional.h that is
// dedicated to the Tensor class NEUROTENSOR_API

#ifndef NT_TENSORGAD_FUNCTIONAL_H__
#define NT_TENSORGAD_FUNCTIONAL_H__

#include "functional_class.h"
#include "../utils/collect_ri.hpp"
#include "../functional/tensor_files/normalize.h"
#include "../utils/optional_tensor_variant.h"
#include "../utils/type_traits.h"

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

inline TensorGrad& matmult(const TensorGrad &a, const TensorGrad &b, TensorGrad& out, bool transpose_a = false, bool transpose_b = false) {
    return TensorGrad_Functional_Class::matmult(a, b, out, transpose_a, transpose_b);
}

inline TensorGrad& matmult(const Tensor &a, const TensorGrad &b, TensorGrad& out, bool transpose_a = false, bool transpose_b = false) {
    return TensorGrad_Functional_Class::matmult(a, b, out, transpose_a, transpose_b);
}

inline TensorGrad& matmult(const TensorGrad &a, const Tensor &b, TensorGrad& out, bool transpose_a = false, bool transpose_b = false) {
    return TensorGrad_Functional_Class::matmult(a, b, out, transpose_a, transpose_b);
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

inline TensorGrad unfold2d(const TensorGrad &a, utils::my_tuple kernel_size,
                         utils::my_tuple dilation = 1,
                         utils::my_tuple padding = 0,
                         utils::my_tuple stride = 1,
                         bool transpose_out = true) {
    return TensorGrad_Functional_Class::unfold2d(a, kernel_size, dilation,
                                               padding, stride, transpose_out);
}

inline TensorGrad
unfold3d(const TensorGrad &a, utils::my_n_tuple<3> kernel_size,
         utils::my_n_tuple<3> dilation = 1, utils::my_n_tuple<3> padding = 0,
         utils::my_n_tuple<3> stride = 1, bool transpose_out = true) {
    return TensorGrad_Functional_Class::unfold3d(
        a, kernel_size, dilation, padding, stride, transpose_out);
}

inline TensorGrad unfoldnd(const TensorGrad &a, int64_t dim, utils::optional_list kernel_size,
                           utils::optional_list dilation = 1, utils::optional_list padding = 0,
                           utils::optional_list stride = 1, bool transpose_out = true, bool test_mode = false){
    return TensorGrad_Functional_Class::unfoldnd(a, dim, kernel_size, dilation, padding, stride, transpose_out, test_mode);
}

inline TensorGrad fold1d(const TensorGrad &a, Tensor::size_value_t output_size,
                          Tensor::size_value_t kernel_size, Tensor::size_value_t dilation=1,
                          Tensor::size_value_t padding=0, Tensor::size_value_t stride=1){
    
    return TensorGrad_Functional_Class::fold1d(a, output_size, kernel_size,
                                             dilation, padding, stride);
    
}


inline TensorGrad fold2d(const TensorGrad &a, utils::my_tuple output_size,
                       utils::my_tuple kernel_size,
                       utils::my_tuple dilation = 1,
                       utils::my_tuple padding = 0,
                       utils::my_tuple stride = 1) {
    return TensorGrad_Functional_Class::fold2d(a, output_size, kernel_size,
                                             dilation, padding, stride);
}

inline TensorGrad fold3d(const TensorGrad &a, utils::my_n_tuple<3> output_size, 
                         utils::my_n_tuple<3> kernel_size,
                         utils::my_n_tuple<3> dilation = 1,
                         utils::my_n_tuple<3> padding = 0, 
                         utils::my_n_tuple<3> stride = 1){
    return TensorGrad_Functional_Class::fold3d(a, output_size, kernel_size,
                                             dilation, padding, stride);
    
}
inline TensorGrad foldnd(const TensorGrad &a, int64_t dim, utils::optional_list output_size, 
                         utils::optional_list kernel_size,
                         utils::optional_list dilation = 1, 
                         utils::optional_list padding = 0, 
                         utils::optional_list stride = 1,
                         bool test_mode = false){
    return TensorGrad_Functional_Class::foldnd(a, dim, output_size, kernel_size,
                                             dilation, padding, stride, test_mode);

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

inline TensorGrad convnd(const TensorGrad &image, const TensorGrad &kernel, int64_t dim,
                         utils::optional_list stride = 1, utils::optional_list padding = 0,
                         utils::optional_list dilation = 1, int64_t groups = 1) {
    return TensorGrad_Functional_Class::convnd(image, kernel, dim, stride,
                                                   padding, dilation, groups);
}

inline TensorGrad convnd(const Tensor &image, const TensorGrad &kernel, int64_t dim,
                         utils::optional_list stride = 1, utils::optional_list padding = 0,
                         utils::optional_list dilation = 1, int64_t groups = 1) {
    return TensorGrad_Functional_Class::convnd(image, kernel, dim, stride,
                                                   padding, dilation, groups);
}

inline TensorGrad convnd(const TensorGrad &image, const Tensor &kernel, int64_t dim,
                         utils::optional_list stride = 1, utils::optional_list padding = 0,
                         utils::optional_list dilation = 1, int64_t groups = 1) {
    return TensorGrad_Functional_Class::convnd(image, kernel, dim, stride,
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

inline TensorGrad conv_transposend(const TensorGrad &image, const TensorGrad &kernel,
                                                   int64_t dim, utils::optional_list stride = 1, utils::optional_list padding = 0,
                                                   utils::optional_list output_padding = 0, utils::optional_list dilation = 1, int64_t groups = 1){
    return TensorGrad_Functional_Class::conv_transposend(image, kernel, dim, stride, padding, output_padding, dilation, groups);
}
inline TensorGrad conv_transposend(const Tensor &image, const TensorGrad &kernel,
                                                   int64_t dim, utils::optional_list stride = 1, utils::optional_list padding = 0,
                                                   utils::optional_list output_padding = 0, utils::optional_list dilation = 1, int64_t groups = 1){
    return TensorGrad_Functional_Class::conv_transposend(image, kernel, dim, stride, padding, output_padding, dilation, groups);
}
inline TensorGrad conv_transposend(const TensorGrad &image, const Tensor &kernel,
                                                   int64_t dim, utils::optional_list stride = 1, utils::optional_list padding = 0,
                                                   utils::optional_list output_padding = 0, utils::optional_list dilation = 1, int64_t groups = 1){
    return TensorGrad_Functional_Class::conv_transposend(image, kernel, dim, stride, padding, output_padding, dilation, groups);
}


inline TensorGrad clamp(const TensorGrad &a,
                        std::optional<Scalar> min = std::nullopt,
                        std::optional<Scalar> max = std::nullopt) {
    return TensorGrad_Functional_Class::clamp(a, min, max);
}

inline TensorGrad& clamp_(TensorGrad &a,
                        std::optional<Scalar> min = std::nullopt,
                        std::optional<Scalar> max = std::nullopt) {
    return TensorGrad_Functional_Class::clamp_(a, min, max);
}

inline TensorGrad var(const TensorGrad &a, utils::optional_list dim = nullptr,
                      int64_t correction = 1, bool keepdim = false) {
    return TensorGrad_Functional_Class::var(a, dim, correction, keepdim);
}

namespace details{

// this is a function that forces a tensograd optional
// by default the utils::optional_tensorgrad(var) will be none if var is holding only a tensor
// This will make it so that if it does not have a value only then it will be none
inline utils::optional_tensorgrad force_optional_tg(utils::optional_tensor_variant var){
    if(!var.has_value()) return utils::optional_tensorgrad(nullptr);
    if(var.tracking_grad()) return utils::optional_tensorgrad(var);
    return utils::optional_tensorgrad(TensorGrad(var.value<Tensor>(), false));
}

}

inline TensorGrad batch_norm(const TensorGrad & x, const Tensor & running_mean, const Tensor & running_var,
                                             utils::optional_tensor_variant weight = nullptr,
                                             utils::optional_tensor_variant bias = nullptr,
                                             bool training = false, Scalar momentum = 0.1, Scalar eps = 1e-05){
    return TensorGrad_Functional_Class::batch_norm(x, running_mean, running_var,
                                                   utils::details::force_optional_tg(weight),
                                                   utils::details::force_optional_tg(bias),
                                                   training, momentum, eps);
    
}


namespace details{


template<typename WT, typename BT>
struct norm_variant_output{
    using type = typename std::conditional_t<
                std::is_same_v<WT, TensorGrad> || std::is_same_v<BT, TensorGrad>,
                TensorGrad, Tensor>;
};

}

template<typename WT, typename BT>
inline typename details::norm_variant_output<WT, BT>::type batch_norm(const Tensor & x, 
                                             const Tensor & running_mean, const Tensor & running_var,
                                             WT weight = nullptr, BT bias = nullptr,
                                             bool training = false, Scalar momentum = 0.1, Scalar eps = 1e-05){
    static_assert(utils::details::valid_optional_tensor_variant_type<type_traits::remove_cvref_t<WT>>::value
                  && utils::details::valid_optional_tensor_variant_type<type_traits::remove_cvref_t<BT>>::value,
                "Needed weight and bias type to be tensor, tensorgrad, or null");
    using out_type = typename details::norm_variant_output<WT, BT>::type;
    if constexpr (std::is_same_v<out_type, Tensor>){
        return no_grad::batch_norm(x, running_mean, running_var,
                                   utils::optional_tensor(weight), utils::optional_tensor(bias),
                                   training, momentum, eps);
    }else{
        return TensorGrad_Functional_Class::batch_norm(x, running_mean, running_var,
                                                       utils::optional_tensorgrad(weight),
                                                       utils::optional_tensorgrad(bias),
                                                       training, momentum, eps);
    }
}

inline TensorGrad batch_norm(const TensorGrad & x, const TensorGrad & running_mean, const Tensor & running_var,
                                             utils::optional_tensor_variant weight = nullptr,
                                             utils::optional_tensor_variant bias = nullptr,
                                             bool training = false, Scalar momentum = 0.1, Scalar eps = 1e-05){
    utils::throw_exception(!running_mean.track_grad(),
                           "Error, running mean cannot require a gradient for batch norm");

    return TensorGrad_Functional_Class::batch_norm(x, running_mean.detach(), running_var,
                                                   utils::details::force_optional_tg(weight),
                                                   utils::details::force_optional_tg(bias),
                                                   training, momentum, eps);
    
}



inline TensorGrad batch_norm(const TensorGrad & x, const TensorGrad & running_mean, const TensorGrad & running_var,
                                             utils::optional_tensor_variant weight = nullptr,
                                             utils::optional_tensor_variant bias = nullptr,
                                             bool training = false, Scalar momentum = 0.1, Scalar eps = 1e-05){
    utils::throw_exception(!running_mean.track_grad() && !running_var.track_grad(),
                           "Error, running mean and running var cannot require a gradient for batch norm");
    return TensorGrad_Functional_Class::batch_norm(x, running_mean.detach(), running_var.detach(),
                                                   utils::details::force_optional_tg(weight),
                                                   utils::details::force_optional_tg(bias),
                                                   training, momentum, eps);
    
}

inline TensorGrad batch_norm(const TensorGrad & x, const Tensor & running_mean, const TensorGrad & running_var,
                                             utils::optional_tensor_variant weight = nullptr,
                                             utils::optional_tensor_variant bias = nullptr,
                                             bool training = false, Scalar momentum = 0.1, Scalar eps = 1e-05){
    utils::throw_exception(!running_var.track_grad(),
                           "Error, running var cannot require a gradient for batch norm");

    return TensorGrad_Functional_Class::batch_norm(x, running_mean, running_var.detach(),
                                                   utils::details::force_optional_tg(weight),
                                                   utils::details::force_optional_tg(bias),
                                                   training, momentum, eps);
    
}

template<typename WT, typename BT>
inline typename details::norm_variant_output<WT, BT>::type batch_norm(const Tensor & x, 
                                            const TensorGrad & running_mean, const Tensor & running_var,
                                             WT weight = nullptr,
                                             BT bias = nullptr,
                                             bool training = false, Scalar momentum = 0.1, Scalar eps = 1e-05){
    utils::throw_exception(!running_mean.track_grad(),
                           "Error, running mean cannot require a gradient for batch norm");
    static_assert(utils::details::valid_optional_tensor_variant_type<type_traits::remove_cvref_t<WT>>::value
                  && utils::details::valid_optional_tensor_variant_type<type_traits::remove_cvref_t<BT>>::value,
                "Needed weight and bias type to be tensor, tensorgrad, or null");
    using out_type = typename details::norm_variant_output<WT, BT>::type;
    if constexpr (std::is_same_v<out_type, Tensor>){
        return no_grad::batch_norm(x, running_mean.detach(), running_var,
                                   utils::optional_tensor(weight), utils::optional_tensor(bias),
                                   training, momentum, eps);
    }else{
        return TensorGrad_Functional_Class::batch_norm(x, running_mean.detach(), running_var,
                                                       utils::optional_tensorgrad(weight),
                                                       utils::optional_tensorgrad(bias),
                                                       training, momentum, eps);
    }
}


template<typename WT, typename BT>
inline typename details::norm_variant_output<WT, BT>::type batch_norm(const Tensor & x, 
                                             const TensorGrad & running_mean, const TensorGrad & running_var,
                                             WT weight = nullptr,
                                             BT bias = nullptr,
                                             bool training = false, Scalar momentum = 0.1, Scalar eps = 1e-05){
    utils::throw_exception(!running_mean.track_grad() && !running_var.track_grad(),
                           "Error, running mean and running var cannot require a gradient for batch norm");
    static_assert(utils::details::valid_optional_tensor_variant_type<type_traits::remove_cvref_t<WT>>::value
                  && utils::details::valid_optional_tensor_variant_type<type_traits::remove_cvref_t<BT>>::value,
                "Needed weight and bias type to be tensor, tensorgrad, or null");
    using out_type = typename details::norm_variant_output<WT, BT>::type;
    if constexpr (std::is_same_v<out_type, Tensor>){
        return no_grad::batch_norm(x, running_mean.detach(), running_var.detach(),
                                   utils::optional_tensor(weight), utils::optional_tensor(bias),
                                   training, momentum, eps);
    }else{
        return TensorGrad_Functional_Class::batch_norm(x, running_mean.detach(), running_var.detach(),
                                                       utils::optional_tensorgrad(weight),
                                                       utils::optional_tensorgrad(bias),
                                                       training, momentum, eps);
    }

}




template<typename WT, typename BT>
inline typename details::norm_variant_output<WT, BT>::type batch_norm(const Tensor & x, 
                                             const Tensor & running_mean, const TensorGrad & running_var,
                                             WT weight = nullptr,
                                             BT bias = nullptr,
                                             bool training = false, Scalar momentum = 0.1, Scalar eps = 1e-05){
    utils::throw_exception(!running_var.track_grad(),
                           "Error, running var cannot require a gradient for batch norm");
    static_assert(utils::details::valid_optional_tensor_variant_type<type_traits::remove_cvref_t<WT>>::value
                  && utils::details::valid_optional_tensor_variant_type<type_traits::remove_cvref_t<BT>>::value,
                "Needed weight and bias type to be tensor, tensorgrad, or null");
    using out_type = typename details::norm_variant_output<WT, BT>::type;
    if constexpr (std::is_same_v<out_type, Tensor>){
        return no_grad::batch_norm(x, running_mean, running_var.detach(),
                                   utils::optional_tensor(weight), utils::optional_tensor(bias),
                                   training, momentum, eps);
    }else{
        return TensorGrad_Functional_Class::batch_norm(x, running_mean, running_var.detach(),
                                                       utils::optional_tensorgrad(weight),
                                                       utils::optional_tensorgrad(bias),
                                                       training, momentum, eps);
    }
}



inline TensorGrad group_norm(const TensorGrad & input, int64_t num_groups,
                                             utils::optional_tensor_variant weight = nullptr,
                                             utils::optional_tensor_variant bias = nullptr,
                                             Scalar eps = 1e-05){
    return TensorGrad_Functional_Class::group_norm(input, num_groups,
                                                   utils::details::force_optional_tg(weight),
                                                   utils::details::force_optional_tg(bias),
                                                   eps);
    
}

template<typename WT, typename BT>
inline typename details::norm_variant_output<WT, BT>::type group_norm(const Tensor & input, 
                                             int64_t num_groups,
                                             WT weight = nullptr, BT bias = nullptr,
                                             Scalar eps = 1e-05){
    static_assert(utils::details::valid_optional_tensor_variant_type<type_traits::remove_cvref_t<WT>>::value
                  && utils::details::valid_optional_tensor_variant_type<type_traits::remove_cvref_t<BT>>::value,
                "Needed weight and bias type to be tensor, tensorgrad, or null");
    using out_type = typename details::norm_variant_output<WT, BT>::type;
    if constexpr (std::is_same_v<out_type, Tensor>){
        return no_grad::group_norm(input, num_groups,
                                   utils::optional_tensor(weight), utils::optional_tensor(bias),
                                   eps);
    }else{
        return TensorGrad_Functional_Class::group_norm(input, num_groups,
                                                       utils::optional_tensorgrad(weight),
                                                       utils::optional_tensorgrad(bias),
                                                       eps);
    }
}


//activation_functions.cpp

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

inline TensorGrad pow(const TensorGrad& x, Scalar exponent){
    return TensorGrad_Functional_Class::pow(x, exponent);
}

inline TensorGrad abs(const TensorGrad &input){
    return TensorGrad_Functional_Class::abs(input);
}

inline TensorGrad softplus(const TensorGrad &input, Scalar beta=1.0, Scalar threshold=20.0){
    return TensorGrad_Functional_Class::softplus(input, beta, threshold);
}

inline TensorGrad sigmoid(const TensorGrad &a) {
    return TensorGrad_Functional_Class::sigmoid(a);
}
inline TensorGrad relu(const TensorGrad &a) {
    return TensorGrad_Functional_Class::relu(a);
}


inline TensorGrad& sqrt_(TensorGrad &a) {
    return TensorGrad_Functional_Class::sqrt_(a);
}

inline TensorGrad& invsqrt_(TensorGrad &a) {
    return TensorGrad_Functional_Class::invsqrt_(a);
}

inline TensorGrad& silu_(TensorGrad &a) {
    return TensorGrad_Functional_Class::silu_(a);
}
inline TensorGrad& gelu_(TensorGrad &a) {
    return TensorGrad_Functional_Class::gelu_(a);
}

inline TensorGrad& pow_(TensorGrad& x, Scalar exponent){
    return TensorGrad_Functional_Class::pow_(x, exponent);
}

inline TensorGrad& abs_(TensorGrad &input){
    return TensorGrad_Functional_Class::abs_(input);
}

inline TensorGrad& softplus_(TensorGrad &input, Scalar beta=1.0, Scalar threshold=20.0){
    return TensorGrad_Functional_Class::softplus_(input, beta, threshold);
}

inline TensorGrad& sigmoid_(TensorGrad &a) {
    return TensorGrad_Functional_Class::sigmoid_(a);
}

inline TensorGrad& relu_(TensorGrad &a) {
    return TensorGrad_Functional_Class::relu_(a);
}

inline TensorGrad cat(std::vector<TensorGrad> tgs, int64_t dim = 0) {
    return TensorGrad_Functional_Class::cat(std::move(tgs), dim);
}

inline TensorGrad cat(TensorGrad tg, int64_t dim = 0) {
    return TensorGrad_Functional_Class::cat(std::move(tg), dim);
}



inline TensorGrad stack(TensorGrad input, int64_t dim){
    return TensorGrad_Functional_Class::stack(std::move(input), dim);
}

inline TensorGrad stack(std::vector<TensorGrad> input, int64_t dim){
    return TensorGrad_Functional_Class::stack(std::move(input), dim);
}

inline TensorGrad real(const TensorGrad& tg){return TensorGrad_Functional_Class::real(tg);}
inline TensorGrad imag(const TensorGrad& tg){return TensorGrad_Functional_Class::imag(tg);}
inline TensorGrad to_complex_from_real(const TensorGrad& tg){return TensorGrad_Functional_Class::to_complex_from_real(tg);} 
inline TensorGrad to_complex_from_imag(const TensorGrad& tg){return TensorGrad_Functional_Class::to_complex_from_imag(tg);}
inline TensorGrad to(const TensorGrad& tg, DType dt){return TensorGrad_Functional_Class::to(tg, dt);}

// inline TensorGrad dilate(const TensorGrad& tg, Tensor::size_value_t a){
//     return TensorGrad_Functional_Class::dilate(tg, a);
// }
// inline TensorGrad dilate(const TensorGrad& tg, Tensor::size_value_t a, Tensor::size_value_t b){
//     return TensorGrad_Functional_Class::dilate(tg, a, b);
// }
// inline TensorGrad dilate(const TensorGrad& tg, Tensor::size_value_t a, Tensor::size_value_t b, Tensor::size_value_t c){
//     return TensorGrad_Functional_Class::dilate(tg, a, b, c);
// }
inline TensorGrad dilate(const TensorGrad& tg, std::vector<Tensor::size_value_t> dil, bool test = false){
    return TensorGrad_Functional_Class::dilate(tg, std::move(dil), test);
}
template<typename... Args>
inline TensorGrad dilate(const TensorGrad& t, Tensor::size_value_t i, Args&&... args){
    std::vector<Tensor::size_value_t> vec;
    vec.reserve(sizeof...(Args) + 1);
    utils::collect_integers_impl(vec, i, std::forward<Args>(args)...);
    return dilate(t, std::move(vec));
}

// inline TensorGrad undilate(const TensorGrad& tg, Tensor::size_value_t a){
//     return TensorGrad_Functional_Class::undilate(tg, a);
// }
// inline TensorGrad undilate(const TensorGrad& tg, Tensor::size_value_t a, Tensor::size_value_t b){
//     return TensorGrad_Functional_Class::undilate(tg, a, b);
// }
// inline TensorGrad undilate(const TensorGrad& tg, Tensor::size_value_t a, Tensor::size_value_t b, Tensor::size_value_t c){
//     return TensorGrad_Functional_Class::undilate(tg, a, b, c);
// }
inline TensorGrad undilate(const TensorGrad& tg, std::vector<Tensor::size_value_t> dil){
    return TensorGrad_Functional_Class::undilate(tg, std::move(dil));
}
template<typename... Args>
inline TensorGrad undilate(const TensorGrad& t, Tensor::size_value_t i, Args&&... args){
    std::vector<Tensor::size_value_t> vec;
    vec.reserve(sizeof...(Args) + 1);
    utils::collect_integers_impl(vec, i, std::forward<Args>(args)...);
    return undilate(t, std::move(vec));
}


// inline TensorGrad undilate_(const TensorGrad& tg, Tensor::size_value_t a){
//     return TensorGrad_Functional_Class::undilate_(tg, a);
// }
// inline TensorGrad undilate_(const TensorGrad& tg, Tensor::size_value_t a, Tensor::size_value_t b){
//     return TensorGrad_Functional_Class::undilate_(tg, a, b);
// }
// inline TensorGrad undilate_(const TensorGrad& tg, Tensor::size_value_t a, Tensor::size_value_t b, Tensor::size_value_t c){
//     return TensorGrad_Functional_Class::undilate_(tg, a, b, c);
// }
inline TensorGrad undilate_(const TensorGrad& tg, std::vector<Tensor::size_value_t> dil, bool test = false){
    return TensorGrad_Functional_Class::undilate_(tg, std::move(dil), test);
}
template<typename... Args>
inline TensorGrad undilate_(const TensorGrad& t, Tensor::size_value_t i, Args&&... args){
    std::vector<Tensor::size_value_t> vec;
    vec.reserve(sizeof...(Args) + 1);
    utils::collect_integers_impl(vec, i, std::forward<Args>(args)...);
    return undilate_(t, std::move(vec));
}

inline TensorGrad zeros_like(const TensorGrad& tg){ return TensorGrad_Functional_Class::zeros_like(tg); }
inline TensorGrad ones_like(const TensorGrad& tg){ return TensorGrad_Functional_Class::ones_like(tg); }
inline TensorGrad nums_like(const TensorGrad& tg, Scalar s){ return TensorGrad_Functional_Class::nums_like(tg, s); }
inline TensorGrad& fill_diagonal_(TensorGrad& tg, Scalar s){ return TensorGrad_Functional_Class::fill_diagonal_(tg, s); }
inline TensorGrad& fill_(TensorGrad& tg, Scalar s){ return TensorGrad_Functional_Class::fill_(tg, s); }
inline TensorGrad& set_(TensorGrad& tg, const Tensor& t){ return TensorGrad_Functional_Class::set_(tg, t); }
inline TensorGrad& set_(TensorGrad& tg, const TensorGrad& t){ return TensorGrad_Functional_Class::set_(tg, t); }

inline TensorGrad dropout(const TensorGrad &input, double p) {
    return TensorGrad_Functional_Class::dropout(input, p);
}

inline TensorGrad dropout2d(const TensorGrad &input, double p) {
    return TensorGrad_Functional_Class::dropout(input, p);
}

inline TensorGrad dropout3d(const TensorGrad &input, double p) {
    return TensorGrad_Functional_Class::dropout(input, p);
}

//by default is stable
inline TensorGrad softmax(const TensorGrad& input, std::optional<int64_t> dim = std::nullopt, bool stable = true){
    if(dim.has_value()){
        return TensorGrad_Functional_Class::softmax(input, dim.value(), stable);
    }
    return TensorGrad_Functional_Class::softmax(input, stable);
}


inline TensorGrad softmax_unstable(const TensorGrad& input){
    return TensorGrad_Functional_Class::softmax(input, false);
}

inline TensorGrad softmax_unstable(const TensorGrad& input, typename SizeRef::value_type dim){
    return TensorGrad_Functional_Class::softmax(input, dim, false);
}

inline TensorGrad gumbel_softmax(const TensorGrad& input, Scalar tau, bool hard, int64_t dim = -1, bool stable = true){
    return TensorGrad_Functional_Class::gumbel_softmax(input, tau, hard, dim, stable);
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
                                 utils::tuple_or_var<double, 2> output_ratio = double(-1.0), bool return_indices = false){
    return TensorGrad_Functional_Class::fractional_max_pool2d(input, kernel_size, output_size, 
                         output_ratio, return_indices); 
}
inline TensorGrad fractional_max_pool3d(TensorGrad input, utils::my_n_tuple<3> kernel_size, utils::my_n_tuple<3> output_size = -1, 
                                 utils::tuple_or_var<double, 3> output_ratio = double(-1.0), bool return_indices = false){
    return TensorGrad_Functional_Class::fractional_max_pool3d(input, kernel_size, output_size, 
                         output_ratio, return_indices);
}


//flip.cpp
inline TensorGrad flip(const TensorGrad& input, utils::optional_list list = nullptr){return TensorGrad_Functional_Class::flip(input, list);}
inline TensorGrad flip_view(const TensorGrad& input, utils::optional_list list){return TensorGrad_Functional_Class::flip_view(input, list);}

// Fused.cpp
//returns c + (a * b);
inline TensorGrad fused_multiply_add(const TensorGrad& c, const TensorGrad& a, const TensorGrad& b){return TensorGrad_Functional_Class::fused_multiply_add(c, a, b);} 
inline TensorGrad fused_multiply_add(const Tensor& c, const TensorGrad& a, const TensorGrad& b){return TensorGrad_Functional_Class::fused_multiply_add(c, a, b);} 
inline TensorGrad fused_multiply_add(const Tensor& c, const Tensor& a, const TensorGrad& b){return TensorGrad_Functional_Class::fused_multiply_add(c, a, b);} 
inline TensorGrad fused_multiply_add(const Tensor& c, const TensorGrad& a, const Tensor& b){return TensorGrad_Functional_Class::fused_multiply_add(c, a, b);} 
inline TensorGrad fused_multiply_add(const TensorGrad& c, const Tensor& a, const TensorGrad& b){return TensorGrad_Functional_Class::fused_multiply_add(c, a, b);} 
inline TensorGrad fused_multiply_add(const TensorGrad& c, const Tensor& a, const Tensor& b){return TensorGrad_Functional_Class::fused_multiply_add(c, a, b);} 
inline TensorGrad fused_multiply_add(const TensorGrad& c, const TensorGrad& a, const Tensor& b){return TensorGrad_Functional_Class::fused_multiply_add(c, a, b);} 

inline TensorGrad fused_multiply_add(const TensorGrad& c, const TensorGrad& a, Scalar b){return TensorGrad_Functional_Class::fused_multiply_add(c, a, b);} 
inline TensorGrad fused_multiply_add(const TensorGrad& c, const Tensor& a, Scalar b){return TensorGrad_Functional_Class::fused_multiply_add(c, a, b);} 
inline TensorGrad fused_multiply_add(const Tensor& c, const TensorGrad& a, Scalar b){return TensorGrad_Functional_Class::fused_multiply_add(c, a, b);} 
//returns c += (a * b);
inline TensorGrad& fused_multiply_add_(TensorGrad& c, const TensorGrad& a, const TensorGrad& b){return TensorGrad_Functional_Class::fused_multiply_add_(c, a, b);} 
inline TensorGrad& fused_multiply_add_(TensorGrad& c, const Tensor& a, const TensorGrad& b){return TensorGrad_Functional_Class::fused_multiply_add_(c, a, b);} 
inline TensorGrad& fused_multiply_add_(TensorGrad& c, const TensorGrad& a, const Tensor& b){return TensorGrad_Functional_Class::fused_multiply_add_(c, a, b);} 
inline TensorGrad& fused_multiply_add_(TensorGrad& c, const Tensor& a, const Tensor& b){return TensorGrad_Functional_Class::fused_multiply_add_(c, a, b);} 

inline TensorGrad& fused_multiply_add_(TensorGrad& c, const TensorGrad& a, Scalar b){return TensorGrad_Functional_Class::fused_multiply_add_(c, a, b);} 
inline TensorGrad& fused_multiply_add_(TensorGrad& c, const Tensor& a, Scalar b){return TensorGrad_Functional_Class::fused_multiply_add_(c, a, b);} 

//returns c - (a * b);
inline TensorGrad fused_multiply_subtract(const TensorGrad& c, const TensorGrad& a, const TensorGrad& b){return TensorGrad_Functional_Class::fused_multiply_subtract(c, a, b);} 
inline TensorGrad fused_multiply_subtract(const Tensor& c, const TensorGrad& a, const TensorGrad& b){return TensorGrad_Functional_Class::fused_multiply_subtract(c, a, b);} 
inline TensorGrad fused_multiply_subtract(const Tensor& c, const Tensor& a, const TensorGrad& b){return TensorGrad_Functional_Class::fused_multiply_subtract(c, a, b);} 
inline TensorGrad fused_multiply_subtract(const Tensor& c, const TensorGrad& a, const Tensor& b){return TensorGrad_Functional_Class::fused_multiply_subtract(c, a, b);} 
inline TensorGrad fused_multiply_subtract(const TensorGrad& c, const Tensor& a, const TensorGrad& b){return TensorGrad_Functional_Class::fused_multiply_subtract(c, a, b);} 
inline TensorGrad fused_multiply_subtract(const TensorGrad& c, const Tensor& a, const Tensor& b){return TensorGrad_Functional_Class::fused_multiply_subtract(c, a, b);} 
inline TensorGrad fused_multiply_subtract(const TensorGrad& c, const TensorGrad& a, const Tensor& b){return TensorGrad_Functional_Class::fused_multiply_subtract(c, a, b);} 

inline TensorGrad fused_multiply_subtract(const TensorGrad& c, const TensorGrad& a, Scalar b){return TensorGrad_Functional_Class::fused_multiply_subtract(c, a, b);} 
inline TensorGrad fused_multiply_subtract(const TensorGrad& c, const Tensor& a, Scalar b){return TensorGrad_Functional_Class::fused_multiply_subtract(c, a, b);} 
inline TensorGrad fused_multiply_subtract(const Tensor& c, const TensorGrad& a, Scalar b){return TensorGrad_Functional_Class::fused_multiply_subtract(c, a, b);} 

//returns c -= (a * b);
inline TensorGrad& fused_multiply_subtract_(TensorGrad& c, const TensorGrad& a, const TensorGrad& b){return TensorGrad_Functional_Class::fused_multiply_subtract_(c, a, b);} 
inline TensorGrad& fused_multiply_subtract_(TensorGrad& c, const Tensor& a, const TensorGrad& b){return TensorGrad_Functional_Class::fused_multiply_subtract_(c, a, b);} 
inline TensorGrad& fused_multiply_subtract_(TensorGrad& c, const Tensor& a, const Tensor& b){return TensorGrad_Functional_Class::fused_multiply_subtract_(c, a, b);} 
inline TensorGrad& fused_multiply_subtract_(TensorGrad& c, const TensorGrad& a, const Tensor& b){return TensorGrad_Functional_Class::fused_multiply_subtract_(c, a, b);} 

inline TensorGrad& fused_multiply_subtract_(TensorGrad& c, const TensorGrad& a, Scalar b){return TensorGrad_Functional_Class::fused_multiply_subtract_(c, a, b);} 
inline TensorGrad& fused_multiply_subtract_(TensorGrad& c, const Tensor& a, Scalar b){return TensorGrad_Functional_Class::fused_multiply_subtract_(c, a, b);} 

//Index.cpp
inline TensorGrad at(const TensorGrad& input, Tensor::size_value_t index){return TensorGrad_Functional_Class::at(input, index);}
inline TensorGrad at(const TensorGrad& input, const Tensor& index){return TensorGrad_Functional_Class::at(input, index);}
inline TensorGrad at(const TensorGrad& input, const TensorGrad& index){return TensorGrad_Functional_Class::at(input, index);}
inline Tensor at(const Tensor& input, const TensorGrad& index){return TensorGrad_Functional_Class::at(input, index);}
inline TensorGrad at(const TensorGrad& input, std::vector<Tensor::size_value_t> index){return TensorGrad_Functional_Class::at(input, std::move(index));}
inline TensorGrad at_tensor_split(const TensorGrad & input, const TensorGrad & index, Tensor::size_value_t splitting){return TensorGrad_Functional_Class::at_tensor_split(input, index, splitting);} 
inline TensorGrad at_tensor_split(const TensorGrad & input, const Tensor & index, Tensor::size_value_t splitting){return TensorGrad_Functional_Class::at_tensor_split(input, index, splitting);} 
inline TensorGrad &at_tensor_split(const TensorGrad & input, const TensorGrad & index, Tensor::size_value_t splitting,
                    TensorGrad & output){return TensorGrad_Functional_Class::at_tensor_split(input, index, splitting, output);} 
inline TensorGrad &at_tensor_split(const TensorGrad & input, const Tensor & index, Tensor::size_value_t splitting,
                    TensorGrad & output){return TensorGrad_Functional_Class::at_tensor_split(input, index, splitting, output);} 
inline TensorGrad index_except(const TensorGrad & input, int64_t dim, Tensor::size_value_t index){return TensorGrad_Functional_Class::index_except(input, dim, index);} 
inline TensorGrad index_select(const TensorGrad & input, int64_t dim, const Tensor& index){return TensorGrad_Functional_Class::index_select(input, dim, index);} 
inline TensorGrad index_select(const TensorGrad & input, int64_t dim, const TensorGrad& index){return TensorGrad_Functional_Class::index_select(input, dim, index);} 
inline TensorGrad select(const TensorGrad& input, Tensor::size_value_t dim, Tensor::size_value_t index){return TensorGrad_Functional_Class::select(input, dim, index);} 

//min_max.cpp
inline result_types::max<TensorGrad, Tensor> max(const TensorGrad& input, utils::optional_list dim = nullptr, bool keepdim = false){
    return TensorGrad_Functional_Class::max(input, dim, keepdim);
}
inline result_types::max<TensorGrad, Tensor> min(const TensorGrad& input, utils::optional_list dim = nullptr, bool keepdim = false){
    return TensorGrad_Functional_Class::min(input, dim, keepdim);
}
inline TensorGrad maximum(const std::vector<TensorGrad>& tgs, const std::vector<Tensor>& ts = {}, const std::vector<Scalar>& scalars = {}){
    return TensorGrad_Functional_Class::maximum(tgs, ts, scalars);
}
inline TensorGrad minimum(const std::vector<TensorGrad>& tgs, const std::vector<Tensor>& ts = {}, const std::vector<Scalar>& scalars = {}){
    return TensorGrad_Functional_Class::minimum(tgs, ts, scalars);
    
}


//operators.cpp


inline TensorGrad add(const TensorGrad& a, const TensorGrad& b){return TensorGrad_Functional_Class::add(a, b);}\
inline TensorGrad add(const TensorGrad& a, const Tensor& b){return TensorGrad_Functional_Class::add(a, b);}\
inline TensorGrad add(const Tensor& a, const TensorGrad& b){return TensorGrad_Functional_Class::add(a, b);}\
inline TensorGrad add(const TensorGrad& a, const Scalar& b){return TensorGrad_Functional_Class::add(a, b);}\
inline TensorGrad add(const Scalar& a, const TensorGrad& b){return TensorGrad_Functional_Class::add(a, b);}\
inline TensorGrad& add_(TensorGrad& a, const TensorGrad& b){return TensorGrad_Functional_Class::add_(a, b);}\
inline TensorGrad& add_(TensorGrad& a, const Tensor& b){return TensorGrad_Functional_Class::add_(a, b);}\
inline Tensor& add_(Tensor& a, const TensorGrad& b){return TensorGrad_Functional_Class::add_(a, b);}\
inline TensorGrad& add_(TensorGrad& a, const Scalar& b){return TensorGrad_Functional_Class::add_(a, b);}\


inline TensorGrad multiply(const TensorGrad& a, const TensorGrad& b){return TensorGrad_Functional_Class::multiply(a, b);}\
inline TensorGrad multiply(const TensorGrad& a, const Tensor& b){return TensorGrad_Functional_Class::multiply(a, b);}\
inline TensorGrad multiply(const Tensor& a, const TensorGrad& b){return TensorGrad_Functional_Class::multiply(a, b);}\
inline TensorGrad multiply(const TensorGrad& a, const Scalar& b){return TensorGrad_Functional_Class::multiply(a, b);}\
inline TensorGrad multiply(const Scalar& a, const TensorGrad& b){return TensorGrad_Functional_Class::multiply(a, b);}\
inline TensorGrad& multiply_(TensorGrad& a, const TensorGrad& b){return TensorGrad_Functional_Class::multiply_(a, b);}\
inline TensorGrad& multiply_(TensorGrad& a, const Tensor& b){return TensorGrad_Functional_Class::multiply_(a, b);}\
inline Tensor& multiply_(Tensor& a, const TensorGrad& b){return TensorGrad_Functional_Class::multiply_(a, b);}\
inline TensorGrad& multiply_(TensorGrad& a, const Scalar& b){return TensorGrad_Functional_Class::multiply_(a, b);}\

inline TensorGrad subtract(const TensorGrad& a, const TensorGrad& b){return TensorGrad_Functional_Class::subtract(a, b);}\
inline TensorGrad subtract(const TensorGrad& a, const Tensor& b){return TensorGrad_Functional_Class::subtract(a, b);}\
inline TensorGrad subtract(const Tensor& a, const TensorGrad& b){return TensorGrad_Functional_Class::subtract(a, b);}\
inline TensorGrad subtract(const TensorGrad& a, const Scalar& b){return TensorGrad_Functional_Class::subtract(a, b);}\
inline TensorGrad subtract(const Scalar& a, const TensorGrad& b){return TensorGrad_Functional_Class::subtract(a, b);}\
inline TensorGrad& subtract_(TensorGrad& a, const TensorGrad& b){return TensorGrad_Functional_Class::subtract_(a, b);}\
inline TensorGrad& subtract_(TensorGrad& a, const Tensor& b){return TensorGrad_Functional_Class::subtract_(a, b);}\
inline Tensor& subtract_(Tensor& a, const TensorGrad& b){return TensorGrad_Functional_Class::subtract_(a, b);}\
inline TensorGrad& subtract_(TensorGrad& a, const Scalar& b){return TensorGrad_Functional_Class::subtract_(a, b);}\

inline TensorGrad divide(const TensorGrad& a, const TensorGrad& b){return TensorGrad_Functional_Class::divide(a, b);}\
inline TensorGrad divide(const TensorGrad& a, const Tensor& b){return TensorGrad_Functional_Class::divide(a, b);}\
inline TensorGrad divide(const Tensor& a, const TensorGrad& b){return TensorGrad_Functional_Class::divide(a, b);}\
inline TensorGrad divide(const TensorGrad& a, const Scalar& b){return TensorGrad_Functional_Class::divide(a, b);}\
inline TensorGrad divide(const Scalar& a, const TensorGrad& b){return TensorGrad_Functional_Class::divide(a, b);}\
inline TensorGrad& divide_(TensorGrad& a, const TensorGrad& b){return TensorGrad_Functional_Class::divide_(a, b);}\
inline TensorGrad& divide_(TensorGrad& a, const Tensor& b){return TensorGrad_Functional_Class::divide_(a, b);}\
inline Tensor& divide_(Tensor& a, const TensorGrad& b){return TensorGrad_Functional_Class::divide_(a, b);}\
inline TensorGrad& divide_(TensorGrad& a, const Scalar& b){return TensorGrad_Functional_Class::divide_(a, b);}\


inline TensorGrad fmod(const TensorGrad& input, const TensorGrad& other){return TensorGrad_Functional_Class::fmod(input, other);}
inline TensorGrad fmod(const TensorGrad& input, const Tensor& other){return TensorGrad_Functional_Class::fmod(input, other);}
inline TensorGrad fmod(const Tensor& input, const TensorGrad& other){return TensorGrad_Functional_Class::fmod(input, other);}
inline TensorGrad fmod(const TensorGrad& input, const Scalar& other){return TensorGrad_Functional_Class::fmod(input, other);}
inline TensorGrad fmod(const Scalar& input, const TensorGrad& other){return TensorGrad_Functional_Class::fmod(input, other);}


inline TensorGrad remainder(const TensorGrad& input, const TensorGrad& other){return TensorGrad_Functional_Class::remainder(input, other);}
inline TensorGrad remainder(const TensorGrad& input, const Tensor& other){return TensorGrad_Functional_Class::remainder(input, other);}
inline TensorGrad remainder(const Tensor& input, const TensorGrad& other){return TensorGrad_Functional_Class::remainder(input, other);}
inline TensorGrad remainder(const TensorGrad& input, const Scalar& other){return TensorGrad_Functional_Class::remainder(input, other);}
inline TensorGrad remainder(const Scalar& input, const TensorGrad& other){return TensorGrad_Functional_Class::remainder(input, other);}

inline TensorGrad inverse(const TensorGrad& input){return TensorGrad_Functional_Class::inverse(input);}
inline TensorGrad& inverse_(TensorGrad& input){return TensorGrad_Functional_Class::inverse_(input);}


//padding.cpp
inline TensorGrad pad(const TensorGrad& input, std::vector<Tensor::size_value_t> padding, const char* mode = "constant", Scalar value = 0){
    return TensorGrad_Functional_Class::pad(input, std::move(padding), mode, value);
}
inline TensorGrad unpad(const TensorGrad& input, std::vector<Tensor::size_value_t> padding, bool no_contiguous = false){
    return TensorGrad_Functional_Class::unpad(input, std::move(padding), no_contiguous);
}



//repeat.cpp
inline TensorGrad repeat_(const TensorGrad& input, Tensor::size_value_t dim, Tensor::size_value_t amt){
    return TensorGrad_Functional_Class::repeat_(input, dim, amt);
}
inline TensorGrad repeat_(const TensorGrad& input, Tensor::size_value_t amt){
    return TensorGrad_Functional_Class::repeat_(input, amt);
}
inline TensorGrad expand(const TensorGrad& input, SizeRef size){
    return TensorGrad_Functional_Class::expand(input, size);
}
inline TensorGrad expand_as(const TensorGrad& a, const TensorGrad& b){
    return TensorGrad_Functional_Class::expand_as(a, b);
}
inline TensorGrad expand_as(const TensorGrad& a, const Tensor& b){
    return TensorGrad_Functional_Class::expand_as(a, b);
}
inline Tensor expand_as(const Tensor& a, const TensorGrad& b){
    return TensorGrad_Functional_Class::expand_as(a, b);
}

inline TensorGrad round(const TensorGrad& input){
    return TensorGrad_Functional_Class::round(input); 
}
inline TensorGrad trunc(const TensorGrad& input){
    return TensorGrad_Functional_Class::trunc(input); 
}
inline TensorGrad floor(const TensorGrad& input){
    return TensorGrad_Functional_Class::floor(input); 
}
inline TensorGrad ceil(const TensorGrad& input){
    return TensorGrad_Functional_Class::ceil(input); 
}



//sort.cpp
inline TensorGrad sort(const TensorGrad& input, const Tensor::size_value_t dim = -1,
        bool descending = false, bool return_sorted = true,
        bool return_indices = true){
    return TensorGrad_Functional_Class::sort(input, dim, descending, return_sorted, return_indices);
}

inline TensorGrad coordsort(const TensorGrad& input, const Tensor::size_value_t dim = -2, bool descending = false, 
                                                bool return_sorted = true, bool return_indices = true){
    return TensorGrad_Functional_Class::coordsort(input, dim, descending, return_sorted, return_indices);

}

inline TensorGrad split(const TensorGrad& input, int64_t dim, utils::optional_list splitting = nullptr){
    return TensorGrad_Functional_Class::split(input, dim, splitting); 
}
inline TensorGrad chunk(const TensorGrad& input, const Tensor::size_value_t chunks, int64_t dim = 0){
    return TensorGrad_Functional_Class::chunk(input, chunks, dim); 
}

inline TensorGrad diagonal(const TensorGrad& input, bool keep_dims = false){
    return TensorGrad_Functional_Class::diagonal(input, keep_dims);
}
inline TensorGrad as_strided(const TensorGrad &input, const SizeRef n_size, SizeRef n_stride,
                  const int64_t storage_offset = 0, bool whole_tensor = false){
    return TensorGrad_Functional_Class::as_strided(input, n_size, n_stride, storage_offset, whole_tensor);
}
   
inline TensorGrad log(const TensorGrad &input){
    return TensorGrad_Functional_Class::log(input); 
}
inline TensorGrad& log_(TensorGrad& input){
    return TensorGrad_Functional_Class::log_(input); 
}
inline TensorGrad exp(const TensorGrad &input){
    return TensorGrad_Functional_Class::exp(input); 
}
inline TensorGrad& exp_(TensorGrad& input){
    return TensorGrad_Functional_Class::exp_(input); 
}
inline TensorGrad sum(const TensorGrad &input,
                            utils::optional_list list = nullptr,
                            bool keepdim = true){
    return TensorGrad_Functional_Class::sum(input, list, keepdim);
}
inline TensorGrad logsumexp(const TensorGrad &input,
                            utils::optional_list list = nullptr,
                            bool keepdim = true){
    return TensorGrad_Functional_Class::logsumexp(input, list, keepdim);
}

inline TensorGrad transpose(const TensorGrad& input, Tensor::size_value_t a, Tensor::size_value_t b){
    return TensorGrad_Functional_Class::transpose(input, a, b);
}
inline TensorGrad permute(const TensorGrad& input, std::vector<Tensor::size_value_t> permutations){
    return TensorGrad_Functional_Class::permute(input, std::move(permutations));
}
inline TensorGrad& row_col_swap_(TensorGrad& input){
    return TensorGrad_Functional_Class::row_col_swap_(input);
}

//trig.cpp
inline TensorGrad tan(const TensorGrad &input){
    return TensorGrad_Functional_Class::tan(input); 
}
inline TensorGrad tanh(const TensorGrad &input){
    return TensorGrad_Functional_Class::tanh(input); 
}
inline TensorGrad atan(const TensorGrad &input){
    return TensorGrad_Functional_Class::atan(input); 
}
inline TensorGrad atanh(const TensorGrad &input){
    return TensorGrad_Functional_Class::atanh(input); 
}
inline TensorGrad cotan(const TensorGrad &input){
    return TensorGrad_Functional_Class::cotan(input); 
}
inline TensorGrad cotanh(const TensorGrad &input){
    return TensorGrad_Functional_Class::cotanh(input); 
}

inline TensorGrad sin(const TensorGrad &input){
    return TensorGrad_Functional_Class::sin(input); 
}
inline TensorGrad sinh(const TensorGrad &input){
    return TensorGrad_Functional_Class::sinh(input); 
}
inline TensorGrad asin(const TensorGrad &input){
    return TensorGrad_Functional_Class::asin(input); 
}
inline TensorGrad asinh(const TensorGrad &input){
    return TensorGrad_Functional_Class::asinh(input); 
}
inline TensorGrad csc(const TensorGrad &input){
    return TensorGrad_Functional_Class::csc(input); 
}
inline TensorGrad csch(const TensorGrad &input){
    return TensorGrad_Functional_Class::csch(input); 
}

inline TensorGrad cos(const TensorGrad &input){
    return TensorGrad_Functional_Class::cos(input); 
}
inline TensorGrad cosh(const TensorGrad &input){
    return TensorGrad_Functional_Class::cosh(input); 
}
inline TensorGrad acos(const TensorGrad &input){
    return TensorGrad_Functional_Class::acos(input); 
}
inline TensorGrad acosh(const TensorGrad &input){
    return TensorGrad_Functional_Class::acosh(input); 
}
inline TensorGrad sec(const TensorGrad &input){
    return TensorGrad_Functional_Class::sec(input); 
}
inline TensorGrad sech(const TensorGrad &input){
    return TensorGrad_Functional_Class::sech(input); 
}


inline TensorGrad& tan_(TensorGrad &input){
    return TensorGrad_Functional_Class::tan_(input); 
}
inline TensorGrad& tanh_(TensorGrad &input){
    return TensorGrad_Functional_Class::tanh_(input); 
}
inline TensorGrad& atan_(TensorGrad &input){
    return TensorGrad_Functional_Class::atan_(input); 
}
inline TensorGrad& atanh_(TensorGrad &input){
    return TensorGrad_Functional_Class::atanh_(input); 
}
inline TensorGrad& cotan_(TensorGrad &input){
    return TensorGrad_Functional_Class::cotan_(input); 
}
inline TensorGrad& cotanh_(TensorGrad &input){
    return TensorGrad_Functional_Class::cotanh_(input); 
}

inline TensorGrad& sin_(TensorGrad &input){
    return TensorGrad_Functional_Class::sin_(input); 
}
inline TensorGrad& sinh_(TensorGrad &input){
    return TensorGrad_Functional_Class::sinh_(input); 
}
inline TensorGrad& asin_(TensorGrad &input){
    return TensorGrad_Functional_Class::asin_(input); 
}
inline TensorGrad& asinh_(TensorGrad &input){
    return TensorGrad_Functional_Class::asinh_(input); 
}
inline TensorGrad& csc_(TensorGrad &input){
    return TensorGrad_Functional_Class::csc_(input); 
}
inline TensorGrad& csch_(TensorGrad &input){
    return TensorGrad_Functional_Class::csch_(input); 
}

inline TensorGrad& cos_(TensorGrad &input){
    return TensorGrad_Functional_Class::cos_(input); 
}
inline TensorGrad& cosh_(TensorGrad &input){
    return TensorGrad_Functional_Class::cosh_(input); 
}
inline TensorGrad& acos_(TensorGrad &input){
    return TensorGrad_Functional_Class::acos_(input); 
}
inline TensorGrad& acosh_(TensorGrad &input){
    return TensorGrad_Functional_Class::acosh_(input); 
}
inline TensorGrad& sec_(TensorGrad &input){
    return TensorGrad_Functional_Class::sec_(input); 
}
inline TensorGrad& sech_(TensorGrad &input){
    return TensorGrad_Functional_Class::sech_(input); 
}

inline TensorGrad unique(const TensorGrad& input, int64_t dim, bool return_sorted = true, bool return_indice = true){
    return TensorGrad_Functional_Class::unique(input, dim, return_sorted, return_indice);
}
} // namespace functional
} // namespace nt

#include "functional/functional_list.h"

#endif // NT_TENSORGAD_FUNCTIONAL_H__
