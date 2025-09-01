// this is a friend class to TensorGrad that holds the functional functions
// dedicated to tensorgrad this is meant to re-create the functional.h that is
// dedicated to the Tensor class NEUROTENSOR_API
#ifndef NT_TENSORGRAD_FUNCTIONAL_CLASS_H__
#define NT_TENSORGRAD_FUNCTIONAL_CLASS_H__

namespace nt {
namespace functional {
class NEUROTENSOR_API TensorGrad_Functional_Class; // forward declaration

} // namespace functional
} // namespace nt

#include "TensorGrad.h"
#include "../utils/tuple_or_var.h"
namespace nt {
namespace functional {

class NEUROTENSOR_API TensorGrad_Functional_Class {
  public:
    TensorGrad_Functional_Class() = default;
    static NEUROTENSOR_API TensorGrad matmult(const TensorGrad &, const TensorGrad &, bool,
                              bool);
    static NEUROTENSOR_API TensorGrad matmult(const Tensor &, const TensorGrad &, bool, bool);
    static NEUROTENSOR_API TensorGrad matmult(const TensorGrad &a, const Tensor &b, bool, bool);

    static NEUROTENSOR_API TensorGrad& matmult(const TensorGrad &, const TensorGrad &, TensorGrad&, bool,
                              bool);
    static NEUROTENSOR_API TensorGrad& matmult(const Tensor &, const TensorGrad &, TensorGrad&, bool, bool);
    static NEUROTENSOR_API TensorGrad& matmult(const TensorGrad &a, const Tensor &b, TensorGrad&, bool, bool);

    static NEUROTENSOR_API TensorGrad linear(const TensorGrad&, const TensorGrad&, const TensorGrad&, bool, bool);
    static NEUROTENSOR_API TensorGrad linear(const Tensor&, const TensorGrad&, const TensorGrad&, bool, bool);
    static NEUROTENSOR_API TensorGrad linear(const TensorGrad&, const Tensor&, const TensorGrad&, bool, bool);
    static NEUROTENSOR_API TensorGrad linear(const TensorGrad&, const TensorGrad&, const Tensor&, bool, bool);
    static NEUROTENSOR_API TensorGrad linear(const TensorGrad&, const Tensor&, const Tensor&, bool, bool);
    static NEUROTENSOR_API TensorGrad linear(const Tensor&, const TensorGrad&, const Tensor&, bool, bool);
    static NEUROTENSOR_API TensorGrad linear(const Tensor&, const Tensor&, const TensorGrad&, bool, bool);

    static NEUROTENSOR_API TensorGrad unfold1d(const TensorGrad &, Tensor::size_value_t,
                               Tensor::size_value_t, Tensor::size_value_t,
                               Tensor::size_value_t, bool);
    static NEUROTENSOR_API TensorGrad unfold2d(const TensorGrad &, utils::my_tuple,
                             utils::my_tuple, utils::my_tuple, utils::my_tuple,
                             bool);
    static NEUROTENSOR_API TensorGrad unfold3d(const TensorGrad &, utils::my_n_tuple<3>,
                               utils::my_n_tuple<3>, utils::my_n_tuple<3>,
                               utils::my_n_tuple<3>, bool);
    static NEUROTENSOR_API TensorGrad unfoldnd(const TensorGrad &, int64_t dim, utils::optional_list,
                               utils::optional_list, utils::optional_list,
                               utils::optional_list, bool, bool);
    static NEUROTENSOR_API TensorGrad fold1d(const TensorGrad &, Tensor::size_value_t, Tensor::size_value_t,
                           Tensor::size_value_t, Tensor::size_value_t, Tensor::size_value_t);
    static NEUROTENSOR_API TensorGrad fold2d(const TensorGrad &, utils::my_tuple, utils::my_tuple,
                           utils::my_tuple, utils::my_tuple, utils::my_tuple);
    static NEUROTENSOR_API TensorGrad fold3d(const TensorGrad &, utils::my_n_tuple<3>, utils::my_n_tuple<3>,
                           utils::my_n_tuple<3>, utils::my_n_tuple<3>, utils::my_n_tuple<3>);
    static NEUROTENSOR_API TensorGrad foldnd(const TensorGrad &, int64_t, utils::optional_list, utils::optional_list,
                           utils::optional_list, utils::optional_list, utils::optional_list,bool);
    // image, kernel, stride, padding, dilation, groups
    static NEUROTENSOR_API TensorGrad conv1d(const TensorGrad &, const TensorGrad &, int64_t,
                             int64_t, int64_t, int64_t);
    static NEUROTENSOR_API TensorGrad conv1d(const Tensor &, const TensorGrad &, int64_t,
                             int64_t, int64_t, int64_t);
    static NEUROTENSOR_API TensorGrad conv1d(const TensorGrad &, const Tensor &, int64_t,
                             int64_t, int64_t, int64_t);
    static NEUROTENSOR_API TensorGrad conv2d(const TensorGrad &, const TensorGrad &,
                             utils::my_tuple, utils::my_tuple, utils::my_tuple,
                             int64_t);
    static NEUROTENSOR_API TensorGrad conv2d(const Tensor &, const TensorGrad &,
                             utils::my_tuple, utils::my_tuple, utils::my_tuple,
                             int64_t);
    static NEUROTENSOR_API TensorGrad conv2d(const TensorGrad &, const Tensor &,
                             utils::my_tuple, utils::my_tuple, utils::my_tuple,
                             int64_t);
    static NEUROTENSOR_API TensorGrad conv3d(const TensorGrad &, const TensorGrad &,
                             utils::my_n_tuple<3>, utils::my_n_tuple<3>,
                             utils::my_n_tuple<3>, int64_t);
    static NEUROTENSOR_API TensorGrad conv3d(const Tensor &, const TensorGrad &,
                             utils::my_n_tuple<3>, utils::my_n_tuple<3>,
                             utils::my_n_tuple<3>, int64_t);
    static NEUROTENSOR_API TensorGrad conv3d(const TensorGrad &, const Tensor &,
                             utils::my_n_tuple<3>, utils::my_n_tuple<3>,
                             utils::my_n_tuple<3>, int64_t);
    static NEUROTENSOR_API TensorGrad convnd(const TensorGrad &, const TensorGrad &,
                             int64_t, utils::optional_list, utils::optional_list,
                             utils::optional_list, int64_t);
    static NEUROTENSOR_API TensorGrad convnd(const Tensor &, const TensorGrad &,
                             int64_t, utils::optional_list, utils::optional_list,
                             utils::optional_list, int64_t);
    static NEUROTENSOR_API TensorGrad convnd(const TensorGrad &, const Tensor &,
                             int64_t, utils::optional_list, utils::optional_list,
                             utils::optional_list, int64_t);


    static NEUROTENSOR_API TensorGrad conv_transpose1d(const TensorGrad &, const TensorGrad &,
                                       int64_t, int64_t, int64_t, int64_t,
                                       int64_t);
    static NEUROTENSOR_API TensorGrad conv_transpose1d(const Tensor &, const TensorGrad &,
                                       int64_t, int64_t, int64_t, int64_t,
                                       int64_t);
    static NEUROTENSOR_API TensorGrad conv_transpose1d(const TensorGrad &, const Tensor &,
                                       int64_t, int64_t, int64_t, int64_t,
                                       int64_t);
    static NEUROTENSOR_API TensorGrad conv_transpose2d(const TensorGrad &, const TensorGrad &,
                                       utils::my_tuple, utils::my_tuple,
                                       utils::my_tuple, utils::my_tuple,
                                       int64_t);
    static NEUROTENSOR_API TensorGrad conv_transpose2d(const Tensor &, const TensorGrad &,
                                       utils::my_tuple, utils::my_tuple,
                                       utils::my_tuple, utils::my_tuple,
                                       int64_t);
    static NEUROTENSOR_API TensorGrad conv_transpose2d(const TensorGrad &, const Tensor &,
                                       utils::my_tuple, utils::my_tuple,
                                       utils::my_tuple, utils::my_tuple,
                                       int64_t);
    static NEUROTENSOR_API TensorGrad conv_transpose3d(const TensorGrad &, const TensorGrad &,
                                       utils::my_n_tuple<3>,
                                       utils::my_n_tuple<3>,
                                       utils::my_n_tuple<3>,
                                       utils::my_n_tuple<3>, int64_t);
    static NEUROTENSOR_API TensorGrad conv_transpose3d(const Tensor &, const TensorGrad &,
                                       utils::my_n_tuple<3>,
                                       utils::my_n_tuple<3>,
                                       utils::my_n_tuple<3>,
                                       utils::my_n_tuple<3>, int64_t);
    static NEUROTENSOR_API TensorGrad conv_transpose3d(const TensorGrad &, const Tensor &,
                                       utils::my_n_tuple<3>,
                                       utils::my_n_tuple<3>,
                                       utils::my_n_tuple<3>,
                                       utils::my_n_tuple<3>, int64_t);


    static NEUROTENSOR_API TensorGrad clamp(const TensorGrad &, std::optional<Scalar>,
                            std::optional<Scalar>);
    static NEUROTENSOR_API TensorGrad& clamp_(TensorGrad &, std::optional<Scalar>,
                            std::optional<Scalar>);

    static NEUROTENSOR_API TensorGrad var(const TensorGrad &, utils::optional_list, int64_t,
                          bool);
    //activation_functions.cpp
    static NEUROTENSOR_API TensorGrad sqrt(const TensorGrad &);
    static NEUROTENSOR_API TensorGrad invsqrt(const TensorGrad &);
    static NEUROTENSOR_API TensorGrad pow(const TensorGrad&, Scalar);
    static NEUROTENSOR_API TensorGrad silu(const TensorGrad &);
    static NEUROTENSOR_API TensorGrad gelu(const TensorGrad &);
    static NEUROTENSOR_API TensorGrad abs(const TensorGrad &);
    static NEUROTENSOR_API TensorGrad softplus(const TensorGrad &, Scalar, Scalar);
    static NEUROTENSOR_API TensorGrad sigmoid(const TensorGrad &);
    static NEUROTENSOR_API TensorGrad relu(const TensorGrad &);
    static NEUROTENSOR_API TensorGrad& sqrt_(TensorGrad &);
    static NEUROTENSOR_API TensorGrad& invsqrt_(TensorGrad &);
    static NEUROTENSOR_API TensorGrad& pow_(TensorGrad&, Scalar);
    static NEUROTENSOR_API TensorGrad& silu_(TensorGrad &);
    static NEUROTENSOR_API TensorGrad& gelu_(TensorGrad &);
    static NEUROTENSOR_API TensorGrad& abs_(TensorGrad &);
    static NEUROTENSOR_API TensorGrad& softplus_(TensorGrad &, Scalar, Scalar);
    static NEUROTENSOR_API TensorGrad& sigmoid_(TensorGrad &);
    static NEUROTENSOR_API TensorGrad& relu_(TensorGrad &);



    static NEUROTENSOR_API TensorGrad cat(std::vector<TensorGrad>, int64_t);
    static NEUROTENSOR_API TensorGrad cat(TensorGrad, int64_t);
    static NEUROTENSOR_API TensorGrad stack(TensorGrad, int64_t);
    static NEUROTENSOR_API TensorGrad stack(std::vector<TensorGrad>, int64_t);

    static NEUROTENSOR_API TensorGrad real(const TensorGrad&);
    static NEUROTENSOR_API TensorGrad imag(const TensorGrad&);
    static NEUROTENSOR_API TensorGrad to_complex_from_real(const TensorGrad&);
    static NEUROTENSOR_API TensorGrad to_complex_from_imag(const TensorGrad&);
    static NEUROTENSOR_API TensorGrad to(const TensorGrad&, DType);

    static NEUROTENSOR_API TensorGrad dilate(const TensorGrad&, Tensor::size_value_t);
    static NEUROTENSOR_API TensorGrad dilate(const TensorGrad&, Tensor::size_value_t, Tensor::size_value_t);
    static NEUROTENSOR_API TensorGrad dilate(const TensorGrad&, Tensor::size_value_t, Tensor::size_value_t, Tensor::size_value_t);
    static NEUROTENSOR_API TensorGrad undilate_(const TensorGrad&, Tensor::size_value_t);
    static NEUROTENSOR_API TensorGrad undilate_(const TensorGrad&, Tensor::size_value_t, Tensor::size_value_t);
    static NEUROTENSOR_API TensorGrad undilate_(const TensorGrad&, Tensor::size_value_t, Tensor::size_value_t, Tensor::size_value_t);
    static NEUROTENSOR_API TensorGrad undilate(const TensorGrad&, Tensor::size_value_t);
    static NEUROTENSOR_API TensorGrad undilate(const TensorGrad&, Tensor::size_value_t, Tensor::size_value_t);
    static NEUROTENSOR_API TensorGrad undilate(const TensorGrad&, Tensor::size_value_t, Tensor::size_value_t, Tensor::size_value_t);
    
    static NEUROTENSOR_API TensorGrad zeros_like(const TensorGrad&);
    static NEUROTENSOR_API TensorGrad ones_like(const TensorGrad&);
    static NEUROTENSOR_API TensorGrad nums_like(const TensorGrad&, Scalar);
    static NEUROTENSOR_API TensorGrad& fill_diagonal_(TensorGrad&, Scalar);
    static NEUROTENSOR_API TensorGrad& fill_(TensorGrad&, Scalar);
    static NEUROTENSOR_API TensorGrad& set_(TensorGrad&, const Tensor&);
    static NEUROTENSOR_API TensorGrad& set_(TensorGrad&, const TensorGrad&);


    static NEUROTENSOR_API TensorGrad dropout(const TensorGrad &, double);
    static NEUROTENSOR_API TensorGrad dropout2d(const TensorGrad &, double);
    static NEUROTENSOR_API TensorGrad dropout3d(const TensorGrad &, double);


    static NEUROTENSOR_API TensorGrad softmax(const TensorGrad &, bool stable);
    static NEUROTENSOR_API TensorGrad softmax(const TensorGrad &,
                              typename SizeRef::value_type dim, bool stable);
    static NEUROTENSOR_API TensorGrad gumbel_softmax(const TensorGrad &, Scalar tau, bool hard,
                                     int64_t dim, bool stable);
    static NEUROTENSOR_API TensorGrad symmetric_bilinear(const TensorGrad&, const TensorGrad&, const TensorGrad&);
    static NEUROTENSOR_API TensorGrad symmetric_bilinear(const TensorGrad&, const TensorGrad&, const Tensor&);
    static NEUROTENSOR_API TensorGrad symmetric_bilinear(const TensorGrad&, const Tensor&, const TensorGrad&);
    static NEUROTENSOR_API TensorGrad symmetric_bilinear(const Tensor&, const TensorGrad&, const TensorGrad&);
    static NEUROTENSOR_API TensorGrad symmetric_bilinear(const TensorGrad&, const Tensor&, const Tensor&);
    static NEUROTENSOR_API TensorGrad symmetric_bilinear(const Tensor&, const Tensor&, const TensorGrad&);
    static NEUROTENSOR_API TensorGrad symmetric_bilinear(const Tensor&, const TensorGrad&, const Tensor&);
    
    //pooling
    static NEUROTENSOR_API TensorGrad avg_pool1d(TensorGrad input, int64_t kernel_size, int64_t stride, int64_t padding, bool ceil_mode, bool count_include_pad);
    static NEUROTENSOR_API TensorGrad adaptive_avg_pool1d(TensorGrad x, int64_t l_out);

    static NEUROTENSOR_API TensorGrad avg_pool2d(TensorGrad input, utils::my_tuple kernel_size, utils::my_tuple stride, utils::my_tuple padding, 
                          bool ceil_mode, bool count_include_pad);
    static NEUROTENSOR_API TensorGrad adaptive_avg_pool2d(TensorGrad x, utils::my_tuple out_shape);

    static NEUROTENSOR_API TensorGrad avg_pool3d(TensorGrad input, utils::my_n_tuple<3> kernel_size, utils::my_n_tuple<3> stride, utils::my_n_tuple<3> padding, 
                          bool ceil_mode, bool count_include_pad);
    static NEUROTENSOR_API TensorGrad adaptive_avg_pool3d(TensorGrad x, utils::my_n_tuple<3> out_shape);

    //LP Pooling
    static NEUROTENSOR_API TensorGrad lp_pool1d(TensorGrad input, Scalar power, int64_t kernel_size, int64_t stride, bool ceil_mode);
    static NEUROTENSOR_API TensorGrad adaptive_lp_pool1d(TensorGrad x, int64_t l_out, Scalar power);

    static NEUROTENSOR_API TensorGrad lp_pool2d(TensorGrad input, Scalar power, utils::my_tuple kernel_size, utils::my_tuple stride, bool ceil_mode);
    static NEUROTENSOR_API TensorGrad adaptive_lp_pool2d(TensorGrad x, utils::my_tuple out_shape, Scalar power);

    static NEUROTENSOR_API TensorGrad lp_pool3d(TensorGrad input, Scalar power, utils::my_n_tuple<3> kernel_size, utils::my_n_tuple<3> stride, bool ceil_mode);
    static NEUROTENSOR_API TensorGrad adaptive_lp_pool3d(TensorGrad x, utils::my_n_tuple<3> out_shape, Scalar power);


    //Max Pooling
    static NEUROTENSOR_API TensorGrad max_pool1d(TensorGrad input, int64_t kernel_size, int64_t stride, int64_t padding, int64_t dilation, bool ceil_mode, bool return_indices);
    static NEUROTENSOR_API TensorGrad max_unpool1d(TensorGrad input, Tensor indices, int64_t kernel_size, int64_t stride, int64_t padding, int64_t output_size);
    static NEUROTENSOR_API TensorGrad max_unpool1d(TensorGrad input, TensorGrad indices, int64_t kernel_size, int64_t stride, int64_t padding, int64_t output_size);
    static NEUROTENSOR_API TensorGrad adaptive_max_pool1d(TensorGrad x, int64_t l_out, bool return_indices);

    static NEUROTENSOR_API TensorGrad max_pool2d(TensorGrad input, 
                          utils::my_tuple kernel_size, utils::my_tuple stride, utils::my_tuple padding,
                          utils::my_tuple dilation, bool ceil_mode, bool return_indices);
    static NEUROTENSOR_API TensorGrad max_unpool2d(TensorGrad input, Tensor indices, utils::my_tuple kernel_size, utils::my_tuple stride, utils::my_tuple padding, utils::my_tuple output_size);
    static NEUROTENSOR_API TensorGrad max_unpool2d(TensorGrad input, TensorGrad indices, utils::my_tuple kernel_size, utils::my_tuple stride, utils::my_tuple padding, utils::my_tuple output_size);
    static NEUROTENSOR_API TensorGrad adaptive_max_pool2d(TensorGrad x, utils::my_tuple out_shape, bool return_indices);

    static NEUROTENSOR_API TensorGrad max_pool3d(TensorGrad input, 
                          utils::my_n_tuple<3> kernel_size, utils::my_n_tuple<3> stride, utils::my_n_tuple<3> padding,
                          utils::my_n_tuple<3> dilation, bool ceil_mode, bool return_indices);
    static NEUROTENSOR_API TensorGrad max_unpool3d(TensorGrad input, Tensor indices, utils::my_n_tuple<3> kernel_size, utils::my_n_tuple<3> stride, utils::my_n_tuple<3> padding, utils::my_n_tuple<3> output_size);
    static NEUROTENSOR_API TensorGrad max_unpool3d(TensorGrad input, TensorGrad indices, utils::my_n_tuple<3> kernel_size, utils::my_n_tuple<3> stride, utils::my_n_tuple<3> padding, utils::my_n_tuple<3> output_size);
    static NEUROTENSOR_API TensorGrad adaptive_max_pool3d(TensorGrad x, utils::my_n_tuple<3> out_shape, bool return_indices);

    //Fractional Max Pooling
    static NEUROTENSOR_API TensorGrad fractional_max_pool2d(TensorGrad input, utils::my_tuple kernel_size, utils::my_tuple output_size, 
                                     utils::tuple_or_var<double, 2> output_ratio, bool return_indices);
    static NEUROTENSOR_API TensorGrad fractional_max_pool3d(TensorGrad input, utils::my_n_tuple<3> kernel_size, utils::my_n_tuple<3> output_size, 
                                     utils::tuple_or_var<double, 3> output_ratio, bool return_indices);

    // Flip.cpp
    static NEUROTENSOR_API TensorGrad flip(const TensorGrad&, utils::optional_list);
    static NEUROTENSOR_API TensorGrad flip_view(const TensorGrad&, utils::optional_list);

    // Fused.cpp
    //returns c + (a * b);
    static NEUROTENSOR_API TensorGrad fused_multiply_add(const TensorGrad& c, const TensorGrad& a, const TensorGrad& b);
    static NEUROTENSOR_API TensorGrad fused_multiply_add(const Tensor& c, const TensorGrad& a, const TensorGrad& b);
    static NEUROTENSOR_API TensorGrad fused_multiply_add(const Tensor& c, const Tensor& a, const TensorGrad& b);
    static NEUROTENSOR_API TensorGrad fused_multiply_add(const Tensor& c, const TensorGrad& a, const Tensor& b);
    static NEUROTENSOR_API TensorGrad fused_multiply_add(const TensorGrad& c, const Tensor& a, const TensorGrad& b);
    static NEUROTENSOR_API TensorGrad fused_multiply_add(const TensorGrad& c, const Tensor& a, const Tensor& b);
    static NEUROTENSOR_API TensorGrad fused_multiply_add(const TensorGrad& c, const TensorGrad& a, const Tensor& b);

    static NEUROTENSOR_API TensorGrad fused_multiply_add(const TensorGrad& c, const TensorGrad& a, Scalar b);
    static NEUROTENSOR_API TensorGrad fused_multiply_add(const TensorGrad& c, const Tensor& a, Scalar b);
    static NEUROTENSOR_API TensorGrad fused_multiply_add(const Tensor& c, const TensorGrad& a, Scalar b);
    //returns c += (a * b);
    static NEUROTENSOR_API TensorGrad& fused_multiply_add_(TensorGrad& c, const TensorGrad& a, const TensorGrad& b);
    static NEUROTENSOR_API TensorGrad& fused_multiply_add_(TensorGrad& c, const Tensor& a, const TensorGrad& b);
    static NEUROTENSOR_API TensorGrad& fused_multiply_add_(TensorGrad& c, const TensorGrad& a, const Tensor& b);
    static NEUROTENSOR_API TensorGrad& fused_multiply_add_(TensorGrad& c, const Tensor& a, const Tensor& b);

    static NEUROTENSOR_API TensorGrad& fused_multiply_add_(TensorGrad& c, const TensorGrad& a, Scalar b);
    static NEUROTENSOR_API TensorGrad& fused_multiply_add_(TensorGrad& c, const Tensor& a, Scalar b);

    //returns c - (a * b);
    static NEUROTENSOR_API TensorGrad fused_multiply_subtract(const TensorGrad& c, const TensorGrad& a, const TensorGrad& b);
    static NEUROTENSOR_API TensorGrad fused_multiply_subtract(const Tensor& c, const TensorGrad& a, const TensorGrad& b);
    static NEUROTENSOR_API TensorGrad fused_multiply_subtract(const Tensor& c, const Tensor& a, const TensorGrad& b);
    static NEUROTENSOR_API TensorGrad fused_multiply_subtract(const Tensor& c, const TensorGrad& a, const Tensor& b);
    static NEUROTENSOR_API TensorGrad fused_multiply_subtract(const TensorGrad& c, const Tensor& a, const TensorGrad& b);
    static NEUROTENSOR_API TensorGrad fused_multiply_subtract(const TensorGrad& c, const Tensor& a, const Tensor& b);
    static NEUROTENSOR_API TensorGrad fused_multiply_subtract(const TensorGrad& c, const TensorGrad& a, const Tensor& b);

    static NEUROTENSOR_API TensorGrad fused_multiply_subtract(const TensorGrad& c, const TensorGrad& a, Scalar b);
    static NEUROTENSOR_API TensorGrad fused_multiply_subtract(const TensorGrad& c, const Tensor& a, Scalar b);
    static NEUROTENSOR_API TensorGrad fused_multiply_subtract(const Tensor& c, const TensorGrad& a, Scalar b);

    //returns c -= (a * b);
    static NEUROTENSOR_API TensorGrad& fused_multiply_subtract_(TensorGrad& c, const TensorGrad& a, const TensorGrad& b);
    static NEUROTENSOR_API TensorGrad& fused_multiply_subtract_(TensorGrad& c, const Tensor& a, const TensorGrad& b);
    static NEUROTENSOR_API TensorGrad& fused_multiply_subtract_(TensorGrad& c, const Tensor& a, const Tensor& b);
    static NEUROTENSOR_API TensorGrad& fused_multiply_subtract_(TensorGrad& c, const TensorGrad& a, const Tensor& b);

    static NEUROTENSOR_API TensorGrad& fused_multiply_subtract_(TensorGrad& c, const TensorGrad& a, Scalar b);
    static NEUROTENSOR_API TensorGrad& fused_multiply_subtract_(TensorGrad& c, const Tensor& a, Scalar b);

    //Index.cpp
    static NEUROTENSOR_API TensorGrad at(const TensorGrad&, Tensor::size_value_t);
    static NEUROTENSOR_API TensorGrad at(const TensorGrad&, const Tensor&);
    static NEUROTENSOR_API TensorGrad at(const TensorGrad&, const TensorGrad&);
    static NEUROTENSOR_API Tensor at(const Tensor&, const TensorGrad&);
    static NEUROTENSOR_API TensorGrad at(const TensorGrad&, std::vector<Tensor::size_value_t>);
    static NEUROTENSOR_API TensorGrad at_tensor_split(const TensorGrad &, const TensorGrad &, Tensor::size_value_t);
    static NEUROTENSOR_API TensorGrad at_tensor_split(const TensorGrad &, const Tensor &, Tensor::size_value_t);
    static NEUROTENSOR_API TensorGrad &at_tensor_split(const TensorGrad &, const TensorGrad &, Tensor::size_value_t,
                        TensorGrad &);
    static NEUROTENSOR_API TensorGrad &at_tensor_split(const TensorGrad &, const Tensor &, Tensor::size_value_t,
                        TensorGrad &);
    static NEUROTENSOR_API TensorGrad index_except(const TensorGrad &, int64_t, Tensor::size_value_t);
    static NEUROTENSOR_API TensorGrad index_select(const TensorGrad &, int64_t, const Tensor& index);
    static NEUROTENSOR_API TensorGrad index_select(const TensorGrad &, int64_t, const TensorGrad& index);
    static NEUROTENSOR_API TensorGrad select(const TensorGrad& input, Tensor::size_value_t dim, Tensor::size_value_t index);
    
    //min max
    static NEUROTENSOR_API result_types::max<TensorGrad, Tensor> max(const TensorGrad& input, utils::optional_list dim, bool keepdim);
    static NEUROTENSOR_API result_types::max<TensorGrad, Tensor> min(const TensorGrad& input, utils::optional_list dim, bool keepdim);
    static NEUROTENSOR_API TensorGrad maximum(const std::vector<TensorGrad>&, const std::vector<Tensor>& ts = {}, const std::vector<Scalar>& ss = {});
    static NEUROTENSOR_API TensorGrad minimum(const std::vector<TensorGrad>&, const std::vector<Tensor>& ts = {}, const std::vector<Scalar>& ss = {});

    //operators.cpp

    static NEUROTENSOR_API TensorGrad add(const TensorGrad&, const TensorGrad&);
    static NEUROTENSOR_API TensorGrad add(const TensorGrad&, const Tensor&);
    static NEUROTENSOR_API TensorGrad add(const Tensor&, const TensorGrad&);
    static NEUROTENSOR_API TensorGrad add(const TensorGrad&, const Scalar&);
    static NEUROTENSOR_API TensorGrad add(const Scalar&, const TensorGrad&);
    static NEUROTENSOR_API TensorGrad& add_(TensorGrad&, const TensorGrad&);
    static NEUROTENSOR_API TensorGrad& add_(TensorGrad&, const Tensor&);
    static NEUROTENSOR_API Tensor& add_(Tensor&, const TensorGrad&);
    static NEUROTENSOR_API TensorGrad& add_(TensorGrad&, const Scalar&);


    static NEUROTENSOR_API TensorGrad multiply(const TensorGrad&, const TensorGrad&);
    static NEUROTENSOR_API TensorGrad multiply(const TensorGrad&, const Tensor&);
    static NEUROTENSOR_API TensorGrad multiply(const Tensor&, const TensorGrad&);
    static NEUROTENSOR_API TensorGrad multiply(const TensorGrad&, const Scalar&);
    static NEUROTENSOR_API TensorGrad multiply(const Scalar&, const TensorGrad&);
    static NEUROTENSOR_API TensorGrad& multiply_(TensorGrad&, const TensorGrad&);
    static NEUROTENSOR_API TensorGrad& multiply_(TensorGrad&, const Tensor&);
    static NEUROTENSOR_API Tensor& multiply_(Tensor&, const TensorGrad&);
    static NEUROTENSOR_API TensorGrad& multiply_(TensorGrad&, const Scalar&);


    static NEUROTENSOR_API TensorGrad subtract(const TensorGrad&, const TensorGrad&);
    static NEUROTENSOR_API TensorGrad subtract(const TensorGrad&, const Tensor&);
    static NEUROTENSOR_API TensorGrad subtract(const Tensor&, const TensorGrad&);
    static NEUROTENSOR_API TensorGrad subtract(const TensorGrad&, const Scalar&);
    static NEUROTENSOR_API TensorGrad subtract(const Scalar&, const TensorGrad&);
    static NEUROTENSOR_API TensorGrad& subtract_(TensorGrad&, const TensorGrad&);
    static NEUROTENSOR_API TensorGrad& subtract_(TensorGrad&, const Tensor&);
    static NEUROTENSOR_API Tensor& subtract_(Tensor&, const TensorGrad&);
    static NEUROTENSOR_API TensorGrad& subtract_(TensorGrad&, const Scalar&);

    static NEUROTENSOR_API TensorGrad divide(const TensorGrad&, const TensorGrad&);
    static NEUROTENSOR_API TensorGrad divide(const TensorGrad&, const Tensor&);
    static NEUROTENSOR_API TensorGrad divide(const Tensor&, const TensorGrad&);
    static NEUROTENSOR_API TensorGrad divide(const TensorGrad&, const Scalar&);
    static NEUROTENSOR_API TensorGrad divide(const Scalar&, const TensorGrad&);
    static NEUROTENSOR_API TensorGrad& divide_(TensorGrad&, const TensorGrad&);
    static NEUROTENSOR_API TensorGrad& divide_(TensorGrad&, const Tensor&);
    static NEUROTENSOR_API Tensor& divide_(Tensor&, const TensorGrad&);
    static NEUROTENSOR_API TensorGrad& divide_(TensorGrad&, const Scalar&);

    static NEUROTENSOR_API TensorGrad fmod(const TensorGrad&, const TensorGrad&);
    static NEUROTENSOR_API TensorGrad fmod(const TensorGrad&, const Tensor&);
    static NEUROTENSOR_API TensorGrad fmod(const Tensor&, const TensorGrad&);
    static NEUROTENSOR_API TensorGrad fmod(const TensorGrad&, const Scalar&);
    static NEUROTENSOR_API TensorGrad fmod(const Scalar&, const TensorGrad&);

    static NEUROTENSOR_API TensorGrad remainder(const TensorGrad&, const TensorGrad&);
    static NEUROTENSOR_API TensorGrad remainder(const TensorGrad&, const Tensor&);
    static NEUROTENSOR_API TensorGrad remainder(const Tensor&, const TensorGrad&);
    static NEUROTENSOR_API TensorGrad remainder(const TensorGrad&, const Scalar&);
    static NEUROTENSOR_API TensorGrad remainder(const Scalar&, const TensorGrad&);

    static NEUROTENSOR_API TensorGrad inverse(const TensorGrad&);
    static NEUROTENSOR_API TensorGrad& inverse_(TensorGrad&);
    
    //padding.cpp
    static NEUROTENSOR_API TensorGrad pad(const TensorGrad&, std::vector<Tensor::size_value_t>, const char*, Scalar);
    static NEUROTENSOR_API TensorGrad unpad(const TensorGrad&, std::vector<Tensor::size_value_t>, bool no_contiguous);

    //repeat.cpp
    static NEUROTENSOR_API TensorGrad repeat_(const TensorGrad&, Tensor::size_value_t dim, Tensor::size_value_t amt);
    static NEUROTENSOR_API TensorGrad repeat_(const TensorGrad&, Tensor::size_value_t amt);
    static NEUROTENSOR_API TensorGrad expand(const TensorGrad&, SizeRef);
    static NEUROTENSOR_API TensorGrad expand_as(const TensorGrad&, const TensorGrad&);
    static NEUROTENSOR_API TensorGrad expand_as(const TensorGrad&, const Tensor&);
    static NEUROTENSOR_API Tensor expand_as(const Tensor&, const TensorGrad&);

    //round.cpp
    static NEUROTENSOR_API TensorGrad round(const TensorGrad&);
    static NEUROTENSOR_API TensorGrad trunc(const TensorGrad&);
    static NEUROTENSOR_API TensorGrad floor(const TensorGrad&);
    static NEUROTENSOR_API TensorGrad ceil(const TensorGrad&);

    //sort.cpp
    static NEUROTENSOR_API TensorGrad sort(const TensorGrad&, const Tensor::size_value_t dim,
            bool descending, bool return_sorted,
            bool return_indices);

    static NEUROTENSOR_API TensorGrad coordsort(const TensorGrad&, const Tensor::size_value_t dim, bool descending, 
                                                bool return_sorted, bool return_indices);

    // split.cpp
    static NEUROTENSOR_API TensorGrad split(const TensorGrad&, int64_t dim, utils::optional_list splitting = nullptr);
    static NEUROTENSOR_API TensorGrad chunk(const TensorGrad&, const Tensor::size_value_t chunks, int64_t dim = 0);
    
    //stride.h
    static NEUROTENSOR_API TensorGrad diagonal(const TensorGrad&, bool keep_dims = false);
    static NEUROTENSOR_API TensorGrad as_strided(const TensorGrad &input, const SizeRef n_size, SizeRef n_stride,
                  const int64_t storage_offset = 0, bool whole_tensor = false);

    //sum_exp_log.cpp
    static NEUROTENSOR_API TensorGrad log(const TensorGrad &);
    static NEUROTENSOR_API TensorGrad& log_(TensorGrad &);
    static NEUROTENSOR_API TensorGrad exp(const TensorGrad &);
    static NEUROTENSOR_API TensorGrad& exp_(TensorGrad &);
    static NEUROTENSOR_API TensorGrad sum(const TensorGrad &,
                                utils::optional_list list = nullptr,
                                bool keepdim = true);
    static NEUROTENSOR_API TensorGrad logsumexp(const TensorGrad &,
                                utils::optional_list list = nullptr,
                                bool keepdim = true);

    // transpose.h
    static NEUROTENSOR_API TensorGrad transpose(const TensorGrad&, Tensor::size_value_t a, Tensor::size_value_t b);
    static NEUROTENSOR_API TensorGrad permute(const TensorGrad&, std::vector<Tensor::size_value_t> permutations);
    static NEUROTENSOR_API TensorGrad& row_col_swap_(TensorGrad&);

    //trig.cpp
    static NEUROTENSOR_API TensorGrad tan(const TensorGrad &);
    static NEUROTENSOR_API TensorGrad tanh(const TensorGrad &);
    static NEUROTENSOR_API TensorGrad atan(const TensorGrad &);
    static NEUROTENSOR_API TensorGrad atanh(const TensorGrad &);
    static NEUROTENSOR_API TensorGrad cotan(const TensorGrad &);
    static NEUROTENSOR_API TensorGrad cotanh(const TensorGrad &);

    static NEUROTENSOR_API TensorGrad sin(const TensorGrad &);
    static NEUROTENSOR_API TensorGrad sinh(const TensorGrad &);
    static NEUROTENSOR_API TensorGrad asin(const TensorGrad &);
    static NEUROTENSOR_API TensorGrad asinh(const TensorGrad &);
    static NEUROTENSOR_API TensorGrad csc(const TensorGrad &);
    static NEUROTENSOR_API TensorGrad csch(const TensorGrad &);

    static NEUROTENSOR_API TensorGrad cos(const TensorGrad &);
    static NEUROTENSOR_API TensorGrad cosh(const TensorGrad &);
    static NEUROTENSOR_API TensorGrad acos(const TensorGrad &);
    static NEUROTENSOR_API TensorGrad acosh(const TensorGrad &);
    static NEUROTENSOR_API TensorGrad sec(const TensorGrad &);
    static NEUROTENSOR_API TensorGrad sech(const TensorGrad &);


    static NEUROTENSOR_API TensorGrad& tan_(TensorGrad &);
    static NEUROTENSOR_API TensorGrad& tanh_(TensorGrad &);
    static NEUROTENSOR_API TensorGrad& atan_(TensorGrad &);
    static NEUROTENSOR_API TensorGrad& atanh_(TensorGrad &);
    static NEUROTENSOR_API TensorGrad& cotan_(TensorGrad &);
    static NEUROTENSOR_API TensorGrad& cotanh_(TensorGrad &);

    static NEUROTENSOR_API TensorGrad& sin_(TensorGrad &);
    static NEUROTENSOR_API TensorGrad& sinh_(TensorGrad &);
    static NEUROTENSOR_API TensorGrad& asin_(TensorGrad &);
    static NEUROTENSOR_API TensorGrad& asinh_(TensorGrad &);
    static NEUROTENSOR_API TensorGrad& csc_(TensorGrad &);
    static NEUROTENSOR_API TensorGrad& csch_(TensorGrad &);

    static NEUROTENSOR_API TensorGrad& cos_(TensorGrad &);
    static NEUROTENSOR_API TensorGrad& cosh_(TensorGrad &);
    static NEUROTENSOR_API TensorGrad& acos_(TensorGrad &);
    static NEUROTENSOR_API TensorGrad& acosh_(TensorGrad &);
    static NEUROTENSOR_API TensorGrad& sec_(TensorGrad &);
    static NEUROTENSOR_API TensorGrad& sech_(TensorGrad &);
    
    // unique.cpp
    static NEUROTENSOR_API TensorGrad unique(const TensorGrad&, std::optional<int64_t> dim = std::nullopt, bool return_sorted = true, bool return_indice = true);
}; // TensorGrad_Functional_Class 



} // namespace functional
} // namespace nt

#endif
