// this is a friend class to TensorGrad that holds the functional functions
// dedicated to tensorgrad this is meant to re-create the functional.h that is
// dedicated to the Tensor class
#ifndef _NT_TENSORGRAD_FUNCTIONAL_CLASS_H_
#define _NT_TENSORGRAD_FUNCTIONAL_CLASS_H_

namespace nt {
namespace functional {
class TensorGrad_Functional_Class; // forward declaration

} // namespace functional
} // namespace nt

#include "TensorGrad.h"
namespace nt {
namespace functional {
class TensorGrad_Functional_Class {
  public:
    TensorGrad_Functional_Class() = default;
    static TensorGrad matmult(const TensorGrad &, const TensorGrad &, bool,
                              bool);
    static TensorGrad matmult(const Tensor &, const TensorGrad &, bool, bool);
    static TensorGrad matmult(const TensorGrad &a, const Tensor &b, bool, bool);
    static TensorGrad linear(const TensorGrad&, const TensorGrad&, const TensorGrad&, bool, bool);
    static TensorGrad linear(const Tensor&, const TensorGrad&, const TensorGrad&, bool, bool);
    static TensorGrad linear(const TensorGrad&, const Tensor&, const TensorGrad&, bool, bool);
    static TensorGrad linear(const TensorGrad&, const TensorGrad&, const Tensor&, bool, bool);
    static TensorGrad linear(const TensorGrad&, const Tensor&, const Tensor&, bool, bool);
    static TensorGrad linear(const Tensor&, const TensorGrad&, const Tensor&, bool, bool);
    static TensorGrad linear(const Tensor&, const Tensor&, const TensorGrad&, bool, bool);

    static TensorGrad unfold1d(const TensorGrad &, Tensor::size_value_t,
                               Tensor::size_value_t, Tensor::size_value_t,
                               Tensor::size_value_t, bool);
    static TensorGrad unfold(const TensorGrad &, utils::my_tuple,
                             utils::my_tuple, utils::my_tuple, utils::my_tuple,
                             bool);
    static TensorGrad unfold3d(const TensorGrad &, utils::my_n_tuple<3>,
                               utils::my_n_tuple<3>, utils::my_n_tuple<3>,
                               utils::my_n_tuple<3>, bool);
    static TensorGrad fold(const TensorGrad &, utils::my_tuple, utils::my_tuple,
                           utils::my_tuple, utils::my_tuple, utils::my_tuple);
    // image, kernel, stride, padding, dilation, groups
    static TensorGrad conv1d(const TensorGrad &, const TensorGrad &, int64_t,
                             int64_t, int64_t, int64_t);
    static TensorGrad conv1d(const Tensor &, const TensorGrad &, int64_t,
                             int64_t, int64_t, int64_t);
    static TensorGrad conv1d(const TensorGrad &, const Tensor &, int64_t,
                             int64_t, int64_t, int64_t);
    static TensorGrad conv2d(const TensorGrad &, const TensorGrad &,
                             utils::my_tuple, utils::my_tuple, utils::my_tuple,
                             int64_t);
    static TensorGrad conv2d(const Tensor &, const TensorGrad &,
                             utils::my_tuple, utils::my_tuple, utils::my_tuple,
                             int64_t);
    static TensorGrad conv2d(const TensorGrad &, const Tensor &,
                             utils::my_tuple, utils::my_tuple, utils::my_tuple,
                             int64_t);
    static TensorGrad conv3d(const TensorGrad &, const TensorGrad &,
                             utils::my_n_tuple<3>, utils::my_n_tuple<3>,
                             utils::my_n_tuple<3>, int64_t);
    static TensorGrad conv3d(const Tensor &, const TensorGrad &,
                             utils::my_n_tuple<3>, utils::my_n_tuple<3>,
                             utils::my_n_tuple<3>, int64_t);
    static TensorGrad conv3d(const TensorGrad &, const Tensor &,
                             utils::my_n_tuple<3>, utils::my_n_tuple<3>,
                             utils::my_n_tuple<3>, int64_t);

    static TensorGrad conv_transpose1d(const TensorGrad &, const TensorGrad &,
                                       int64_t, int64_t, int64_t, int64_t,
                                       int64_t);
    static TensorGrad conv_transpose1d(const Tensor &, const TensorGrad &,
                                       int64_t, int64_t, int64_t, int64_t,
                                       int64_t);
    static TensorGrad conv_transpose1d(const TensorGrad &, const Tensor &,
                                       int64_t, int64_t, int64_t, int64_t,
                                       int64_t);
    static TensorGrad conv_transpose2d(const TensorGrad &, const TensorGrad &,
                                       utils::my_tuple, utils::my_tuple,
                                       utils::my_tuple, utils::my_tuple,
                                       int64_t);
    static TensorGrad conv_transpose2d(const Tensor &, const TensorGrad &,
                                       utils::my_tuple, utils::my_tuple,
                                       utils::my_tuple, utils::my_tuple,
                                       int64_t);
    static TensorGrad conv_transpose2d(const TensorGrad &, const Tensor &,
                                       utils::my_tuple, utils::my_tuple,
                                       utils::my_tuple, utils::my_tuple,
                                       int64_t);
    static TensorGrad conv_transpose3d(const TensorGrad &, const TensorGrad &,
                                       utils::my_n_tuple<3>,
                                       utils::my_n_tuple<3>,
                                       utils::my_n_tuple<3>,
                                       utils::my_n_tuple<3>, int64_t);
    static TensorGrad conv_transpose3d(const Tensor &, const TensorGrad &,
                                       utils::my_n_tuple<3>,
                                       utils::my_n_tuple<3>,
                                       utils::my_n_tuple<3>,
                                       utils::my_n_tuple<3>, int64_t);
    static TensorGrad conv_transpose3d(const TensorGrad &, const Tensor &,
                                       utils::my_n_tuple<3>,
                                       utils::my_n_tuple<3>,
                                       utils::my_n_tuple<3>,
                                       utils::my_n_tuple<3>, int64_t);

    static TensorGrad sigmoid(const TensorGrad &);
    static TensorGrad clamp(const TensorGrad &, std::optional<int64_t>,
                            std::optional<int64_t>);
    static TensorGrad relu(const TensorGrad &);
    static TensorGrad var(const TensorGrad &, utils::optional_list, int64_t,
                          bool);
    static TensorGrad sqrt(const TensorGrad &);
    static TensorGrad invsqrt(const TensorGrad &);
    static TensorGrad silu(const TensorGrad &);
    static TensorGrad gelu(const TensorGrad &);
    static TensorGrad tanh(const TensorGrad &);
    static TensorGrad tan(const TensorGrad &);
    static TensorGrad cosh(const TensorGrad &);
    static TensorGrad cos(const TensorGrad &);
    static TensorGrad sinh(const TensorGrad &);
    static TensorGrad sin(const TensorGrad &);
    static TensorGrad atanh(const TensorGrad &);
    static TensorGrad atan(const TensorGrad &);
    static TensorGrad acosh(const TensorGrad &);
    static TensorGrad acos(const TensorGrad &);
    static TensorGrad asinh(const TensorGrad &);
    static TensorGrad asin(const TensorGrad &);
    static TensorGrad cotanh(const TensorGrad &);
    static TensorGrad cotan(const TensorGrad &);
    static TensorGrad sech(const TensorGrad &);
    static TensorGrad sec(const TensorGrad &);
    static TensorGrad csch(const TensorGrad &);
    static TensorGrad csc(const TensorGrad &);
    static TensorGrad cat(std::vector<TensorGrad>, int64_t);
    static TensorGrad cat(TensorGrad, int64_t);
    static TensorGrad stack(TensorGrad, int64_t);
    static TensorGrad stack(std::vector<TensorGrad>, int64_t);
    static TensorGrad chunk(TensorGrad, typename Tensor::size_value_t,
                            int64_t); // splits into that many chunks
    static TensorGrad split(TensorGrad input,
                            typename Tensor::size_value_t split_size, int64_t);
    static TensorGrad
    split(TensorGrad input,
          std::vector<typename Tensor::size_value_t> split_sections, int64_t);
    static TensorGrad log(const TensorGrad &);
    static TensorGrad logsumexp(const TensorGrad &,
                                utils::optional_list list = nullptr,
                                bool keepdim = true);

    static TensorGrad dropout(const TensorGrad &, double);
    static TensorGrad abs(const TensorGrad &);
    static TensorGrad softplus(const TensorGrad &, Scalar, Scalar);
    static TensorGrad softmax(const TensorGrad &, bool stable);
    static TensorGrad softmax(const TensorGrad &,
                              typename SizeRef::value_type dim, bool stable);
    static TensorGrad gumbel_softmax(const TensorGrad &, Scalar tau, bool hard,
                                     bool stable);
    static TensorGrad symmetric_bilinear(const TensorGrad&, const TensorGrad&, const TensorGrad&);
    static TensorGrad symmetric_bilinear(const TensorGrad&, const TensorGrad&, const Tensor&);
    static TensorGrad symmetric_bilinear(const TensorGrad&, const Tensor&, const TensorGrad&);
    static TensorGrad symmetric_bilinear(const Tensor&, const TensorGrad&, const TensorGrad&);
    static TensorGrad symmetric_bilinear(const TensorGrad&, const Tensor&, const Tensor&);
    static TensorGrad symmetric_bilinear(const Tensor&, const Tensor&, const TensorGrad&);
    static TensorGrad symmetric_bilinear(const Tensor&, const TensorGrad&, const Tensor&);
    static TensorGrad avg_pool1d(TensorGrad input, int64_t kernel_size, int64_t stride, int64_t padding, bool ceil_mode, bool count_include_pad);
    static TensorGrad adaptive_avg_pool1d(TensorGrad x, int64_t l_out);

    static TensorGrad avg_pool2d(TensorGrad input, utils::my_tuple kernel_size, utils::my_tuple stride, utils::my_tuple padding, 
                          bool ceil_mode, bool count_include_pad);
    static TensorGrad adaptive_avg_pool2d(TensorGrad x, utils::my_tuple out_shape);

    static TensorGrad avg_pool3d(TensorGrad input, utils::my_n_tuple<3> kernel_size, utils::my_n_tuple<3> stride, utils::my_n_tuple<3> padding, 
                          bool ceil_mode, bool count_include_pad);
    static TensorGrad adaptive_avg_pool3d(TensorGrad x, utils::my_n_tuple<3> out_shape);

    //LP Pooling
    static TensorGrad lp_pool1d(TensorGrad input, Scalar power, int64_t kernel_size, int64_t stride, bool ceil_mode);
    static TensorGrad adaptive_lp_pool1d(TensorGrad x, int64_t l_out, Scalar power);

    static TensorGrad lp_pool2d(TensorGrad input, Scalar power, utils::my_tuple kernel_size, utils::my_tuple stride, bool ceil_mode);
    static TensorGrad adaptive_lp_pool2d(TensorGrad x, utils::my_tuple out_shape, Scalar power);

    static TensorGrad lp_pool3d(TensorGrad input, Scalar power, utils::my_n_tuple<3> kernel_size, utils::my_n_tuple<3> stride, bool ceil_mode);
    static TensorGrad adaptive_lp_pool3d(TensorGrad x, utils::my_n_tuple<3> out_shape, Scalar power);


    //Max Pooling
    static TensorGrad max_pool1d(TensorGrad input, int64_t kernel_size, int64_t stride, int64_t padding, int64_t dilation, bool ceil_mode, bool return_indices);
    static TensorGrad max_unpool1d(TensorGrad input, Tensor indices, int64_t kernel_size, int64_t stride, int64_t padding, int64_t output_size);
    static TensorGrad max_unpool1d(TensorGrad input, TensorGrad indices, int64_t kernel_size, int64_t stride, int64_t padding, int64_t output_size);
    static TensorGrad adaptive_max_pool1d(TensorGrad x, int64_t l_out, bool return_indices);

    static TensorGrad max_pool2d(TensorGrad input, 
                          utils::my_tuple kernel_size, utils::my_tuple stride, utils::my_tuple padding,
                          utils::my_tuple dilation, bool ceil_mode, bool return_indices);
    static TensorGrad max_unpool2d(TensorGrad input, Tensor indices, utils::my_tuple kernel_size, utils::my_tuple stride, utils::my_tuple padding, utils::my_tuple output_size);
    static TensorGrad max_unpool2d(TensorGrad input, TensorGrad indices, utils::my_tuple kernel_size, utils::my_tuple stride, utils::my_tuple padding, utils::my_tuple output_size);
    static TensorGrad adaptive_max_pool2d(TensorGrad x, utils::my_tuple out_shape, bool return_indices);

    static TensorGrad max_pool3d(TensorGrad input, 
                          utils::my_n_tuple<3> kernel_size, utils::my_n_tuple<3> stride, utils::my_n_tuple<3> padding,
                          utils::my_n_tuple<3> dilation, bool ceil_mode, bool return_indices);
    static TensorGrad max_unpool3d(TensorGrad input, Tensor indices, utils::my_n_tuple<3> kernel_size, utils::my_n_tuple<3> stride, utils::my_n_tuple<3> padding, utils::my_n_tuple<3> output_size);
    static TensorGrad max_unpool3d(TensorGrad input, TensorGrad indices, utils::my_n_tuple<3> kernel_size, utils::my_n_tuple<3> stride, utils::my_n_tuple<3> padding, utils::my_n_tuple<3> output_size);
    static TensorGrad adaptive_max_pool3d(TensorGrad x, utils::my_n_tuple<3> out_shape, bool return_indices);

    //Fractional Max Pooling
    static TensorGrad fractional_max_pool2d(TensorGrad input, utils::my_tuple kernel_size, utils::my_tuple output_size, 
                                     std::variant<double, std::tuple<double, double>> output_ratio, bool return_indices);
    static TensorGrad fractional_max_pool3d(TensorGrad input, utils::my_n_tuple<3> kernel_size, utils::my_n_tuple<3> output_size, 
                                     std::variant<double, std::tuple<double, double, double>> output_ratio, bool return_indices);


}; // TensorGrad_Functional_Class

} // namespace functional
} // namespace nt

#endif
