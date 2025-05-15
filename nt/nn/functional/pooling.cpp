#include "../../Tensor.h"
#include "../../dtype/ArrayVoid.hpp"
#include "../../functional/functional.h"
#include "../../functional/tensor_files/exceptions.hpp"
#include "../../intrusive_ptr/intrusive_ptr.hpp"
#include "../TensorGrad.h"
#include "../functional_class.h"
#include "functional_list.h"

namespace nt {
namespace functional {

inline int64_t find_pooling_kernel_size(const int64_t &output_size,
                                        const int64_t &input_size,
                                        const int64_t &stride,
                                        const int64_t &padding) noexcept {

    return -((output_size - 1) * stride - (2 * padding) - input_size);
}

inline void find_adaptive(int64_t output_size, int64_t input_size,
                          int64_t &kernel_size, int64_t &stride,
                          int64_t &padding) {
    stride = input_size / output_size;
    padding = 0;
    kernel_size =
        find_pooling_kernel_size(output_size, input_size, stride, padding);
}

TensorGrad TensorGrad_Functional_Class::avg_pool1d(
    TensorGrad input, int64_t kernel_size, int64_t stride, int64_t padding,
    bool ceil_mode, bool count_include_pad) {
    if (!input.do_track_grad) {
        Tensor out =
            ::nt::functional::avg_pool1d(input.tensor, kernel_size, stride,
                                         padding, ceil_mode, count_include_pad);
        TensorGrad result(std::move(out), false);
        result.do_track_grad = false;
        return std::move(result);
    }

    SizeRef in_shape = input.shape().clone();
    TensorGrad result(::nt::functional::avg_pool1d(input.tensor, kernel_size,
                                                   stride, padding, ceil_mode,
                                                   count_include_pad),
                      true);
    result.track_tensors(input);
    result.create_backward_function(
        [in_shape, kernel_size, stride, padding, ceil_mode,
         count_include_pad](
            const Tensor &grad,
            std::vector<intrusive_ptr<TensorGrad>> &parents) {
            Tensor n_grad = ::nt::functional::backward_avg_pool1d(
                in_shape, grad, kernel_size, stride, padding, ceil_mode,
                count_include_pad);
            parents[0]->grad->tensor += n_grad;
        });
    return std::move(result);
}
TensorGrad TensorGrad_Functional_Class::adaptive_avg_pool1d(TensorGrad x,
                                                            int64_t l_out) {
    utils::throw_exception(
        l_out <= x.shape()[-1],
        "Expected the output from adaptive pooling ($) to be less than or "
        "equal to the specified input ($) at the dimension",
        l_out, x.shape()[-1]);
    utils::throw_exception(
        l_out != 0, "Cannot find adaptive for an output of 0 at any dimension");
    if (l_out == 1) {
        return x.mean(-1);
    }
    if (l_out == x.shape()[-1])
        return x;
    int64_t kernel_size, stride, padding;
    find_adaptive(l_out, x.shape()[-1], kernel_size, stride, padding);
    return avg_pool1d(x, kernel_size, stride, padding, false, true);
}

TensorGrad TensorGrad_Functional_Class::avg_pool2d(
    TensorGrad input, utils::my_tuple kernel_size, utils::my_tuple stride,
    utils::my_tuple padding, bool ceil_mode, bool count_include_pad) {
    if (!input.do_track_grad) {
        Tensor out =
            ::nt::functional::avg_pool2d(input.tensor, kernel_size, stride,
                                         padding, ceil_mode, count_include_pad);
        TensorGrad result(std::move(out), false);
        result.do_track_grad = false;
        return std::move(result);
    }

    SizeRef in_shape = input.shape().clone();
    TensorGrad result(::nt::functional::avg_pool2d(input.tensor, kernel_size,
                                                   stride, padding, ceil_mode,
                                                   count_include_pad),
                      true);
    result.track_tensors(input);
    result.create_backward_function(
        [in_shape, kernel_size, stride, padding, ceil_mode,
         count_include_pad](
            const Tensor &grad,
            std::vector<intrusive_ptr<TensorGrad>> &parents) {
            Tensor n_grad = ::nt::functional::backward_avg_pool2d(
                in_shape, grad, kernel_size, stride, padding, ceil_mode,
                count_include_pad);
            parents[0]->grad->tensor += n_grad;
        });
    return std::move(result);
}

TensorGrad TensorGrad_Functional_Class::adaptive_avg_pool2d(TensorGrad x,
                               utils::my_tuple out_shape) {
    int64_t c_out = out_shape[1];
    int64_t r_out = out_shape[0];
    utils::throw_exception(x.dims() >= 2,
                           "Expected dimensions of input for adapting average "
                           "2d pooling to be less than or equal to 2 got $",
                           x.dims());
    utils::throw_exception(
        c_out <= x.shape()[-1],
        "Expected the output from adaptive pooling ($) to be less than or "
        "equal to the specified input ($) at the dimension",
        c_out, x.shape()[-1]);
    utils::throw_exception(
        c_out != 0, "Cannot find adaptive for an output of 0 at any dimension");
    utils::throw_exception(
        r_out <= x.shape()[-2],
        "Expected the output from adaptive pooling ($) to be less than or "
        "equal to the specified input ($) at the dimension",
        r_out, x.shape()[-2]);
    utils::throw_exception(
        r_out != 0, "Cannot find adaptive for an output of 0 at any dimension");

    int64_t kernel_size_r, stride_r, padding_r;
    int64_t kernel_size_c, stride_c, padding_c;
    find_adaptive(r_out, x.shape()[-2], kernel_size_r, stride_r, padding_r);
    find_adaptive(c_out, x.shape()[-1], kernel_size_c, stride_c, padding_c);
    return avg_pool2d(x, {kernel_size_r, kernel_size_c}, {stride_r, stride_c},
                      {padding_r, padding_c}, false, true);
}

TensorGrad TensorGrad_Functional_Class::avg_pool3d(
    TensorGrad input, utils::my_n_tuple<3> kernel_size,
    utils::my_n_tuple<3> stride, utils::my_n_tuple<3> padding, bool ceil_mode,
    bool count_include_pad) {
    if (!input.do_track_grad) {
        Tensor out =
            ::nt::functional::avg_pool3d(input.tensor, kernel_size, stride,
                                         padding, ceil_mode, count_include_pad);
        TensorGrad result(std::move(out), false);
        result.do_track_grad = false;
        return std::move(result);
    }

    SizeRef in_shape = input.shape().clone();
    TensorGrad result(::nt::functional::avg_pool3d(input.tensor, kernel_size,
                                                   stride, padding, ceil_mode,
                                                   count_include_pad),
                      true);
    result.track_tensors(input);
    result.create_backward_function(
        [in_shape, kernel_size, stride, padding, ceil_mode,
         count_include_pad](
            const Tensor &grad,
            std::vector<intrusive_ptr<TensorGrad>> &parents) {
            Tensor n_grad = ::nt::functional::backward_avg_pool3d(
                in_shape, grad, kernel_size, stride, padding, ceil_mode,
                count_include_pad);
            parents[0]->grad->tensor += n_grad;
        });
    return std::move(result);
}
TensorGrad TensorGrad_Functional_Class::adaptive_avg_pool3d(
    TensorGrad x, utils::my_n_tuple<3> out_shape) {
    int64_t d_out = out_shape[0];
    int64_t c_out = out_shape[2];
    int64_t r_out = out_shape[1];
    utils::throw_exception(x.dims() >= 3,
                           "Expected dimensions of input for adapting average "
                           "2d pooling to be less than or equal to 2 got $",
                           x.dims());
    utils::throw_exception(
        c_out <= x.shape()[-1],
        "Expected the output from adaptive pooling ($) to be less than or "
        "equal to the specified input ($) at the dimension",
        c_out, x.shape()[-1]);
    utils::throw_exception(
        c_out != 0, "Cannot find adaptive for an output of 0 at any dimension");
    utils::throw_exception(
        r_out <= x.shape()[-2],
        "Expected the output from adaptive pooling ($) to be less than or "
        "equal to the specified input ($) at the dimension",
        r_out, x.shape()[-2]);
    utils::throw_exception(
        r_out != 0, "Cannot find adaptive for an output of 0 at any dimension");
    utils::throw_exception(
        d_out <= x.shape()[-3],
        "Expected the output from adaptive pooling ($) to be less than or "
        "equal to the specified input ($) at the dimension",
        d_out, x.shape()[-3]);
    utils::throw_exception(
        d_out != 0, "Cannot find adaptive for an output of 0 at any dimension");

    int64_t kernel_size_d, stride_d, padding_d;
    int64_t kernel_size_r, stride_r, padding_r;
    int64_t kernel_size_c, stride_c, padding_c;
    find_adaptive(d_out, x.shape()[-3], kernel_size_d, stride_d, padding_d);
    find_adaptive(r_out, x.shape()[-2], kernel_size_r, stride_r, padding_r);
    find_adaptive(c_out, x.shape()[-1], kernel_size_c, stride_c, padding_c);
    return avg_pool3d(x, {kernel_size_d, kernel_size_r, kernel_size_c},
                      {stride_d, stride_r, stride_c},
                      {padding_d, padding_r, padding_c}, false, true);
}

// LP Pooling
TensorGrad TensorGrad_Functional_Class::lp_pool1d(TensorGrad input,
                                                  Scalar power,
                                                  int64_t kernel_size,
                                                  int64_t stride,
                                                  bool ceil_mode) {
    if (!input.do_track_grad) {
        Tensor out = ::nt::functional::lp_pool1d(
            input.tensor, power, kernel_size, stride, ceil_mode);
        TensorGrad result(std::move(out), false);
        result.do_track_grad = false;
        return std::move(result);
    }

    intrusive_ptr<tensor_holder> cpy =
        make_intrusive<tensor_holder>(input.tensor.conditional_mutate_clone());
    TensorGrad result(::nt::functional::lp_pool1d(
                          input.tensor, power, kernel_size, stride, ceil_mode),
                      true);
    result.track_tensors(input);
    result.create_backward_function(
        [power, kernel_size, stride, ceil_mode](
            const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents,
            intrusive_ptr<tensor_holder> cpy) {
            Tensor n_grad = ::nt::functional::backward_lp_pool1d(
                cpy->tensor, grad, power, kernel_size, stride, ceil_mode);
            parents[0]->grad->tensor += n_grad;
        },
        cpy);
    return std::move(result);
}
TensorGrad TensorGrad_Functional_Class::adaptive_lp_pool1d(TensorGrad x,
                                                           int64_t l_out,
                                                           Scalar power) {
    utils::throw_exception(
        l_out <= x.shape()[-1],
        "Expected the output from adaptive pooling ($) to be less than or "
        "equal to the specified input ($) at the dimension",
        l_out, x.shape()[-1]);
    utils::throw_exception(
        l_out != 0, "Cannot find adaptive for an output of 0 at any dimension");
    if (l_out == 1) {
        return x.pow(power).sum(-1).pow(power.inverse());
    }
    if (l_out == x.shape()[-1])
        return x;
    int64_t kernel_size, stride, padding;
    find_adaptive(l_out, x.shape()[-1], kernel_size, stride, padding);
    return lp_pool1d(x, power, kernel_size, stride, false);
}

TensorGrad TensorGrad_Functional_Class::lp_pool2d(TensorGrad input,
                                                  Scalar power,
                                                  utils::my_tuple kernel_size,
                                                  utils::my_tuple stride,
                                                  bool ceil_mode) {
    if (!input.do_track_grad) {
        Tensor out = ::nt::functional::lp_pool2d(
            input.tensor, power, kernel_size, stride, ceil_mode);
        TensorGrad result(std::move(out), false);
        result.do_track_grad = false;
        return std::move(result);
    }

    intrusive_ptr<tensor_holder> cpy =
        make_intrusive<tensor_holder>(input.tensor.conditional_mutate_clone());
    TensorGrad result(::nt::functional::lp_pool2d(
                          input.tensor, power, kernel_size, stride, ceil_mode),
                      true);
    result.track_tensors(input);
    result.create_backward_function(
        [power, kernel_size, stride, ceil_mode](
            const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents,
            intrusive_ptr<tensor_holder> cpy) {
            Tensor n_grad = ::nt::functional::backward_lp_pool2d(
                cpy->tensor, grad, power, kernel_size, stride, ceil_mode);
            parents[0]->grad->tensor += n_grad;
        },
        cpy);
    return std::move(result);
}
TensorGrad TensorGrad_Functional_Class::adaptive_lp_pool2d(
    TensorGrad x, utils::my_tuple out_shape, Scalar power) {
    int64_t c_out = out_shape[1];
    int64_t r_out = out_shape[0];
    utils::throw_exception(x.dims() >= 2,
                           "Expected dimensions of input for adapting average "
                           "2d pooling to be less than or equal to 2 got $",
                           x.dims());
    utils::throw_exception(
        c_out <= x.shape()[-1],
        "Expected the output from adaptive pooling ($) to be less than or "
        "equal to the specified input ($) at the dimension",
        c_out, x.shape()[-1]);
    utils::throw_exception(
        c_out != 0, "Cannot find adaptive for an output of 0 at any dimension");
    utils::throw_exception(
        r_out <= x.shape()[-2],
        "Expected the output from adaptive pooling ($) to be less than or "
        "equal to the specified input ($) at the dimension",
        r_out, x.shape()[-2]);
    utils::throw_exception(
        r_out != 0, "Cannot find adaptive for an output of 0 at any dimension");

    int64_t kernel_size_r, stride_r, padding_r;
    int64_t kernel_size_c, stride_c, padding_c;
    find_adaptive(r_out, x.shape()[-2], kernel_size_r, stride_r, padding_r);
    find_adaptive(c_out, x.shape()[-1], kernel_size_c, stride_c, padding_c);
    return lp_pool2d(x, power, {kernel_size_r, kernel_size_c},
                     {stride_r, stride_c}, false);
}

TensorGrad TensorGrad_Functional_Class::lp_pool3d(
    TensorGrad input, Scalar power, utils::my_n_tuple<3> kernel_size,
    utils::my_n_tuple<3> stride, bool ceil_mode) {
    if (!input.do_track_grad) {
        Tensor out = ::nt::functional::lp_pool3d(
            input.tensor, power, kernel_size, stride, ceil_mode);
        TensorGrad result(std::move(out), false);
        result.do_track_grad = false;
        return std::move(result);
    }

    intrusive_ptr<tensor_holder> cpy =
        make_intrusive<tensor_holder>(input.tensor.conditional_mutate_clone());
    TensorGrad result(::nt::functional::lp_pool3d(
                          input.tensor, power, kernel_size, stride, ceil_mode),
                      true);
    result.track_tensors(input);
    result.create_backward_function(
        [power, kernel_size, stride, ceil_mode](
            const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents,
            intrusive_ptr<tensor_holder> cpy) {
            Tensor n_grad = ::nt::functional::backward_lp_pool3d(
                cpy->tensor, grad, power, kernel_size, stride, ceil_mode);
            parents[0]->grad->tensor += n_grad;
        },
        cpy);
    return std::move(result);
}
TensorGrad TensorGrad_Functional_Class::adaptive_lp_pool3d(
    TensorGrad x, utils::my_n_tuple<3> out_shape, Scalar power) {
    int64_t c_out = out_shape[2];
    int64_t r_out = out_shape[1];
    int64_t d_out = out_shape[0];
    utils::throw_exception(x.dims() >= 2,
                           "Expected dimensions of input for adapting average "
                           "2d pooling to be less than or equal to 2 got $",
                           x.dims());
    utils::throw_exception(
        c_out <= x.shape()[-1],
        "Expected the output from adaptive pooling ($) to be less than or "
        "equal to the specified input ($) at the dimension",
        c_out, x.shape()[-1]);
    utils::throw_exception(
        c_out != 0, "Cannot find adaptive for an output of 0 at any dimension");
    utils::throw_exception(
        r_out <= x.shape()[-2],
        "Expected the output from adaptive pooling ($) to be less than or "
        "equal to the specified input ($) at the dimension",
        r_out, x.shape()[-2]);
    utils::throw_exception(
        r_out != 0, "Cannot find adaptive for an output of 0 at any dimension");
    utils::throw_exception(
        d_out <= x.shape()[-3],
        "Expected the output from adaptive pooling ($) to be less than or "
        "equal to the specified input ($) at the dimension",
        d_out, x.shape()[-3]);
    utils::throw_exception(
        d_out != 0, "Cannot find adaptive for an output of 0 at any dimension");

    int64_t kernel_size_d, stride_d, padding_d;
    int64_t kernel_size_r, stride_r, padding_r;
    int64_t kernel_size_c, stride_c, padding_c;

    find_adaptive(d_out, x.shape()[-3], kernel_size_d, stride_d, padding_d);
    find_adaptive(r_out, x.shape()[-2], kernel_size_r, stride_r, padding_r);
    find_adaptive(c_out, x.shape()[-1], kernel_size_c, stride_c, padding_c);
    return lp_pool3d(x, power, {kernel_size_d, kernel_size_r, kernel_size_c},
                     {stride_d, stride_r, stride_c}, false);
}

// Max Pooling
TensorGrad TensorGrad_Functional_Class::max_pool1d(
    TensorGrad input, int64_t kernel_size, int64_t stride, int64_t padding,
    int64_t dilation, bool ceil_mode, bool return_indices) {
    if (!input.do_track_grad) {
        Tensor out = ::nt::functional::max_pool1d(input.tensor, kernel_size,
                                                  stride, padding, dilation,
                                                  ceil_mode, return_indices);
        TensorGrad result(std::move(out), false);
        result.do_track_grad = false;
        return std::move(result);
    }

    auto [output, indices] = get<2>(::nt::functional::max_pool1d(
        input.tensor, kernel_size, stride, padding, dilation, ceil_mode, true));
    SizeRef in_shape = input.shape().clone();
    intrusive_ptr<tensor_holder> cpy =
        make_intrusive<tensor_holder>(indices.clone());
    TensorGrad result(output, true);
    result.track_tensors(input);
    result.create_backward_function(
        [in_shape](const Tensor &grad,
                   std::vector<intrusive_ptr<TensorGrad>> &parents,
                   intrusive_ptr<tensor_holder> indices) {
            Tensor n_grad = ::nt::functional::backward_max_pool1d(
                in_shape, grad, indices->tensor);
            parents[0]->grad->tensor += n_grad;
        },
        cpy);
    if (!return_indices)
        return std::move(result);
    return ::nt::functional::list(result, TensorGrad(indices, true));
}

TensorGrad TensorGrad_Functional_Class::max_unpool1d(
    TensorGrad input, Tensor indices, int64_t kernel_size, int64_t stride,
    int64_t padding, int64_t output_size) {
    if (!input.do_track_grad) {
        Tensor out = ::nt::functional::max_unpool1d(
            input.tensor, indices, kernel_size, stride, padding, output_size);
        TensorGrad result(std::move(out), false);
        result.do_track_grad = false;
        return std::move(result);
    }

    intrusive_ptr<tensor_holder> cpy = make_intrusive<tensor_holder>(indices);
    SizeRef in_shape = input.shape().clone();
    TensorGrad result(::nt::functional::max_unpool1d(input.tensor, indices,
                                                     kernel_size, stride,
                                                     padding, output_size),
                      true);
    result.track_tensors(input);
    result.create_backward_function(
        [in_shape, padding](const Tensor &grad,
                            std::vector<intrusive_ptr<TensorGrad>> &parents,
                            intrusive_ptr<tensor_holder> indices) {
            Tensor n_grad = ::nt::functional::backward_max_unpool1d(
                in_shape, grad, indices->tensor, padding);
            parents[0]->grad->tensor += n_grad;
        },
        cpy);
    return std::move(result);
}

TensorGrad TensorGrad_Functional_Class::max_unpool1d(
    TensorGrad input, TensorGrad _indices, int64_t kernel_size, int64_t stride,
    int64_t padding, int64_t output_size) {
    return max_unpool1d(input, _indices.tensor, kernel_size, stride, padding, output_size);
}
TensorGrad
TensorGrad_Functional_Class::adaptive_max_pool1d(TensorGrad x, int64_t l_out,
                                                 bool return_indices) {
    utils::throw_exception(
        l_out <= x.shape()[-1],
        "Expected the output from adaptive pooling ($) to be less than or "
        "equal to the specified input ($) at the dimension",
        l_out, x.shape()[-1]);
    utils::throw_exception(
        l_out != 0, "Cannot find adaptive for an output of 0 at any dimension");
    if (l_out == 1) {
        if (return_indices) {
            auto out = x.max(-1);
            return functional::list(out.values, TensorGrad(out.indices, true));
        }
        return x.max(-1).values;
    }
    if (l_out == x.shape()[-1])
        return x;
    int64_t kernel_size, stride, padding;
    find_adaptive(l_out, x.shape()[-1], kernel_size, stride, padding);
    return max_pool1d(x, kernel_size, stride, padding, 1, false,
                      return_indices);
}

TensorGrad TensorGrad_Functional_Class::max_pool2d(
    TensorGrad input, utils::my_tuple kernel_size, utils::my_tuple stride,
    utils::my_tuple padding, utils::my_tuple dilation, bool ceil_mode,
    bool return_indices) {
    if (!input.do_track_grad) {
        Tensor out = ::nt::functional::max_pool2d(input.tensor, kernel_size,
                                                  stride, padding, dilation,
                                                  ceil_mode, return_indices);
        TensorGrad result(std::move(out), false);
        result.do_track_grad = false;
        return std::move(result);
    }

    auto [output, indices] = get<2>(::nt::functional::max_pool2d(
        input.tensor, kernel_size, stride, padding, dilation, ceil_mode, true));
    SizeRef in_shape = input.shape().clone();
    intrusive_ptr<tensor_holder> cpy =
        make_intrusive<tensor_holder>(indices.clone());
    TensorGrad result(output, true);
    result.track_tensors(input);
    result.create_backward_function(
        [in_shape](const Tensor &grad,
                   std::vector<intrusive_ptr<TensorGrad>> &parents,
                   intrusive_ptr<tensor_holder> indices) {
            Tensor n_grad = ::nt::functional::backward_max_pool2d(
                in_shape, grad, indices->tensor);
            parents[0]->grad->tensor += n_grad;
        },
        cpy);
    if (!return_indices)
        return std::move(result);
    return ::nt::functional::list(result, TensorGrad(indices, true));
}

TensorGrad TensorGrad_Functional_Class::max_unpool2d(
    TensorGrad input, Tensor indices, utils::my_tuple kernel_size,
    utils::my_tuple stride, utils::my_tuple padding,
    utils::my_tuple output_size) {
    if (!input.do_track_grad) {
        Tensor out = ::nt::functional::max_unpool2d(
            input.tensor, indices, kernel_size, stride, padding, output_size);
        TensorGrad result(std::move(out), false);
        result.do_track_grad = false;
        return std::move(result);
    }

    intrusive_ptr<tensor_holder> cpy = make_intrusive<tensor_holder>(indices);
    SizeRef in_shape = input.shape().clone();
    TensorGrad result(::nt::functional::max_unpool2d(input.tensor, indices,
                                                     kernel_size, stride,
                                                     padding, output_size),
                      true);
    result.track_tensors(input);
    result.create_backward_function(
        [in_shape, padding](const Tensor &grad,
                            std::vector<intrusive_ptr<TensorGrad>> &parents,
                            intrusive_ptr<tensor_holder> indices) {
            Tensor n_grad = ::nt::functional::backward_max_unpool2d(
                in_shape, grad, indices->tensor, padding);
            parents[0]->grad->tensor += n_grad;
        },
        cpy);
    return std::move(result);
}
TensorGrad TensorGrad_Functional_Class::max_unpool2d(
    TensorGrad input, TensorGrad _indices, utils::my_tuple kernel_size,
    utils::my_tuple stride, utils::my_tuple padding,
    utils::my_tuple output_size) {
    return max_unpool2d(input, _indices.tensor, kernel_size, stride, padding,
                        output_size);
}

TensorGrad TensorGrad_Functional_Class::adaptive_max_pool2d(
    TensorGrad x, utils::my_tuple out_shape, bool return_indices) {
    int64_t r_out = out_shape[0];
    int64_t c_out = out_shape[1];
    utils::throw_exception(
        c_out <= x.shape()[-1],
        "Expected the output from adaptive pooling ($) to be less than or "
        "equal to the specified input ($) at the dimension",
        c_out, x.shape()[-1]);
    utils::throw_exception(
        c_out != 0, "Cannot find adaptive for an output of 0 at any dimension");
    utils::throw_exception(
        r_out <= x.shape()[-2],
        "Expected the output from adaptive pooling ($) to be less than or "
        "equal to the specified input ($) at the dimension",
        r_out, x.shape()[-2]);
    utils::throw_exception(
        r_out != 0, "Cannot find adaptive for an output of 0 at any dimension");
    int64_t kernel_size_c, stride_c, padding_c;
    int64_t kernel_size_r, stride_r, padding_r;
    find_adaptive(c_out, x.shape()[-1], kernel_size_c, stride_c, padding_c);
    find_adaptive(r_out, x.shape()[-2], kernel_size_r, stride_r, padding_r);
    return max_pool2d(x, {kernel_size_r, kernel_size_c}, {stride_r, stride_c},
                      {padding_r, padding_c}, 1, false, return_indices);
}

TensorGrad TensorGrad_Functional_Class::max_pool3d(
    TensorGrad input, utils::my_n_tuple<3> kernel_size,
    utils::my_n_tuple<3> stride, utils::my_n_tuple<3> padding,
    utils::my_n_tuple<3> dilation, bool ceil_mode, bool return_indices) {
    if (!input.do_track_grad) {
        Tensor out = ::nt::functional::max_pool3d(input.tensor, kernel_size,
                                                  stride, padding, dilation,
                                                  ceil_mode, return_indices);
        TensorGrad result(std::move(out), false);
        result.do_track_grad = false;
        return std::move(result);
    }

    auto [output, indices] = get<2>(::nt::functional::max_pool3d(
        input.tensor, kernel_size, stride, padding, dilation, ceil_mode, true));
    SizeRef in_shape = input.shape().clone();
    intrusive_ptr<tensor_holder> cpy =
        make_intrusive<tensor_holder>(indices.clone());
    TensorGrad result(output, true);
    result.track_tensors(input);
    result.create_backward_function(
        [in_shape](const Tensor &grad,
                   std::vector<intrusive_ptr<TensorGrad>> &parents,
                   intrusive_ptr<tensor_holder> indices) {
            Tensor n_grad = ::nt::functional::backward_max_pool3d(
                in_shape, grad, indices->tensor);
            parents[0]->grad->tensor += n_grad;
        },
        cpy);
    if (!return_indices)
        return std::move(result);
    return ::nt::functional::list(result, TensorGrad(indices, true));
}

TensorGrad TensorGrad_Functional_Class::max_unpool3d(
    TensorGrad input, Tensor indices, utils::my_n_tuple<3> kernel_size,
    utils::my_n_tuple<3> stride, utils::my_n_tuple<3> padding,
    utils::my_n_tuple<3> output_size) {
    if (!input.do_track_grad) {
        Tensor out = ::nt::functional::max_unpool3d(
            input.tensor, indices, kernel_size, stride, padding, output_size);
        TensorGrad result(std::move(out), false);
        result.do_track_grad = false;
        return std::move(result);
    }

    intrusive_ptr<tensor_holder> cpy = make_intrusive<tensor_holder>(indices);
    SizeRef in_shape = input.shape().clone();
    TensorGrad result(::nt::functional::max_unpool3d(input.tensor, indices,
                                                     kernel_size, stride,
                                                     padding, output_size),
                      true);
    result.track_tensors(input);
    result.create_backward_function(
        [in_shape, padding](const Tensor &grad,
                            std::vector<intrusive_ptr<TensorGrad>> &parents,
                            intrusive_ptr<tensor_holder> indices) {
            Tensor n_grad = ::nt::functional::backward_max_unpool3d(
                in_shape, grad, indices->tensor, padding);
            parents[0]->grad->tensor += n_grad;
        },
        cpy);
    return std::move(result);
}

TensorGrad TensorGrad_Functional_Class::max_unpool3d(
    TensorGrad input, TensorGrad indices, utils::my_n_tuple<3> kernel_size,
    utils::my_n_tuple<3> stride, utils::my_n_tuple<3> padding,
    utils::my_n_tuple<3> output_size) {
    return max_unpool3d(input, indices.tensor, kernel_size, stride, padding,
                        output_size);
}
TensorGrad TensorGrad_Functional_Class::adaptive_max_pool3d(
    TensorGrad x, utils::my_n_tuple<3> out_shape, bool return_indices) {
    int64_t d_out = out_shape[0];
    int64_t r_out = out_shape[1];
    int64_t c_out = out_shape[2];
    utils::throw_exception(
        c_out <= x.shape()[-1],
        "Expected the output from adaptive pooling ($) to be less than or "
        "equal to the specified input ($) at the dimension",
        c_out, x.shape()[-1]);
    utils::throw_exception(
        c_out != 0, "Cannot find adaptive for an output of 0 at any dimension");
    utils::throw_exception(
        r_out <= x.shape()[-2],
        "Expected the output from adaptive pooling ($) to be less than or "
        "equal to the specified input ($) at the dimension",
        r_out, x.shape()[-2]);
    utils::throw_exception(
        r_out != 0, "Cannot find adaptive for an output of 0 at any dimension");
    utils::throw_exception(
        d_out <= x.shape()[-3],
        "Expected the output from adaptive pooling ($) to be less than or "
        "equal to the specified input ($) at the dimension",
        d_out, x.shape()[-3]);
    utils::throw_exception(
        d_out != 0, "Cannot find adaptive for an output of 0 at any dimension");
    int64_t kernel_size_c, stride_c, padding_c;
    int64_t kernel_size_r, stride_r, padding_r;
    int64_t kernel_size_d, stride_d, padding_d;
    find_adaptive(c_out, x.shape()[-1], kernel_size_c, stride_c, padding_c);
    find_adaptive(r_out, x.shape()[-2], kernel_size_r, stride_r, padding_r);
    find_adaptive(d_out, x.shape()[-3], kernel_size_r, stride_r, padding_r);
    return max_pool3d(x, {kernel_size_d, kernel_size_r, kernel_size_c},
                      {stride_d, stride_r, stride_c},
                      {padding_d, padding_r, padding_c}, 1, false,
                      return_indices);
}

inline std::vector<int64_t> getSlidingWindowSizesOutput(int64_t kernel_size,
                                                        int64_t input_size,
                                                        int64_t output_size) {
    std::vector<int64_t> window_sizes(output_size, 0);
    double stride =
        static_cast<double>(input_size - kernel_size) / (output_size - 1);

    int64_t at = 0;

    for (int64_t i = 0; i < output_size; ++i) {
        window_sizes[i] =
            static_cast<int64_t>(std::round(kernel_size + i * stride)) - at;
        at += window_sizes[i];
    }

    return window_sizes;
}

// Function 2: Sliding window sizes based on ratio
inline std::vector<int64_t> getSlidingWindowSizesRatio(int64_t kernel_size,
                                                       int64_t input_size,
                                                       double ratio) {
    // Calculate the output size based on the ratio
    int64_t output_size = static_cast<int64_t>(std::round(input_size * ratio));
    return getSlidingWindowSizesOutput(kernel_size, input_size, output_size);
}


inline void check_parameters_fractional(const Tensor &input,
                                        int64_t output_size,
                                        int64_t kernel_size, int64_t dim) {
    _NT_FUNCTIONAL_ALWAYS_CHECK_(input);
    utils::throw_exception(
        !input.is_null(),
        "Cannot perform fractional maxpooling on a null tensor");
    dim = (dim < 0) ? dim + input.dims() : dim;
    utils::throw_exception(
        dim >= 0 && dim < input.dims(),
        "Expected dimensions of input tensor $ to be greater than or equal to "
        "at least $ for fractional max pooling",
        input.dims(), dim);
    utils::throw_exception(
        output_size + kernel_size - 1 <= input.shape()[dim],
        "Error output_size ($) + kernel_size ($) - 1 <= input_shape at $ ($)",
        output_size, kernel_size, dim, input.shape()[dim]);
}

inline void check_parameters_fractional(const Tensor &input,
                                        double output_ratio,
                                        int64_t kernel_size, int64_t dim) {
    _NT_FUNCTIONAL_ALWAYS_CHECK_(input);
    check_parameters_fractional(
        input, static_cast<int64_t>(input.shape()[dim] * output_ratio),
        kernel_size, dim);
}

// Fractional Max Pooling
TensorGrad TensorGrad_Functional_Class::fractional_max_pool2d(
    TensorGrad _input, utils::my_tuple kernel_size, utils::my_tuple output_size,
    std::variant<double, std::tuple<double, double>> output_ratio,
    bool return_indices) {
    const Tensor &input = _input.tensor;
    std::vector<int64_t> row_bounds, col_bounds;
    if (!(output_size == -1)) {
        utils::throw_exception(
            output_ratio.index() == 0 && std::get<0>(output_ratio) == -1.0,
            "Expected if output size is defined, then output ratio is not "
            "defined [and is equal to -1]");
        check_parameters_fractional(input, output_size[0], kernel_size[0], -2);
        check_parameters_fractional(input, output_size[1], kernel_size[1], -1);
        row_bounds = getSlidingWindowSizesOutput(
            kernel_size[0], input.shape()[-2], output_size[0]);
        col_bounds = getSlidingWindowSizesOutput(
            kernel_size[1], input.shape()[-1], output_size[1]);
    } else {
        std::tuple<double, double> output_ratio_tup;
        if (output_ratio.index() == 0) {
            output_ratio_tup = std::tuple<double, double>{
                std::get<0>(output_ratio), std::get<0>(output_ratio)};
        } else {
            output_ratio_tup = std::get<1>(output_ratio);
        }

        utils::throw_exception(std::get<0>(output_ratio_tup) > 0.0 &&
                                   std::get<0>(output_ratio_tup) < 1.0 &&
                                   std::get<1>(output_ratio_tup) > 0.0 &&
                                   std::get<1>(output_ratio_tup) < 1.0,
                               "Error, expected output ratio to be between 0 "
                               "and 1 for both arguments, but got {$,$}",
                               std::get<0>(output_ratio_tup),
                               std::get<1>(output_ratio_tup));
        output_size =
            utils::my_tuple(static_cast<int64_t>(std::get<0>(output_ratio_tup) *
                                                 input.shape()[-2]),
                            static_cast<int64_t>(std::get<1>(output_ratio_tup) *
                                                 input.shape()[-1]));
        check_parameters_fractional(input, output_size[0], kernel_size[0], -2);
        check_parameters_fractional(input, output_size[1], kernel_size[1], -1);
        row_bounds = getSlidingWindowSizesOutput(
            kernel_size[0], input.shape()[-2], output_size[0]);
        col_bounds = getSlidingWindowSizesOutput(
            kernel_size[1], input.shape()[-1], output_size[1]);
    }

    Tensor bools = ::nt::functional::extract_sliding_windows_max_2d(
        row_bounds, col_bounds, input);
    bools.set_mutability(false);
    TensorGrad out_max =
        _input[bools].view(input.shape()
                               .redo_index(-2, output_size[0])
                               .redo_index(-1, output_size[1]));
    if (!return_indices)
        return out_max;
    Tensor indices =
        where(bools.flatten(-2, -1))[-1].item<Tensor>().view(out_max.shape());
    return ::nt::functional::list(out_max, TensorGrad(indices, true));
}

TensorGrad TensorGrad_Functional_Class::fractional_max_pool3d(
    TensorGrad _input, utils::my_n_tuple<3> kernel_size,
    utils::my_n_tuple<3> output_size,
    std::variant<double, std::tuple<double, double, double>> output_ratio,
    bool return_indices) {

    const Tensor &input = _input.tensor;
    std::vector<int64_t> chan_bounds, row_bounds, col_bounds;
    if (!(output_size == -1)) {
        utils::throw_exception(
            output_ratio.index() == 0 && std::get<0>(output_ratio) == -1.0,
            "Expected if output size is defined, then output ratio is not "
            "defined [and is equal to -1]");
        check_parameters_fractional(input, output_size[0], kernel_size[0], -3);
        check_parameters_fractional(input, output_size[1], kernel_size[1], -2);
        check_parameters_fractional(input, output_size[2], kernel_size[2], -1);
        chan_bounds = getSlidingWindowSizesOutput(
            kernel_size[0], input.shape()[-3], output_size[0]);
        row_bounds = getSlidingWindowSizesOutput(
            kernel_size[1], input.shape()[-2], output_size[1]);
        col_bounds = getSlidingWindowSizesOutput(
            kernel_size[2], input.shape()[-1], output_size[2]);
    } else {
        std::tuple<double, double, double> output_ratio_tup;
        if (output_ratio.index() == 0) {
            output_ratio_tup = std::tuple<double, double, double>{
                std::get<0>(output_ratio), std::get<0>(output_ratio),
                std::get<0>(output_ratio)};
        } else {
            output_ratio_tup = std::get<1>(output_ratio);
        }

        utils::throw_exception(std::get<0>(output_ratio_tup) > 0.0 &&
                                   std::get<0>(output_ratio_tup) < 1.0 &&
                                   std::get<1>(output_ratio_tup) > 0.0 &&
                                   std::get<1>(output_ratio_tup) < 1.0 &&
                                   std::get<2>(output_ratio_tup) > 0.0 &&
                                   std::get<2>(output_ratio_tup) < 1.0,
                               "Error, expected output ratio to be between 0 "
                               "and 1 for both arguments, but got {$,$,$}",
                               std::get<0>(output_ratio_tup),
                               std::get<1>(output_ratio_tup),
                               std::get<2>(output_ratio_tup));
        output_size = utils::my_n_tuple<3>(
            static_cast<int64_t>(std::get<0>(output_ratio_tup) *
                                 input.shape()[-3]),
            static_cast<int64_t>(std::get<1>(output_ratio_tup) *
                                 input.shape()[-2]),
            static_cast<int64_t>(std::get<2>(output_ratio_tup) *
                                 input.shape()[-1]));
        check_parameters_fractional(input, output_size[0], kernel_size[0], -3);
        check_parameters_fractional(input, output_size[1], kernel_size[1], -2);
        check_parameters_fractional(input, output_size[2], kernel_size[2], -1);
        chan_bounds = getSlidingWindowSizesOutput(
            kernel_size[0], input.shape()[-3], output_size[0]);
        row_bounds = getSlidingWindowSizesOutput(
            kernel_size[1], input.shape()[-2], output_size[1]);
        col_bounds = getSlidingWindowSizesOutput(
            kernel_size[2], input.shape()[-1], output_size[2]);
    }

    Tensor bools = extract_sliding_windows_max_3d(chan_bounds, row_bounds,
                                                  col_bounds, input);
    bools.set_mutability(false);
    TensorGrad out_max =
        _input[bools].view(input.shape()
                               .redo_index(-3, output_size[0])
                               .redo_index(-2, output_size[1])
                               .redo_index(-1, output_size[2]));
    if (!return_indices)
        return out_max;
    Tensor indices =
        where(bools.flatten(-3, -1))[-1].item<Tensor>().view(out_max.shape());
    return list(out_max, TensorGrad(indices, true));
}

} // namespace functional
} // namespace nt
