#include "pool_utilities.hpp"

namespace nt {
namespace functional {

Tensor lp_pool3d(Tensor input, Scalar power, utils::my_n_tuple<3> kernel_size,
                 utils::my_n_tuple<3> stride = -1, bool ceil_mode = false) {
    _NT_FUNCTIONAL_ALWAYS_CHECK_(input);
    // if(!DTypeFuncs::is_floating(input.dtype()) ||
    // !DTypeFuncs::is_complex(input)) input = input.to(DType::Float32);
    if (stride == -1)
        stride = kernel_size;
    utils::my_n_tuple<3> padding = 0;
    check_pool_args(input, -1, kernel_size[2], stride[2], padding[2]);
    check_pool_args(input, -2, kernel_size[1], stride[1], padding[1]);
    check_pool_args(input, -3, kernel_size[0], stride[0], padding[0]);
    if (ceil_mode) {
        const int64_t &cin = input.shape()[-1];
        const int64_t cout =
            find_pooling_size_ceil(cin, kernel_size[2], stride[2], padding[2]);
        const int64_t &rin = input.shape()[-2];
        const int64_t rout =
            find_pooling_size_ceil(rin, kernel_size[1], stride[1], padding[1]);
        const int64_t &din = input.shape()[-3];
        const int64_t dout =
            find_pooling_size_ceil(din, kernel_size[0], stride[0], padding[0]);
        int64_t modC, modR, modD;
        if ((cout - 1) * stride[2] >= (cin + padding[2]))
            modC = 0;
        else
            modC = kernel_size[2] -
                   ((cin + (2 * padding[2]) - kernel_size[2]) % stride[2]);
        if ((rout - 1) * stride[1] >= (rin + padding[1]))
            modR = 0;
        else
            modR = kernel_size[1] -
                   ((rin + (2 * padding[1]) - kernel_size[1]) % stride[1]);
        if ((dout - 1) * stride[0] >= (din + padding[0]))
            modD = 0;
        else
            modD = kernel_size[0] -
                   ((din + (2 * padding[0]) - kernel_size[0]) % stride[0]);

        if (modD == kernel_size[0])
            modD = 0;
        if (modR == kernel_size[1])
            modR = 0;
        if (modC == kernel_size[2])
            modC = 0;

        if (modR != 0 || modC != 0 || modD != 0)
            input = input.pad({0, modD, 0, modR, 0, modC});
    }

    // if(padding != 0) input = input.pad({padding, padding});
    Scalar one(complex_64(1, 1));
    Tensor strided = input.unfold(-3, kernel_size[0], stride[0]);
    strided = strided.unfold(-3, kernel_size[1], stride[1])
                  .unfold(-3, kernel_size[2], stride[2]);
    strided = strided.flatten(-3, -1);
    if (power.isEqual(one)) {
        return strided.sum(-1, false);
    }
    Tensor out = strided.pow(power).sum(-1, true).pow(power.inverse());
    return out.view(strided.shape().delete_index(-1));
}

Tensor backward_lp_pool3d(Tensor input, Tensor output_grad, Scalar power,
                          utils::my_n_tuple<3> kernel_size,
                          utils::my_n_tuple<3> stride = -1,
                          bool ceil_mode = false) {
    _NT_FUNCTIONAL_ALWAYS_CHECK_(input, output_grad);
    utils::my_n_tuple<3> padding = 0;
    SizeRef in_shape = input.shape();
    // const SizeRef& in_shape = input.shape();
    int64_t modR = 0, modC = 0, modD = 0;
    if (stride == -1)
        stride = kernel_size;
    if (ceil_mode) {
        const int64_t &cin = in_shape[-1];
        const int64_t cout =
            find_pooling_size_ceil(cin, kernel_size[2], stride[2], padding[2]);
        const int64_t &rin = in_shape[-2];
        const int64_t rout =
            find_pooling_size_ceil(rin, kernel_size[1], stride[1], padding[1]);
        const int64_t &din = in_shape[-3];
        const int64_t dout =
            find_pooling_size_ceil(din, kernel_size[0], stride[0], padding[0]);
        if ((cout - 1) * stride[2] >= (cin + padding[2]))
            modC = 0;
        else
            modC = kernel_size[2] -
                   ((cin + (2 * padding[2]) - kernel_size[2]) % stride[2]);
        if ((rout - 1) * stride[1] >= (rin + padding[1]))
            modR = 0;
        else
            modR = kernel_size[1] -
                   ((rin + (2 * padding[1]) - kernel_size[1]) % stride[1]);
        if ((dout - 1) * stride[0] >= (din + padding[0]))
            modR = 0;
        else
            modR = kernel_size[0] -
                   ((din + (2 * padding[0]) - kernel_size[0]) % stride[0]);

        if (modD == kernel_size[0])
            modD = 0;
        if (modR == kernel_size[1])
            modR = 0;
        if (modC == kernel_size[2])
            modC = 0;
        if (modD > 0) {
            in_shape = in_shape.redo_index(-3, in_shape[-3] + modD);
        }
        if (modR > 0) {
            in_shape = in_shape.redo_index(-2, in_shape[-2] + modR);
        }
        if (modC > 0) {
            in_shape = in_shape.redo_index(-1, in_shape[-1] + modC);
        }
        if (modR == 0 && modC == 0 && modD == 0)
            ceil_mode = false;
    }

    if (ceil_mode) {
        input = input.pad({0, modD, 0, modR, 0, modC});
        in_shape = input.shape().clone();
    }
    Scalar one(complex_64(1, 1));

    if (power.isEqual(one)) {
        Tensor grad = zeros(in_shape, output_grad.dtype());
        Tensor strided = grad.unfold(-3, kernel_size[0], stride[0]);
        strided = strided.unfold(-3, kernel_size[1], stride[1])
                      .unfold(-3, kernel_size[2], stride[2]);
        strided = strided.flatten(-3, -1);
        while (output_grad.dims() < strided.dims()) {
            output_grad = output_grad.unsqueeze(-1);
            // output = output.unsqueeze(-1);
        }

        strided += output_grad.expand_as(strided);
        if (!ceil_mode)
            return grad;
        return unpad(grad, {0, modD, 0, modR, 0, modC});
    }
    Tensor grad = ceil_mode ? input : input.clone();
    Tensor strided_a = grad.unfold(-3, kernel_size[0], stride[0]);
    strided_a = strided_a.unfold(-3, kernel_size[1], stride[1])
                    .unfold(-3, kernel_size[2], stride[2]);
    strided_a = strided_a.flatten(-3, -1);
    while (output_grad.dims() < strided_a.dims()) {
        output_grad = output_grad.unsqueeze(-1);
        // output = output.unsqueeze(-1);
    }
    strided_a.pow_(power - one);
    strided_a *= power;
    // above is gradient of the first operation
    //  Tensor strided_b = strided_a.sum(-1, true);
    Tensor strided_c = strided_a.sum(-1, true);
    strided_c.pow_(power.inverse() - one);
    strided_c *= power.inverse();
    // Tensor strided_c = (power.inverse()) * strided_b.pow(power.inverse()-1);
    strided_c *= output_grad;
    // Tensor grad_c = strided_c * output_grad.unsqueeze(-1);
    // Tensor grad_b = grad_c.expand_as(strided_a);
    strided_a *= strided_c.expand_as(strided_a);
    if (!ceil_mode)
        return grad;
    return unpad(grad, {0, modD, 0, modR, 0, modC});
}

Tensor adaptive_lp_pool3d(Tensor x, utils::my_n_tuple<3> out_shape,
                          Scalar power) {
    _NT_FUNCTIONAL_ALWAYS_CHECK_(x);
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
                     {stride_d, stride_r, stride_c});
}

} // namespace functional
} // namespace nt
