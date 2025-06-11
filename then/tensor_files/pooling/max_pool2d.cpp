#include "../combine.h"
#include "../compare.h"
#include "../index.h"
#include "../min_max.h"
#include "pool_utilities.hpp"

namespace nt {
namespace functional {

Tensor max_pool2d(Tensor input, utils::my_tuple kernel_size,
                  utils::my_tuple stride = -1, utils::my_tuple padding = 0,
                  utils::my_tuple dilation = 1, bool ceil_mode = false,
                  bool return_indices = false, bool get_bools = false) {
    _NT_FUNCTIONAL_ALWAYS_CHECK_(input);
    assert_dilation(dilation);
    if (stride == -1)
        stride = kernel_size;
    check_pool_args(input, -1, kernel_size[1], stride[1], padding[1]);
    check_pool_args(input, -2, kernel_size[0], stride[0], padding[0]);
    int64_t modC = 0, modR = 0;
    if (ceil_mode) {
        const int64_t &cin = input.shape()[-1];
        const int64_t cout =
            find_pooling_size_ceil(cin, kernel_size[1], stride[1], padding[1]);
        const int64_t &rin = input.shape()[-2];
        const int64_t rout =
            find_pooling_size_ceil(rin, kernel_size[0], stride[0], padding[0]);
        if ((cout - 1) * stride[1] >= (cin + padding[1]))
            modC = 0;
        else
            modC = kernel_size[1] -
                   ((cin + (2 * padding[1]) - kernel_size[1]) % stride[1]);
        if ((rout - 1) * stride[0] >= (rin + padding[0]))
            modR = 0;
        else
            modR = kernel_size[0] -
                   ((rin + (2 * padding[0]) - kernel_size[0]) % stride[0]);

        if (modR == kernel_size[0])
            modR = 0;
        if (modC == kernel_size[1])
            modC = 0;
        if (modR == 0 && modC == 0)
            ceil_mode = false;
    }
    if (!(padding == 0) || ceil_mode)
        input = input.pad(
            {padding[0], padding[0] + modR, padding[1], padding[1] + modC},
            "constant", -inf);

    if (dilation > 1) {
        if (!return_indices) {
            return max_pool2d(input.undilate_(dilation[0], dilation[1]),
                              kernel_size, stride, 0, 1, false, false);
        }
        auto [output, indices_] =
            get<2>(max_pool2d(input.undilate_(dilation[0], dilation[1]),
                              kernel_size, stride, 0, 1, false, true, true));
        auto indices = indices_.dilate(dilation[0], dilation[1]);
        auto outShape = indices.shape().delete_index(-1);
        if (!(padding == 0) || ceil_mode)
            indices = unpad(indices, {padding[0], padding[0] + modR, padding[1],
                                      padding[1] + modC});
        Tensor out_indices = where(indices.flatten(-2, -1))[-1].item<Tensor>();
        return list(output, out_indices.view(outShape));
    }

    Tensor strided = input.unfold(-2, kernel_size[0], stride[0]);
    strided = strided.unfold(-2, kernel_size[1], stride[1]).flatten(-2, -1);
    // std::cout << strided.shape() << std::endl;
    //  Tensor arg_max = argmax(strided, -1, true);
    //  std::cout << arg_max << std::endl;
    if (!return_indices)
        return strided.max(-1).values;
    if (return_indices) {
        Tensor bools(input.shape(), DType::Bool);
        bools.fill_(false);
        Tensor strided_bools = bools.unfold(-2, kernel_size[0], stride[0]);
        strided_bools =
            strided_bools.unfold(-2, kernel_size[1], stride[1]).flatten(-2, -1);
        max_indices(strided, strided_bools, -1);
        auto out_shape = strided.shape().delete_index(-1);
        Tensor out_values = input[bools].contiguous().view(out_shape);
        if (get_bools)
            return list(out_values, bools);
        // this is basically just an argmax
        if (!(padding == 0) || ceil_mode)
            bools = unpad(bools, {padding[0], padding[0] + modR, padding[1],
                                  padding[1] + modC});
        Tensor out = where(bools.flatten(-2, -1))[-1].item<Tensor>();
        return list(out_values, out.view(out_shape));
    }
    return strided.max(-1).values;
}

Tensor backward_max_pool2d(SizeRef input_shape, Tensor dldg, Tensor indices) {
    _NT_FUNCTIONAL_ALWAYS_CHECK_(dldg, indices);
    Tensor grad = zeros(input_shape, dldg.dtype);
    Tensor f_grad = grad.flatten(-1, -2);
    Tensor setting =
        at_tensor_split(grad.flatten(-1, -2), indices.flatten(-1, -2), -2);
    setting.set_(dldg.flatten(-1, -2));
    return std::move(grad);
}

Tensor max_unpool2d(Tensor input, Tensor indices, utils::my_tuple kernel_size,
                    utils::my_tuple stride, utils::my_tuple padding,
                    utils::my_tuple output_size = -1) {
    _NT_FUNCTIONAL_ALWAYS_CHECK_(input, indices);
    utils::throw_exception(indices.dtype == DType::int64,
                           "Max unpool requires indices to be int64 got $",
                           indices.dtype);
    if (stride == -1)
        stride = kernel_size;
    int64_t c_out =
        (output_size[1] != -1 ? output_size[-1]
                              : (input.shape()[-1] - 1) * stride[1] + kernel_size[1]);
    int64_t r_out =
        (output_size[0] != -1 ? output_size[-2]
                              : (input.shape()[-2] - 1) * stride[0] + kernel_size[0]);
    utils::throw_exception(
        c_out > 0, "Error, output size $ cannot be less than or equal to 0",
        c_out);
    utils::throw_exception(
        r_out > 0, "Error, output size $ cannot be less than or equal to 0",
        r_out);

    // if(!(padding == 0)){
    //     int64_t padd_r = (kernel_size[1] * stride[1]) + (2*padding[1]);
    //     indices = indices + ((padd_r * padding[0]) + padding[1]);
    // }
    Tensor unpooled = zeros(
        input.shape().redo_index(-1, c_out).redo_index(-2, r_out), input.dtype);
    std::cout << "unpooled shape: "<<unpooled.shape() << std::endl;
    Tensor setting =
        at_tensor_split(unpooled.flatten(-2, -1), indices.flatten(-2, -1), -2);
    setting.set_(input.view(setting.shape()));
    if(padding != 0){
        return unpad(unpooled, {padding[0], padding[0], padding[1], padding[1]});
    }
    return std::move(unpooled);
}

Tensor backward_max_unpool2d(SizeRef input_shape, Tensor dldg, Tensor indices,
                             utils::my_tuple padding) {
    _NT_FUNCTIONAL_ALWAYS_CHECK_(dldg, indices);
    // if(!(padding == 0)){
    //     indices = indices + ((dldg.shape()[-2] * padding[0]) + padding[1]);
    // }
    if(!(padding == 0)){
        dldg = pad(dldg, {padding[0], padding[0], padding[1], padding[1]});
    }
    Tensor grad = zeros(input_shape, dldg.dtype);
    Tensor f_grad = grad.flatten(-1, -2);
    at_tensor_split(dldg.flatten(-1, -2), indices.flatten(-1, -2), -2, f_grad);
    return std::move(grad);
}

Tensor adaptive_max_pool2d(Tensor x, utils::my_tuple out_shape,
                           bool return_indices = false) {
    _NT_FUNCTIONAL_ALWAYS_CHECK_(x);
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

} // namespace functional
} // namespace nt
