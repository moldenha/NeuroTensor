#include <nt/Tensor.h>
#include <nt/functional/functional.h>
#include "pool_sources.h"

nt::Tensor max_pool1d(nt::Tensor input, int64_t kernel_size, int64_t stride = -1, int64_t padding = 0, int64_t dilation=1, bool ceil_mode = false, bool return_indices = false, bool get_bools = false){
    using namespace nt;
    assert_dilation(dilation);
    if(stride == -1) stride = kernel_size;
    check_pool_args(input, -1, kernel_size, stride, padding);
    int64_t mod = (ceil_mode ? kernel_size - ((input.shape()[-1]+(2*padding) - kernel_size) % stride) : 0);
    if(padding != 0 || mod != 0) input = input.pad({padding, padding+mod}, "constant", -nt::inf);

    if(dilation > 1){
        if(!return_indices){
            return max_pool1d(input.undilate_(dilation), kernel_size, stride, 0, 1, false, false);
        }
        auto [output, indices_] = get<2>(max_pool1d(input.undilate_(dilation), kernel_size, stride, 0, 1, false, true, true));
        auto indices = indices_.dilate(dilation);
        auto outShape = indices.shape().delete_index(-1);
        if(padding != 0) indices = unpad(indices, {padding, padding});
        Tensor out_indices = functional::where(indices.flatten(-2, -1))[-1].item<Tensor>();
        return functional::list(output, out_indices.view(outShape));
    }


    Tensor strided = input.unfold(-1, kernel_size, stride);
    //std::cout << strided.shape() << std::endl;
    // Tensor arg_max = functional::argmax(strided, -1, true);
    // std::cout << arg_max << std::endl;
    auto max_output = strided.max(-1);
    if(return_indices){
        if(get_bools)
            return functional::list(max_output.values, max_output.indices);
        //this is basically just an argmax
        // std::cout << max_output.indices << std::endl;

        Tensor out = functional::where(max_output.indices.flatten(-2, -1))[-1].item<Tensor>();
        auto out_shape = strided.shape().delete_index(-1);
        return functional::list(max_output.values, out.view(out_shape));
    }
    return max_output.values;
}

nt::Tensor backward_max_pool1d(nt::SizeRef input_shape, nt::Tensor dldg, nt::Tensor indices){
    nt::Tensor grad = nt::functional::zeros(input_shape, dldg.dtype);
    nt::Tensor setting = nt::functional::at_tensor_split(grad, indices, -2);
    setting.set_(dldg);
    return std::move(grad);
}


nt::Tensor max_unpool1d(nt::Tensor input, nt::Tensor indices, int64_t kernel_size, int64_t stride = -1, int64_t padding = 0, int64_t output_size=-1){
    using namespace nt;
    utils::throw_exception(indices.dtype == DType::int64,
        "max_unpool requires indices to be int64 got $", indices.dtype);
    if(stride == -1) stride = kernel_size;
    int64_t h_out = (output_size != -1 ? output_size : (input.shape()[-1] - 1) * stride - 2 * padding + kernel_size);
    utils::throw_exception(h_out > 0,
                           "Error output size $ cannot be less than or equal to 0", h_out);
    if(padding > 0) indices = indices + padding;
    Tensor unpooled = functional::zeros(input.shape().redo_index(-1, h_out), input.dtype);
    Tensor setting = functional::at_tensor_split(unpooled, indices, -2);
    setting.set_(input);
    return std::move(unpooled);
}

nt::Tensor backward_max_unpool1d(nt::SizeRef input_shape, nt::Tensor dldg, nt::Tensor indices, int64_t padding){
    using namespace nt;
    if(padding > 0) indices = indices + padding;
    Tensor grad = functional::zeros(input_shape, dldg.dtype);
    functional::at_tensor_split(dldg, indices, -2, grad);
    return std::move(grad);
}

nt::Tensor adaptive_max_pool1d(nt::Tensor x, int64_t l_out, bool return_indices = false){
    using namespace nt;
    utils::throw_exception(l_out <= x.shape()[-1], 
                           "Expected the output from adaptive pooling ($) to be less than or equal to the specified input ($) at the dimension",
                           l_out, x.shape()[-1]);
    utils::throw_exception(l_out != 0,
                           "Cannot find adaptive for an output of 0 at any dimension");
    if(l_out == 1){
        if(return_indices){
            auto out = x.max(-1);
            return functional::list(out.values, out.indices);
        }
        return x.max(-1).values;
    }
    if(l_out == x.shape()[-1]) return x;
    int64_t kernel_size, stride, padding;
    find_adaptive(l_out, x.shape()[-1], kernel_size, stride, padding);
    return max_pool1d(x, kernel_size, stride, padding, 1, false, return_indices);
}

