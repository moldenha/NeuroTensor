#include "pool_utilities.hpp"

namespace nt{
namespace functional{

Tensor lp_pool1d(Tensor input, Scalar power, int64_t kernel_size, int64_t stride = -1, bool ceil_mode = false){
     _NT_FUNCTIONAL_ALWAYS_CHECK_(input);
    // if(!DTypeFuncs::is_floating(input.dtype) || !DTypeFuncs::is_complex(input)) input = input.to(DType::Float32);
    if(stride == -1) stride = kernel_size;
    int64_t padding = 0;
    check_pool_args(input, -1, kernel_size, stride, padding);
    if(ceil_mode){
        const int64_t& lin = input.shape()[-1];
        const int64_t lout = find_pooling_size_ceil(lin, kernel_size, stride, padding);
        if(!((lout-1) * stride >= (lin+padding)))
            input = input.pad({0, kernel_size - ((lin+(2*padding) - kernel_size) % stride)});
    }

    // if(padding != 0) input = input.pad({padding, padding});
    Scalar one(complex_64(1,1));
    Tensor strided = input.unfold(-1, kernel_size, stride);
    if(power.isEqual(one)){
        return strided.sum(-1, false);
    }
    Tensor out = strided.pow(power).sum(-1, true).pow(power.inverse());
    return out.view(strided.shape().delete_index(-1));
}

Tensor backward_lp_pool1d(Tensor input, Tensor output_grad, 
                                Scalar power, int64_t kernel_size, int64_t stride = -1,
                                bool ceil_mode = false){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(input, output_grad);
    SizeRef in_shape = input.shape().clone();
    int64_t padding = 0;
    // const SizeRef& in_shape = input.shape();
    int64_t mod = 0;
    if(stride == -1) stride = kernel_size;
    if(ceil_mode){
        const int64_t& lin = in_shape[-1];
        // const int64_t& lout = output_grad.shape()[-1];
        const int64_t lout = find_pooling_size_ceil(lin, kernel_size, stride, padding);
        if(!((lout-1) * stride >= (lin+padding))){
            mod = kernel_size - ((lin+(2*padding) - kernel_size) % stride);
            in_shape = in_shape.redo_index(-1, in_shape[-1] + mod);
            // input = input.pad({0, mod});
        }
        else {
            ceil_mode = false;
        }
    }

    if (ceil_mode){input = input.pad({0, mod}); in_shape = input.shape().clone();}
    Scalar one(complex_64(1,1));



    if(power.isEqual(one)){
        Tensor grad = zeros(in_shape, output_grad.dtype);
        Tensor strided = grad.unfold(-1, kernel_size, stride);
        while(output_grad.dims() < strided.dims()){
            output_grad = output_grad.unsqueeze(-1);
            // output = output.unsqueeze(-1);
        }

        strided += output_grad.expand_as(strided);
        if(!ceil_mode) return grad;
        return unpad(grad, {0, mod});
    }

    Tensor grad = input.clone();
    Tensor strided_a = grad.unfold(-1, kernel_size, stride);
    while(output_grad.dims() < strided_a.dims()){
        output_grad = output_grad.unsqueeze(-1);
        //output = output.unsqueeze(-1);
    }
    strided_a.pow_(power-one);
    strided_a *= power;
    //above is gradient of the first operation
    // Tensor strided_b = strided_a.sum(-1, true);
    Tensor strided_c = strided_a.sum(-1, true);
    strided_c.pow_(power.inverse()-one);
    strided_c *= power.inverse();
    // Tensor strided_c = (power.inverse()) * strided_b.pow(power.inverse()-1);
    strided_c *= output_grad;
    //Tensor grad_c = strided_c * output_grad.unsqueeze(-1);
    //Tensor grad_b = grad_c.expand_as(strided_a);
    strided_a *= strided_c.expand_as(strided_a);     
    if(!ceil_mode) return grad;
    return unpad(grad, {0, mod});
}


Tensor adaptive_lp_pool1d(Tensor x, int64_t l_out, Scalar power){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(x);
    utils::throw_exception(l_out <= x.shape()[-1], 
                           "Expected the output from adaptive pooling ($) to be less than or equal to the specified input ($) at the dimension",
                           l_out, x.shape()[-1]);
    utils::throw_exception(l_out != 0,
                           "Cannot find adaptive for an output of 0 at any dimension");
    if(l_out == 1){
        return x.pow(power).sum(-1).pow(power.inverse());
    }
    if(l_out == x.shape()[-1]) return x;
    int64_t kernel_size, stride, padding;
    find_adaptive(l_out, x.shape()[-1], kernel_size, stride, padding);
    return lp_pool1d(x, power, kernel_size, stride);
}

}} //nt::functional::
