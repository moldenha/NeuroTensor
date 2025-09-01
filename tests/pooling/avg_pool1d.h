#include <nt/Tensor.h>
#include <nt/functional/functional.h>
#include "pool_sources.h"


nt::Tensor avg_pool1d_ceil(nt::Tensor input, int64_t kernel_size, int64_t stride, int64_t padding, bool count_include_pad, int64_t mod){
    using namespace nt;
    if(padding != 0) input = input.pad({padding, padding+mod});
    else input = input.pad({0, mod});
    Tensor strided = input.unfold(-1, kernel_size, stride);
    Tensor out = strided.sum(-1, true);
    int64_t div_a = kernel_size;
    int64_t div_b = div_a-(count_include_pad ? mod : padding+mod);
    int64_t div_c = kernel_size - (count_include_pad ? 0 : padding);
    Scalar num_a = (DTypeFuncs::is_complex(input.dtype()) ? Scalar(complex_64(div_a, div_a)).inverse() : Scalar(div_a));
    Scalar num_b = (DTypeFuncs::is_complex(input.dtype()) ? Scalar(complex_64(div_b, div_b)).inverse() : Scalar(div_b));
    Scalar num_c = (DTypeFuncs::is_complex(input.dtype()) ? Scalar(complex_64(div_c, div_c)).inverse() : Scalar(div_c));
    if(DTypeFuncs::is_floating(input.dtype())){
        num_a = num_a.inverse();
        num_b = num_b.inverse();
        num_c = num_c.inverse();
    }

    Tensor div = functional::nums({out.shape()[-2], 1}, num_a);
    div[-1] = num_b;
    div[0] = num_c;
    if(DTypeFuncs::is_floating(input.dtype()) || DTypeFuncs::is_complex(input.dtype()))
        out *= div;
    else
        out /= div;
    
    return out.view(strided.shape().delete_index(-1));
    
}

nt::Tensor avg_pool1d(nt::Tensor input, int64_t kernel_size, int64_t stride = -1, int64_t padding = 0, bool ceil_mode = false, bool count_include_pad = true){
    using namespace nt;
    // if(!DTypeFuncs::is_floating(input.dtype()) || !DTypeFuncs::is_complex(input)) input = input.to(DType::Float32);
    if(stride == -1) stride = kernel_size;
    check_pool_args(input, -1, kernel_size, stride, padding);
    if(ceil_mode){
        const int64_t& lin = input.shape()[-1];
        const int64_t lout = find_pooling_size_ceil(lin, kernel_size, stride, padding);
        if((lout-1) * stride >= (lin+padding))
            return avg_pool1d(input, kernel_size, stride, padding, false, count_include_pad);
        int64_t mod = kernel_size - ((lin+(2*padding) - kernel_size) % stride);
        std::cout << "avg pool mod: "<<mod<<std::endl;
        if(mod == kernel_size)
            return avg_pool1d(input, kernel_size, stride, padding, false, count_include_pad);
        return avg_pool1d_ceil(input, kernel_size, stride, padding, count_include_pad, mod); 
    }

    if(padding != 0) input = input.pad({padding, padding});
    Tensor strided = input.unfold(-1, kernel_size, stride);
    if(padding == 0 || count_include_pad == true){
        return strided.mean(-1).view(strided.shape().delete_index(-1));
    }

    Tensor out = strided.sum(-1, true);
    int64_t div_a = kernel_size;
    int64_t div_b = div_a-(padding);
    Scalar num_a = (DTypeFuncs::is_complex(input.dtype()) ? Scalar(complex_64(div_a, div_a)).inverse() : Scalar(div_a));
    Scalar num_b = (DTypeFuncs::is_complex(input.dtype()) ? Scalar(complex_64(div_b, div_b)).inverse() : Scalar(div_b));
    if(DTypeFuncs::is_floating(input.dtype())){
        num_a = num_a.inverse();
        num_b = num_b.inverse();
    }

    Tensor div = functional::nums({out.shape()[-2], 1}, num_a);
    div[-1] = num_b;
    div[0] = num_b;
    if(DTypeFuncs::is_floating(input.dtype()) || DTypeFuncs::is_complex(input.dtype()))
        out *= div;
    else
        out /= div;
    return out.view(strided.shape().delete_index(-1));

}


nt::Tensor backward_avg_pool1d_ceil(nt::SizeRef in_shape, nt::Tensor output_grad, int64_t kernel_size, int64_t stride, int64_t padding, bool count_include_pad, int64_t mod){
    using namespace nt;
    if(padding > 0) in_shape = in_shape.redo_index(-1, in_shape[-1] + 2 * padding + mod);
    else in_shape = in_shape.redo_index(-1, in_shape[-1] + mod);
    if(stride == -1) stride = kernel_size;
    Tensor grad = functional::zeros(in_shape, output_grad.dtype());
    Tensor strided = grad.unfold(-1, kernel_size, stride);
    while(output_grad.dims() < strided.dims()){
        output_grad = output_grad.unsqueeze(-1);
    }

    int64_t div_a = kernel_size;
    int64_t div_b = div_a-(count_include_pad ? mod : padding+mod);
    int64_t div_c = kernel_size - (count_include_pad ? 0 : padding);
    Scalar num_a = (DTypeFuncs::is_complex(output_grad.dtype()) ? Scalar(complex_64(div_a, div_a)).inverse() : Scalar(div_a));
    Scalar num_b = (DTypeFuncs::is_complex(output_grad.dtype()) ? Scalar(complex_64(div_b, div_b)).inverse() : Scalar(div_b));
    Scalar num_c = (DTypeFuncs::is_complex(output_grad.dtype()) ? Scalar(complex_64(div_c, div_c)).inverse() : Scalar(div_c));
    if(DTypeFuncs::is_floating(output_grad.dtype())){
        num_a = num_a.inverse();
        num_b = num_b.inverse();
        num_c = num_c.inverse();
    }

    Tensor div = functional::nums({output_grad.shape()[-2], 1}, num_a);
    div[-1] = num_b;
    div[0] = num_c;
    Tensor dl_dp = (DTypeFuncs::is_floating(output_grad.dtype()) || DTypeFuncs::is_complex(output_grad.dtype())) ? output_grad * div : output_grad / div;  
    strided += dl_dp.expand_as(strided);
    if(padding > 0){
        std::vector<range_> ranges(grad.dims(), range_(0, -1));
        for(int64_t i = 0; i < ranges.size()-1; ++i){
            ranges[i].end = in_shape[i];
        }
        ranges.back() = range_(padding, in_shape[-1]-(padding+mod));
        return grad[std::move(ranges)].contiguous();

    }
    std::vector<range_> ranges(grad.dims(), range_(0, -1));
    for(int64_t i = 0; i < ranges.size()-1; ++i){
        ranges[i].end = in_shape[i];
    }
    ranges.back() = range_(0, in_shape[-1]-mod);
    return grad[std::move(ranges)].contiguous();
}


nt::Tensor backward_avg_pool1d(nt::SizeRef in_shape, nt::Tensor output_grad, int64_t kernel_size, int64_t stride = -1, int64_t padding = 0, bool ceil_mode = false, bool count_include_pad = true){
    using namespace nt;


    if(stride == -1) stride = kernel_size;
    if(ceil_mode){
        const int64_t& lin = in_shape[-1];
        // const int64_t& lout = output_grad.shape()[-1];
        const int64_t lout = find_pooling_size_ceil(lin, kernel_size, stride, padding);
        if((lout-1) * stride >= (lin+padding))
            return backward_avg_pool1d(in_shape, output_grad, kernel_size, stride, padding, false, count_include_pad);
        int64_t mod = kernel_size - ((lin+(2*padding) - kernel_size) % stride);
        if(mod == kernel_size)
            return backward_avg_pool1d(in_shape, output_grad, kernel_size, stride, padding, false, count_include_pad);
        return backward_avg_pool1d_ceil(in_shape, output_grad, kernel_size, stride, padding, count_include_pad, mod); 
    }
    if(padding > 0) in_shape = in_shape.redo_index(-1, in_shape[-1] + 2 * padding);

    Tensor grad = functional::zeros(in_shape, output_grad.dtype());
    Tensor strided = grad.unfold(-1, kernel_size, stride);
    while(output_grad.dims() < strided.dims()){
        output_grad = output_grad.unsqueeze(-1);
    }

    if(padding == 0 || count_include_pad == true){
        strided += (output_grad / kernel_size).expand_as(strided);
        if(padding == 0) return std::move(grad);
        std::vector<range_> ranges(grad.dims(), range_(0, -1));
        for(int64_t i = 0; i < ranges.size()-1; ++i){
            ranges[i].end = in_shape[i];
        }
        ranges.back() = range_(padding, in_shape[-1]-padding);
        return grad[std::move(ranges)].contiguous();
    }

    int64_t div_a = kernel_size;
    int64_t div_b = div_a-(padding);
    Scalar num_a = (DTypeFuncs::is_complex(output_grad.dtype()) ? Scalar(complex_64(div_a, div_a)).inverse() : Scalar(div_a));
    Scalar num_b = (DTypeFuncs::is_complex(output_grad.dtype()) ? Scalar(complex_64(div_b, div_b)).inverse() : Scalar(div_b));
    if(DTypeFuncs::is_floating(output_grad.dtype())){
        num_a = num_a.inverse();
        num_b = num_b.inverse();
    }

    Tensor div = functional::nums({output_grad.shape()[-2], 1}, num_a);
    div[-1] = num_b;
    div[0] = num_b;
    Tensor dl_dp = (DTypeFuncs::is_floating(output_grad.dtype()) || DTypeFuncs::is_complex(output_grad.dtype())) ? output_grad * div : output_grad / div;  
    strided += dl_dp.expand_as(strided);
    if(padding > 0){
        std::vector<range_> ranges(grad.dims(), range_(0, -1));
        for(int64_t i = 0; i < ranges.size()-1; ++i){
            ranges[i].end = in_shape[i];
        }
        ranges.back() = range_(padding, in_shape[-1]-(padding));
        return grad[std::move(ranges)].contiguous();
    }
    return std::move(grad);
}


nt::Tensor adaptive_avg_pool1d(nt::Tensor x, int64_t l_out){
    using namespace nt;
    utils::throw_exception(l_out <= x.shape()[-1], 
                           "Expected the output from adaptive pooling ($) to be less than or equal to the specified input ($) at the dimension",
                           l_out, x.shape()[-1]);
    utils::throw_exception(l_out != 0,
                           "Cannot find adaptive for an output of 0 at any dimension");
    if(l_out == 1){
        return x.mean(-1);
    }
    if(l_out == x.shape()[-1]) return x;
    int64_t kernel_size, stride, padding;
    find_adaptive(l_out, x.shape()[-1], kernel_size, stride, padding);
    return avg_pool1d(x, kernel_size, stride, padding);
}

