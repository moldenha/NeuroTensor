#include "pool_utilities.hpp"

namespace nt{
namespace functional{


Tensor avg_pool3d_ceil(Tensor input, utils::my_n_tuple<3> kernel_size, utils::my_n_tuple<3> stride, utils::my_n_tuple<3> padding,
                           bool count_include_pad, int64_t modD, int64_t modR, int64_t modC){
    if(!(padding == 0)) input = input.pad({padding[0], padding[0] + modD, padding[1], padding[1]+modR, padding[2], padding[2]+modC});
    else{
        input = input.pad({0, modD, 0, modR, 0, modC});
    }
    Tensor strided = input.unfold(-3, kernel_size[0], stride[0]);
    strided = strided.unfold(-3, kernel_size[1], stride[1]).unfold(-3, kernel_size[2], stride[2]).flatten(-3, -1);
    Tensor out = strided.sum(-1, true);
    int64_t div_a = kernel_size[0] * kernel_size[1] * kernel_size[2];
    //modR (modifying rows) is multiplied by the number of columns 
    //modC (modified cols) is multiplied by the number of rows
    int64_t div_b = div_a-(count_include_pad ? (modD * (kernel_size[2] * kernel_size[1]) + modR * kernel_size[2] + modC * kernel_size[1])
                            : ((modD+padding[0]) * (kernel_size[2] * kernel_size[1]) + (modR+padding[1]) * kernel_size[2] + (modC+padding[2]) * kernel_size[1]));
    int64_t div_c = div_a - (count_include_pad ? 0 : ((padding[0] * kernel_size[1] * kernel_size[2]) + 
                               padding[1] * kernel_size[2] + padding[2] * kernel_size[1]));
    Scalar num_a = (DTypeFuncs::is_complex(input.dtype()) ? Scalar(complex_64(div_a, div_a)).inverse() : Scalar(div_a));
    Scalar num_b = (DTypeFuncs::is_complex(input.dtype()) ? Scalar(complex_64(div_b, div_b)).inverse() : Scalar(div_b));
    Scalar num_c = (DTypeFuncs::is_complex(input.dtype()) ? Scalar(complex_64(div_c, div_c)).inverse() : Scalar(div_c));
    if(DTypeFuncs::is_floating(input.dtype())){
        num_a = num_a.inverse();
        num_b = num_b.inverse();
        num_c = num_c.inverse();
    }

    Tensor div = nums({out.shape()[-2], 1}, num_a);
    div[-1] = num_b;
    div[0] = num_c;
    if(DTypeFuncs::is_floating(input.dtype()) || DTypeFuncs::is_complex(input.dtype()))
        out *= div;
    else
        out /= div;
    
    return out.view(strided.shape().delete_index(-1));
    
}

Tensor avg_pool3d(Tensor input, utils::my_n_tuple<3> kernel_size, utils::my_n_tuple<3> stride = -1, utils::my_n_tuple<3> padding = 0, 
                      bool ceil_mode = false, bool count_include_pad = true){
     _NT_FUNCTIONAL_ALWAYS_CHECK_(input);
    // if(!DTypeFuncs::is_floating(input.dtype()) || !DTypeFuncs::is_complex(input)) input = input.to(DType::Float32);
    if(stride == -1) stride = kernel_size;
    check_pool_args(input, -1, kernel_size[2], stride[2], padding[2]);
    check_pool_args(input, -2, kernel_size[1], stride[1], padding[1]);
    check_pool_args(input, -3, kernel_size[0], stride[0], padding[0]);
    if(ceil_mode){
        const int64_t& cin = input.shape()[-1];
        const int64_t cout = find_pooling_size_ceil(cin, kernel_size[2], stride[2], padding[2]);
        const int64_t& rin = input.shape()[-2];
        const int64_t rout = find_pooling_size_ceil(rin, kernel_size[1], stride[1], padding[1]);
        const int64_t& din = input.shape()[-3];
        const int64_t dout = find_pooling_size_ceil(rin, kernel_size[0], stride[0], padding[0]);
        int64_t modC, modR, modD;
        if((cout-1) * stride[2] >= (cin+padding[2]))
            modC = 0;
        else
            modC = kernel_size[2] - ((cin+(2*padding[2]) - kernel_size[2]) % stride[2]);
        if((rout-1) * stride[1] >= (rin+padding[1]))
            modR = 0;
        else
            modR = kernel_size[1] - ((rin+(2*padding[1]) - kernel_size[1]) % stride[1]);
        if((dout-1) * stride[0] >= (din+padding[0]))
            modD = 0;
        else
            modD = kernel_size[0] - ((din+(2*padding[0]) - kernel_size[0]) % stride[0]);
        
        if(modD == kernel_size[0]) modD = 0;
        if(modR == kernel_size[1]) modR = 0;
        if(modC == kernel_size[2]) modC = 0;

        if(modR == 0 && modC == 0 && modD == 0)
            return avg_pool3d(input, kernel_size, stride, padding, false, count_include_pad);
        return avg_pool3d_ceil(input, kernel_size, stride, padding, count_include_pad, modD, modR, modC);
    }
    if(!(padding == 0)){
        input = input.pad({padding[0], padding[0], padding[1], padding[1], padding[2], padding[2]});
    }
    Tensor strided = input.unfold(-3, kernel_size[0], stride[0]);
    strided = strided.unfold(-3, kernel_size[1], stride[1]).unfold(-3, kernel_size[2], kernel_size[2]).flatten(-3, -1);
    if(padding == 0 || count_include_pad == true){
        return strided.mean(-1).view(strided.shape().delete_index(-1));
    }

    Tensor out = strided.sum(-1, true);
    int64_t div_a = kernel_size[0] * kernel_size[1] * kernel_size[2];
    //modR (modifying rows) is multiplied by the number of columns 
    //modC (modified cols) is multiplied by the number of rows
    int64_t div_b = div_a - (count_include_pad ? 0 : ((padding[0] * kernel_size[1] * kernel_size[2]) + padding[2] * kernel_size[1] + padding[1] * kernel_size[2]));
    Scalar num_a = (DTypeFuncs::is_complex(input.dtype()) ? Scalar(complex_64(div_a, div_a)).inverse() : Scalar(div_a));
    Scalar num_b = (DTypeFuncs::is_complex(input.dtype()) ? Scalar(complex_64(div_b, div_b)).inverse() : Scalar(div_b));
    if(DTypeFuncs::is_floating(input.dtype())){
        num_a = num_a.inverse();
        num_b = num_b.inverse();
    }

    Tensor div = nums({out.shape()[-2], 1}, num_a);
    div[-1] = num_b;
    div[0] = num_b;
    if(DTypeFuncs::is_floating(input.dtype()) || DTypeFuncs::is_complex(input.dtype()))
        out *= div;
    else
        out /= div;
    return out.view(strided.shape().delete_index(-1));

}


Tensor backward_avg_pool3d_ceil(SizeRef in_shape, Tensor output_grad, utils::my_n_tuple<3> kernel_size, utils::my_n_tuple<3> stride, 
                                    utils::my_n_tuple<3> padding,
                                    bool count_include_pad, int64_t modD, int64_t modR, int64_t modC){
    if(!(padding == 0)){ in_shape = in_shape.redo_index(-1, in_shape[-1] + 2 * padding[2] + modC)
                                            .redo_index(-2, in_shape[-2] + 2 * padding[1] + modR)
                                            .redo_index(-3, in_shape[-3] + 2 * padding[0] + modD);}
    else{ in_shape = in_shape.redo_index(-1, in_shape[-1] + modC)
                        .redo_index(-2, in_shape[-2] + modR)
                        .redo_index(-3, in_shape[-3] + modD);}
    if(stride == -1) stride = kernel_size;
    Tensor grad = zeros(in_shape, output_grad.dtype());
    Tensor strided = grad.unfold(-3, kernel_size[0], stride[0]);
    strided = strided.unfold(-3, kernel_size[1], stride[1]).unfold(-3, kernel_size[2], kernel_size[2]).flatten(-3, -1);
    while(output_grad.dims() < strided.dims()){
        output_grad = output_grad.unsqueeze(-1);
    }

    int64_t div_a = kernel_size[0] * kernel_size[1] * kernel_size[2];
    //modR (modifying rows) is multiplied by the number of columns 
    //modC (modified cols) is multiplied by the number of rows
    int64_t div_b = div_a-(count_include_pad ? (modD * (kernel_size[2] * kernel_size[1]) + modR * kernel_size[2] + modC * kernel_size[1])
                            : ((modD+padding[0]) * (kernel_size[2] * kernel_size[1]) + (modR+padding[1]) * kernel_size[2] + (modC+padding[2]) * kernel_size[1]));
    int64_t div_c = div_a - (count_include_pad ? 0 : ((padding[0] * kernel_size[1] * kernel_size[2]) + 
                               padding[1] * kernel_size[2] + padding[2] * kernel_size[1]));
    Scalar num_a = (DTypeFuncs::is_complex(output_grad.dtype()) ? Scalar(complex_64(div_a, div_a)).inverse() : Scalar(div_a));
    Scalar num_b = (DTypeFuncs::is_complex(output_grad.dtype()) ? Scalar(complex_64(div_b, div_b)).inverse() : Scalar(div_b));
    Scalar num_c = (DTypeFuncs::is_complex(output_grad.dtype()) ? Scalar(complex_64(div_c, div_c)).inverse() : Scalar(div_c));
    if(DTypeFuncs::is_floating(output_grad.dtype())){
        num_a = num_a.inverse();
        num_b = num_b.inverse();
        num_c = num_c.inverse();
    }



    Tensor div = nums({output_grad.shape()[-2], 1}, num_a);
    div[-1] = num_b;
    div[0] = num_c;
    Tensor dl_dp = (DTypeFuncs::is_floating(output_grad.dtype()) || DTypeFuncs::is_complex(output_grad.dtype())) ? output_grad * div : output_grad / div;  
    strided += dl_dp.expand_as(strided);
    if(!(padding == 0)){return unpad(grad, {padding[0], padding[0] + modD, padding[1], padding[1]+modR, padding[2], padding[2]+modC}).contiguous();}
    return unpad(grad, {0, modD, 0, modR, 0, modC}).contiguous();
}


Tensor backward_avg_pool3d(SizeRef in_shape, Tensor output_grad, 
                               utils::my_n_tuple<3> kernel_size, utils::my_n_tuple<3> stride = -1, utils::my_n_tuple<3> padding = 0, 
                               bool ceil_mode = false, bool count_include_pad = true){

    _NT_FUNCTIONAL_ALWAYS_CHECK_(output_grad);
    if(stride == -1) stride = kernel_size;
    if(ceil_mode){
        const int64_t& cin = in_shape[-1];
        const int64_t cout = find_pooling_size_ceil(cin, kernel_size[2], stride[2], padding[2]);
        const int64_t& rin = in_shape[-2];
        const int64_t rout = find_pooling_size_ceil(rin, kernel_size[1], stride[1], padding[1]);
        const int64_t& din = in_shape[-3];
        const int64_t dout = find_pooling_size_ceil(din, kernel_size[0], stride[0], padding[0]);
        int64_t modC, modR, modD;
        if((cout-1) * stride[2] >= (cin+padding[2]))
            modC = 0;
        else
            modC = kernel_size[2] - ((cin+(2*padding[2]) - kernel_size[2]) % stride[2]);
        if((rout-1) * stride[1] >= (rin+padding[1]))
            modR = 0;
        else
            modR = kernel_size[1] - ((rin+(2*padding[1]) - kernel_size[1]) % stride[1]);
        if((dout-1) * stride[0] >= (din+padding[0]))
            modD = 0;
        else
            modD = kernel_size[0] - ((din+(2*padding[0]) - kernel_size[0]) % stride[0]);
        
        if(modD == kernel_size[0]) modD = 0;
        if(modR == kernel_size[1]) modR = 0;
        if(modC == kernel_size[2]) modC = 0;
        if(modR == 0 && modC == 0 && modD == 0)
            return backward_avg_pool3d(in_shape, output_grad, kernel_size, stride, padding, false, count_include_pad);
        return backward_avg_pool3d_ceil(in_shape, output_grad, kernel_size, stride, padding, count_include_pad, modD, modR, modC);
    }
    if(!(padding == 0)){
        in_shape = in_shape.redo_index(-1, in_shape[-1] + 2 * padding[2])
                            .redo_index(-2, in_shape[-2] + 2 * padding[1])
                            .redo_index(-3, in_shape[-3] + 2 * padding[0]);
    }

    Tensor grad = zeros(in_shape, output_grad.dtype());
    Tensor strided = grad.unfold(-3, kernel_size[0], stride[0]);
    strided = strided.unfold(-3, kernel_size[1], stride[1]).unfold(-3, kernel_size[2], stride[2]).flatten(-3, -1);
    while(output_grad.dims() < strided.dims()){
        output_grad = output_grad.unsqueeze(-1);
    }

    if(padding == 0 || count_include_pad == true){
        strided += (output_grad / (kernel_size[0] * kernel_size[1] * kernel_size[2])).expand_as(strided);
        if(padding == 0) return std::move(grad);
        return unpad(grad, {padding[0], padding[0], padding[1], padding[1], padding[2], padding[2]}).contiguous();
    }

    int64_t div_a = kernel_size[0] * kernel_size[1] * kernel_size[2];
    //modR (modifying rows) is multiplied by the number of columns 
    //modC (modified cols) is multiplied by the number of rows
    int64_t div_b = div_a - (count_include_pad ? 0 : ((padding[0] * kernel_size[1] * kernel_size[2]) + padding[2] * kernel_size[1] + padding[1] * kernel_size[2]));
    Scalar num_a = (DTypeFuncs::is_complex(output_grad.dtype()) ? Scalar(complex_64(div_a, div_a)).inverse() : Scalar(div_a));
    Scalar num_b = (DTypeFuncs::is_complex(output_grad.dtype()) ? Scalar(complex_64(div_b, div_b)).inverse() : Scalar(div_b));
    if(DTypeFuncs::is_floating(output_grad.dtype())){
        num_a = num_a.inverse();
        num_b = num_b.inverse();
    }

    Tensor div = nums({output_grad.shape()[-2], 1}, num_a);
    div[-1] = num_b;
    div[0] = num_b;
    Tensor dl_dp = (DTypeFuncs::is_floating(output_grad.dtype()) || DTypeFuncs::is_complex(output_grad.dtype())) ? output_grad * div : output_grad / div;  
    strided += dl_dp.expand_as(strided);
    if(padding == 0) return std::move(grad);
    return unpad(grad, {padding[0], padding[0], padding[1], padding[1], padding[2], padding[2]});
}


Tensor adaptive_avg_pool3d(Tensor x, utils::my_n_tuple<3> out_shape){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(x);
    int64_t d_out = out_shape[0];
    int64_t c_out = out_shape[2];
    int64_t r_out = out_shape[1];
    utils::throw_exception(x.dims() >= 3,
                           "Expected dimensions of input for adapting average 2d pooling to be less than or equal to 2 got $",
                           x.dims());
    utils::throw_exception(c_out <= x.shape()[-1], 
                           "Expected the output from adaptive pooling ($) to be less than or equal to the specified input ($) at the dimension",
                           c_out, x.shape()[-1]);
    utils::throw_exception(c_out != 0,
                           "Cannot find adaptive for an output of 0 at any dimension");
    utils::throw_exception(r_out <= x.shape()[-2], 
                           "Expected the output from adaptive pooling ($) to be less than or equal to the specified input ($) at the dimension",
                           r_out, x.shape()[-2]);
    utils::throw_exception(r_out != 0,
                           "Cannot find adaptive for an output of 0 at any dimension");
    utils::throw_exception(d_out <= x.shape()[-3], 
                           "Expected the output from adaptive pooling ($) to be less than or equal to the specified input ($) at the dimension",
                           d_out, x.shape()[-3]);
    utils::throw_exception(d_out != 0,
                           "Cannot find adaptive for an output of 0 at any dimension");


    int64_t kernel_size_d, stride_d, padding_d;
    int64_t kernel_size_r, stride_r, padding_r;
    int64_t kernel_size_c, stride_c, padding_c;
    find_adaptive(d_out, x.shape()[-3], kernel_size_d, stride_d, padding_d);
    find_adaptive(r_out, x.shape()[-2], kernel_size_r, stride_r, padding_r);
    find_adaptive(c_out, x.shape()[-1], kernel_size_c, stride_c, padding_c);
    return avg_pool3d(x, {kernel_size_d, kernel_size_r, kernel_size_c}, {stride_d, stride_r, stride_c}, {padding_d, padding_r, padding_c});
}


}} //nt::functional::
