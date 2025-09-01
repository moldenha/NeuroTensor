#include "conv.h"
//#include "conv_dnn.cpp"
#include "../../Tensor.h"
#include <memory>
#include <algorithm>
#include "exceptions.hpp"

namespace nt{
namespace functional{

//image input shape: (batch, in_channels, i_cols)
//kernel input shape: (out_channels, in_channels/groups, k_cols)
Tensor conv1d(const Tensor& image, const Tensor& kernel, Tensor::size_value_t stride, Tensor::size_value_t padding, Tensor::size_value_t dilation, int64_t groups,
              intrusive_ptr<tensor_holder> original_x, intrusive_ptr<tensor_holder> original_w){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(image, kernel);
	utils::THROW_EXCEPTION(image.dims() >= 2, "Expected to get a 2D or greater tensor as the input for a 1d convolution, but got $D", image.dims());
	utils::THROW_EXCEPTION(kernel.dims() == 3 || kernel.dims() == 2, "Expected to get a 3D or 2D tensor as the kernel for a 1d convolution, but got $D", kernel.dims());
	utils::THROW_EXCEPTION(image.dtype() == kernel.dtype(), "Expected both kernel and image to have the same dtype for conv1d but got $ and $", kernel.dtype(), image.dtype());
	utils::THROW_EXCEPTION(image.dtype() != DType::Bool && image.dtype() != DType::TensorObj, "Expected to get a number type for image and kernel but got $", image.dtype());
    utils::THROW_EXCEPTION(stride >= 1, "Conv1d stride cannot be less than 1 but got $", stride);
    utils::THROW_EXCEPTION(dilation >= 1, "Conv1d dilation cannot be less than 1 but got $", dilation);
    utils::THROW_EXCEPTION(padding >= 0, "Conv1d padding cannot be less than 1 but got $", padding);

	Tensor x = (image.dims() == 2) ? image.unsqueeze(0) : image.flatten(0, -3);
	Tensor w = kernel.dims() == 2 ? kernel.unsqueeze(0) : kernel;
    utils::THROW_EXCEPTION(groups > 0, "Expected groups for a convolution to be greater than 0 but got $", groups); 
	utils::THROW_EXCEPTION(w.shape()[1] * groups == x.shape()[1], 
                    "Expected channels of the kernel to equal the input channels of the image but got $ and $", w.shape()[1], x.shape()[1]);
	utils::THROW_EXCEPTION(w.shape()[0] % groups == 0, 
                        "Expected the output channels, being the kernel's shape at dimension 0 ($) to be divisible by groups ($) but is not", w.shape()[0], groups);

	
	const int64_t Cout = ((image.shape()[-1] + 2 * padding - dilation * (w.shape()[-1] - 1) - 1) / stride) + 1;
	Tensor inp_unfold = unfold1d(x, w.shape()[-1], dilation, padding, stride, true);
	if(groups == 1){
        if(original_x != nullptr){
            original_x->tensor = inp_unfold.clone();
        }
        if(original_w != nullptr){
            original_w->tensor = w.view(w.shape()[0], -1).clone();
        }
		Tensor outp_unfold = matmult(inp_unfold, w.view(w.shape()[0], -1), true, true).RowColSwap(); //contiguous in place transpose
        return outp_unfold.view(x.shape()[0], -1, Cout);
	}
    utils::THROW_EXCEPTION(x.shape()[1] % groups == 0, "Expected image channels to be divisible by groups in Conv1d but got $ % $ != 0", x.shape()[1], groups);
    utils::THROW_EXCEPTION(w.shape()[1] * groups == x.shape()[1], "Expected kernel rows times groups to be equal to rows of image Conv1d but got $ * $ != $", w.shape()[1], groups, x.shape()[1]);


    int64_t add = int64_t(x.shape()[1]/groups) * (w.shape()[-1]);
    int64_t k_add = w.shape()[0] / groups;
    Tensor x_parts = inp_unfold.split_axis({range, 0 < range > add }).clone();
    Tensor k_parts = w.split_axis({0 < range > k_add}).view_Tensors(k_add, -1).clone();
    if(original_x != nullptr){
        if(x.shape()[0] > 1){
            original_x->tensor = inp_unfold;
        }else{
            original_x->tensor = x_parts;
        }
    }
    if(original_w != nullptr){
        original_w->tensor = k_parts;
    }
    Tensor output = matmult(x_parts, k_parts, true, true).RowColSwap();
    int64_t per_row = output.numel() / (groups * Cout * x.shape()[0]);
    output = output.view(groups, -1, per_row, Cout).transpose(0,1).view(x.shape()[0], -1, Cout).contiguous();
    return std::move(output);
}

Tensor conv2d(const Tensor &image, const Tensor &kernel, utils::my_tuple stride, utils::my_tuple padding, utils::my_tuple dilation, int64_t groups,
              intrusive_ptr<tensor_holder> original_x, intrusive_ptr<tensor_holder> original_w){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(image, kernel);
	utils::THROW_EXCEPTION(image.dims() >= 3, "Expected to get a 3D or greater tensor as the input for a 2d convolution, but got $D", image.dims());
	utils::THROW_EXCEPTION(kernel.dims() == 3 || kernel.dims() == 4, "Expected to get a 4D or 3D tensor as the kernel for a 2d convolution, but got $D", kernel.dims());
	utils::THROW_EXCEPTION(image.dtype() == kernel.dtype(), "Expected both kernel and image to have the same dtype for conv2d but got $ and $", kernel.dtype(), image.dtype());
	utils::THROW_EXCEPTION(image.dtype() != DType::Bool && image.dtype() != DType::TensorObj, "Expected to get a number type for image and kernel but got $", image.dtype());
    utils::THROW_EXCEPTION(stride >= 1, "Conv2d stride cannot be less than 1 but got $", stride);
    utils::THROW_EXCEPTION(dilation >= 1, "Conv2d dilation cannot be less than 1 but got $", dilation);
    utils::THROW_EXCEPTION(padding >= 0, "Conv2d padding cannot be less than 1 but got $", padding);
	
    Tensor x = (image.dims() == 3) ? image.unsqueeze(0) : image.flatten(0, -4);
	/* x = x.pad({padding[0], padding[0], padding[1], padding[1], padding[2], padding[2]}); */
	Tensor w = kernel.dims() == 3 ? kernel.unsqueeze(0) : kernel;
	
	const int64_t Rout = ((image.shape()[-2] + 2 * padding[0] - dilation[0] * (w.shape()[-2] - 1) - 1) / stride[0]) + 1;
	const int64_t Cout = ((image.shape()[-1] + 2 * padding[1] - dilation[1] * (w.shape()[-1] - 1) - 1) / stride[1]) + 1;
	Tensor inp_unfold = unfold2d(x, {w.shape()[-2], w.shape()[-1]}, dilation, padding, stride, true);
	if(groups == 1){
        if(original_x != nullptr){
            original_x->tensor = inp_unfold;
        }
        if(original_w != nullptr){
            original_w->tensor = w.view(w.shape()[0], -1).clone();
        }
		Tensor outp_unfold = matmult(inp_unfold, w.view(w.shape()[0], -1), true, true).RowColSwap(); //contiguous in place transpose
		return outp_unfold.view(x.shape()[0], -1, Rout, Cout);
	}
    utils::THROW_EXCEPTION(x.shape()[1] % groups == 0, "Expected image channels to be divisible by groups in Conv2d but got $ % $ != 0", x.shape()[1], groups);
    utils::THROW_EXCEPTION(w.shape()[1] * groups == x.shape()[1], "Expected kernel rows times groups to be equal to rows of image Conv2d but got $ * $ != $", w.shape()[1], groups, x.shape()[1]);


    int64_t add = int64_t(x.shape()[1]/groups) * (w.shape()[-1] * w.shape()[-2]);
    int64_t k_add = w.shape()[0] / groups;
    Tensor x_parts = inp_unfold.split_axis({range_(0, inp_unfold.shape()[0]), range_(0, add)}).clone();
    Tensor k_parts = w.split_axis({range_(0, k_add)}).view_Tensors(k_add, -1).clone();
    if(original_x != nullptr){
        if(x.shape()[0] > 1){
            original_x->tensor = inp_unfold;
        }else{
            original_x->tensor = x_parts;
        }
    }
    if(original_w != nullptr){
        original_w->tensor = k_parts;
    }
    Tensor output = matmult(x_parts, k_parts, true, true).RowColSwap();
    int64_t per_row = output.numel() / (groups * Rout * Cout * x.shape()[0]);
    output = output.view(groups, -1, per_row, Rout, Cout).transpose(0,1).view(x.shape()[0], -1, Rout, Cout).contiguous();
    return std::move(output);
}


Tensor conv3d(const Tensor& image, const Tensor& kernel, utils::my_n_tuple<3> stride, utils::my_n_tuple<3> padding, utils::my_n_tuple<3> dilation, int64_t groups,
              intrusive_ptr<tensor_holder> original_x, intrusive_ptr<tensor_holder> original_w){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(image, kernel);
	utils::THROW_EXCEPTION(image.dims() >= 4, "Expected to get a 4D or greater tensor as the input for a 3d convolution, but got $D", image.dims());
	utils::THROW_EXCEPTION(kernel.dims() == 5 || kernel.dims() == 4, "Expected to get a 4D or 5D tensor as the kernel for a 3d convolution, but got $D", kernel.dims());
	utils::THROW_EXCEPTION(image.dtype() == kernel.dtype(), "Expected both kernel and image to have the same dtype for conv3d but got $ and $", kernel.dtype(), image.dtype());
	utils::THROW_EXCEPTION(image.dtype() != DType::Bool && image.dtype() != DType::TensorObj, "Expected to get a number type for image and kernel but got $", image.dtype());
    utils::THROW_EXCEPTION(stride[0] >= 1 && stride[1] >= 1 && stride[2] >= 1, "Conv3d stride cannot be less than 1 but got $", stride);
    utils::THROW_EXCEPTION(dilation[0] >= 1 && dilation[1] >= 1 && stride[2] >= 1, "Conv3d Transpose dilation cannot be less than 1 but got $", dilation);
    utils::THROW_EXCEPTION(padding[0] >= 0 && padding[1] >= 0 && padding[2] >= 0, "Conv3d padding cannot be less than 1 but got $", padding);
	
    Tensor x = (image.dims() == 4) ? image.unsqueeze(0) : image.flatten(0, -5);
	/* x = x.pad({padding[0], padding[0], padding[1], padding[1], padding[2], padding[2]}); */
	Tensor w = kernel.dims() == 4 ? kernel.unsqueeze(0) : kernel;
	
	const int64_t Dout = ((image.shape()[-3] + 2 * padding[0] - dilation[0] * (w.shape()[-3] - 1) - 1) / stride[0]) + 1;
	const int64_t Rout = ((image.shape()[-2] + 2 * padding[1] - dilation[1] * (w.shape()[-2] - 1) - 1) / stride[1]) + 1;
	const int64_t Cout = ((image.shape()[-1] + 2 * padding[2] - dilation[2] * (w.shape()[-1] - 1) - 1) / stride[2]) + 1;
	Tensor inp_unfold = unfold3d(x, {w.shape()[-3], w.shape()[-2], w.shape()[-1]}, dilation, padding, stride, true);
	if(groups == 1){
        if(original_x != nullptr){
            original_x->tensor = inp_unfold;
        }
        if(original_w != nullptr){
            original_w->tensor = w.view(w.shape()[0], -1).clone();
        }
		Tensor outp_unfold = matmult(inp_unfold, w.view(w.shape()[0], -1), true, true).RowColSwap(); //contiguous in place transpose
		return outp_unfold.view(x.shape()[0], -1, Dout, Rout, Cout);
	}
    utils::THROW_EXCEPTION(x.shape()[1] % groups == 0, "Expected image channels to be divisible by groups in Conv3d but got $ % $ != 0", x.shape()[1], groups);
    utils::THROW_EXCEPTION(w.shape()[1] * groups == x.shape()[1], "Expected kernel rows times groups to be equal to rows of image Conv3d but got $ * $ != $", w.shape()[1], groups, x.shape()[1]);


    int64_t add = int64_t(x.shape()[1]/groups) * (w.shape()[-1] * w.shape()[-2] * w.shape()[-3]);
    int64_t k_add = w.shape()[0] / groups;
    Tensor x_parts = inp_unfold.split_axis({range_(0, inp_unfold.shape()[0]), range_(0, add)}).clone();
    Tensor k_parts = w.split_axis({range_(0, k_add)}).view_Tensors(k_add, -1).clone();
    if(original_x != nullptr){
        if(x.shape()[0] > 1){
            original_x->tensor = inp_unfold;
        }else{
            original_x->tensor = x_parts;
        }
    }
    if(original_w != nullptr){
        original_w->tensor = k_parts;
    }

    Tensor output = matmult(x_parts, k_parts, true, true).RowColSwap();
    int64_t per_row = output.numel() / (groups * Dout * Rout * Cout * x.shape()[0]);
    output = output.view(groups, -1, per_row, Dout, Rout, Cout).transpose(0,1).view(x.shape()[0], -1, Dout, Rout, Cout).contiguous();
    return std::move(output);
}


Tensor convnd(const Tensor& image, const Tensor& kernel, int64_t dim, utils::optional_list _stride, 
              utils::optional_list _padding, utils::optional_list _dilation, int64_t groups,
              intrusive_ptr<tensor_holder> original_x, intrusive_ptr<tensor_holder> original_w){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(image, kernel);
    utils::THROW_EXCEPTION(dim > 0, "Error, cannot do a dynamic ConvND where N < 0 got (N = $)", dim);
	utils::THROW_EXCEPTION(image.dims() >= (dim+1), "Expected to get a $D or greater tensor as the input for a $d convolution, but got $D", dim+1, dim, image.dims());
	utils::THROW_EXCEPTION(kernel.dims() == (dim+2) || kernel.dims() == (dim+1), "Expected to get a $D or $D tensor as the kernel for a $d convolution, but got $D", dim+1, dim+2, dim, kernel.dims());
	utils::THROW_EXCEPTION(image.dtype() == kernel.dtype(), "Expected both kernel and image to have the same dtype for conv$d but got $ and $", dim, kernel.dtype(), image.dtype());
	utils::THROW_EXCEPTION(image.dtype() != DType::Bool && image.dtype() != DType::TensorObj, "Expected to get a number type for image and kernel but got $", image.dtype());
    utils::throw_exception(_dilation.has_value() && (_dilation.is_scalar() || _dilation->size() == dim),
                           "Error, expected dilation to be a single value or equal to the number of dimensions for unfold $d", dim);
    utils::throw_exception(_padding.has_value() && (_padding.is_scalar() || _padding->size() == dim),
                           "Error, expected padding to be a single value or equal to the number of dimensions for unfold $d", dim);
    utils::throw_exception(_stride.has_value() && (_stride.is_scalar() || _stride->size() == dim),
                           "Error, expected stride to be a single value or equal to the number of dimensions for unfold $d", dim);
    if(_dilation.is_scalar()){
        utils::throw_exception(_dilation.get_scalar() >= 1, "Conv$D dilation cannot be less than 1 but got $", dim, _dilation.get_scalar());
    }else{
        utils::throw_exception(std::all_of(_dilation.cbegin(), _dilation.cend(), [](const int64_t& val){return val >= 1;}), 
                                           "Conv$D dilation cannot be less than 1", dim);
    }
    if(_stride.is_scalar()){
        utils::throw_exception(_stride.get_scalar() >= 1, "Conv$D stride cannot be less than 1 but got $", dim, _stride.get_scalar());
    }else{
        utils::throw_exception(std::all_of(_stride.cbegin(), _stride.cend(), [](const int64_t& val){return val >= 1;}), 
                                           "Conv$D stride cannot be less than 1", dim);
    }
    if(_padding.is_scalar()){
        utils::throw_exception(_padding.get_scalar() >= 0, "Conv$D padding cannot be less than 0 but got $", dim, _padding.get_scalar());
    }else{
        utils::throw_exception(std::all_of(_padding.cbegin(), _padding.cend(), [](const int64_t& val){return val >= 0;}), 
                                           "Conv$D padding cannot be less than 0", dim);
    }

    std::vector<int64_t> padding = _padding.to_repeat_vector(dim);
    std::vector<int64_t> dilation = _dilation.to_repeat_vector(dim);
    std::vector<int64_t> stride = _stride.to_repeat_vector(dim);

    Tensor x = (image.dims() == dim+1) ? image.unsqueeze(0) : image.flatten(0, -1 * (dim+2));
	/* x = x.pad({padding[0], padding[0], padding[1], padding[1], padding[2], padding[2]}); */
	Tensor w = kernel.dims() == (dim+1) ? kernel.unsqueeze(0) : kernel;
    std::vector<int64_t> Nout(dim);
    int64_t NMultiply = 1;
    for(int64_t i = 0; i < dim; ++i){
        Nout[i] = ((image.shape()[(-1 * dim)+i] + 2 * padding[i] - dilation[i] * (w.shape()[(-1 * dim) + i] - 1) -1) / stride[i]) + 1;
        NMultiply *= Nout[i];
    }
    
    // std::cout << w.shape()[range_(-1 * (dim+1), -1)] << std::endl;
	Tensor inp_unfold = unfoldnd(x, dim, utils::optional_list(w.shape()[range_(-1 * (dim+1), -1)].Vec()), _dilation, _padding, _stride, true);
	if(groups == 1){
        if(original_x != nullptr){
            original_x->tensor = inp_unfold;
        }
        if(original_w != nullptr){
            original_w->tensor = w.view(w.shape()[0], -1).clone();
        }
		Tensor outp_unfold = matmult(inp_unfold, w.view(w.shape()[0], -1), true, true).RowColSwap(); //contiguous in place transpose
        std::vector<int64_t> out_shape(dim+2);
        out_shape[0] = x.shape()[0];
        out_shape[1] = outp_unfold.numel() / (out_shape[0] * NMultiply);
        for(int64_t i = 0; i < dim; ++i){
            out_shape[i+2] = Nout[i];
        }
        return outp_unfold.view(SizeRef(std::move(out_shape)));
	}
    utils::THROW_EXCEPTION(x.shape()[1] % groups == 0, "Expected image channels to be divisible by groups in Conv$d but got $ % $ != 0", dim, x.shape()[1], groups);
    utils::THROW_EXCEPTION(w.shape()[1] * groups == x.shape()[1], "Expected kernel rows times groups to be equal to rows of image Conv$d but got $ * $ != $", dim, w.shape()[1], groups, x.shape()[1]);


    int64_t add = int64_t(x.shape()[1]/groups) * w.shape()[range_(-1 * (dim+1), -1)].multiply();
    int64_t k_add = w.shape()[0] / groups;
    Tensor x_parts = inp_unfold.split_axis({range_(0, inp_unfold.shape()[0]), range_(0, add)}).clone();
    Tensor k_parts = w.split_axis({range_(0, k_add)}).view_Tensors(k_add, -1).clone();
    if(original_x != nullptr){
        if(x.shape()[0] > 1){
            original_x->tensor = inp_unfold;
        }else{
            original_x->tensor = x_parts;
        }
    }
    if(original_w != nullptr){
        original_w->tensor = k_parts;
    }

    Tensor output = matmult(x_parts, k_parts, true, true).RowColSwap();
    std::vector<int64_t> out_shape(dim+2);
    out_shape[0] = x.shape()[0];
    out_shape[1] = output.numel() / (out_shape[0] * NMultiply);
    for(int64_t i = 0; i < dim; ++i){
        out_shape[i+2] = Nout[i];
    }
    int64_t per_row = output.numel() / (groups * NMultiply * x.shape()[0]);
    output = output.view(groups, -1, per_row, NMultiply).transpose(0,1).view(SizeRef(std::move(out_shape))).contiguous();
    return std::move(output);
}



// !-- CONVOLUTION GRADIENT FUNCTIONS --!

//this finds the gradient of the image with respect to the gradient
//works for any dimension of the convolution operations
void conv_dimage(Tensor grad, Tensor kernel, Tensor& d_img, std::vector<int64_t> kernel_size, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> dilation, int64_t groups){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(grad, kernel, d_img);
    utils::THROW_EXCEPTION(d_img.is_mutable(),
                           "Cannot set tensor for gradient of convolution that is immutible");
    utils::throw_exception(stride.size() == padding.size() && padding.size() == dilation.size(),
                           "Expected to get same amount for padding ($) , stride ($) , and dilation ($) for conv dimage ", 
                           padding.size(), stride.size(), dilation.size());
    utils::throw_exception(grad.dims()-2 == stride.size(), 
                           "For a grad dims of $ indicating a convolution backward of Conv$d, expected to have parameter size of $ but got $",
                           grad.dims(), grad.dims()-2, grad.dims()-2, stride.size());
    int64_t dim = grad.dims()-2;
    utils::throw_exception(dim > 0, "Expected to do a backward convolution for ConvND N > 0, but got for $ dimensions", dim);
    if(dim == 1){
        const int64_t Cout = grad.shape().back();
        const int64_t& Cin = d_img.shape()[-1];
        const int64_t out_channels = grad.shape()[1];
        const int64_t batch_size = grad.shape()[0];
        const int64_t& kC = kernel_size.back();
        bool transpose_kernel = false;
        bool transpose_grad = true;
        if(groups > 1){
            int64_t k_add = out_channels / groups;
            int64_t per_row = grad.numel() / (groups * Cout * batch_size);
            grad = grad.view(batch_size, groups, 1, k_add, (Cout)).contiguous().split_axis(1).flatten(0, -1);
            // kernel.RowColSwap_Tensors();
            transpose_kernel = false;
            if(batch_size > 1){
                kernel = kernel.repeat_(batch_size);
            }
        }else{
            grad = grad.view(batch_size, out_channels, Cout);
            // grad = grad.transpose(-1, -2).clone();
        }
        Tensor d_unfold = matmult(grad, kernel, transpose_grad, transpose_kernel);
        d_unfold.RowColSwap();
        unfold1d_backward(d_unfold, d_img, Cin, kC, dilation[0],
                        padding[0], stride[0], true);
    }
    else if(dim == 2){
        const int64_t Cout = grad.shape().back();
        const int64_t Rout = grad.shape()[2];
        const int64_t& Cin = d_img.shape()[-1];
        const int64_t& Rin = d_img.shape()[-2];
        const int64_t out_channels = grad.shape()[1];
        const int64_t batch_size = grad.shape()[0];
        const int64_t& kC = kernel_size.back();
        const int64_t& kR = kernel_size[kernel_size.size()-2];
        bool transpose_kernel = false;
        bool transpose_grad = true;
        if(groups > 1){
            int64_t k_add = out_channels / groups;
            int64_t per_row = grad.numel() / (groups * Rout * Cout * batch_size);
            // grad = grad.view(batch_size, groups, 1, k_add, (Rout * Cout)).split_axis(1).flatten(0, -1);
            grad = grad.view(batch_size, groups, 1, k_add, (Rout * Cout)).split_axis(1).flatten(0, -1);
            transpose_kernel = false;
            if(batch_size > 1){
                kernel = kernel.repeat_(batch_size);
            }
        }else{
            grad = grad.view(batch_size, out_channels, Rout * Cout);
        }
        Tensor d_unfold = matmult(grad, kernel, transpose_grad, transpose_kernel);
        d_unfold.RowColSwap();
        unfold2d_backward(d_unfold, d_img, {Rin, Cin}, {kR, kC}, {dilation[0], dilation[1]},
                        {padding[0], padding[1]}, {stride[0], stride[1]}, true);
    }
    else if(dim == 3){
        const int64_t Cout = grad.shape().back();
        const int64_t Rout = grad.shape()[-2];
        const int64_t Dout = grad.shape()[-3];
        const int64_t& Cin = d_img.shape()[-1]; 
        const int64_t& Rin = d_img.shape()[-2];
        const int64_t& Din = d_img.shape()[-3];
        const int64_t out_channels = grad.shape()[1];
        const int64_t batch_size = grad.shape()[0];
        const int64_t& kC = kernel_size.back();
        const int64_t& kR = kernel_size[kernel_size.size()-2];
        const int64_t& kD = kernel_size[kernel_size.size()-3];
        bool transpose_kernel = false;
        bool transpose_grad = true;
        if(groups > 1){
            int64_t k_add = out_channels / groups;
            int64_t per_row = grad.numel() / (groups * Dout * Rout * Cout * batch_size);
            grad = grad.view(batch_size, groups, 1, k_add, (Dout * Rout * Cout)).split_axis(1).flatten(0, -1);
            transpose_kernel = false;
            if(batch_size > 1){
                kernel = kernel.repeat_(batch_size);
            }

        }else{
            grad = grad.view(batch_size, out_channels, Dout * Rout * Cout);
        }
        Tensor d_unfold = matmult(grad, kernel, transpose_grad, transpose_kernel);
        d_unfold.RowColSwap();
        unfold3d_backward(d_unfold, d_img, {Din, Rin, Cin}, {kD, kR, kC}, {dilation[0], dilation[1], dilation[2]},
                        {padding[0], padding[1], padding[2]}, {stride[0], stride[1], stride[2]}, true);
    }else{
        // dim > 3
        std::vector<int64_t> Nout = grad.shape()[range_((-1 * (dim+1)), -1)].Vec();
        std::vector<int64_t> Nin = d_img.shape()[range_((-1 * (dim+1)), -1)].Vec();
        const int64_t out_channels = grad.shape()[1];
        const int64_t batch_size = grad.shape()[0];
        //Kn = kernel_size[n];
        bool transpose_kernel = false;
        bool transpose_grad = true;
        const int64_t NMultiply = std::accumulate(Nout.cbegin(), Nout.cend(), 1, std::multiplies<int64_t>());
        if(groups > 1){
            int64_t k_add = out_channels / groups;
            int64_t per_row = grad.numel() / (groups * batch_size * NMultiply);
            grad = grad.view(batch_size, groups, 1, k_add, NMultiply).split_axis(1).flatten(0, -1);
            transpose_kernel = false;
            if(batch_size > 1){
                kernel = kernel.repeat_(batch_size);
            }

        }else {
            grad = grad.view(batch_size, out_channels, NMultiply);
        }
        Tensor d_unfold = matmult(grad, kernel, transpose_grad, transpose_kernel);
        d_unfold.RowColSwap();
        unfoldnd_backward(d_unfold, d_img, dim, Nin, utils::optional_list(kernel_size),
                                            utils::optional_list(dilation), utils::optional_list(padding),
                                            utils::optional_list(stride), true, false);
    }
}

//this finds the gradient of the kernel with respect to the gradient
//works for any dimension of the convolution operations
void conv_dkernel(Tensor grad, Tensor image, Tensor& d_kernel, std::vector<int64_t> img_size, int64_t groups){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(grad, image, d_kernel);
    utils::THROW_EXCEPTION(d_kernel.is_mutable(),
                           "Cannot set tensor for gradient of convolution that is immutible");
    int64_t dim = grad.dims()-2;
    utils::throw_exception(dim == img_size.size(),
                           "Expected img_size.size() to be equal to the number of dims for the convolution which has been evaluated at $d, but got $",
                           dim, img_size.size());
    utils::throw_exception(dim > 0, "Expected to do a backward convolution for ConvND N > 0, but got for $ dimensions", dim);
   if(dim == 1){
        const int64_t Cout = grad.shape().back();
        const int64_t& Cin = img_size.back(); 
        const int64_t out_channels = grad.shape()[1];
        const int64_t batch_size = grad.shape()[0];
        const int64_t& kC = d_kernel.shape().back();

        if(groups > 1){
            int64_t k_add = out_channels / groups;
            int64_t per_row = grad.numel() / (groups *  Cout * batch_size);
            grad = grad.view(batch_size, groups, 1, k_add, Cout).contiguous().split_axis(1).flatten(0, -1);
            if(batch_size > 1){
                int64_t add = d_kernel.shape()[1] * (kC);
                image = image.split_axis({0 <range> 1, 0 <range> add}).clone();
            }
        }else{
            grad = grad.view(batch_size, out_channels, Cout);
        }
        bool transpose_img = true;
        if(batch_size > 1){
            d_kernel += matmult(grad, image, false, transpose_img).view(batch_size, -1).sum(0, false).view(d_kernel.shape());
        }else{
            d_kernel += matmult(grad, image, false, transpose_img).view(d_kernel.shape());
        }
    }   
    else if(dim == 2){
        const int64_t Cout = grad.shape().back();
        const int64_t Rout = grad.shape()[2];
        const int64_t& Cin = img_size[img_size.size()-1]; 
        const int64_t& Rin = img_size[img_size.size()-2]; 
        const int64_t out_channels = grad.shape()[1];
        const int64_t batch_size = grad.shape()[0];
        const int64_t& kC = d_kernel.shape()[-1];
        const int64_t& kR = d_kernel.shape()[-2];
        if(groups > 1){
            int64_t k_add = out_channels / groups;
            int64_t per_row = grad.numel() / (groups * Rout * Cout * batch_size);
            grad = grad.view(batch_size, groups, 1, k_add, (Rout * Cout)).contiguous().split_axis(1).flatten(0, -1);
            if(batch_size > 1){
                int64_t add = d_kernel.shape()[1] * (kC * kR);
                image = image.split_axis({0 <range> 1, 0 <range> add}).clone();
            }
        }else{
            grad = grad.view(batch_size, out_channels, Rout * Cout);
            // grad = grad.transpose(-1, -2).contiguous();
        }
        bool transpose_img = true;
        if(batch_size > 1){
            d_kernel += matmult(grad, image, false, transpose_img).view(batch_size, -1).sum(0, false).view(d_kernel.shape());
        }else{
            d_kernel += matmult(grad, image, false, transpose_img).view(d_kernel.shape());
        }
    }
    else if(dim == 3){
        const int64_t Cout = grad.shape().back();
        const int64_t Rout = grad.shape()[-2];
        const int64_t Dout = grad.shape()[-3];
        const int64_t& Cin = img_size[img_size.size()-1]; 
        const int64_t& Rin = img_size[img_size.size()-2]; 
        const int64_t& Din = img_size[img_size.size()-3]; 
        const int64_t out_channels = grad.shape()[1];
        const int64_t batch_size = grad.shape()[0];
        const int64_t& kC = d_kernel.shape()[-1];
        const int64_t& kR = d_kernel.shape()[-2];
        const int64_t& kD = d_kernel.shape()[-3];
        if(groups > 1){
            int64_t k_add = out_channels / groups;
            int64_t per_row = grad.numel() / (groups * Dout * Rout * Cout * batch_size);
            grad = grad.view(batch_size, groups, 1, k_add, (Dout * Rout * Cout)).contiguous().split_axis(1).flatten(0, -1);
            if(batch_size > 1){
                int64_t add = d_kernel.shape()[1] * (kC * kR * kD);
                image = image.split_axis({0 <range> 1, 0 <range> add}).clone();
            }
            // image.RowColSwap_Tensors();
        }else{
            grad = grad.view(batch_size, out_channels, Dout * Rout * Cout);
            // grad = grad.transpose(-1, -2).contiguous();
        }
        bool transpose_img = true;
        if(batch_size > 1){
            d_kernel += matmult(grad, image, false, transpose_img).view(batch_size, -1).sum(0, false).view(d_kernel.shape());
        }else{
            d_kernel += matmult(grad, image, false, transpose_img).view(d_kernel.shape());
        }
    }
    else{
        // dim > 3
        std::vector<int64_t> Nout = grad.shape()[range_((-1 * (dim+1)), -1)].Vec();
        std::vector<int64_t> Nin(img_size.cbegin() + (img_size.size() - dim), img_size.cend());
        const int64_t out_channels = grad.shape()[1];
        const int64_t batch_size = grad.shape()[0];
        SizeRef kernel_size = d_kernel.shape()[range_((-1 * (dim+1)), -1)].clone();
        //Kn = kernel_size[n];
        const int64_t NMultiply = std::accumulate(Nout.cbegin(), Nout.cend(), 1, std::multiplies<int64_t>());
        const int64_t KMultiply = std::accumulate(kernel_size.cbegin(), kernel_size.cend(), 1, std::multiplies<int64_t>());
        if(groups > 1){
            int64_t k_add = out_channels / groups;
            int64_t per_row = grad.numel() / (groups * NMultiply * batch_size);
            grad = grad.view(batch_size, groups, 1, k_add, NMultiply).contiguous().split_axis(1).flatten(0, -1);
            if(batch_size > 1){
                int64_t add = d_kernel.shape()[1] * KMultiply;
                image = image.split_axis({0 <range> 1, 0 <range> add}).clone();
            }
            // image.RowColSwap_Tensors();
        }else{
            grad = grad.view(batch_size, out_channels, NMultiply);
            // grad = grad.transpose(-1, -2).contiguous();
        }
        bool transpose_img = true;
        if(batch_size > 1){
            d_kernel += matmult(grad, image, false, transpose_img).view(batch_size, -1).sum(0, false).view(d_kernel.shape());
        }else{
            d_kernel += matmult(grad, image, false, transpose_img).view(d_kernel.shape());
        }
    }
}

// !-- CONVOLUTION TRANSPOSE IN MATRIX MULTIPLICATION FUNCTIONS --!

Tensor conv_transpose1d(const Tensor& image, const Tensor& kernel, Tensor::size_value_t stride, Tensor::size_value_t padding, Tensor::size_value_t output_padding, Tensor::size_value_t dilation, int64_t groups, intrusive_ptr<tensor_holder> original_x, intrusive_ptr<tensor_holder> original_w){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(image, kernel);
	utils::THROW_EXCEPTION(image.dims() >= 2, "Expected to get a 2D or greater tensor as the input for a conv1d transpose, but got $D", image.dims());
	utils::THROW_EXCEPTION(kernel.dims() == 3 || kernel.dims() == 2, "Expected to get a 3D or 2D tensor as the kernel for a conv1d transpose, but got $D", kernel.dims());
	utils::THROW_EXCEPTION(image.dtype() == kernel.dtype(), 
                        "Expected both kernel and image to have the same dtype for conv1d transpose but got $ and $", kernel.dtype(), image.dtype());
	utils::THROW_EXCEPTION(image.dtype() != DType::Bool && image.dtype() != DType::TensorObj, "Expected to get a number type for image and kernel but got $", image.dtype());
    utils::THROW_EXCEPTION(stride >= 1, "Conv1d Transpose stride cannot be less than 1 but got $", stride);
    utils::THROW_EXCEPTION(dilation >= 1, "Conv1d Transpose dilation cannot be less than 1 but got $", dilation);
    utils::THROW_EXCEPTION(output_padding >= 0, "Conv1d Transpose output_padding cannot be less than 1 but got $", output_padding);
    utils::THROW_EXCEPTION(padding >= 0, "Conv1d Transpose padding cannot be less than 1 but got $", padding);
    Tensor x = (image.dims() == 2) ? image.unsqueeze(0) : image.flatten(0, -3);
	Tensor w = kernel.dims() == 2 ? kernel.unsqueeze(0) : kernel;
    const int64_t batch_size = x.shape()[0];
    utils::THROW_EXCEPTION(x.shape()[1] == w.shape()[0], 
                           "Expected image rows to be the same as the kernel channels for conv_transpose2d but got $ and $", x.shape(), w.shape());
    utils::THROW_EXCEPTION(groups > 0, "Expected groups for a convolution to be greater than 0 but got $", groups); 
    if(stride > 1){
        x = x.dilate(1, stride); // take into account stride
    }
    if(output_padding > 0){
        x = x.pad({0, output_padding});
    }

    int64_t _pad = ((w.shape()[-1] - 1) * dilation - padding);
    Tensor inp_unfold = unfold1d(x, w.shape()[-1], dilation, _pad, 1);
    const int64_t Cout = (image.shape()[-1] - 1) * stride - 2 * padding + dilation * (kernel.shape()[-1] - 1) + 1 + output_padding;
    if(groups == 1){
        w = w.transpose(0, 1).flip(-1);
        if(original_x != nullptr){
            original_x->tensor = inp_unfold;
        }
        if(original_w != nullptr){
            original_w->tensor = w.view(w.shape()[0], -1).clone();
        }

		Tensor outp_unfold = matmult(w.view(w.shape()[0], -1), inp_unfold);
		if(padding != 0){
            return outp_unfold.view(x.shape()[0], -1, Cout).pad({padding, padding});
        }
        return outp_unfold.view(x.shape()[0], -1, Cout);
	}
    utils::THROW_EXCEPTION(x.shape()[1] % groups == 0, "Expected image rows to be divisible by groups in Conv1d Transpose but got $ % $ != 0", x.shape()[1], groups);
    // utils::THROW_EXCEPTION(w.shape()[1] * groups == x.shape()[1], "Expected kernel rows times groups to be equal to rows of image Conv1d Transpose but got $ * $ != $", w.shape()[1], groups, x.shape()[1]);

    
    int64_t add = int64_t(x.shape()[1]/groups) * (w.shape()[-1]);
    int64_t k_add = w.shape()[0] / groups;
    Tensor x_parts = inp_unfold.split_axis({range_(0, inp_unfold.shape()[0]), range_(0, add)});
    w = w.transpose(0, 1).flip(-1);
    Tensor k_parts = w.split_axis({range_(0, w.shape()[0]), range_(0, k_add)}).view_Tensors(w.shape()[0], -1).contiguous();
    utils::THROW_EXCEPTION(k_parts.numel() == x_parts.numel() && x_parts.numel() == groups, "Expected to be able to split the image and kernel into ($) groups but got $ groups for the image and $ groups for the kernel", groups, k_parts.numel(), x_parts.numel());
    if(original_x != nullptr){
        original_x->tensor = x_parts;
    }
    if(original_w != nullptr){
        original_w->tensor = k_parts;
    }
    // Tensor output = matmult(k_parts.view_Tensors(k_add, -1), x_parts).view(-1, x.shape()[0], Cout);
    // if(padding == 0){return output.transpose(0, 1).contiguous();}
    // return output.transpose(0,1).pad({padding, padding});
    Tensor output = matmult(k_parts, x_parts);
    int64_t per_row = output.numel() / (groups * Cout * batch_size); // the amount of rows in the correct order
    output = output.view(groups, -1, per_row, Cout).transpose(0,1).view(x.shape()[0], -1, Cout);
    
    //this is the old way of doing it:
// 	std::unique_ptr<Tensor> tensors(new Tensor(Tensor::makeNullTensorArray(groups)));
// 	// Tensor fKernel = w.view(w.shape()[0], -1);
// 	Tensor* t_begin = reinterpret_cast<Tensor*>(x_parts.data_ptr());
// 	Tensor* t_end = t_begin + x_parts.numel();
//     Tensor* k_begin = reinterpret_cast<Tensor*>(k_parts.data_ptr());
// 	Tensor* begin = reinterpret_cast<Tensor*>(tensors->data_ptr());
//     std::cout << t_begin->shape() << std::endl;
// 	for(;t_begin != t_end; ++t_begin, ++k_begin, ++begin)
// 		*begin = matmult(*k_begin, *t_begin).view(x.shape()[0], -1, Cout);
// 	Tensor catted = cat(*tensors, 1);
// 	tensors.reset(nullptr); //release memory, this way when contiguous is called there isn't double the memory used
	if(padding == 0)
        return output.contiguous();
    return output.pad({padding, padding});
}



Tensor conv_transpose2d(const Tensor& image, const Tensor& kernel, utils::my_tuple stride, utils::my_tuple padding, utils::my_tuple output_padding, utils::my_tuple dilation, int64_t groups, intrusive_ptr<tensor_holder> original_x, intrusive_ptr<tensor_holder> original_w){

    _NT_FUNCTIONAL_ALWAYS_CHECK_(image, kernel);
    utils::THROW_EXCEPTION(image.dims() >= 3, "Expected to get a 3D or greater tensor as the input for a conv2d transpose, but got $D", image.dims());
	utils::THROW_EXCEPTION(kernel.dims() == 3 || kernel.dims() == 4, "Expected to get a 3D or 4D tensor as the kernel for a conv2d transpose, but got $D", kernel.dims());
	utils::THROW_EXCEPTION(image.dtype() == kernel.dtype(), 
                        "Expected both kernel and image to have the same dtype for conv2d transpose but got $ and $", kernel.dtype(), image.dtype());
	utils::THROW_EXCEPTION(image.dtype() != DType::Bool && image.dtype() != DType::TensorObj, "Expected to get a number type for image and kernel but got $", image.dtype());
    utils::THROW_EXCEPTION(stride >= 1, "Conv2d Transpose stride cannot be less than 1 but got $", stride);
    utils::THROW_EXCEPTION(dilation >= 1, "Conv2d Transpose dilation cannot be less than 1 but got $", dilation);
    utils::THROW_EXCEPTION(output_padding >= 0, "Conv2d Transpose output_padding cannot be less than 1 but got $", output_padding);
    utils::THROW_EXCEPTION(padding >= 0, "Conv2d Transpose padding cannot be less than 1 but got $", padding);
    Tensor x = (image.dims() == 3) ? image.unsqueeze(0) : image.flatten(0, -4);
	Tensor w = kernel.dims() == 3 ? kernel.unsqueeze(0) : kernel;
    const int64_t batch_size = x.shape()[0];
    utils::THROW_EXCEPTION(x.shape()[1] == w.shape()[0], 
                           "Expected image channels to be the same as the kernel batches for conv_transpose2d but got $ and $", x.shape(), w.shape());
    utils::THROW_EXCEPTION(groups > 0, "Expected groups for a convolution to be greater than 0 but got $", groups); 
    if(stride != 1){
        x = x.dilate(stride[0], stride[1]); // take into account stride
    }
    if(output_padding != 0){
        x = x.pad({0, output_padding[0], 0, output_padding[1]});
    }

    const int64_t _ph = ((w.shape()[-2] - 1) * dilation[0] - padding[0]);
    const int64_t _pw = ((w.shape()[-1] - 1) * dilation[1] - padding[1]);
    Tensor inp_unfold = unfold2d(x, {w.shape()[-2], w.shape()[-1]}, dilation, {_ph, _pw}, 1);
    const int64_t Rout = (image.shape()[-2] - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kernel.shape()[-2] - 1) + 1 + output_padding[0];
    const int64_t Cout = (image.shape()[-1] - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kernel.shape()[-1] - 1) + 1 + output_padding[1];
    if(groups == 1){
        w = w.transpose(0, 1).flip(-1).flip(-2);
        if(original_x != nullptr){
            original_x->tensor = inp_unfold;
        }
        if(original_w != nullptr){
            original_w->tensor = w.view(w.shape()[0], -1).clone();
        }
		Tensor outp_unfold = matmult(original_w != nullptr ? original_w->tensor : w.view(w.shape()[0], -1), inp_unfold); 
		if(padding != 0){
            return outp_unfold.view(x.shape()[0], -1, Rout, Cout).pad({padding[0], padding[0], padding[1], padding[1]});
        }
        return outp_unfold.view(x.shape()[0], -1, Rout, Cout).contiguous();
	}
    utils::THROW_EXCEPTION(x.shape()[1] % groups == 0, "Expected image channels to be divisible by groups in Conv2d Transpose but got $ % $ != 0", x.shape()[1], groups);
    // utils::THROW_EXCEPTION(w.shape()[1] * groups == x.shape()[1], "Expected kernel channels times groups to be equal to channels of image Conv2d Transpose but got $ * $ != $", w.shape()[1], groups, x.shape()[1]);

    
    int64_t add = int64_t(x.shape()[1]/groups) * (w.shape()[-1] * w.shape()[-2]);
    int64_t k_add = w.shape()[0] / groups;
	Tensor x_parts = inp_unfold.split_axis({range_(0, inp_unfold.shape()[0]), range_(0, add)});
    w = w.transpose(0, 1).flip(-1).flip(-2);
    Tensor k_parts = w.split_axis({range_(0, w.shape()[0]), range_(0, k_add)}).view_Tensors(w.shape()[0], -1).contiguous();
    utils::THROW_EXCEPTION(k_parts.numel() == x_parts.numel() && x_parts.numel() == groups, "Expected to be able to split the image and kernel into ($) groups but got $ groups for the image and $ groups for the kernel [logic error Conv2d Transpose]", groups, k_parts.numel(), x_parts.numel());
    if(original_x != nullptr){
        original_x->tensor = x_parts;
    }
    if(original_w != nullptr){
        original_w->tensor = k_parts;
    }
    Tensor output = matmult(k_parts, x_parts);
    int64_t per_row = output.numel() / (groups * Rout * Cout * batch_size); // the amount of channels in the correct order
    output = output.view(groups, -1, per_row, Rout, Cout).transpose(0,1).view(x.shape()[0], -1, Rout, Cout);
    if(padding == 0)
        return output.contiguous();
    return output.pad({padding[0], padding[0], padding[1], padding[1]});

}

Tensor conv_transpose3d(const Tensor& image, const Tensor& kernel, utils::my_n_tuple<3> stride, utils::my_n_tuple<3> padding, utils::my_n_tuple<3> output_padding, utils::my_n_tuple<3> dilation, int64_t groups, intrusive_ptr<tensor_holder> original_x, intrusive_ptr<tensor_holder> original_w){

    _NT_FUNCTIONAL_ALWAYS_CHECK_(image, kernel);
    utils::THROW_EXCEPTION(image.dims() >= 4, "Expected to get a 4D or greater tensor as the input for a conv3d transpose, but got $D", image.dims());
	utils::THROW_EXCEPTION(kernel.dims() == 5 || kernel.dims() == 4, "Expected to get a 5D or 4D tensor as the kernel for a conv3d transpose, but got $D", kernel.dims());
	utils::THROW_EXCEPTION(image.dtype() == kernel.dtype(), 
                        "Expected both kernel and image to have the same dtype for conv3d transpose but got $ and $", kernel.dtype(), image.dtype());
	utils::THROW_EXCEPTION(image.dtype() != DType::Bool && image.dtype() != DType::TensorObj, "Expected to get a number type for image and kernel but got $", image.dtype());
    utils::THROW_EXCEPTION(stride[0] >= 1 && stride[1] >= 1 && stride[2] >= 1, "Conv3d Transpose stride cannot be less than 1 but got $", stride);
    utils::THROW_EXCEPTION(dilation[0] >= 1 && dilation[1] >= 1 && stride[2] >= 1, "Conv3d Transpose dilation cannot be less than 1 but got $", dilation);
    utils::THROW_EXCEPTION(output_padding[0] >= 0 && output_padding[1] >= 0 && output_padding[2] >= 0, 
                           "Conv3d Transpose output_padding cannot be less than 1 but got $", output_padding);
    utils::THROW_EXCEPTION(padding[0] >= 0 && padding[1] >= 0 && padding[2] >= 0, "Conv3d Transpose padding cannot be less than 1 but got $", padding);
    Tensor x = (image.dims() == 4) ? image.unsqueeze(0) : image.flatten(0, -5);
	Tensor w = kernel.dims() == 4 ? kernel.unsqueeze(0) : kernel;
    const int64_t batch_size = x.shape()[0];
    utils::THROW_EXCEPTION(groups > 0, "Expected groups for a convolution to be greater than 0 but got $", groups); 
    utils::THROW_EXCEPTION(x.shape()[1] == w.shape()[0], 
                           "Expected image channels to be the same as the kernel batches for conv_transpose3d but got $ and $", x.shape(), w.shape());
    if(stride != 1){
        x = x.dilate(stride[0], stride[1], stride[3]); // take into account stride
    }
    if(output_padding != 0){
        x = x.pad({0, output_padding[0], 0, output_padding[1], 0, output_padding[2]});
    }

    const int64_t _pd = ((w.shape()[-3] - 1) * dilation[0] - padding[0]);
    const int64_t _ph = ((w.shape()[-2] - 1) * dilation[1] - padding[1]);
    const int64_t _pw = ((w.shape()[-1] - 1) * dilation[2] - padding[2]);
    Tensor inp_unfold = unfold3d(x, {w.shape()[-3], w.shape()[-2], w.shape()[-1]}, dilation, {_pd, _ph, _pw}, 1);
    const int64_t Dout = (image.shape()[-3] - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kernel.shape()[-3] - 1) + 1 + output_padding[0]; 
    const int64_t Rout = (image.shape()[-2] - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kernel.shape()[-2] - 1) + 1 + output_padding[1];
    const int64_t Cout = (image.shape()[-1] - 1) * stride[2] - 2 * padding[2] + dilation[2] * (kernel.shape()[-1] - 1) + 1 + output_padding[2];

    if(groups == 1){
        w = w.transpose(0, 1).flip(-1).flip(-2).flip(-3);
        if(original_x != nullptr){
            original_x->tensor = inp_unfold;
        }
        if(original_w != nullptr){
            original_w->tensor = w.view(w.shape()[0], -1).clone();
        }
		Tensor outp_unfold = matmult(original_w != nullptr ? original_w->tensor : w.view(w.shape()[0], -1), inp_unfold); 
		if(padding != 0){
            return outp_unfold.view(x.shape()[0], -1, Dout, Rout, Cout).pad({padding[0], padding[0], padding[1], padding[1], padding[2], padding[2]});
        }
        return outp_unfold.view(x.shape()[0], -1, Dout, Rout, Cout);
	}
    utils::THROW_EXCEPTION(x.shape()[1] % groups == 0, "Expected image channels to be divisible by groups in Conv3d Transpose but got $ % $ != 0", x.shape()[1], groups);
    // utils::THROW_EXCEPTION(w.shape()[1] * groups == x.shape()[1], "Expected kernel channels times groups to be equal to channels of image Conv3d Transpose but got $ * $ != $", w.shape()[1], groups, x.shape()[1]);


    int64_t add = int64_t(x.shape()[1]/groups) * (w.shape()[-1] * w.shape()[-2] * w.shape()[-3]);
    int64_t k_add = w.shape()[0] / groups;
	Tensor x_parts = inp_unfold.split_axis({range_(0, inp_unfold.shape()[0]), range_(0, add)});
    w = w.transpose(0, 1).flip(-1).flip(-2).flip(-3);
    Tensor k_parts = w.split_axis({range_(0, w.shape()[0]), range_(0, k_add)}).view_Tensors(w.shape()[0], -1).contiguous();
   utils::THROW_EXCEPTION(k_parts.numel() == x_parts.numel() && x_parts.numel() == groups, "Expected to be able to split the image and kernel into ($) groups but got $ groups for the image and $ groups for the kernel [logic error Conv3d Transpose]", groups, k_parts.numel(), x_parts.numel());
    if(original_x != nullptr){
        original_x->tensor = x_parts;
    }
    if(original_w != nullptr){
        original_w->tensor = k_parts;
    }
    Tensor output = matmult(k_parts, x_parts);
    int64_t per_row = output.numel() / (groups * Dout * Rout * Cout * batch_size); // the amount of channels in the correct order
    output = output.view(groups, -1, per_row, Dout, Rout, Cout).transpose(0,1).view(x.shape()[0], -1, Dout, Rout, Cout);

	if(padding == 0)
        return output.contiguous();
    return output.pad({padding[0], padding[0], padding[1], padding[1], padding[2], padding[3]});

}


// !-- CONVOLUTION TRANSPOSE GRADIENT FUNCTIONS --!


std::vector<range_> make_ranges(const std::vector<int64_t>& padding, const SizeRef& s, const int64_t& dim, bool first_z = false){
    std::vector<range_> ranges(dim+2);
    ranges[0] = range_(0, s[0]);
    ranges[1] = range_(0, s[1]);
    if(dim == 1){
        ranges[2] = range_(first_z ? 0 : padding[0], s[2]-padding[0]);
    }
    else if(dim == 2){
        ranges[2] = range_(first_z ? 0 : padding[0], s[2]-padding[0]);
        ranges[3] = range_(first_z ? 0 : padding[1], s[3]-padding[1]);
    }
    else if(dim == 3){
        ranges[2] = range_(first_z ? 0 : padding[0], s[2]-padding[0]);
        ranges[3] = range_(first_z ? 0 : padding[1], s[3]-padding[1]);
        ranges[4] = range_(first_z ? 0 : padding[2], s[4]-padding[2]);
    }
    return std::move(ranges);
}

void convt_dimage(Tensor grad, Tensor kernel, Tensor& d_img, std::vector<int64_t> kernel_size, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> output_padding, std::vector<int64_t> dilation, int64_t groups){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(grad, kernel, d_img);
    utils::THROW_EXCEPTION(d_img.is_mutable(),
                           "Cannot set tensor for gradient of convolution that is immutible");
    utils::throw_exception(stride.size() == padding.size() && padding.size() == dilation.size() && padding.size() == output_padding.size(),
                           "Expected to get same amount for  padding ($) , output_padding ($), stride ($) , and dilation ($) for conv dimage ", 
                           padding.size(), output_padding.size(), stride.size(), dilation.size());
    utils::throw_exception(grad.dims()-2 == stride.size(), 
                           "For a grad dims of $ indicating a convolution backward of Conv$d, expected to have parameter size of $ but got $",
                           grad.dims(), grad.dims()-2, grad.dims()-2, stride.size());
    int64_t dim = grad.dims()-2;
    utils::throw_exception(dim == 1 || dim == 2 || dim == 3, "Expected to do a backward convolution for Conv1d, Conv2d, or Conv3d, but got for $ dimensions", dim);
    if(std::any_of(padding.cbegin(), padding.cend(), [](const int64_t& p){return p != 0;})){
        std::vector<range_> ranges = make_ranges(padding, grad.shape(), dim);
        grad = grad[ranges];
    }
    if(dim == 1){
        const int64_t Cout = grad.shape().back();
        int64_t Cin = d_img.shape()[-1];
        //take stride into account
        Cin *= stride[0];
        Cin -= (stride[0]-1);
        //take output_padding into account
        Cin += output_padding[0];

        const int64_t out_channels = grad.shape()[1];
        const int64_t batch_size = grad.shape()[0];
        const int64_t& kC = kernel_size.back();
        const int64_t _pad = ((kC - 1) * dilation[0] - padding[0]);
        if(groups > 1){
            int64_t k_add = out_channels / groups;
            int64_t per_row = grad.numel() / (groups * Cout * batch_size);
            grad = grad.view(-1, groups, per_row, Cout).transpose(0, 1);
            grad = grad.view(groups, -1, k_add, (Cout)).transpose(-1,-2).contiguous().split_axis(0);
            // kernel.RowColSwap_Tensors();
        }else{
            grad = grad.view(batch_size, out_channels, Cout);
            grad = grad.transpose(-1,-2).contiguous();
        }
        Tensor d_matmult = matmult(grad, kernel, false, false);
        d_matmult.RowColSwap();
        Tensor d_unfold = unfold1d_backward(d_matmult, Cin, kC, dilation[0], _pad, 1, true).view(batch_size, -1, Cin);
        //backward output padding
        if(output_padding[0] > 0){
            std::vector<range_> ranges = make_ranges(output_padding, d_unfold.shape(), dim, true);
            d_unfold = d_unfold[ranges];
        }
        //backward stride
        if(stride[0] > 0){
            d_unfold = d_unfold.undilate(1, stride[0]);
        }
        d_img += d_unfold;
    }
    else if(dim == 2){
        const int64_t Cout = grad.shape().back();
        const int64_t Rout = grad.shape()[2];
        int64_t Cin = d_img.shape()[-1];
        //take stride into account
        Cin *= stride[1];
        Cin -= (stride[1]-1);
        //take output_padding into account
        Cin += output_padding[1];

        int64_t Rin = d_img.shape()[-2];
        //take stride into account
        Rin *= stride[0];
        Rin -= (stride[0]-1);
        //take output_padding into account
        Rin += output_padding[0];
        const int64_t out_channels = grad.shape()[1];
        const int64_t batch_size = grad.shape()[0];
        const int64_t& kC = kernel_size.back();
        const int64_t& kR = kernel_size[kernel_size.size()-2];
        const int64_t _pR = ((kR - 1) * dilation[0] - padding[0]);
        const int64_t _pC = ((kC - 1) * dilation[1] - padding[1]);
        if(groups > 1){
            int64_t k_add = out_channels / groups;
            int64_t per_row = grad.numel() / (groups * Rout * Cout * batch_size);
            grad = grad.view(-1, groups, per_row, Rout, Cout);
            grad = grad.transpose(0, 1);
            grad = grad.view(groups, -1, k_add, (Rout * Cout)).transpose(-1,-2).contiguous().split_axis(0);
            // kernel.RowColSwap_Tensors();

        }else{
            grad = grad.view(batch_size, out_channels, Rout * Cout);
            grad = grad.transpose(-1,-2).contiguous();
        }
        Tensor d_matmult = matmult(grad, kernel, false, false);
        //Tensor inp_unfold = unfold1d(x, w.shape()[-1], dilation, _pad, 1);
        d_matmult.RowColSwap();
        Tensor d_unfold = unfold2d_backward(d_matmult, {Rin, Cin}, {kR, kC}, {dilation[0], dilation[1]},
                        {_pR, _pC}, 1, true).view(batch_size, -1, Rin, Cin);
        //backward output padding
        if(output_padding[0] > 0 || output_padding[1] > 0){
            std::vector<range_> ranges = make_ranges(output_padding, d_unfold.shape(), dim, true);
            d_unfold = d_unfold[ranges];
        }
        //backward stride
        if(stride[0] > 0 || stride[1] > 0){
            d_unfold = d_unfold.undilate(stride[0], stride[1]);
        }
        d_img += d_unfold; 
    }
    else if(dim == 3){
        const int64_t Cout = grad.shape().back();
        const int64_t Rout = grad.shape()[-2];
        const int64_t Dout = grad.shape()[-3];
        int64_t Cin = d_img.shape()[-1];
        //take stride into account
        Cin *= stride[2];
        Cin -= (stride[2]-1);
        //take output_padding into account
        Cin += output_padding[2];

        int64_t Rin = d_img.shape()[-2];
        //take stride into account
        Rin *= stride[1];
        Rin -= (stride[1]-1);
        //take output_padding into account
        Rin += output_padding[1];
        
        int64_t Din = d_img.shape()[-3];
        //take stride into account
        Din *= stride[0];
        Din -= (stride[0]-1);
        //take output_padding into account
        Din += output_padding[0];

        const int64_t out_channels = grad.shape()[1];
        const int64_t batch_size = grad.shape()[0];
        const int64_t& kC = kernel_size.back();
        const int64_t& kR = kernel_size[kernel_size.size()-2];
        const int64_t& kD = kernel_size[kernel_size.size()-3];
        const int64_t _pD = ((kD - 1) * dilation[0] - padding[0]);
        const int64_t _pR = ((kR - 1) * dilation[1] - padding[1]);
        const int64_t _pC = ((kC - 1) * dilation[2] - padding[2]);
        if(groups > 1){
            int64_t k_add = out_channels / groups;
            int64_t per_row = grad.numel() / (groups * Dout * Rout * Cout * batch_size);
            grad = grad.view(-1, groups, per_row, Dout, Rout, Cout).transpose(0, 1);
            grad = grad.view(groups, -1, k_add, (Dout * Rout * Cout)).transpose(-1,-2).contiguous().split_axis(0);
            // kernel.RowColSwap_Tensors();
        }else{
            grad = grad.view(batch_size, out_channels, Rout * Cout);
            grad = grad.transpose(-1,-2).contiguous();
        }
        Tensor d_matmult = matmult(grad, kernel, false, false);
        d_matmult.RowColSwap();
        Tensor d_unfold = unfold3d_backward(d_matmult, {Din, Rin, Cin}, {kD, kR, kC}, {dilation[0], dilation[1], dilation[2]},
                        {_pD, _pR, _pC}, 1, true).view(batch_size, -1, Din, Rin, Cin);
        //backward output padding
        if(output_padding[0] > 0 || output_padding[1] > 0 || output_padding[2] > 0){
            std::vector<range_> ranges = make_ranges(output_padding, d_unfold.shape(), dim, true);
            d_unfold = d_unfold[ranges];
        }
        //backward stride
        if(stride[0] > 0 || stride[1] > 0 || stride[2] > 0){
            d_unfold = d_unfold.undilate(stride[0], stride[1], stride[2]);
        }
        d_img += d_unfold; 
    }
}

//this finds the gradient of the kernel with respect to the gradient
//works for any dimension of the convolution operations
void convt_dkernel(Tensor grad, Tensor image, Tensor& d_kernel, std::vector<int64_t> padding,  std::vector<int64_t> img_size, int64_t groups){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(grad, image, d_kernel);
    utils::THROW_EXCEPTION(d_kernel.is_mutable(),
                           "Cannot set tensor for gradient of convolution that is immutible");
    int64_t dim = grad.dims()-2;
    utils::throw_exception(dim == img_size.size(),
                           "Expected img_size.size() to be equal to the number of dims for the convolution which has been evaluated at $d, but got $",
                           dim, img_size.size());
    utils::throw_exception(dim == 1 || dim == 2 || dim == 3, "Expected to do a backward convolution for Conv1d, Conv2d, or Conv3d, but got for $ dimensions", dim);
    if(std::any_of(padding.cbegin(), padding.cend(), [](const int64_t& p){return p != 0;})){
        std::vector<range_> ranges = make_ranges(padding, grad.shape(), dim);
        grad = grad[ranges];
    }
    if(dim == 1){
        const int64_t Cout = grad.shape().back();
        const int64_t out_channels = grad.shape()[1];
        const int64_t batch_size = grad.shape()[0];
        const int64_t& kC = d_kernel.shape().back();
        if(groups > 1){
            int64_t k_add = out_channels / groups;
            int64_t per_row = grad.numel() / (groups *  Cout * batch_size);
            grad = grad.view(-1, groups, per_row, Cout).transpose(0, 1);
            grad = grad.view(groups, -1, k_add, Cout).transpose(-1,-2).contiguous().split_axis(0);
            // image.RowColSwap_Tensors();
        }else{
            grad = grad.view(batch_size, out_channels, Cout);
            grad = grad.transpose(-1, -2).contiguous();
        }
        Tensor d_matmult = matmult(image, grad, false, false);
        if(batch_size > 1){
            d_matmult = d_matmult.view(batch_size, -1).sum(0, false); //take into account batches
        }
        //take into account the transpose and flip
        d_matmult = d_matmult.view(out_channels / groups, -1, kC).transpose(0, 1).flip(-1);
        d_kernel += d_matmult;
    }   
    else if(dim == 2){
        const int64_t Cout = grad.shape().back();
        const int64_t Rout = grad.shape()[2];
        const int64_t out_channels = grad.shape()[1];
        const int64_t batch_size = grad.shape()[0];
        const int64_t& kC = d_kernel.shape()[-1];
        const int64_t& kR = d_kernel.shape()[-2];
        if(groups > 1){
            int64_t k_add = out_channels / groups;
            int64_t per_row = grad.numel() / (groups * Rout * Cout * batch_size);
            grad = grad.view(-1, groups, per_row, Rout, Cout).transpose(0, 1);
            grad = grad.view(groups, -1, k_add, (Rout * Cout)).transpose(-1,-2).contiguous().split_axis(0);
            // image.RowColSwap_Tensors();
        }else{
            grad = grad.view(batch_size, out_channels, Rout * Cout);
            grad = grad.transpose(-1, -2).contiguous();
        }

        Tensor d_matmult = matmult(image, grad, false, false);
        if(batch_size > 1){
            d_matmult = d_matmult.view(batch_size, -1).sum(0, false); //take into account batches
        }
        //take into account the transpose and flip
        d_matmult = d_matmult.view(out_channels / groups, -1, kR, kC).transpose(0, 1).flip(-2).flip(-1);
        d_kernel += d_matmult;
    }
    else if(dim == 3){
        const int64_t Cout = grad.shape().back();
        const int64_t Rout = grad.shape()[-2];
        const int64_t Dout = grad.shape()[-3];
        const int64_t out_channels = grad.shape()[1];
        const int64_t batch_size = grad.shape()[0];
        const int64_t& kC = d_kernel.shape()[-1];
        const int64_t& kR = d_kernel.shape()[-2];
        const int64_t& kD = d_kernel.shape()[-3];
        if(groups > 1){
            int64_t k_add = out_channels / groups;
            int64_t per_row = grad.numel() / (groups * Dout * Rout * Cout * batch_size);
            grad = grad.view(-1, groups, per_row, Dout, Rout, Cout).transpose(0, 1);
            grad = grad.view(groups, -1, k_add, (Dout * Rout * Cout)).transpose(-1,-2).contiguous().split_axis(0);
            // image.RowColSwap_Tensors();
        }else{
            grad = grad.view(batch_size, out_channels, Rout * Cout);
            grad = grad.transpose(-1, -2).contiguous();
        }
        Tensor d_matmult = matmult(image, grad, false, false);
        if(batch_size > 1){
            d_matmult = d_matmult.view(batch_size, -1).sum(0, false); //take into account batches
        }
        //take into account the transpose and flip
        d_matmult = d_matmult.view(out_channels / groups, -1, kR, kC).transpose(0, 1).flip(-3).flip(-2).flip(-1);
        d_kernel += d_matmult;
    }
 
}

}
}
