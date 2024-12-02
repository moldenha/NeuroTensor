#include "functional_conv.h"
#include "../layers/TensorGrad.h"
#include "../Tensor.h"
#include <memory>

namespace nt{
namespace functional{
namespace functional_std{



//grouped convolution numpy implementation
/*
 * def conv2d_grouped(input, kernel, groups, stride=(1, 1), padding=(0, 0)):
    # Input: (batch_size, in_channels, height, width)
    # Kernel: (out_channels, in_channels/groups, kernel_height, kernel_width)
    batch_size, in_channels, height, width = input.shape
    out_channels, _, kernel_height, kernel_width = kernel.shape
    
    # Ensure in_channels and out_channels are divisible by the number of groups
    group_in_channels = in_channels // groups
    group_out_channels = out_channels // groups
    
    # Apply padding
    padded_input = np.pad(input, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])), mode='constant')

    # Calculate output dimensions
    out_height = (height + 2 * padding[0] - kernel_height) // stride[0] + 1
    out_width = (width + 2 * padding[1] - kernel_width) // stride[1] + 1

    # Initialize output
    output = np.zeros((batch_size, out_channels, out_height, out_width))

    # Perform grouped convolution
    for b in range(batch_size):
        for g in range(groups):
            # Select input and kernel slices corresponding to the current group
            input_group = padded_input[b, g * group_in_channels:(g + 1) * group_in_channels, :, :]
            kernel_group = kernel[g * group_out_channels:(g + 1) * group_out_channels, :, :, :]
            
            for o in range(group_out_channels):
                for i in range(0, out_height):
                    for j in range(0, out_width):
                        h_start = i * stride[0]
                        w_start = j * stride[1]
                        h_end = h_start + kernel_height
                        w_end = w_start + kernel_width
                        # Sum over all channels in the group
                        output[b, g * group_out_channels + o, i, j] = np.sum(
                            input_group[:, h_start:h_end, w_start:w_end] * kernel_group[o, :, :, :]
                        )

    return output*/


Tensor grouped_conv2d(const Tensor &image, const Tensor &kernel, utils::my_tuple& stride, utils::my_tuple& padding, utils::my_tuple& dilation, int64_t& groups){
	utils::throw_exception(kernel.dims() == 3 || kernel.dims() == 4, "Expected the input kernel to convolution to be 3D or 4D but is $D", kernel.dims());
	utils::throw_exception(image.dims() == 3 || image.dims() == 4, "Expected input image to convolution to be 3D or 4D but got $D", image.dims());
	Tensor x = (image.dims() == 3) ? image.unsqueeze(0) : image;
	Tensor w = kernel.dims() == 3 ? kernel.unsqueeze(0) : kernel;
	utils::throw_exception(w.shape()[1] * groups == x.shape()[1], "Expected channels of the kernel to equal the input channels of the image but got $ and $", w.shape()[1], x.shape()[1]);
	utils::throw_exception(w.shape()[0] % groups == 0, "Expected the output channels, being the kernel's shape at dimension 0 ($) to be divisible by groups ($) but is not", w.shape()[0], groups);

	int64_t Rout = ((image.shape()[-2] + 2 * padding[0] - dilation[0] * (w.shape()[-2] - 1) - 1) / stride[0]) + 1;
	int64_t Cout = ((image.shape()[-1] + 2 * padding[1] - dilation[1] * (w.shape()[-1] - 1) - 1) / stride[1]) + 1;
	//set padding to 0
	Tensor inp_unfold = unfold(x, {kernel.shape()[-2], kernel.shape()[-1]}, dilation, padding, stride, true);
	int64_t add = int(x.shape()[1] / groups) * (w.shape()[-1] * w.shape()[-2]);
	//should be kernel channels * kernel_rows * kernel_cols
	//this is based on after unfolding, there is a shape change to where the channels are now multiplied by the kernel (r,c)
	int64_t k_add = w.shape()[0] / groups;
	//this is what to seperate the kernel by
	//the really nice thing about the way this is split
	//is that as far as the multiplication is concerned, because the only thing not contiguous will be the batches,
	//this basically is contiguous because of the way it is handeled with pointers
	Tensor x_parts = inp_unfold.split_axis({my_range(0, inp_unfold.shape()[0]), my_range(0, add)});
	//this is just going to be contiguous assuming the kernel is contiguous
	Tensor k_parts = w.split_axis({my_range(0, k_add)});
	//I added an optimized way to just do a straight forward multiplication of tensors
	Tensor output = matmult(x_parts, k_parts.view_Tensors(k_add, -1), true, true).RowColSwap().view(-1, x.shape()[0], Rout, Cout);
	utils::throw_exception(output.dtype != DType::TensorObj, "Should not have a tensor object exit a matmult"); //
	return output.transpose(0,1).contiguous();

}

Tensor conv2d(const Tensor &image, const Tensor &kernel, utils::my_tuple stride, utils::my_tuple padding, utils::my_tuple dilation, int64_t groups){
	utils::throw_exception(groups >= 1, "Cannot have less than 1 group for convolution");
	if(groups > 1){
		return grouped_conv2d(image, kernel, stride, padding, dilation, groups);
	}
	utils::throw_exception(kernel.dims() == 3 || kernel.dims() == 4, "Expected the input kernel to convolution to be 3D or 4D but is $D", kernel.dims());
	utils::throw_exception(image.dims() == 3 || image.dims() == 4, "Expected input image to convolution to be 3D or 4D but got $D", image.dims());
	Tensor x = (image.dims() == 3) ? image.unsqueeze(0) : image;
	/* x = x.pad({padding[0], padding[0], padding[1], padding[1]}); */
	Tensor w = kernel.dims() == 3 ? kernel.unsqueeze(0) : kernel;
	utils::throw_exception(w.shape()[1] == x.shape()[1], "Expected channels of the kernel to equal the input channels of the image but got $ and $", w.shape()[1], x.shape()[1]);
	int64_t Rout = ((image.shape()[-2] + 2 * padding[0] - dilation[0] * (w.shape()[-2] - 1) - 1) / stride[0]) + 1;
	int64_t Cout = ((image.shape()[-1] + 2 * padding[1] - dilation[1] * (w.shape()[-1] - 1) - 1) / stride[1]) + 1;
	//set padding to 0
	Tensor inp_unfold = unfold(x, {kernel.shape()[-2], kernel.shape()[-1]}, dilation, padding, stride, true);
	//the false means inp_unfold is not transposed. while w.view(w.shape()[0], -1) is transposed
	Tensor outp_unfold = matmult(inp_unfold, w.view(w.shape()[0], -1), true, true).RowColSwap(); //contiguous in place transpose
	return outp_unfold.view(x.shape()[0], -1, Rout, Cout);

}


//prep to this point:
//Tensor outp_unfold = matmult(inp_unfold, w.view(w.shape()[0], -1), false, true).RowColSwap();
//SizeRef s1 = outp_unfold.shape();
//dz = dz.view(s1).RowColSwap();
//
TensorGrad conv2d(const TensorGrad& image, const TensorGrad& kernel, utils::my_tuple stride, utils::my_tuple padding, utils::my_tuple dilation, int64_t groups){
	utils::throw_exception(groups >= 1, "Cannot have less than 1 group for convolution");
	utils::throw_exception(kernel.dims() == 3 || kernel.dims() == 4, "Expected the input kernel to convolution to be 3D or 4D but is $D", kernel.dims());
	utils::throw_exception(image.dims() == 3 || image.dims() == 4, "Expected input image to convolution to be 3D or 4D but got $D", image.dims());
	Tensor x = (image.dims() == 3) ? image.tensor.unsqueeze(0) : image.tensor;
	/* x = x.pad({padding[0], padding[0], padding[1], padding[1]}); */
	Tensor w = kernel.dims() == 3 ? kernel.tensor.unsqueeze(0) : kernel.tensor;
	utils::throw_exception(w.shape()[1] * groups == x.shape()[1], "Expected channels of the kernel to equal the input channels of the image but got $ and $", w.shape()[1], x.shape()[1]);
	utils::throw_exception(w.shape()[0] % groups == 0, "Expected the output channels, being the kernel's shape at dimension 0 ($) to be divisible by groups ($) but is not", w.shape()[0], groups);

	int64_t Rout = ((image.shape()[-2] + 2 * padding[0] - dilation[0] * (w.shape()[-2] - 1) - 1) / stride[0]) + 1;
	int64_t Cout = ((image.shape()[-1] + 2 * padding[1] - dilation[1] * (w.shape()[-1] - 1) - 1) / stride[1]) + 1;
	//set padding to 0
	Tensor inp_unfold = unfold(x, {kernel.shape()[-2], kernel.shape()[-1]}, dilation, padding, stride, true);
	if(groups > 1){
		SizeRef d_unfold_shape = inp_unfold.shape();
		int64_t add = int(x.shape()[1] / groups) * (w.shape()[-1] * w.shape()[-2]);
		//should be kernel channels * kernel_rows * kernel_cols
		//this is based on after unfolding, there is a shape change to where the channels are now multiplied by the kernel (r,c)
		int64_t k_add = w.shape()[0] / groups;
		//this is what to seperate the kernel by
		//the really nice thing about the way this is split
		//is that as far as the multiplication is concerned, because the only thing not contiguous will be the batches,
		//this basically is contiguous because of the way it is handeled with pointers
		Tensor x_parts = inp_unfold.split_axis({my_range(0, inp_unfold.shape()[0]), my_range(0, add)});
		//this is just going to be contiguous assuming the kernel is contiguous
		Tensor k_parts = w.split_axis({my_range(0, k_add)});

		intrusive_ptr<tensor_holder> original_w = make_intrusive<tensor_holder>(k_parts.clone());
		intrusive_ptr<tensor_holder> original_x = make_intrusive<tensor_holder>(x_parts);
		//I added an optimized way to just do a straight forward multiplication of tensors
		Tensor output = matmult(x_parts, k_parts.view_Tensors(k_add, -1), true, true).RowColSwap();
		const SizeRef& s1 = output.shape();
		output = output.view(-1, x.shape()[0], Rout, Cout);
		utils::throw_exception(output.dtype != DType::TensorObj, "Should not have a tensor object exit a matmult"); //
		TensorGrad result(output.transpose(0,1).contiguous());
		result.track_tensors(image, kernel);
		result.create_backward_function([d_unfold_shape, s1, stride, padding, dilation, add, k_add](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
						intrusive_ptr<tensor_holder> w, intrusive_ptr<tensor_holder> img){
			Tensor grad_s1 = grad.transpose(0,1).split_axis(0).view_Tensors(s1.pop_front());
			grad_s1.RowColSwap_Tensors();
			Tensor dw = matmult(img->tensor, grad_s1, true, false);
			parents[1]->grad->tensor.split_axis({my_range(0, k_add)}) += dw;
		
			Tensor d_unfold = zeros(d_unfold_shape, grad.dtype);
			d_unfold.split_axis({my_range(0,d_unfold.shape()[0]), my_range(0, add)}).set_(matmult(grad_s1, w->tensor, false, true));
			
			utils::my_tuple output_size(parents[0]->grad->tensor.shape()[-2], parents[0]->grad->tensor.shape()[-1]);
			unfold_backward(d_unfold, parents[0]->grad->tensor, output_size, 
					{parents[1]->grad->tensor.shape()[-2], parents[1]->grad->tensor.shape()[-1]},
					dilation, padding, stride, true);

		}, k_parts, x_parts);	
		return std::move(result);
	}

	intrusive_ptr<tensor_holder> original_w = make_intrusive<tensor_holder>(w.view(w.shape()[0], -1).clone());
	intrusive_ptr<tensor_holder> original_x = make_intrusive<tensor_holder>(inp_unfold);
	Tensor outp_unfold = matmult(inp_unfold, w.view(w.shape()[0], -1), true, true).RowColSwap();
	const SizeRef& s1 = outp_unfold.shape();
	TensorGrad result(outp_unfold.view(x.shape()[0], -1, Rout, Cout));
	result.track_tensors(image, kernel);
	result.create_backward_function([s1, stride, padding, dilation](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
				intrusive_ptr<tensor_holder> w, intrusive_ptr<tensor_holder> original_x){
		Tensor grad_s1 = grad.view(s1);
		grad_s1.RowColSwap();
		Tensor dw = matmult(original_x->tensor, grad_s1, false, true);
		parents[1]->grad->tensor += dw.view(parents[1]->grad->tensor.shape());

		Tensor d_unfold = matmult(grad_s1, w->tensor, true, false);
		utils::my_tuple output_size(parents[0]->grad->tensor.shape()[-2], parents[0]->grad->tensor.shape()[-1]);
		unfold_backward(d_unfold, parents[0]->grad->tensor, output_size, 
				{parents[1]->grad->tensor.shape()[-2], parents[1]->grad->tensor.shape()[-1]},
				dilation, padding, stride, true);
					
	
	}, original_w, original_x);
	
	return std::move(result);


}

//this is going to be the forward and backward functions of conv2d for TensorGrad
/* TensorGrad conv2d_oneGroup(const TensorGrad& image, const TensorGrad& kernel, utils::my_tuple stride, utils::my_tuple padding, utils::my_tuple dilation){ */
/* 	utils::throw_exception(kernel.dims() == 3 || kernel.dims() == 4, "Expected the input kernel to convolution to be 3D or 4D but is $D", kernel.dims()); */
/* 	utils::throw_exception(image.dims() == 3 || image.dims() == 4, "Expected input image to convolution to be 3D or 4D but got $D", image.dims()); */

/* 	Tensor x = (image.dims() == 3) ? image.tensor.unsqueeze(0) : image.tensor; */
/* 	Tensor w = kernel.dims() == 3 ? kernel.tensor.unsqueeze(0) : kernel.tensor; */
/* 	intrusive_ptr<tensor_holder> original_w = make_intrusive<tensor_holder>(w.view(w.shape()[0], -1).clone()); */

/* 	int64_t Rout = ((image.shape()[-2] + 2 * padding[0] - dilation[0] * (w.shape()[-2] - 1) - 1) / stride[0]) + 1; */
/* 	int64_t Cout = ((image.shape()[-1] + 2 * padding[1] - dilation[1] * (w.shape()[-1] - 1) - 1) / stride[1]) + 1; */
	
/* 	Tensor inp_unfold = unfold(x, {kernel.shape()[-2], kernel.shape()[-1]}, dilation, padding, stride, true); */

/* 	intrusive_ptr<tensor_holder> original_x = make_intrusive<tensor_holder>(inp_unfold); */
/* 	Tensor outp_unfold = matmult(inp_unfold, w.view(w.shape()[0], -1), true, true).RowColSwap(); */
/* 	const SizeRef& s1 = outp_unfold.shape(); */
/* 	TensorGrad result(outp_unfold.view(x.shape()[0], -1, Rout, Cout)); */
/* 	result.track_tensors(image, kernel); */
/* 	result.create_backward_function([s1, stride, padding, dilation](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents, */
/* 				intrusive_ptr<tensor_holder> w, intrusive_ptr<tensor_holder> original_x){ */
/* 		Tensor grad_s1 = grad.view(s1); */
/* 		grad_s1.RowColSwap(); */
/* 		Tensor dw = matmult(original_x->tensor, grad_s1, false, true); */
/* 		parents[1]->grad->tensor += dw.view(parents[1]->grad->tensor.shape()); */

/* 		Tensor d_unfold = matmult(grad_s1, w->tensor, true, false); */
/* 		utils::my_tuple output_size(parents[0]->grad->tensor.shape()[-2], parents[0]->grad->tensor.shape()[-1]); */
/* 		unfold_backward(d_unfold, parents[0]->grad->tensor, output_size, */ 
/* 				{parents[1]->grad->tensor.shape()[-2], parents[1]->grad->tensor.shape()[-1]}, */
/* 				dilation, padding, stride, true); */
					
	
/* 	}, original_w, original_x); */
	
/* 	return std::move(result); */
/* } */

/* TensorGrad conv2d_multiGroup(const TensorGrad& image, const TensorGrad& kernel, utils::my_tuple stride, utils::my_tuple padding, utils::my_tuple dilation, int64_t groups){ */
	
/* 	utils::throw_exception(kernel.dims() == 3 || kernel.dims() == 4, "Expected the input kernel to convolution to be 3D or 4D but is $D", kernel.dims()); */
/* 	utils::throw_exception(image.dims() == 3 || image.dims() == 4, "Expected input image to convolution to be 3D or 4D but got $D", image.dims()); */
/* 	Tensor x = (image.dims() == 3) ? image.tensor.unsqueeze(0) : image.tensor; */
/* 	/1* x = x.pad({padding[0], padding[0], padding[1], padding[1]}); *1/ */
/* 	Tensor w = kernel.dims() == 3 ? kernel.tensor.unsqueeze(0) : kernel.tensor; */
/* 	utils::throw_exception(w.shape()[1] * groups == x.shape()[1], "Expected channels of the kernel to equal the input channels of the image but got $ and $", w.shape()[1], x.shape()[1]); */
/* 	utils::throw_exception(w.shape()[0] % groups == 0, "Expected the output channels, being the kernel's shape at dimension 0 ($) to be divisible by groups ($) but is not", w.shape()[0], groups); */

/* 	int64_t Rout = ((image.shape()[-2] + 2 * padding[0] - dilation[0] * (w.shape()[-2] - 1) - 1) / stride[0]) + 1; */
/* 	int64_t Cout = ((image.shape()[-1] + 2 * padding[1] - dilation[1] * (w.shape()[-1] - 1) - 1) / stride[1]) + 1; */
/* 	//set padding to 0 */
/* 	Tensor inp_unfold = unfold(x, {kernel.shape()[-2], kernel.shape()[-1]}, dilation, padding, stride, true); */
/* 	SizeRef d_unfold_shape = inp_unfold.shape(); */

/* 	int64_t add = int(x.shape()[1] / groups) * (w.shape()[-1] * w.shape()[-2]); */
/* 	//should be kernel channels * kernel_rows * kernel_cols */
/* 	//this is based on after unfolding, there is a shape change to where the channels are now multiplied by the kernel (r,c) */
/* 	int64_t k_add = w.shape()[0] / groups; */
/* 	//this is what to seperate the kernel by */
/* 	//the really nice thing about the way this is split */
/* 	//is that as far as the multiplication is concerned, because the only thing not contiguous will be the batches, */
/* 	//this basically is contiguous because of the way it is handeled with pointers */
/* 	Tensor x_parts = inp_unfold.split_axis({my_range(0, -1), my_range(0, add)}); */
/* 	//this is just going to be contiguous assuming the kernel is contiguous */
/* 	Tensor k_parts = w.split_axis({my_range(0, k_add)}); */

/* 	intrusive_ptr<tensor_holder> original_w(k_parts); */
/* 	intrusive_ptr<tensor_holder> original_x(x_parts); */
/* 	//I added an optimized way to just do a straight forward multiplication of tensors */
/* 	Tensor output = matmult(x_parts, k_parts.view_Tensors(k_add, -1), true, true).RowColSwap(); */
/* 	const SizeRef& s1 = output.shape(); */
/* 	output = output.view(x.shape()[0], -1, Rout, Cout); */
/* 	utils::throw_exception(output.dtype != DType::TensorObj, "Should not have a tensor object exit a matmult"); // */
/* 	TensorGrad result(output.transpose(0,1).contiguous()); */
/* 	result.track_tensors(image, kernel); */
/* 	result.create_backward_function([d_unfold_shape, s1, stride, padding, dilation, add, k_add](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents, */
/* 					intrusive_ptr<tensor_holder> w, intrusive_ptr<tensor_holder> img){ */
/* 		Tensor grad_s1 = grad.transpose(0,1).split_axis(0).view_Tensors(s1[my_range(1,-1)]); */
/* 		grad_s1.RowColSwap_Tensors(); */
/* 		Tensor dw = matmult(img->tensor, grad_s1, true, false); */
/* 		parents[1]->grad->tensor.split_axis({my_range(0, k_add)}) += dw; */
	
/* 		Tensor d_unfold = zeros(d_unfold_shape, grad.dtype); */
/* 		d_unfold.split_axis({my_range(0,-1), my_range(0, add)}).set_(matmult(grad_s1, w->tensor, false, true)); */
		
/* 		utils::my_tuple output_size(parents[0]->grad->tensor.shape()[-2], parents[0]->grad->tensor.shape()[-1]); */
/* 		unfold_backward(d_unfold, parents[0]->grad->tensor, output_size, */ 
/* 				{parents[1]->grad->tensor.shape()[-2], parents[1]->grad->tensor.shape()[-1]}, */
/* 				dilation, padding, stride, true); */

/* 	}, k_parts, x_parts); */	
/* 	return std::move(result); */
/* } */




Tensor conv3d(const Tensor& image, const Tensor& kernel, utils::my_n_tuple<3> stride, utils::my_n_tuple<3> padding, utils::my_n_tuple<3> dilation, int64_t groups){
	utils::throw_exception(image.dims() >= 4, "Expected to get a 4D or greater tensor as the input for a 3d convolution, but got $D", image.dims());
	utils::throw_exception(kernel.dims() == 5 || kernel.dims() == 4, "Expected to get a 4D or 5D tensor as the kernel for a 3d convolution, but got $D", kernel.dims());
	utils::throw_exception(image.dtype == kernel.dtype, "Expected both kernel and image to have the same dtype for conv3d but got $ and $", kernel.dtype, image.dtype);
	utils::throw_exception(image.dtype != DType::Bool && image.dtype != DType::TensorObj, "Expected to get a number type for image and kernel but got $", image.dtype);
	Tensor x = (image.dims() == 4) ? image.unsqueeze(0) : image.flatten(0, -5);
	/* x = x.pad({padding[0], padding[0], padding[1], padding[1], padding[2], padding[2]}); */
	Tensor w = kernel.dims() == 4 ? kernel.unsqueeze(0) : kernel;
	
	const int64_t Dout = ((image.shape()[-3] + 2 * padding[0] - dilation[0] * (w.shape()[-3] - 1) - 1) / stride[0]) + 1;
	const int64_t Rout = ((image.shape()[-2] + 2 * padding[1] - dilation[1] * (w.shape()[-2] - 1) - 1) / stride[1]) + 1;
	const int64_t Cout = ((image.shape()[-1] + 2 * padding[2] - dilation[2] * (w.shape()[-1] - 1) - 1) / stride[2]) + 1;
	Tensor inp_unfold = unfold3d(x, {w.shape()[-3], w.shape()[-2], w.shape()[-1]}, dilation, padding, stride, false);
	if(groups == 1){
		Tensor outp_unfold = matmult(inp_unfold, w.view(w.shape()[0], -1), false, true).RowColSwap(); //contiguous in place transpose
		return outp_unfold.view(x.shape()[0], -1, Dout, Rout, Cout);
	}

	int64_t add = int64_t(x.shape()[1]/groups) * (w.shape()[-1] * w.shape()[-2]);
	std::unique_ptr<Tensor> tensors(new Tensor(Tensor::makeNullTensorArray(groups)));
	Tensor inp_unf_parts = inp_unfold.split_axis({my_range(0, -1), my_range(0, add)});
	Tensor fKernel = w.view(w.shape()[0], -1);
	Tensor* t_begin = reinterpret_cast<Tensor*>(inp_unf_parts.data_ptr());
	Tensor* t_end = t_begin + inp_unf_parts.numel();
	Tensor* begin = reinterpret_cast<Tensor*>(tensors->data_ptr());
	for(;t_begin != t_end; ++t_begin, ++begin)
		*begin = matmult(*t_begin, fKernel, false, true).RowColSwap().view(x.shape()[0], -1, Dout, Rout, Cout);
	Tensor catted = cat(*tensors, 1);
	tensors.reset(nullptr); //release memory, this way when contiguous is called there isn't double the memory used
	return catted.contiguous();
}


Tensor conv1d(const Tensor& image, const Tensor& kernel, Tensor::size_value_t stride, Tensor::size_value_t padding, Tensor::size_value_t dilation, int64_t groups){
	utils::throw_exception(image.dims() >= 2, "Expected to get a 2D or greater tensor as the input for a 1d convolution, but got $D", image.dims());
	utils::throw_exception(kernel.dims() == 3 || kernel.dims() == 2, "Expected to get a 3D or 2D tensor as the kernel for a 1d convolution, but got $D", kernel.dims());
	utils::throw_exception(image.dtype == kernel.dtype, "Expected both kernel and image to have the same dtype for conv1d but got $ and $", kernel.dtype, image.dtype);
	utils::throw_exception(image.dtype != DType::Bool && image.dtype != DType::TensorObj, "Expected to get a number type for image and kernel but got $", image.dtype);
	Tensor x = (image.dims() == 2) ? image.unsqueeze(0) : image.flatten(0, -3);
	/* x = x.pad({padding[0], padding[0], padding[1], padding[1], padding[2], padding[2]}); */
	Tensor w = kernel.dims() == 2 ? kernel.unsqueeze(0) : kernel;
	
	const int64_t Cout = ((image.shape()[-1] + 2 * padding - dilation * (w.shape()[-1] - 1) - 1) / stride) + 1;
	Tensor inp_unfold = unfold1d(x, w.shape()[-1], dilation, padding, stride, false);
	if(groups == 1){
		Tensor outp_unfold = matmult(inp_unfold, w.view(w.shape()[0], -1), false, true).RowColSwap(); //contiguous in place transpose
		return outp_unfold.view(x.shape()[0], -1, Cout);
	}

	int64_t add = int64_t(x.shape()[1]/groups) * (w.shape()[-1] * w.shape()[-2]);
	std::unique_ptr<Tensor> tensors(new Tensor(Tensor::makeNullTensorArray(groups)));
	Tensor inp_unf_parts = inp_unfold.split_axis({my_range(0, -1), my_range(0, add)});
	Tensor fKernel = w.view(w.shape()[0], -1);
	Tensor* t_begin = reinterpret_cast<Tensor*>(inp_unf_parts.data_ptr());
	Tensor* t_end = t_begin + inp_unf_parts.numel();
	Tensor* begin = reinterpret_cast<Tensor*>(tensors->data_ptr());
	for(;t_begin != t_end; ++t_begin, ++begin)
		*begin = matmult(*t_begin, fKernel, false, true).RowColSwap().view(x.shape()[0], -1, Cout);
	Tensor catted = cat(*tensors, 1);
	tensors.reset(nullptr); //release memory, this way when contiguous is called there isn't double the memory used
	return catted.contiguous();
}
/*
'''
a reiteration of a pytorch version:
def my_conv_transpose2d(x, w, stride, padding, output_padding, dilation):
	#just going to assume it has the proper dimensions heading into it
	kernel_shape = (w.shape[-2], w.shape[-1])
	Hout = (x.shape[-2] - 1) * stride[0] - 2 * padding[0] + dilation[0] * (w.shape[-2] - 1) + 1 
	Wout = (x.shape[-1] - 1) * stride[1] - 2 * padding[1] + dilation[1] * (w.shape[-1] - 1) + 1 
	w_transpose = torch.flip(w.transpose(0, 1), [2,3])
	print("w transpose shape: {}, x shape: {}".format(w_transpose.shape, x.shape))
	inp_unfold = torch.nn.functional.unfold(x, (w.shape[-2], w.shape[-1]), dilation, (kernel_shape[0] - 1 + padding[0], kernel_shape[1] - 1 + padding[1]), stride)
	print("inp_unfold shape: {}, w_transpose re-shape: {}".format(inp_unfold.shape, w_transpose.reshape(w_transpose.shape[0], -1).transpose(-1,-2).shape))
	print("inp_unfold shape: {}, w_transpose re-shape: {}".format(inp_unfold.shape, w_transpose.reshape(w_transpose.shape[0], -1).shape))
	outp_unfold = torch.matmul(w_transpose.reshape(w_transpose.shape[0], -1), inp_unfold)
	print("outp_unfold shape: {}".format(outp_unfold.shape))
	outp_fold = outp_unfold.reshape(x.shape[0], w.shape[1], Hout, Wout)
	print("outp_fold shape: {}".format(outp_fold.shape))
	if(output_padding != (0,0)):
		return outp_fold.pad(((output_padding[0], output_padding[0]), (output_padding[1], output_padding[1])))
	return outp_fold



def test_my_conv_transpose2d():
	def run_test(x, w, stride, padding, output_padding, dilation):
		custom_output = my_conv_transpose2d(x, w, stride, padding, output_padding, dilation)
		torch_output = torch.nn.functional.conv_transpose2d(x, w, stride=stride, padding=padding, output_padding=output_padding, dilation=dilation)
		print("custom output shape: {}, torch output size: {}".format(custom_output.shape, torch_output.shape))
		assert torch.allclose(custom_output, torch_output, atol=1e-5), \
		f"Test failed for input {x.shape}, weight {w.shape}, stride {stride}, padding {padding}, output_padding {output_padding}, dilation {dilation}"
	# Example test cases
	x = torch.randn(1, 3, 5, 5)
	w = torch.randn(3, 2, 3, 3)
	run_test(x, w, (1, 1), (0, 0), (0, 0), (1, 1))
	x = torch.randn(2, 3, 10, 10)
	w = torch.randn(3, 4, 5, 5)
	run_test(x, w, (2, 2), (1, 1), (0, 0), (1, 1))
	x = torch.randn(1, 3, 8, 8)
	w = torch.randn(3, 1, 4, 4)
	run_test(x, w, (1, 1), (1, 1), (1, 1), (2, 2))
	print("All tests passed!")


'''
*/

/* Tensor conv_transpose2d(const Tensor &image, const Tensor &kernel, utils::my_tuple stride, utils::my_tuple padding, utils::my_tuple output_padding, utils::my_tuple dilation, int64_t groups) { */
/*     utils::throw_exception(kernel.dims() == 3 || kernel.dims() == 4, "Expected the input kernel to convolution to be 3D or 4D but is $D", kernel.dims()); */
/*     utils::throw_exception(image.dims() == 3 || image.dims() == 4, "Expected input image to convolution to be 3D or 4D but got $D", image.dims()); */
/*     Tensor x = (image.dims() == 3) ? image.unsqueeze(0) : image; */
/*     Tensor w = kernel.dims() == 3 ? kernel.unsqueeze(0) : kernel; */

/*     /1* int64_t Hout = (x.shape()[-2] - 1) * stride[0] - 2 * padding[0] + dilation[0] * (w.shape()[-2] - 1) + 1 + output_padding[0]; *1/ */
/*     /1* int64_t Wout = (x.shape()[-1] - 1) * stride[1] - 2 * padding[1] + dilation[1] * (w.shape()[-1] - 1) + 1 + output_padding[1]; *1/ */
/*     int64_t Hout = (x.shape()[-2] - 1) * stride[0] - 2 * padding[0] + dilation[0] * (w.shape()[-2] - 1) + 1; */
/*     int64_t Wout = (x.shape()[-1] - 1) * stride[1] - 2 * padding[1] + dilation[1] * (w.shape()[-1] - 1) + 1; */

/*     Tensor w_transpose = w.transpose(0, 1).flip(2).flip(3);  // Transpose and flip the kernel */
/*     Tensor inp_unfold = unfold(x, {w_transpose.shape()[-2], w_transpose.shape()[-3]}, dilation, padding, stride, true); */

/*     if (groups == 1) { */
/*         Tensor outp_unfold = matmult(w_transpose.view(w_transpose.shape()[0], -1), inp_unfold, true, false).view(x.shape()[0], w.shape()[1], Hout, Wout); */
/* 	if(output_padding != 0) */
/* 		return outp_unfold.pad({{output_padding[0], output_padding[0]}, {output_padding[1], output_padding[1]}}); */
/* 	return outp_unfold; */
/*     } */

/*     int64_t add = int64_t(x.shape()[1] / groups) * (w_transpose.shape()[-1] * w_transpose.shape()[-2]); */
/*     std::unique_ptr<Tensor> tensors(new Tensor(Tensor::makeNullTensorArray(groups))); */
/*     Tensor inp_unf_parts = inp_unfold.split_axis({my_range(0, -1), my_range(0, add)}); */
/*     Tensor fKernel = w_transpose.view(w_transpose.shape()[0], -1).transpose(1, 0); */
/*     Tensor* t_begin = reinterpret_cast<Tensor*>(inp_unf_parts.data_ptr()); */
/*     Tensor* t_end = t_begin + inp_unf_parts.numel(); */
/*     Tensor* begin = reinterpret_cast<Tensor*>(tensors->data_ptr()); */
/*     for (; t_begin != t_end; ++t_begin, ++begin) */
/*         *begin = matmult(w_transpose.view(w_transpose.shape()[0], -1), inp_unfold, true, false).view(x.shape()[0], -1, Hout, Wout);; */
/*     Tensor catted = cat(*tensors, 1); */
/*     tensors.reset(nullptr);  // Release memory */
/*     if(output_padding == 0) */
/* 	    return catted.contiguous(); */
/*     return catted.pad({{output_padding[0], output_padding[0]}, {output_padding[1], output_padding[1]}}); */
/* } */


/* Tensor conv_transpose3d(const Tensor &image, const Tensor &kernel, utils::my_n_tuple<3> stride, utils::my_n_tuple<3> padding, utils::my_n_tuple<3> output_padding, utils::my_n_tuple<3> dilation, int64_t groups) { */
/*     utils::throw_exception(kernel.dims() == 4 || kernel.dims() == 5, "Expected the input kernel to convolution to be 5D or 4D but is $D", kernel.dims()); */
/*     utils::throw_exception(image.dims() == 4 || image.dims() == 5, "Expected input image to convolution to be 5D or 4D but got $D", image.dims()); */
/*     Tensor x = (image.dims() == 4) ? image.unsqueeze(0) : image; */
/*     Tensor w = kernel.dims() == 4 ? kernel.unsqueeze(0) : kernel; */

/*     /1* int64_t Dout = (x.shape()[-3] - 1) * stride[0] - 2 * padding[0] + dilation[0] * (w.shape()[-3] - 1) + 1 + output_padding[0]; *1/ */
/*     /1* int64_t Hout = (x.shape()[-2] - 1) * stride[1] - 2 * padding[1] + dilation[1] * (w.shape()[-2] - 1) + 1 + output_padding[1]; *1/ */
/*     /1* int64_t Wout = (x.shape()[-1] - 1) * stride[2] - 2 * padding[2] + dilation[2] * (w.shape()[-1] - 1) + 1 + output_padding[2]; *1/ */
/*     int64_t Dout = (x.shape()[-3] - 1) * stride[0] - 2 * padding[0] + dilation[0] * (w.shape()[-3] - 1) + 1; */
/*     int64_t Hout = (x.shape()[-2] - 1) * stride[1] - 2 * padding[1] + dilation[1] * (w.shape()[-2] - 1) + 1; */
/*     int64_t Wout = (x.shape()[-1] - 1) * stride[2] - 2 * padding[2] + dilation[2] * (w.shape()[-1] - 1) + 1; */

/*     Tensor w_transpose = w.transpose(0, 1).flip(2).flip(3).flip(4);  // Transpose and flip the kernel */
/*     Tensor inp_unfold = unfold3d(x, {w_transpose.shape()[-3], w_transpose.shape()[-2], w_transpose.shape()[-3]}, dilation, padding, stride, true); */

/*     if (groups == 1) { */
/*         Tensor outp_unfold = matmult(w_transpose.view(w_transpose.shape()[0], -1), inp_unfold, true, false).view(x.shape()[0], w.shape()[1], Dout, Hout, Wout); */
/*         if (output_padding != 0) */
/*             return outp_unfold.pad({{output_padding[0], output_padding[0]}, {output_padding[1], output_padding[1]}, {output_padding[2], output_padding[2]}}); */
/*         return outp_unfold; */
/*     } */


/*     int64_t add = int64_t(x.shape()[1] / groups) * (w_transpose.shape()[-1] * w_transpose.shape()[-2] * w_transpose.shape()[-3]); */
/*     std::unique_ptr<Tensor> tensors(new Tensor(Tensor::makeNullTensorArray(groups))); */
/*     Tensor inp_unf_parts = inp_unfold.split_axis({my_range(0, -1), my_range(0, add)}); */
/*     Tensor fKernel = w_transpose.view(w_transpose.shape()[0], -1).transpose(1, 0); */
/*     Tensor* t_begin = reinterpret_cast<Tensor*>(inp_unf_parts.data_ptr()); */
/*     Tensor* t_end = t_begin + inp_unf_parts.numel(); */
/*     Tensor* begin = reinterpret_cast<Tensor*>(tensors->data_ptr()); */
/*     for (; t_begin != t_end; ++t_begin, ++begin) */
/*         *begin = matmult(fKernel, *t_begin).view(x.shape()[0], -1, Dout, Hout, Wout); */
/*     Tensor catted = cat(*tensors, 1); */
/*     tensors.reset(nullptr);  // Release memory */
/*     if (output_padding == 0) */
/*         return catted.contiguous(); */
/*     return catted.pad({{output_padding[0], output_padding[0]}, {output_padding[1], output_padding[1]}, {output_padding[2], output_padding[2]}}); */
/* } */
}}} //::nt::functional::functional_std
