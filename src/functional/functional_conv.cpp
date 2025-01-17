#include "functional_conv.h"
#include "conv_std.cpp"
#include "conv_dnn.cpp"
#include "../Tensor.h"
#include <memory>
#include "../layers/functional.h"

namespace nt{
namespace functional{




Tensor conv2d(const Tensor &image, const Tensor &kernel, utils::my_tuple stride, utils::my_tuple padding, utils::my_tuple dilation, int64_t groups){
	/* switch(image.dtype){ */
	/* 	case DType::Float32:{ */
	/* 		return ::nt::functional::functional_dnn::conv2d(image, kernel, stride, padding, dilation, groups); */
	/* 	} */
/* #ifdef _HALF_FLOAT_SUPPORT_ */
	/* 	case DType::Float16: */
	/* 		return ::nt::functional::functional_dnn::conv2d(image, kernel, stride, padding, dilation, groups); */
/* #endif */
	/* 	default: */
	/* 		return ::nt::functional::functional_std::conv2d(image, kernel, stride, padding, dilation, groups); */
	/* } */
	return ::nt::functional::functional_std::conv2d(image, kernel, stride, padding, dilation, groups);
}


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


}
}
