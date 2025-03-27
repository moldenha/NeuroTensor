#include "layers.h"


namespace nt{
namespace layers{


Conv2d::Conv2d(uint32_t in_channels, uint32_t out_channels, 
				utils::my_tuple kernel_size,
				utils::my_tuple dilation,
				utils::my_tuple padding,
				utils::my_tuple stride, 
				uint32_t groups, Scalar learning_rate, bool clip_grads, Scalar minClip, Scalar maxClip, bool bias, DType dtype)
	:Kernel(functional::randn({out_channels, in_channels / groups, kernel_size[0], kernel_size[1]} , dtype)),
	Bias(bias ? functional::randn({out_channels, 1, 1}, dtype) : functional::zeros({1}, dtype)),
	uf(kernel_size, dilation, padding, stride, false),
	groups(groups),
	use_bias(bias),
	u_transposed(false),
	clip(clip_grads),
	x1({0}),
	s1({0}),
	lr(learning_rate),
	clip_max(maxClip),
	clip_min(minClip) 
{
	fKernel = Kernel.view(out_channels, -1).transpose(-1,-2);
}


Tensor Conv2d::forward(const Tensor& x){
	uint32_t Rout = ((x.shape()[-2] + 2 * this->uf.padding[0] - this->uf.dilation[0] * (this->uf.kernel_size[0] - 1) - 1) / this->uf.stride[0]) + 1;
	uint32_t Cout = ((x.shape()[-1] + 2 * this->uf.padding[1] - this->uf.dilation[1] * (this->uf.kernel_size[1] - 1) - 1) / this->uf.stride[1]) + 1;
	x1 = x.shape();
	if(groups == 1){
		Tensor inp_unf = this->uf.forward(x); // this does not transpose out so (unfold = self.unfold(x).transpose(-1,-2)
		std::cout << "unfolded "<<inp_unf.shape() <<" * "<< Kernel.view(Kernel.shape()[0], -1).shape()<<std::endl;
		Tensor out_unf = functional::matmult_cT(inp_unf, Kernel.view(this->Kernel.shape()[0], -1)).RowColSwap();
		std::cout << "got output"<<std::endl;
		this->s1 = out_unf.shape();
		this->unfolded = inp_unf;
		u_transposed = false;
		out_unf = out_unf.view(x.shape()[0], -1, Rout, Cout);
		if(use_bias)
			out_unf += Bias;
		std::cout << "returning"<<std::endl;
		return std::move(out_unf);
	}
	uint32_t add = uint32_t(x.shape()[1]/groups) * this->uf.LKern;
	std::vector<Tensor> tensors(groups);
	Tensor inp_unf = this->uf.forward(x);
	Tensor inp_unf_parts = inp_unf.split_axis({my_range(0, -1), my_range(0, add)});
	Tensor* t_begin = reinterpret_cast<Tensor*>(inp_unf_parts.data_ptr());
	Tensor* t_end = t_begin + inp_unf_parts.numel();
	auto begin = tensors.begin();
	for(;t_begin != t_end; ++t_begin, ++begin)
		*begin = functional::matmult_cT(*t_begin, fKernel).RowColSwap().view(-1, Rout, Cout); 

	Tensor X = functional::cat(std::move(tensors), 1);
	if(use_bias)
		X += Bias;
	return std::move(X);
}


Tensor Conv2d::eval(const Tensor& x) const{
	uint32_t Rout = ((x.shape()[-2] + 2 * this->uf.padding[0] - this->uf.dilation[0] * (this->uf.kernel_size[0] - 1) - 1) / this->uf.stride[0]) + 1;
	uint32_t Cout = ((x.shape()[-1] + 2 * this->uf.padding[1] - this->uf.dilation[1] * (this->uf.kernel_size[1] - 1) - 1) / this->uf.stride[1]) + 1;
	if(groups == 1){
		Tensor inp_unf = this->uf.eval(x); // this does not transpose out
		Tensor out_unf = functional::matmult_cT(inp_unf, Kernel.view(this->Kernel.shape()[0], -1)).RowColSwap();
		out_unf = out_unf.view(-1, Rout, Cout);
		if(use_bias)
			out_unf += Bias;
		return std::move(out_unf);
	}
	uint32_t add = uint32_t(x.shape()[1]/groups) * this->uf.LKern;
	std::vector<Tensor> tensors(groups);
	Tensor inp_unf = this->uf.eval(x);
	Tensor inp_unf_parts = inp_unf.split_axis({my_range(0, -1), my_range(0, add)});
	Tensor* t_begin = reinterpret_cast<Tensor*>(inp_unf_parts.data_ptr());
	Tensor* t_end = t_begin + inp_unf_parts.numel();
	auto begin = tensors.begin();
	for(;t_begin != t_end; ++t_begin, ++begin)
		*begin = functional::matmult_cT(*t_begin, fKernel).RowColSwap().view(-1, Rout, Cout);

	Tensor X = functional::cat(std::move(tensors), 1);
	if(use_bias)
		X += Bias;
	return std::move(X);

}

Tensor Conv2d::backward(const Tensor& dz){
	if(groups == 1){
		std::cout << "backward"<<std::endl;
		Tensor grad_out_unf = dz.view(this->s1).contiguous().RowColSwap(); //undo the last view (which can also be a fold) but it really just changed the view
		std::cout << "grad_out_unf"<<std::endl;
		Tensor grad_input_unf = functional::matmult_cT(grad_out_unf, fKernel);
		grad_input_unf = this->uf.backward(grad_input_unf);
		std::cout << "grad_input_unf"<<std::endl;
		if(use_bias){
			Tensor dB = dz.mean(-4).clip_(clip_min, clip_max);
			std::cout << "dB"<<std::endl;
			Bias -= dB * lr;
		}
		nt::Tensor n_dz = dz.mean(-4).RowColSwap();
		std::cout << "n_dz"<<std::endl;
		if(!u_transposed){
			Kernel -= lr * nt::functional::matmult(n_dz.view(n_dz.shape()[0], -1), this->unfolded.view(-1, this->unfolded.shape()[2])).view(this->Kernel.shape()).clip_(clip_min, clip_max);
			std::cout << "kernel adjusted"<<std::endl;
			u_transposed = true;
		}
		else{
			Kernel -= lr * nt::functional::matmult_cT(dz, this->unfolded.view(-1, this->unfolded.shape()[2])).view(this->Kernel.shape()).clip_(clip_min, clip_max);
		}
		std::cout << "returning"<<std::endl;
		return grad_input_unf.view(-1, this->Kernel.shape()[1], x1[-2], x1[-1]);
	}
	return dz;
}

}
}
