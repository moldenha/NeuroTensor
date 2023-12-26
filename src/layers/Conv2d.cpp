#include "layers.h"


namespace nt{
namespace layers{

void Conv2d::dialate_kernel(uint32_t dilation){
	if(dilation == 1){
		KernelF = Kernel.flip_(-1).flip_(-2);
		return;
	}
	Dilated = Kernel.dilate_(dilation);
	KernelF = Dilated.flip_(-1).flip_(-2);
}


SizeRef Conv2d::get_kernel_size(uint32_t in_channels, uint32_t out_channels, uint32_t groups, std::variant<uint32_t, std::tuple<uint32_t, uint32_t>> &s){
	utils::throw_exception(out_channels % groups == 0 && in_channels % groups == 0, "Runtime Error: Expected in_channels and out_channels to be divisible by groups $", groups);
	in_channels = uint32_t(in_channels / groups);
	out_channels = uint32_t(out_channels / groups);
	if(const auto linearPtr (std::get_if<uint32_t>(&s)); linearPtr){
		return SizeRef({out_channels, in_channels, *linearPtr, *linearPtr});
	}
	std::tuple<uint32_t, uint32_t>& t = std::get<1>(s);
	return SizeRef({out_channels, in_channels, std::get<0>(t), std::get<1>(t)});
	
}

void Conv2d::set_stride(std::variant<uint32_t, std::tuple<uint32_t, uint32_t>> &s){
	if(const auto linearPtr (std::get_if<uint32_t>(&s)); linearPtr){
		s_h = *linearPtr;
		s_w = *linearPtr;
		return;
	}
	std::tuple<uint32_t, uint32_t>& t = std::get<1>(s);
	s_h = std::get<0>(t);
	s_w = std::get<1>(t);
	
}

std::vector<uint32_t> Conv2d::get_padding(std::variant<uint32_t, std::tuple<uint32_t, uint32_t>, std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>> &s){
	if(const auto Element (std::get_if<uint32_t>(&s)); Element){
		pad = *Element == 0;
		return std::vector<uint32_t>({*Element, *Element, *Element, *Element});
	}
	if(const auto Tup1 (std::get_if<std::tuple<uint32_t, uint32_t>>(&s)); Tup1){
		pad = (std::get<0>(*Tup1) == 0 && std::get<1>(*Tup1) == 0);
		return std::vector<uint32_t>({std::get<0>(*Tup1), std::get<0>(*Tup1), std::get<1>(*Tup1), std::get<1>(*Tup1)});
	}
	std::tuple<uint32_t, uint32_t, uint32_t, uint32_t> &Tup2 = std::get<2>(s);
	pad = (std::get<0>(Tup2) == 0 && std::get<1>(Tup2) == 0 && std::get<2>(Tup2) == 0 && std::get<3>(Tup2) == 0);
	return std::vector<uint32_t>({std::get<0>(Tup2), std::get<1>(Tup2), std::get<2>(Tup2), std::get<3>(Tup2)});


	
}

Conv2d::Conv2d(uint32_t in_channels, uint32_t out_channels, 
		std::variant<uint32_t, std::tuple<uint32_t, uint32_t>> kernel_size, 
		std::variant<uint32_t, std::tuple<uint32_t, uint32_t>> stride, 
		std::variant<uint32_t, std::tuple<uint32_t, uint32_t>, std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>> padding, 
		uint32_t dilation, uint32_t groups, bool bias, DType dtype)
	:Kernel(functional::randn(get_kernel_size(in_channels, out_channels, groups, kernel_size) , dtype)),
	Bias(bias ? functional::randn({out_channels, 1, 1}, dtype) : functional::zeros({1}, dtype)),
	groups(groups),
	use_bias(bias),
	padding(get_padding(padding))
{
	dilate_kernel(dilation);
	set_stride(stride);
}


Tensor Conv2d::forward(const Tensor& x){
	APrev = pad ? x.pad(padding) : x.contiguous();
	if(groups == 1){
		Tensor X = pad ? functional::conv2d(x.pad(padding), use_dilation ? Dilated : Kernel, s_h, s_w) : functional::conv2d(x, use_dilation ? Dilated : Kernel, s_h, s_w);
		if(use_bias)
			X += Bias;
		return std::move(X);
	}
	uint32_t add = uint32_t(x.shape()[1]/groups);
	std::vector<my_range> ranges = {my_range(0, x.shape()[0]), my_range(0, add)};
	std::vector<Tensor> tensors(groups);
	for(uint32_t i = 0; i < groups; ++i){
		tensors[i] = pad ? functional::conv2d(x[ranges].pad(padding), use_dilation ? Dilated : Kernel, s_h, s_w) : functional::conv2d(x[ranges], use_dilation ? Dilated : Kernel, s_h, s_w);
		ranges.back().begin += add;
		ranges.back().end += add;
	}

	Tensor X = functional::cat(std::move(tensors), 1);
	if(use_bias)
		X += Bias;
	return std::move(X);
}

}
}
