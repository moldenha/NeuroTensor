#include "layers.h"


namespace nt{
namespace layers{

void Linear::backprop_db(const Tensor &dZ){
	if(!use_bias){return;}
	Tensor dB = dZ.mean(-3).sum(-2);
	if(clip)
		dB.clip_(clip_min, clip_max);
	dB *= lr;
	Bias -= dB;
}

Linear::Linear(uint32_t in_features, uint32_t out_features, Scalar lr, bool use_bias, bool clip, Scalar maxClip, Scalar minClip, DType dt)
	:Weight(functional::randn({in_features, out_features}, dt)),
	Bias(functional::randn({out_features}, dt)),
	lr(lr),
	clip_max(maxClip),
	clip_min(minClip),
	use_bias(use_bias),
	clip(clip)
{
	WeightT = Weight.transpose(-1,-2);
}


Tensor Linear::forward(const Tensor& x){
	APrev = x.contiguous();
	if(use_bias)
		return functional::matmult(x, Weight) + Bias;
	return functional::matmult(x, Weight);
}

Tensor Linear::eval(const Tensor& x) const{
	if(use_bias)
		return functional::matmult(x, Weight) + Bias;
	return functional::matmult(x, Weight);
}


Tensor Linear::backward(const Tensor& dZ){
	backprop_db(dZ);
	Tensor dX = functional::matmult(dZ, Weight, 0, 1);
	Tensor dW = functional::matmult(APrev, dZ, 1, 0).mean(-3);
	if(clip){
		dW.clip_(clip_min, clip_max);
	}
	dW *= lr;
	Weight -= dW;
	return std::move(dX);
}

void Linear::print(){
	std::cout << "Weight: "<<Weight;
	std::cout << "Bias: "<<Bias<<std::endl;
}

}
}
