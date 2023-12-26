#include "layers.h"


namespace nt{
namespace layers{

Tensor Linear::backprop_dx(const Tensor& dZ){
	/* std::cout << WeightT<<std::endl; */
	/* std::cout << Weight << std::endl; */
	return functional::matmult(WeightT, dZ);
}

void Linear::backprop_dw(const Tensor &dZ){
	if(!a_prev_trans){
		APrev = APrev.transpose(-1,-2);
		a_prev_trans = true;
	}
	Tensor dW = functional::matmult(dZ, APrev);
	if(clip){dW.clip_(clip_min, clip_max);}
	/* std::cout << "dW: "<<dW<<std::endl; */
	Weight += dW;
}

void Linear::backprop_db(Tensor &dB){
	if(!use_bias){return;}
	if(clip){dB.clip_(clip_min, clip_max);}
	/* std::cout << "grads: "<<dB<<std::endl; */
	Bias += dB;
}

Linear::Linear(uint32_t in_rows, uint32_t out_rows, Scalar lr, bool use_bias, bool clip, Scalar maxClip, Scalar minClip, DType dt)
	:Weight(functional::randn({out_rows, in_rows}, dt)),
	Bias(functional::randn({out_rows, 1}, dt)),
	lr(lr),
	clip_max(maxClip),
	clip_min(minClip),
	use_bias(use_bias),
	a_prev_trans(false),
	clip(clip)
{
	WeightT = Weight.transpose(-1,-2);
}


Tensor Linear::forward(const Tensor& x){
	APrev = x.contiguous();
	a_prev_trans = false;
	Tensor A = functional::matmult(Weight, x);
	/* std::cout << "after mult: "<<A<<std::endl; */
	/* Weight.print(); */
	if(!use_bias)
		return A;
	A += Bias;
	/* std::cout << "after adding Bias: "<< A<<std::endl; */
	/* Bias.print(); */
	return A;
}

Tensor Linear::eval(const Tensor& x) const{
	Tensor a = functional::matmult(Weight, x);
	if(!use_bias)
		return functional::matmult(Weight, x); 
	return functional::matmult(Weight, x) + Bias;
	/* if(use_bias) */
	/* 	return functional::matmult(Weight, x) + Bias; */
	/* return functional::matmult(Weight, x); */
}


Tensor Linear::backward(const Tensor& dZ){
	const Tensor& dz = dZ[0].item<Tensor>();
	/* std::cout << "Sum: "<<dz.sum().item<float>()<<std::endl; */
	Tensor grads = dZ[1].item<Tensor>() * lr;
	Tensor dX = backprop_dx(dz);
	grads *= lr;
	backprop_dw(grads);
	backprop_db(grads);
	Tensor output({2}, DType::TensorObj);
	output[0] = dX;
	output[1] = dX;
	return std::move(output);
}

void Linear::print(){
	std::cout << "Weight: "<<Weight;
	std::cout << "Bias: "<<Bias<<std::endl;
}

}
}
