#include "layers.h"

#include "../dtype/ArrayVoid.hpp"
#include <functional>

namespace nt{
namespace layers{

ActivationFunction::ActivationFunction(std::function<Tensor(const Tensor&)> f, std::function<Tensor(const Tensor&)> df)
	:func(f),
	dFunc(df)
	{}

Tensor ActivationFunction::forward(const Tensor &x){
	/* std::cout<<"Before activation: "<<x<<std::endl; */
	Tensor A = func(x);
	/* std::cout << "After activation: "<<A<<std::endl; */
	A_Prev = A.contiguous();
	return std::move(A);
}


Tensor ActivationFunction::eval(const Tensor& x) const {
	return func(x);
}

Tensor ActivationFunction::backward(const Tensor &dZ){
	/* Tensor output({2}, DType::TensorObj); */
	/* const Tensor& dz = dZ[1].item<Tensor>(); */
	/* /1* std::cout << "ActivationFunction dZ: "<<dz<<std::endl; *1/ */
	/* output[0] = dZ[0].item<Tensor>(); */
	/* output[1] = dz * A_Prev; */
	/* std::cout << "APrev: "<<A_Prev<<std::endl; */
	/* std::cout << "dAPrev: "<<dA_Prev<<std::endl; */
	return dZ * dFunc(A_Prev);
}


}
}
