#include "layers.h"
#include <_types/_uint32_t.h>
#include <variant>

namespace nt{
namespace layers{


/* template<> */
/* Layer::Layer(Identity layer) */
/* 	:l(layer), */
/* 	l_i(0) */
/* {} */


/* template<> */
/* Layer::Layer(Linear layer) */
/* 	:l(layer), */
/* 	l_i(1) */
/* {} */

/* template<> */
/* Layer::Layer(Sigmoid layer) */
/* 	:l(layer), */
/* 	 l_i(2) */
/* {} */

Layer::Layer()
	:l(Identity())
{}


Tensor Layer::forward(const Tensor& x){
	return std::visit([&x](auto& a) -> Tensor {return a.forward(x);}, l);
	/* switch(l_i){ */
	/* 	case 1: */
	/* 		return l.linear.forward(x); */
	/* 	default: */
	/* 		return x; */
	/* } */
}

Tensor Layer::backward(const Tensor& dZ){
	return std::visit([&dZ](auto& a) -> Tensor{return a.backward(dZ);}, l);

}

Tensor Layer::eval(const Tensor& x) const{
	return std::visit([&x](const auto& a) -> Tensor {return a.eval(x);}, l);
	/* switch(l_i){ */
	/* 	case 1: */
	/* 		return l.linear.eval(x); */
	/* 	default: */
	/* 		return x; */
	/* } */
	
}

uint32_t Layer::parameter_count() const{
	if(std::holds_alternative<Linear>(l))
		return 2;
	return 0;
}


Tensor Layer::parameters(){
	Tensor outp({parameter_count()}, DType::TensorObj);
	if (const auto linearPtr (std::get_if<Linear>(&l)); linearPtr){
		outp[0] = linearPtr->Weight;
		outp[1] = linearPtr->Bias;
	}
	/* switch(l_i){ */
	/* 	case 1: */
	/* 		outp[0] = l.linear.Weight; */
	/* 		outp[1] = l.linear.Bias; */
	/* 		break; */
	/* 	default: */
	/* 		break; */
	/* } */
	return outp;
}


}
}
