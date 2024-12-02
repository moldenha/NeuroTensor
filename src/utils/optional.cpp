#include "optional.h"

namespace nt{
namespace utils{

optional_tensor::optional_tensor(const optional_tensor& op)
	:tensor(op.tensor)
{}

optional_tensor::optional_tensor(optional_tensor&& op)
	:tensor(std::move(op.tensor))
{}

optional_tensor::optional_tensor(const Tensor& t)
	:tensor(make_intrusive<tensor_holder>(t))
{}

optional_tensor::optional_tensor(Tensor&& t)
	:tensor(make_intrusive<tensor_holder>(std::move(t)))
{}

optional_tensor::optional_tensor(nullptr)
	:tensor(nullptr)
{}

optional_tensor::optional_tensor()
	:tensor(nullptr)
{}


optional_tensor& optional_tensor::operator=(const optional_tensor& op){
	tensor = op.tensor;
	return *this;
}

optional_tensor& optional_tensor::operator=(optional_tensor&& op){
	tensor = std::move(op.tensor);
	return *this;
}

optional_tensor& optional_tensor::operator=(const Tensor& t){
	if(*this){
		tensor->tensor = t;
		return *this;
	}
	tensor = make_intrusive<tensor_holder>(t);
	return *this;
}

optional_tensor& optional_tensor::operator=(Tensor&& t){
	if(*this){
		tensor->tensor = std::move(t);
		return *this;
	}
	tensor = make_intrusive<tensor_holder>(std::move(t));
	return *this;
}

optional_tensor& optional_tensor::operator=(nullptr){
	if(*this){this->tensor.reset();}
	return *this;
}

}} //nt::utils::
