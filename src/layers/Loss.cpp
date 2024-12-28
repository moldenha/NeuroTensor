#include "Loss.h"

namespace nt{
namespace loss{

/* Loss::Loss(Layer& layer) */
/* 	:l(layer) */
/* {} */


/* //just a default loss function */
/* void Loss::operator()(const TensorGrad& output, const Tensor& target){ */
/* 	dx = output.tensor - target; */
/* } */

/* void Loss::backward_no_update(){ */
/* 	l.backward(dx); */
/* } */

/* void Loss::backward(){ */
/* 	l.backward(dx); */
/* 	l.update(); */
/* } */

/* void Loss::update(){ */
/* 	l.update(); */
/* } */

/* Scalar Loss::item() const { */
/* 	return dx.sum().toScalar(); */
/* } */

/* MSELoss::MSELoss(Layer& layer) */
/* 	:Loss(layer) */
/* {} */

/* void MSELoss::operator()(const TensorGrad& output, const Tensor& target){ */
/* 	dx = std::pow(output.tensor - target, 2) / target.numel(); */
/* } */

TensorGrad raw_error(const TensorGrad& output, const Tensor& target){
	Tensor dx = output.tensor - target;
	Scalar item = dx.sum().toScalar();
	TensorGrad loss(item);
	loss.grad = nt::make_intrusive<tensor_holder>(dx);
	TensorGrad::redefine_tracking(loss, output, [](const Tensor& grad, intrusive_ptr<TensorGrad>& parent){
		parent->grad->tensor = grad;
	});
	return std::move(loss);
}

TensorGrad MSE(const TensorGrad& output, const Tensor& target){
	Tensor dx = std::pow(output.tensor - target, 2) / target.numel();;
	Scalar item = dx.sum().toScalar();
	TensorGrad loss(item);
	loss.grad = nt::make_intrusive<tensor_holder>(dx);
	TensorGrad::redefine_tracking(loss, output, [](const Tensor& grad, intrusive_ptr<TensorGrad>& parent){
		parent->grad->tensor = grad;
	});
	return std::move(loss);
}



}}
