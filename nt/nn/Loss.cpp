#include "../functional/functional.h"
#include "Loss.h"
#include "ScalarGrad.h"

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

ScalarGrad raw_error(const TensorGrad& output, const Tensor& target){
	Tensor dx = output.tensor - target;
	Scalar item = dx.sum().toScalar();
    return ScalarGrad(item, dx, output);
}

ScalarGrad MSE(const TensorGrad& output, const Tensor& target){
    Tensor diff = output.tensor - target;
    Tensor loss_tensor = std::pow(diff, 2).sum() / target.numel();
	Scalar item = loss_tensor.sum().toScalar();
    Tensor dx = (2.0 * diff) / target.numel();
    return ScalarGrad(item, dx, output);
}



}}
