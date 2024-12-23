#include "Loss.h"

namespace nt{

Loss::Loss(Layer& layer)
	:l(layer)
{}


//just a default loss function
void Loss::operator()(const TensorGrad& output, const Tensor& target){
	dx = output.tensor - target;
}

void Loss::backward_no_update(){
	l.backward(dx);
}

void Loss::backward(){
	l.backward(dx);
	l.update();
}

void Loss::update(){
	l.update();
}

Scalar Loss::item() const {
	return dx.sum().toScalar();
}

MSELoss::MSELoss(Layer& layer)
	:Loss(layer)
{}

void MSELoss::operator()(const TensorGrad& output, const Tensor& target){
	dx = std::pow(output.tensor - target, 2) / target.numel();
}




}
