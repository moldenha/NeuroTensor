#include "../functional/functional.h"
#include "Optimizers.h"

namespace nt{
namespace optimizers{

void SGD::step(){
	for(auto& t : parameters){
		if(t.do_track_grad && t.grad != nullptr && !t.is_null()){
			t.grad->tensor.clip_(-10, 10);
			t.grad->tensor *= learning_rate;
			t.update();
		}
	}
	this->erase_grad_tracking();
}

void SGD::erase_grad_tracking(){
	this->zero_grad();
	for(auto& t : parameters){
		if(t.is_null()){continue;}
		t.parents->clear();
		t.backwardFunc->set(nullptr);
		t.children->clear();
	}
}

void SGD::zero_grad(){
	for(auto& t : parameters){
		if(t.is_null()){continue;}
        if(t.grad == nullptr){t.grad = make_intrusive<tensor_holder>(functional::zeros_like(t.tensor));}
        else{
            t.grad->tensor.fill_(0);
        }
	}
}






void Adam::erase_grad_tracking(){
	this->zero_grad();
	for(auto& t : parameters){
		if(t.is_null()){continue;}
		t.parents->clear();
		t.backwardFunc->set(nullptr);
		t.children->clear();
	}
}


void Adam::zero_grad(){
	for(auto& t : parameters){
		if(t.is_null()){continue;}
        if(t.grad == nullptr){t.grad = make_intrusive<tensor_holder>(functional::zeros_like(t.tensor));}
        else{
            t.grad->tensor.fill_(0);
        }
	}
}

/* grad = p.grad.data */
/* # Update first moment */
/* m.mul_(self.beta1).add_(grad, alpha=1 - self.beta1) */
/* # Update second moment */
/* v.mul_(self.beta2).addcmul_(grad, grad, value=1 - self.beta2) */
/* # Bias correction */
/* m_hat = m / (1 - self.beta1 ** self.t) */
/* v_hat = v / (1 - self.beta2 ** self.t) */
/* # Update parameters */
/* p.data.addcdiv_(m_hat, v_hat.sqrt().add_(self.epsilon), value=-self.lr) */

void Adam::step(){
	Tensor* begin1 = reinterpret_cast<Tensor*>(this->m.data_ptr());
	Tensor* begin2 = reinterpret_cast<Tensor*>(this->v.data_ptr());
	Tensor* end1 = begin1 + this->parameters.size();
	auto begin = this->parameters.begin();
	for(;begin1 != end1; ++begin1, ++begin2, ++begin){
		if(!begin->do_track_grad || begin->grad == nullptr || begin->is_null()){continue;}
		if(begin1->is_null()){
			*begin1 = functional::zeros_like(begin->tensor);
		}
		if(begin2->is_null()){
			*begin2 = functional::zeros_like(begin->tensor);
		}
		Tensor& grad = begin->grad->tensor;
		//first moment
		begin1->multiply_(this->beta1).add_(1-beta1); //m
		begin1->multiply_(grad);
		//second moment
		begin2->multiply_(this->beta2).add_(1-beta2);
		begin2->multiply_(std::pow(grad, 2.0));
		//bias correction
		Tensor m_hat = (*begin1) / std::pow(1 - this->beta1, this->t);
		Tensor v_hat = (*begin2) / std::pow(1 - this->beta2, this->t);
		//self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
		grad.set_((this->learning_rate * m_hat) / (std::sqrt(v_hat) + this->epsilon));
		grad.clip_(-10, 10);
		begin->update();
	}
    this->erase_grad_tracking();
}


}}
