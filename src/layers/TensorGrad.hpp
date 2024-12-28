#ifndef _TENSOR_GRAD_HPP_
#define _TENSOR_GRAD_HPP_

#include "TensorGrad.h"
#include "../intrusive_ptr/intrusive_ptr.hpp"

namespace nt{



template<typename backward_func, typename... Args>
inline void TensorGrad::create_backward_function(backward_func&& func, Args&&... args){
	/* static_assert(all_tensor_grads<Args...>::value, "All arguments must be TensorGrad"); */
	if(!this->do_track_grad){return;}
	this->backwardFunc->set(intrusive_back_func::function_type(std::bind(std::forward<backward_func>(func), 
			std::placeholders::_1, 
			std::placeholders::_2, 
			make_tensor_holder(std::forward<Args>(args))...)));
	utils::THROW_EXCEPTION(this->backwardFunc->index() == 1, "Expected to have an index of 1 in the backward func but got $", this->backwardFunc->index());
}


//children are also tracked for outlining streams
//ex:
//nt::TensorGrad A(nt::functional::randn({3,4,2}));
//A.tensor[A.tensor < 0.01] *= -1;	
//nt::TensorGrad W(nt::functional::randn({3,2,3}));

//A[A <= 0] = 0; <- without children and parents being tracked the = 0 operator wouldn't be tracked
//                  this allows the child to be tracked
//nt::TensorGrad out = nt::functional::matmult(A, W);
//nt::Tensor dt = nt::functional::randn(out.shape());


//a function that specifically tracks modifications to self,
//especially when it needs to track a branching function that involves children 
//when this happens, it basically makes a new self
//it is expected that the first argument would be *this
//but dont actually put *this
//for example:
//this->track_self_mod([](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
/*				const size_t parent_index){
		std::cout << "operator= scalar backward called"<<std::endl;
		std::cout << "parent index is: "<<parent_index<<std::endl;
		std::cout << std::boolalpha << bool(parents[parent_index]) << std::noboolalpha << std::endl;
		parents[parent_index]->grad->tensor.fill_(0);
		
	}, 
 *this); <- dont do this,
 instead do this:

this->track_self_mod([](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,				const size_t parent_index){
		std::cout << "operator= scalar backward called"<<std::endl;
		std::cout << "parent index is: "<<parent_index<<std::endl;
		std::cout << std::boolalpha << bool(parents[parent_index]) << std::noboolalpha << std::endl;
		parents[parent_index]->grad->tensor.fill_(0);
		
	}); (notice how the *this is gone)


there is also the ability to track other tensors
but the "parents" will always have *this as the first parent or in the function below "old_self"

*/
template<typename... Args>
inline void TensorGrad::track_self_mod(std::function<void(const Tensor&, std::vector<intrusive_ptr<TensorGrad>>&, const size_t)> new_backward_func,
		const Args&... args){
	
	if(!this->do_track_grad){return;}
	if(this->grad == nullptr){
		this->grad = nt::make_intrusive<tensor_holder>(nt::functional::zeros_like(this->tensor));
	}


	TensorGrad old_self(tensor, grad, std::move(backwardFunc), std::move(parents), children);
	this->parents = std::vector<intrusive_ptr<TensorGrad> >();
	utils::throw_exception(this->parents.size() == 0, "Expected parent size to be 0 but got $", this->parents.size());
	const size_t other_indice = 0;
	this->track_tensors(old_self, args...);
	this->backwardFunc = make_intrusive<intrusive_back_func>(intrusive_back_func::function_type(std::bind(new_backward_func, std::placeholders::_1, std::placeholders::_2, other_indice)));
	utils::THROW_EXCEPTION(this->backwardFunc->index() == 1, "Expected to have an index of 1 in the backward func but got $", this->backwardFunc->index());


	if(old_self.is_child()){
		/* children->push_back(*this); */
		for(auto& parent : old_self.parents){
			bool b = false;
			if(parent->children->size() > 0){
				for(uint32_t i = 0; i < parent->children->size(); ++i){
					if(parent->children->at(i)->children == this->children){
						parent->children->push_back(*this);
						b = true;
						break;
					}
				}
			}
			if(b){break;}
		}
	}
}

template<typename... Args>
inline void TensorGrad::track_self_mod(std::function<void(const Tensor&, std::vector<intrusive_ptr<TensorGrad>>&, const size_t, bool)> new_backward_func,
		const Args&... args){
	
	if(!this->do_track_grad){return;}
	if(this->grad == nullptr){
		this->grad = nt::make_intrusive<tensor_holder>(nt::functional::zeros_like(this->tensor));
	}


	TensorGrad old_self(tensor, grad, std::move(backwardFunc), std::move(parents), children);
	this->parents = std::vector<intrusive_ptr<TensorGrad> >();
	utils::throw_exception(this->parents.size() == 0, "Expected parent size to be 0 but got $", this->parents.size());
	const size_t other_indice = 0;
	this->track_tensors(old_self, args...);
	this->backwardFunc = make_intrusive<intrusive_back_func>(intrusive_back_func::function_type_b(std::bind(new_backward_func, std::placeholders::_1, std::placeholders::_2, other_indice, std::placeholders::_3)));
	utils::THROW_EXCEPTION(this->backwardFunc->index() == 2, "Expected to have an index of 1 in the backward func but got $", this->backwardFunc->index());


	if(old_self.is_child()){
		/* children->push_back(*this); */
		for(auto& parent : old_self.parents){
			bool b = false;
			if(parent->children->size() > 0){
				for(uint32_t i = 0; i < parent->children->size(); ++i){
					if(parent->children->at(i)->children == this->children){
						parent->children->push_back(*this);
						b = true;
						break;
					}
				}
			}
			if(b){break;}
		}
	}
}




inline void TensorGrad::track_tensors(const TensorGrad& t){
	if(!this->do_track_grad){return;}
	if(t.grad == nullptr){
		t.grad = nt::make_intrusive<tensor_holder>(nt::functional::zeros_like(t.tensor));
	}
	parents.push_back(nt::make_intrusive<TensorGrad>(t));
}


template<typename... Args>
inline void TensorGrad::track_tensors(const TensorGrad& t, const Args&... args){
	if(!this->do_track_grad){return;}
	if(t.grad == nullptr){
		t.grad = nt::make_intrusive<tensor_holder>(nt::functional::zeros_like(t.tensor));
	}
	parents.push_back(nt::make_intrusive<TensorGrad>(t));
	track_tensors(args...);
}


//this is a function to track the same gradient
//what I mean by that is that instead of copying the gradient, this function probably wont have a backward function
//instead its gradient tensor will hold the same original memory as the tensor that is creating this
//this only happens when there is a change to the shape of the gradient
//this function might have to happen on the grad of the incoming tensor when taking into account the overall grad
template<typename OutOperator>
inline void TensorGrad::track_grad(const TensorGrad& t, OutOperator&& op){
	if(!this->do_track_grad){return;}
	if(t.grad == nullptr){
		t.grad = nt::make_intrusive<tensor_holder>(nt::functional::zeros_like(t.tensor));
	}
	parents.push_back(nt::make_intrusive<TensorGrad>(t));
	//just change the gradient views or strides or whatever it may be
	this->grad = nt::make_intrusive<tensor_holder>(op(parents.back()->grad->tensor));
	//log current info as the children
	//something like this:
	//this->backwardFunc->set_child(std::forward(op));
	//wait no, that change will automatically happen because the parent grad will be modified and passed
	//automatically
	t.children->push_back(make_intrusive<TensorGrad>(*this));
}



}

#endif
