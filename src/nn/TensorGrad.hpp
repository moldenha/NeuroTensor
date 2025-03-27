#ifndef _NT_TENSOR_GRAD_HPP_
#define _NT_TENSOR_GRAD_HPP_

#include "TensorGrad.h"
#include "../intrusive_ptr/intrusive_ptr.hpp"

namespace nt{


//this function is used to create backward functions
//it is used in tandem with track_tensors
//it's main use is in functions where knowing if a tensor was used last is irrelevant
template<typename backward_func, typename... Args>
inline void TensorGrad::create_backward_function(backward_func&& func, Args&&... args){
	/* static_assert(all_tensor_grads<Args...>::value, "All arguments must be TensorGrad"); */
	if(!this->do_track_grad){return;}
    if(this->backwardFunc == nullptr) this->backwardFunc = make_intrusive<intrusive_back_func>();

	this->backwardFunc->set(intrusive_back_func::function_type(std::bind(std::forward<backward_func>(func), 
			std::placeholders::_1, 
			std::placeholders::_2, 
			make_tensor_holder(std::forward<Args>(args))...)));
	utils::THROW_EXCEPTION(this->backwardFunc->index() == 1, "Expected to have an index of 1 in the backward func but got $", this->backwardFunc->index());
}

//this function is mainly used when a tensor grad has been modified in onto itself
//then it is needed to know if it was the last function used
template<typename backward_func, typename... Args>
inline void TensorGrad::create_bool_backward_function(backward_func&& func, Args&&... args){
	if(!this->do_track_grad){return;}
    if(this->backwardFunc == nullptr) this->backwardFunc = make_intrusive<intrusive_back_func>();
    this->backwardFunc->set(intrusive_back_func::function_type_b(std::bind(std::forward<backward_func>(func), 
			std::placeholders::_1, 
			std::placeholders::_2, 
            std::placeholders::_3,
			make_tensor_holder(std::forward<Args>(args))...)));
	utils::THROW_EXCEPTION(this->backwardFunc->index() == 2, "Expected to have an index of 2 in the backward func but got $", this->backwardFunc->index()); 
}

//track tensors should be used when an output tensor is created, such that the current one remains un-modified
//ex: TensorGrad A = B + C;
//  - B and C remain unmodified
//  - A will correctly track B and C as parents
//  - B and C will correctly track A as a child
inline void TensorGrad::track_tensors(const TensorGrad& t){
	if(!this->do_track_grad){return;}
    if(this->grad == nullptr){
		this->grad = nt::make_intrusive<tensor_holder>(nt::functional::zeros_like(this->tensor));
    }
	if(t.grad == nullptr){
		t.grad = nt::make_intrusive<tensor_holder>(nt::functional::zeros_like(t.tensor));
	}
    if(t.backwardFunc == nullptr){const_cast<TensorGrad&>(t).backwardFunc = make_intrusive<intrusive_back_func>();}
    t.backwardFunc->un_use();
    if(this->backwardFunc == nullptr){this->backwardFunc = make_intrusive<intrusive_back_func>();}
    this->backwardFunc->un_use();
    this->parents->emplace_back(t);
    const_cast<TensorGrad&>(t).children->emplace_back(*this);
}


template<typename... Args>
inline void TensorGrad::track_tensors(const TensorGrad& t, const Args&... args){
    this->track_tensors(t);
    this->track_tensors(args...);
}



//track tensors should be used when the current tensor is being modified
//ex: TensorGrad A += B;
//  - A will be modified by B
//  - The "resulting A" [look at swap] will track B and the original A as a parent
//  - The original A and B will correctly track the new A as a child

inline void TensorGrad::track_self_mod_tensors(){
	if(!this->do_track_grad){return;}
    TensorGrad n_grad(tensor, make_intrusive<tensor_holder>(functional::zeros_like(tensor)),
                      make_intrusive<intrusive_back_func>(), make_intrusive<tensor_grad_vec>(),
                      make_intrusive<tensor_grad_vec>(), grad_required);
    this->swap(n_grad);
    this->track_tensors(n_grad);
}

template<typename... Args>
inline void TensorGrad::track_self_mod_tensors(const TensorGrad& gr, const Args&... args ){
	if(!this->do_track_grad){return;}
    TensorGrad n_grad(tensor, make_intrusive<tensor_holder>(functional::zeros_like(tensor)),
                      make_intrusive<intrusive_back_func>(), make_intrusive<tensor_grad_vec>(),
                      make_intrusive<tensor_grad_vec>(), grad_required);
    this->swap(n_grad);
    this->track_tensors(n_grad, gr, args...);
}

//this is a function to track the same gradient
//what I mean by that is that instead of copying the gradient, this function probably wont have a backward function
//instead its gradient tensor will hold the same original memory as the tensor that is creating this
//this only happens when there is a change to the shape of the gradient
//this function might have to happen on the grad of the incoming tensor when taking into account the overall grad
template<typename OutOperator>
inline void TensorGrad::track_grad(const TensorGrad& t, OutOperator&& op){
    if(!t.do_track_grad || !this->do_track_grad){
        this->do_track_grad = false;
        return;
    }
	if(t.grad == nullptr){
		t.grad = nt::make_intrusive<tensor_holder>(nt::functional::zeros_like(t.tensor));
	}
    //make sure the backward func and use variables are correctly handled
    if(t.backwardFunc == nullptr){const_cast<TensorGrad&>(t).backwardFunc = make_intrusive<intrusive_back_func>();}
    t.backwardFunc->un_use();
    if(this->backwardFunc == nullptr){this->backwardFunc = make_intrusive<intrusive_back_func>();}
    this->backwardFunc->un_use();

    //making parents and children
    this->parents->emplace_back(t);
    const_cast<TensorGrad&>(t).children->emplace_back(*this);
	
    //just change the gradient views or strides or whatever it may be
	this->grad = nt::make_intrusive<tensor_holder>(op(this->parents->back()->grad->tensor));
}



}

#endif //_NT_TENSOR_GRAD_HPP_
