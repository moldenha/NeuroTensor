
#include "GraphNode.h"

#include "../../intrusive_ptr/intrusive_ptr.hpp"
#include "../../Tensor.h"
#include "../../utils/tensor_holder.h"
#include "../../functional/functional.h"
#include "BackwardFunc.h"
#include "SharedVersion.h"
#include <vector>
#include <memory>
#include <functional>

namespace nt::grad::utility {


void GraphNode::accumulate_gradient(const Tensor& in_grad){
    if(!this->tensor || this->tensor->is_null()){
        // then this Node was temporary, and has already been cleared
        // on the heap
        // for that reason, there is no reason to accumulate the gradient, returning
        return;
    }
    utils::throw_exception(in_grad.dtype() == this->tensor->dtype(),
                           "Error, given gradient does not match tensor dtype ($) !=  ($)", in_grad.dtype(), this->tensor->dtype());
    // Some potential optimal routes, followed by the last one which should work (if given a valid gradient)
    if(in_grad.shape() == this->tensor->shape() && !bool(this->grad)){
        this->grad = std::make_unique<Tensor>(in_grad);
    }else if (in_grad.shape() == this->tensor->shape() && this->grad->is_null()){
        *(this->grad) = in_grad;
    }else{
        this->ensure_gradient_init();
        *(this->grad) += in_grad;
    }
}

void GraphNode::accumulate_gradient(Scalar num){
    if(!this->tensor || this->tensor->is_null()){
        // then this Node was temporary, and has already been cleared
        // on the heap
        // for that reason, there is no reason to accumulate the gradient, returning
        return;
    }
    this->ensure_gradient_init();
    if(!num.isZero()) // if zero, it is just to make sure the gradient is initialized
        *(this->grad) += num;
}

void GraphNode::zero_grad(){
    if(this->grad && !this->grad->is_null()){
        this->grad->fill_(0);
    }else if(this->grad){
        *(this->grad) = ::nt::functional::zeros_like(*(this->tensor));
    }else{
        this->grad = std::make_unique<Tensor>(::nt::functional::zeros_like(*(this->tensor))); 
    }
}

void GraphNode::run_backward() {
    // if(this->name() == "Multiply_Backward"){
    //     std::cout << "run backward called on multiply_ backward node" << std::endl;
    //     std::cout << std::boolalpha << bool(this->grad) << std::endl;
    //     if(this->grad){
    //         std::cout << this->grad->is_null();
    //     }
    //     else
    //         std::cout << " maybe";
    //     std::cout << std::noboolalpha << std::endl;
    // }
    // if(this->edge_type() == EdgeType::View){
    //     if(!this->grad || this->grad->is_null())
    //         std::cout << "Edge Type view would have not been run, and does not have grad defined for " << this->name() << std::endl;
    //     else
    //         std::cout << "Edge Type view would HAVE been run, and HAS grad defined for " << this->name() << std::endl;
    // }
    if(this->edge_type() == EdgeType::Write && this->parents.size() > 0 
            && this->grad && this->grad->is_null()){
        const weak_intrusive_ptr<GraphNode>& weak_parent = this->parents[0].first;
        if(auto parent = weak_parent.lock()){
            if(parent->edge_type() == EdgeType::View){
                // very specific case of when the gradient is not defined, but that it is a write onto a view
                // triggered from something like:
                // a[2] *= 40;
                Tensor cpy = parent->grad->clone();
                parent->grad->fill_(0);
                *(this->grad) = std::move(cpy);
            }
        }
    }
    if(!this->grad || this->grad->is_null()){
        return;
    }
    this->backwardFunc->run(*(this->grad), this->parents); 
}

void GraphNode::run_backward(const Tensor& _grad) { 
    backwardFunc->run(_grad, this->parents); 
}


}
