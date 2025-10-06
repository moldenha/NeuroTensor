#include "AutoGrad.h"

namespace nt::grad{

namespace utility{

void GraphNode::accumulate_gradient(const Tensor& in_grad){
    if(!this->tensor || this->tensor->tensor.is_null()){
        // then this Node was temporary, and has already been cleared
        // on the heap
        // for that reason, there is no reason to accumulate the gradient, returning
        return;
    }
    utils::throw_exception(in_grad.dtype() == this->tensor->tensor.dtype(),
                           "Error, given gradient does not match tensor dtype ($) !=  ($)", in_grad.dtype(), this->tensor->tensor.dtype());
    // Some potential optimal routes, followed by the last one which should work (if given a valid gradient)
    if(in_grad.shape() == this->tensor->tensor.shape() && !bool(this->grad)){
        this->grad = make_intrusive<tensor_holder>(in_grad);
    }else if (in_grad.shape() == this->tensor->tensor.shape() && this->grad->tensor.is_null()){
        this->grad->tensor = in_grad;
    }else{
        this->ensure_gradient_init();
        this->grad->tensor += in_grad;
    }
}

void GraphNode::accumulate_gradient(Scalar num){
    if(!this->tensor || this->tensor->tensor.is_null()){
        // then this Node was temporary, and has already been cleared
        // on the heap
        // for that reason, there is no reason to accumulate the gradient, returning
        return;
    }
    this->ensure_gradient_init();
    if(!num.isZero()) // if zero, it is just to make sure the gradient is initialized
        this->grad->tensor += num;
}


void GraphNode::zero_grad(){
    if(this->grad && !this->grad->tensor.is_null()){
        this->grad->tensor.fill_(0);
    }else if(this->grad){
        this->grad->tensor = ::nt::functional::zeros_like(this->tensor->tensor);
    }else{
        this->grad = make_intrusive<tensor_holder>(::nt::functional::zeros_like(this->tensor->tensor)); 
    }
}

void GraphNode::run_backward() {
    if(!grad || grad->tensor.is_null()){
        return;
    }
    backwardFunc->run(grad->tensor, parents); 
}
void GraphNode::run_backward(const Tensor& _grad) { backwardFunc->run(_grad, parents); }

void GraphNode::ensure_view_backward_initialization(bool zero_if_uninit ){
    //this is a function that can be used to make sure grad and backwardFunc are not nullptr
    if(!this->grad) this->grad = make_intrusive<tensor_holder>(zero_if_uninit ? nt::functional::zeros_like(tensor->tensor) : Tensor::Null());
    if(!this->backwardFunc || !this->backwardFunc->is_view_change()) this->backwardFunc = make_intrusive<view_backward_func>();
}

void GraphNode::ensure_self_mod_backward_initialization(bool zero_if_uninit ) {
    //this is a function that can be used to make sure grad and backwardFunc are not nullptr
    if(!this->grad) this->grad = make_intrusive<tensor_holder>(zero_if_uninit ? nt::functional::zeros_like(tensor->tensor) : Tensor::Null());
    if(!this->backwardFunc || !this->backwardFunc->is_self_mod()) this->backwardFunc = make_intrusive<self_mod_backward_func>();
}

}



template<class HashSet>
void AutoGrad<HashSet>::traverse_graph_impl(const intrusive_ptr<utility::GraphNode>& node,
                                    std::vector<intrusive_ptr<utility::GraphNode>>& result,
                                    HashSet& visited){
    
    if(!node || visited.count(node)) return;
    visited.insert(node);

    // Children are traversed first
    // The backward call of children needs to be called first
    for(auto it = node->children.rbegin(); it != node->children.rend(); ++it){
        traverse_graph_impl(*it, result, visited);
    }
    // for (const auto& child : node->children) {
    //     traverse_graph_impl(child, result, visited);
    // }

    result.push_back(node);

    

    // traverse parents
    for (const auto& weak_parent : node->parents) {
        if (auto parent = weak_parent.lock()) {
            traverse_graph_impl(parent, result, visited);
        }
    }
}

template<class HashSet>
void AutoGrad<HashSet>::zero_grad() {
    for(const auto& node : this->traversed){
        node->zero_grad();
    }
}


inline bool is_child(const intrusive_ptr<utility::GraphNode>& child, const intrusive_ptr<utility::GraphNode>& parent){
    for(const auto& other : parent->children){
        if(other == child) return true;
    }
    return false;
}

inline bool startsWithCompare(const std::string& str, const std::string& prefix) {
    return str.compare(0, prefix.length(), prefix) == 0;
}

inline intrusive_ptr<utility::GraphNode> goto_next_parent(const intrusive_ptr<utility::GraphNode>& child, std::vector<intrusive_ptr<utility::GraphNode>>& parents, size_t& i){
    if(i >= parents.size()) return intrusive_ptr<utility::GraphNode>(nullptr);
    for(const auto& other : parents[i]->children){
        if(other == child) return parents[i];
    }
    ++i;
    return goto_next_parent(child, parents, i);
}
// for example:
//
// t = nt::rand(0, 3, {3, 4, 5});
// t[0].fill_(0)
// y = t + 10

inline void trace_child_view_change(std::vector<intrusive_ptr<utility::GraphNode>>& vec, size_t i, intrusive_ptr<utility::GraphNode>& cur_child){
    // static std::string prefix = "GradTrack";
    intrusive_ptr<utility::GraphNode> next_parent = goto_next_parent(cur_child, vec, i);
    if(!bool(next_parent)){
        return;    
    }

    // this is if the previous one is a view change
    // such as t[0].fill_(0) <- t[0] was the view change, but fill_ was the change on self, t[0] was the view change parent
    if(next_parent->backwardFunc->is_view_change() && next_parent->grad && !(next_parent->grad->tensor.is_null())){
        Tensor sending_back = next_parent->grad->tensor.clone();
        next_parent->grad->tensor.fill_(0); // zero out grad, and have it re-accumulated
        cur_child->run_backward(sending_back);
        return;
    }
    while(bool(next_parent)){
        next_parent = goto_next_parent(next_parent, vec, i);
        if(!bool(next_parent)){return;}
        if(next_parent->backwardFunc->is_view_change() && next_parent->grad && !(next_parent->grad->tensor.is_null())){
            Tensor sending_back = next_parent->grad->tensor.clone();
            next_parent->grad->tensor.fill_(0); // zero out grad, and have it re-accumulated
            cur_child->run_backward(sending_back);
            return;
        }
    }
}

template<class HashSet>
void AutoGrad<HashSet>::backward(const Tensor& initialGrad){
    // Properly setup first gradient pass
    this->traversed[0]->accumulate_gradient(initialGrad); 
    // above ensures gradient is defined
    Tensor cpy_grad = this->traversed[0]->grad->tensor.clone();
    this->traversed[0]->zero_grad();
    this->traversed[0]->run_backward(cpy_grad);
    this->traversed[0]->accumulate_gradient(cpy_grad);
    for(size_t i = 1; i < this->traversed.size(); ++i){
        if(this->traversed[i]->grad && this->traversed[i]->grad->tensor.is_null()){
            trace_child_view_change(this->traversed, i+1, this->traversed[i]);
        }else{
            this->traversed[i]->run_backward();
        }
    }
}

template<class HashSet>
void AutoGrad<HashSet>::backward(){
    utils::throw_exception(bool(this->traversed[0]->grad)
                        && !this->traversed[0]->grad->tensor.is_null(),
                           "Error, backward function not given an initial gradient is expected to already have a gradient defined");

    // above ensures gradient is defined
    Tensor cpy_grad = this->traversed[0]->grad->tensor.clone();
    this->traversed[0]->zero_grad();
    this->traversed[0]->run_backward(cpy_grad);
    this->traversed[0]->accumulate_gradient(cpy_grad);
    for(size_t i = 1; i < this->traversed.size(); ++i){
        if(this->traversed[i]->grad && this->traversed[i]->grad->tensor.is_null()){
            trace_child_view_change(this->traversed, i+1, this->traversed[i]);
        }else{
            this->traversed[i]->run_backward();
        }
    }
}


template<class HashSet>
void AutoGrad<HashSet>::validate_graph(){
    for(const auto& node : this->traversed){
        utils::throw_exception(node->tensor, "Error, expected tensor to be defined");
        utils::throw_exception(node->grad, "Error, expected grad to be defined");
        utils::throw_exception(node->backwardFunc, "Error, expected backwardFunc to be defined");
    }
}




template class AutoGrad<std::unordered_set<intrusive_ptr<utility::GraphNode>>>;

}
