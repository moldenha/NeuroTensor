#include "BackwardFunc.h"
#include "TensorGrad.h"

namespace nt::grad::utility {

void backward_func::run(const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& v){
    // ideally the conditions it would come in
    if(this->Func == nullptr)
        return;
    // This is extremely important:
    for(const auto& parent : v){
        // At this point, because of the TensorGrad::track_grad function setup
        // (v->Node->grad is null) should be impossible, because an error would have been thrown
        // So, this function is really just making sure that v->Node->grad->tensor.is_null() is not true
        // Otherwise, this could cause errors (or un-made errors that become segfaults)
        utils::THROW_EXCEPTION(bool(parent), "Error, parent was nullptr and all strong references have disappeared");
        // if the Node has not gone out of scope already [otherwise no point]
        if(parent->Node->tensor && !parent->Node->tensor->tensor.is_null())
            parent->Node->ensure_gradient_init();
    }
    this->Func(grad, v);
}


void backward_func::run(const Tensor& grad, const std::vector<weak_intrusive_ptr<GraphNode>>& weak_parents){
    if(Func == nullptr)
        return;
    std::vector<intrusive_ptr<TensorGrad>> parents;
    parents.reserve(weak_parents.size());
    // A specific constructor for TensorGrad that takes an intrusive_ptr<GraphNode>, and a bool to track the gradient
    for(const auto& weak_parent : weak_parents){
        if(auto lock = weak_parent.lock()){
            parents.emplace_back(make_intrusive<TensorGrad>(lock));
        }else{
            // a little more tricky
            // but basically means that the calculation of the gradient doesn't matter because it is a temporary tensor
            // So, as such:
            intrusive_ptr<GraphNode> temp_node = make_intrusive<GraphNode>(DontTrackGrad{}); // make a null tensor
            temp_node->grad = make_intrusive<tensor_holder>(Tensor::Null()); // better to call error than potentially seg fault
            parents.emplace_back(make_intrusive<TensorGrad>(std::move(temp_node)));
        }
    }
    this->run(grad, parents);
}


}
