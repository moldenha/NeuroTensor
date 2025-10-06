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


// this basically gets what "self" is for this tensor
// so for example if I had the following:
// auto c = a + b;
// c *= 10;
// auto d = c / 3;
//
// on the divide (d = c / 3) gradient backward step where you are getting the gradient of that and setting the parent's gradient
// the step where c *= 10; that gradient should be updated
// but the problem is that c *= 10 step is a child of the parent c = a + b step
// so when running backward, and getting the original lock on the lines:
// '''
// if(auto lock = weak_parent.lock())
//            parents.emplace_back(make_intrusive<TensorGrad>(lock));
//'''
// it is running the backward function for auto d = c / 3
// however, it is setting the gradient for the c = a + b step
// when in reality, it needs to set the gradient for the c *= 10 step
// where then the c *= 10 step is backpropogated and sets the gradient for the c = a + b step
// So now, the lines are
// '''
// if(auto lock = weak_parent.lock())
//            parents.emplace_back(make_intrusive<TensorGrad>(get_last_self_mod(lock, self)));
// '''
//
// and the following function basically fixes that issue 

// intrusive_ptr<GraphNode> get_last_self_mod(const intrusive_ptr<GraphNode>& original_parent){
//     if(original_parent->children.size() == 0)
//         return original_parent;

//     // it iterates in reverse because it is looking for the most recent self modification
//     // which would have been added to the children vector near the end
//     // if it finds a specific self modification node
//     // then use that one
//     for (auto it = original_parent->children.rbegin(); it != original_parent->children.rend(); ++it) {
//         if((*it)->backwardFunc->is_self_mod()){
//             return *it;
//         }
//     }
//     return original_parent;
// }

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
