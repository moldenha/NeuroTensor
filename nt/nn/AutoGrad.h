#ifndef NT_NN_AUTO_GRAD_H__
#define NT_NN_AUTO_GRAD_H__

//forward declarations
#include <unordered_set>
#include "../intrusive_ptr/intrusive_ptr.hpp"

namespace nt::grad{
namespace utility{ class GraphNode; }
template<class HashSet = std::unordered_set<intrusive_ptr<utility::GraphNode>> >
class AutoGrad;
}

#include "../Tensor.h"
#include "../utils/tensor_holder.h"
#include "../functional/functional.h"
#include "BackwardFunc.h"
#include <vector>


namespace nt::grad {

namespace utility{


// Holds: Current tensor, the gradient, and the backward function
//  - From the original TensorGrad
//
//  Each TensorGrad is just a holder of a GraphNode with additional functionality
struct DontValidateGraph {};
struct DontTrackGrad {};

struct NEUROTENSOR_API GraphNode : public intrusive_ptr_target {
    std::vector<intrusive_ptr<utility::GraphNode>> children;
    std::vector<weak_intrusive_ptr<utility::GraphNode>> parents;

    intrusive_ptr<tensor_holder> tensor, grad;
    intrusive_ptr<backward_func> backwardFunc;

    GraphNode()
    :tensor(make_intrusive<tensor_holder>(Tensor::Null())), 
        grad(make_intrusive<tensor_holder>(Tensor::Null())), 
        backwardFunc(make_intrusive<backward_func>())
    {}

    GraphNode(DontTrackGrad)
    :tensor(make_intrusive<tensor_holder>(Tensor::Null())),
        grad(nullptr),
        backwardFunc(nullptr)
    {}

    GraphNode(intrusive_ptr<tensor_holder> tensor_)
    :tensor(tensor_), 
        grad(make_intrusive<tensor_holder>(Tensor::Null())), 
        backwardFunc(make_intrusive<backward_func>())
    {}

    GraphNode(intrusive_ptr<tensor_holder> tensor_, DontTrackGrad)
    :tensor(tensor_), 
        grad(make_intrusive<tensor_holder>(Tensor::Null())), 
        backwardFunc(make_intrusive<backward_func>())
    {}


    inline void release_resources() override { children.clear(); }
    void run_backward();
    void run_backward(const Tensor& _grad);
    void accumulate_gradient(const Tensor& in_grad);
    void accumulate_gradient(Scalar num);
    void zero_grad();
    inline void ensure_initialization(bool zero_if_uninit = false){
        if(!this->tensor){
            this->tensor = make_intrusive<tensor_holder>(Tensor::Null());
            zero_if_uninit = false;
        }
        this->ensure_backward_initialization(zero_if_uninit && !this->tensor->tensor.is_null());
    }

    // backward refers to the gradient and the backward function
    inline void ensure_backward_initialization(bool zero_if_uninit = false ) {
        //this is a function that can be used to make sure grad and backwardFunc are not nullptr
        if(!this->grad) this->grad = make_intrusive<tensor_holder>(zero_if_uninit ? nt::functional::zeros_like(tensor->tensor) : Tensor::Null());
        if(!this->backwardFunc) this->backwardFunc = make_intrusive<backward_func>();
    }

    inline void ensure_view_backward_initialization(bool zero_if_uninit = false ) {
        //this is a function that can be used to make sure grad and backwardFunc are not nullptr
        if(!this->grad) this->grad = make_intrusive<tensor_holder>(zero_if_uninit ? nt::functional::zeros_like(tensor->tensor) : Tensor::Null());
        if(!this->backwardFunc) this->backwardFunc = make_intrusive<view_backward_func>();
    }

    // This function is to make sure the gradient is initialized
    // So, if !bool(this->grad) it will be initialized, or if this->grad->tensor.is_null()
    // it will also be initialized
    inline void ensure_gradient_init() {
        if(!bool(this->grad))
            this->grad = make_intrusive<tensor_holder>(nt::functional::zeros_like(tensor->tensor));
        else if (this->grad->tensor.is_null())
            this->grad->tensor = nt::functional::zeros_like(tensor->tensor);
    }

};



} // nt::grad::utility::


// This is really more of a helper class designed to make the backward pass more efficient 
template<class HashSet>
class NEUROTENSOR_API AutoGrad{
    intrusive_ptr<utility::GraphNode> start_node;
    std::vector<intrusive_ptr<utility::GraphNode>> traversed;
    static void traverse_graph_impl(const intrusive_ptr<utility::GraphNode>& node,
                                    std::vector<intrusive_ptr<utility::GraphNode>>& result,
                                    HashSet& visited);
    // This is meant to be a function that returns an order of all the graph nodes
    // and the order in which the nodes will have their backward functions called
    inline std::vector<intrusive_ptr<utility::GraphNode>> traverse_graph() const{
        std::vector<intrusive_ptr<utility::GraphNode>> result;
        HashSet visited;
        AutoGrad<HashSet>::traverse_graph_impl(this->start_node, result, visited);
        return std::move(result);
    }

    // ensures graph has all elements defined
    void validate_graph();

public:
    AutoGrad() = delete;
    AutoGrad(const intrusive_ptr<utility::GraphNode>& node)
    :start_node(node) {
        traversed = traverse_graph();
        validate_graph();
    }

    AutoGrad(const intrusive_ptr<utility::GraphNode>& node, utility::DontValidateGraph)
    :start_node(node) {
        traversed = traverse_graph();
    }
    
    void zero_grad();
    void backward(const Tensor& initial_grad);
    void backward();
    const std::vector<intrusive_ptr<utility::GraphNode>>& get_path() const {return traversed;}

};

} // nt:grad
#endif
