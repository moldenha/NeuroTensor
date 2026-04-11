#ifndef NT_TENSOR_GRAD_HPP__
#define NT_TENSOR_GRAD_HPP__

#include "TensorGrad.h"
#include "../functional/tensor_files/fill.h" //zeros_like
#include "../intrusive_ptr/intrusive_ptr.hpp"
#include "../utils/type_traits.h"
#include <type_traits>

namespace nt{


//this function is used to create backward functions
//it's main use is in functions where knowing if a tensor was used last is irrelevant

namespace grad::details{
// Put it in a single function

NT_ALWAYS_INLINE intrusive_ptr<tensor_holder> make_tensor_holder(const TensorGrad &t) {
    return nt::intrusive_ptr<tensor_holder>::make(
        t.detach().conditional_mutate_clone());
}
NT_ALWAYS_INLINE intrusive_ptr<tensor_holder> make_tensor_holder(intrusive_ptr<tensor_holder> t) {
    return t;
}
NT_ALWAYS_INLINE intrusive_ptr<tensor_holder> make_tensor_holder(const Tensor &t) {
    return intrusive_ptr<tensor_holder>::make(t.conditional_mutate_clone());
}

template<typename T>
NT_ALWAYS_INLINE T&& make_tenor_holder(T&& input){return std::forward<T>(input);}

// this makes a backward function for the node on a read or write
template<typename BackFunc, typename... Args>
NT_ALWAYS_INLINE void set_back_func(intrusive_ptr<::nt::grad::utility::GraphNode>& ptr, BackFunc&& func, Args&&... args){
    auto holders = std::make_tuple(make_tensor_holder(std::forward<Args>(args))...);

    ptr->set_func(::nt::grad::utility::backward_func::func_type(
        [func = std::forward<BackFunc>(func), holders = std::move(holders)](auto&& out_grad, auto&& self_grad) mutable {
            std::apply(
                [&](auto&&... unpacked) {
                    func(std::forward<decltype(out_grad)>(out_grad), std::forward<decltype(self_grad)>(self_grad), unpacked...);
                },
                holders
            );
        }
    ));
}

template<typename F, typename Tuple>
struct result_from_tuple;

template<typename F, typename... Args>
struct result_from_tuple<F, std::tuple<Args...>> {
    using type = std::invoke_result_t<F, Args...>;
};

// this makes a backward function for a view
template<typename GradFunc, typename... Args>
NT_ALWAYS_INLINE void set_grad_back_func(intrusive_ptr<::nt::grad::utility::GraphNode>& ptr, GradFunc&& func, Args&&... args){

    auto holders = std::make_tuple(make_tensor_holder(std::forward<Args>(args))...);
    static_assert(type_traits::is_decay_same_v<typename result_from_tuple<GradFunc, decltype(holders)>::type, Tensor>,
            "Error: Grad function to get gradient should return a Tensor");

    ptr->set_func(::nt::grad::utility::backward_func::func_type(
        [func = std::forward<GradFunc>(func), holders = std::move(holders)]
        (const Tensor& out_grad, const std::vector<::nt::intrusive_ptr<::nt::TensorGrad>>& parents) mutable {
            parents[0]->accumulate_gradient(
                std::apply(
                [&](auto&&... unpacked) {
                    func(out_grad, unpacked...);
                },
                holders
            )
            );
        }));
}

}
template<typename backward_func>
inline void TensorGrad::create_read_backward_function(backward_func&& func, const char* func_name){
	/* static_assert(all_tensor_grads<Args...>::value, "All arguments must be TensorGrad"); */
	if(!this->track_grad()){return;}
    this->Node->ensure_backward_initialization();
    this->Node->set_name(std::string(func_name));
    grad::details::set_back_func(this->Node, std::forward<backward_func>(func));

}

template<typename backward_func, typename Arg>
inline void TensorGrad::create_read_backward_function(backward_func&& func, Arg&& arg, const char* func_name){
    // std::cout << "function name was "<<func_name<<std::endl;
	/* static_assert(all_tensor_grads<Args...>::value, "All arguments must be TensorGrad"); */
	if(!this->track_grad()){return;}
    this->Node->ensure_backward_initialization();
    this->Node->set_name(std::string(func_name));
    grad::details::set_back_func(this->Node, std::forward<backward_func>(func), std::forward<Arg>(arg));
}


template<typename backward_func, typename Arg1, typename Arg2>
inline void TensorGrad::create_read_backward_function(backward_func&& func, Arg1&& arg1, Arg2&& arg2, const char* func_name){
	/* static_assert(all_tensor_grads<Args...>::value, "All arguments must be TensorGrad"); */
	if(!this->track_grad()){return;}
    this->Node->ensure_backward_initialization();
    this->Node->set_name(std::string(func_name));
    grad::details::set_back_func(this->Node, std::forward<backward_func>(func), std::forward<Arg1>(arg1), std::forward<Arg2>(arg2));
}


template<typename backward_func, typename Arg1, typename Arg2, typename Arg3>
inline void TensorGrad::create_read_backward_function(backward_func&& func, Arg1&& arg1, Arg2&& arg2, Arg3&& arg3, const char* func_name){
	/* static_assert(all_tensor_grads<Args...>::value, "All arguments must be TensorGrad"); */
	if(!this->track_grad()){return;}
    this->Node->ensure_backward_initialization();
    this->Node->set_name(std::string(func_name));
    grad::details::set_back_func(this->Node, std::forward<backward_func>(func), 
                                 std::forward<Arg1>(arg1), std::forward<Arg2>(arg2), std::forward<Arg3>(arg3));

}

template<typename backward_func, typename Arg1, typename Arg2, typename Arg3, typename Arg4>
inline void TensorGrad::create_read_backward_function(backward_func&& func, Arg1&& arg1, Arg2&& arg2, Arg3&& arg3, Arg4&& arg4, const char* func_name){
	/* static_assert(all_tensor_grads<Args...>::value, "All arguments must be TensorGrad"); */
	if(!this->track_grad()){return;}
    this->Node->ensure_backward_initialization();
    this->Node->set_name(std::string(func_name));
    
    grad::details::set_back_func(this->Node, std::forward<backward_func>(func), 
                                 std::forward<Arg1>(arg1), std::forward<Arg2>(arg2), std::forward<Arg3>(arg3),
                                 std::forward<Arg4>(arg4));
}


// Creates a view of the original gradient
template<typename BackFunc>
inline void TensorGrad::create_view_backward_function(const TensorGrad& original, BackFunc&& func, const char* func_name){
    if(!this->track_grad()) return;
    utils::throw_exception(this->Node->edge_type() == grad::utility::EdgeType::View,
            "Error, expected view backward function to be made for a view edge type");
    utils::throw_exception(original.track_grad(),
            "Error: Trying to track the gradient of a tensor that is not having its gradient tracked");
    original.Node->ensure_gradient_init();
    this->Node->ensure_backward_initialization(false);
    this->Node->set_func(nullptr);
    this->Node->gradient_() = std::forward<BackFunc>(func)(original.Node->gradient_());
    this->Node->set_name(func_name);
}

template<typename... Args>
inline TensorGrad TensorGrad::create_read_node(const Tensor& output_t, const Args&... args){
    const auto process = [](const TensorGrad& val) -> intrusive_ptr<::nt::grad::utility::GraphNode> {return val.Node;};
    node_type new_node = ::nt::grad::utility::makeReadGraphNode(output_t, process(args)...);
    intrusive_ptr<autograd_type> new_tape = make_intrusive<autograd_type>(new_node);
    ([&](auto& arg) {
        // arg is a tensor
        static_assert(type_traits::is_same_v<type_traits::remove_cvref_t<decltype(arg)>, TensorGrad>,
                "Expected read tensor grad to only depend on other tensor grads");
        new_tape->merge(arg.Tape);
        // arg.Tape->add_node(new_node);
    }(args), ...);    
    new_tape->add_node(new_node);
    return TensorGrad(std::move(new_node), std::move(new_tape));

}


// this is used to mark when a modification was made to itself
template<typename BackFunc, typename... Args>
inline TensorGrad& TensorGrad::create_write_node(BackFunc&& func, const char* name, const Args&... args){
    const auto process = [](const TensorGrad& val) -> intrusive_ptr<::nt::grad::utility::GraphNode> {return val.Node;};
    node_type new_node = ::nt::grad::utility::makeWriteGraphNode(
                                std::forward<BackFunc>(func), name, 
                                this->Node, process(args)...);
    
    ([&](auto& arg) {
        // arg is a TensorGrad
        static_assert(type_traits::is_same_v<type_traits::remove_cvref_t<decltype(arg)>, TensorGrad>,
                "Expected writed tensor grad to only depend on other tensor grads");
        this->Tape->merge(arg.Tape);
        // arg.Tape->add_node(new_node);   
    }(args), ...);
    this->Tape->add_node(new_node);
    return *this; 
}

inline TensorGrad TensorGrad::create_view_node(const Tensor& output) const {
    node_type new_node = ::nt::grad::utility::makeViewGraphNode(output, this->Node);
    this->Tape->add_node(new_node);
    return TensorGrad(std::move(new_node), this->Tape); 
}



// this is a function to get the last "self modifying" node
// so basically it will go down the list and get the last tensor that contains what is currently
// pointing to "self"
// inline intrusive_ptr<grad::utility::GraphNode> get_current_self_node(intrusive_ptr<grad::utility::GraphNode> current){
//     if(current->children.size() == 0) return current;
//     for(auto it = current->children.rbegin(); it != current->children.rend(); ++it){
//         if((*it)->backwardFunc->is_self_mod()){
//             return get_current_self_node(*it);
//         }
//     }
//     return current;
// }






//track tensors should be used when the current tensor is being modified
//ex: TensorGrad A += B;

// The way it works:
//  - So when A += B happens, there is a new A that is created.
//  - So 3 tensors, old_A, new_A, and B
//  - old_A needs to be a parent of new_A, but if new_A is
//          returned only, old_A will go out of scope because
//          old_A will only be stored as a weak reference
//  - So, old_A is returned, [ fine because the tensor is modified and the same in both ]
//  - Instead, new_A becomes a child of old_A, so grad tracking wise new_A will still
//          be back propogated before old_A, but old_A won't go out of scope








// template<typename OutOperator>
// inline void TensorGrad::track_old_grad(const TensorGrad& t, OutOperator&& op, const char* func_name){
//     if(!t.track_grad()){
//         this->track_grad_(false);
//         // this->do_track_grad = false;
//         return;
//     }
//     t.Node->ensure_backward_initialization(true);
//     t.Node->ensure_gradient_init();
//     this->Node->ensure_backward_initialization(false);
//     if(!this->Node->backwardFunc->is_view_change()) this->Node->backwardFunc = make_intrusive<grad::utility::view_backward_func>();
//     this->Node->backwardFunc->set_name(func_name);
//     this->Node->backwardFunc->set(nullptr);
//     this->Node->grad->tensor = Tensor::Null();
//     this->Node->grad->tensor = std::forward<OutOperator>(op)(t.Node->grad->tensor);
//     this->Node->parents.emplace_back(t.Node);
//     t.Node->children.emplace_back(this->Node);
// }

template<typename OGFunc>
inline TensorGrad TensorGrad::make_view_grad(Tensor& tensor, const TensorGrad& parent, OGFunc&& func){
    if(!parent.track_grad()) return TensorGrad(tensor, false);
    TensorGrad out = parent.create_view_node(tensor);
    out.create_view_backward_function(parent, std::forward<OGFunc>(func));
    return std::move(out);
}


template<typename... Args>
inline TensorGrad TensorGrad::make_tensor_grad(Tensor& tensor,
                                               std::function<void(const Tensor&, std::vector<intrusive_ptr<TensorGrad>>&)> back_func,
                                               const TensorGrad& parent, Args&&... parents){
    TensorGrad out = TensorGrad::create_read_node(tensor, parent, std::forward<Args>(parents)...);
    out.create_read_backward_function(back_func, "makeTensorGrad");
    return std::move(out);
}


}

#endif //NT_TENSOR_GRAD_HPP__
