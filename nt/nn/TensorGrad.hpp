#ifndef NT_TENSOR_GRAD_HPP__
#define NT_TENSOR_GRAD_HPP__

#include "TensorGrad.h"
#include "../functional/tensor_files/fill.h" //zeros_like
#include "../intrusive_ptr/intrusive_ptr.hpp"

namespace nt{


//this function is used to create backward functions
//it is used in tandem with track_tensors
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

template<typename BackFunc, typename... Args>
NT_ALWAYS_INLINE void set_back_func(intrusive_ptr<::nt::grad::utility::backward_func>& ptr, BackFunc&& func, Args&&... args){
    auto holders = std::make_tuple(make_tensor_holder(std::forward<Args>(args))...);

    ptr->set(::nt::grad::utility::backward_func::func_type(
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

template<typename GradFunc, typename... Args>
NT_ALWAYS_INLINE void set_grad_back_func(intrusive_ptr<::nt::grad::utility::backward_func>& ptr, GradFunc&& func, Args&&... args){
    auto holders = std::make_tuple(make_tensor_holder(std::forward<Args>(args))...);
    ptr->set(::nt::grad::utility::backward_func::func_type(
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
inline void TensorGrad::create_backward_function(backward_func&& func, const char* func_name){
	/* static_assert(all_tensor_grads<Args...>::value, "All arguments must be TensorGrad"); */
	if(!this->track_grad()){return;}
    this->Node->ensure_backward_initialization();
    this->Node->backwardFunc->set_name(std::string(func_name));
    grad::details::set_back_func(this->Node->backwardFunc, std::forward<backward_func>(func));

}

template<typename backward_func, typename Arg>
inline void TensorGrad::create_backward_function(backward_func&& func, Arg&& arg, const char* func_name){
    // std::cout << "function name was "<<func_name<<std::endl;
	/* static_assert(all_tensor_grads<Args...>::value, "All arguments must be TensorGrad"); */
	if(!this->track_grad()){return;}
    this->Node->ensure_backward_initialization();
    this->Node->backwardFunc->set_name(std::string(func_name));
    grad::details::set_back_func(this->Node->backwardFunc, std::forward<backward_func>(func), std::forward<Arg>(arg));
}


template<typename backward_func, typename Arg1, typename Arg2>
inline void TensorGrad::create_backward_function(backward_func&& func, Arg1&& arg1, Arg2&& arg2, const char* func_name){
	/* static_assert(all_tensor_grads<Args...>::value, "All arguments must be TensorGrad"); */
	if(!this->track_grad()){return;}
    this->Node->ensure_backward_initialization();
    this->Node->backwardFunc->set_name(std::string(func_name));
    grad::details::set_back_func(this->Node->backwardFunc, std::forward<backward_func>(func), std::forward<Arg1>(arg1), std::forward<Arg2>(arg2));
}


template<typename backward_func, typename Arg1, typename Arg2, typename Arg3>
inline void TensorGrad::create_backward_function(backward_func&& func, Arg1&& arg1, Arg2&& arg2, Arg3&& arg3, const char* func_name){
	/* static_assert(all_tensor_grads<Args...>::value, "All arguments must be TensorGrad"); */
	if(!this->track_grad()){return;}
    this->Node->ensure_backward_initialization();
    this->Node->backwardFunc->set_name(std::string(func_name));
    grad::details::set_back_func(this->Node->backwardFunc, std::forward<backward_func>(func), 
                                 std::forward<Arg1>(arg1), std::forward<Arg2>(arg2), std::forward<Arg3>(arg3));

}

template<typename backward_func, typename Arg1, typename Arg2, typename Arg3, typename Arg4>
inline void TensorGrad::create_backward_function(backward_func&& func, Arg1&& arg1, Arg2&& arg2, Arg3&& arg3, Arg4&& arg4, const char* func_name){
	/* static_assert(all_tensor_grads<Args...>::value, "All arguments must be TensorGrad"); */
	if(!this->track_grad()){return;}
    this->Node->ensure_backward_initialization();
    this->Node->backwardFunc->set_name(std::string(func_name));
    
    grad::details::set_back_func(this->Node->backwardFunc, std::forward<backward_func>(func), 
                                 std::forward<Arg1>(arg1), std::forward<Arg2>(arg2), std::forward<Arg3>(arg3),
                                 std::forward<Arg4>(arg4));
}



// this is a function to get the last "self modifying" node
// so basically it will go down the list and get the last tensor that contains what is currently
// pointing to "self"
inline intrusive_ptr<grad::utility::GraphNode> get_current_self_node(intrusive_ptr<grad::utility::GraphNode> current){
    if(current->children.size() == 0) return current;
    for(auto it = current->children.rbegin(); it != current->children.rend(); ++it){
        if((*it)->backwardFunc->is_self_mod()){
            return get_current_self_node(*it);
        }
    }
    return current;
}

inline void TensorGrad::track_tensors(const TensorGrad& t){
    if(!this->track_grad()) return;
    utils::THROW_EXCEPTION(t.track_grad(), "\nError: told to track tensor that is not tracking a gradient"
                                            ", this will cause a segmentation fault, please look at function implementation");
    
    // Step1: Make sure that t is having it's gradient tracked (above)
    // Step2: Acknowledge (*this) is the child -> ensure initialization
    // Step3: Make t the weak_intrusive_ptr<graph::utility::GraphNode> parent
    // Step4: Make (*this) the intrusive_ptr<graph::utility::GraphNode> child
    
    this->Node->ensure_backward_initialization();
    intrusive_ptr<grad::utility::GraphNode> last_node = get_current_self_node(t.Node);
    last_node->ensure_backward_initialization();

    this->Node->parents.emplace_back(last_node); // Step3
    last_node->children.emplace_back(this->Node); // Step4
}

template<typename... Args>
inline void TensorGrad::track_tensors(const TensorGrad& t, const Args&... args){
    if(!this->track_grad()) return;
    this->track_tensors(t);
    this->track_tensors(args...);
}


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




template<typename BackFunc>
inline void TensorGrad::track_self_mod_tensors(BackFunc&& func, const char* func_name){
    if(!this->track_grad()) return;
    intrusive_ptr<grad::utility::GraphNode> 
        new_node = make_intrusive<grad::utility::GraphNode>(this->Node->tensor, grad::utility::DontTrackGrad{});
    // if(this->Node->backwardFunc->is_view_change()){
    //     std::cout << "this parent is a view change" << std::endl;
    // }else{
    //     std::cout << "this parent is not a view change" << std::endl;
    // }
    new_node->ensure_self_mod_backward_initialization(true);
    new_node->backwardFunc->set(std::forward<BackFunc>(func));
    new_node->backwardFunc->set_name(func_name);
    intrusive_ptr<grad::utility::GraphNode> last_node = get_current_self_node(this->Node);

    new_node->parents.emplace_back(last_node);
    last_node->children.emplace_back(new_node);
}

template<typename BackFunc, typename... Args>
inline void TensorGrad::track_self_mod_tensors(BackFunc&& func, const char* func_name, const TensorGrad& gr, const Args&... args){
    if(!this->track_grad()) return;
    intrusive_ptr<grad::utility::GraphNode> 
        new_node = make_intrusive<grad::utility::GraphNode>(this->Node->tensor, grad::utility::DontTrackGrad{});
    new_node->ensure_self_mod_backward_initialization(true);
    new_node->backwardFunc->set(std::forward<BackFunc>(func));
    new_node->backwardFunc->set_name(func_name);
    TensorGrad holder(new_node);
    holder.track_tensors(*this, gr, args...);
}



// template<typename OutOperator, typename... Args>
// inline void TensorGrad::track_grad(const TensorGrad& t, OutOperator&& op, const char* func_name, Args&&... args){
//     if(!t.track_grad()){
//         this->track_grad_(false);
//         return;
//     }
//     // Will automatically have *this track the gradient if t is having the gradient tracked
//     // This is because *this should only be a view or stride change of t
//     // Therefore, if t is having its gradient tracked, so should *this
    
//     std::string back_name = std::string("GradTrack") + std::string(func_name);
//     this->Node->ensure_backward_initialization(false);
//     grad::details::set_grad_back_func(this->Node->backwardFunc, std::forward<OutOperator>(op),
//                                       std::forward<Args>(args)...);
//     this->backwardFunc->set_name(std::move(back_name));

//     this->track_tensors(t);
// }


template<typename OutOperator>
inline void TensorGrad::track_grad(const TensorGrad& t, OutOperator&& op, const char* func_name){
    if(!t.track_grad()){
        this->track_grad_(false);
        // this->do_track_grad = false;
        return;
    }
    t.Node->ensure_backward_initialization(true);
    t.Node->ensure_gradient_init();
    this->Node->ensure_backward_initialization(false);
    if(!this->Node->backwardFunc->is_view_change()) this->Node->backwardFunc = make_intrusive<grad::utility::view_backward_func>();
    this->Node->backwardFunc->set_name(func_name);
    this->Node->backwardFunc->set(nullptr);
    this->Node->grad->tensor = Tensor::Null();
    this->Node->grad->tensor = std::forward<OutOperator>(op)(t.Node->grad->tensor);
    this->Node->parents.emplace_back(t.Node);
    t.Node->children.emplace_back(this->Node);
}

template<typename OGFunc>
inline TensorGrad TensorGrad::make_view_grad(Tensor& tensor, const TensorGrad& parent, OGFunc&& func){
    TensorGrad out(tensor, parent.track_grad());
    out.track_grad(parent, std::forward<OGFunc>(func));
    return std::move(out);
}


template<typename... Args>
inline TensorGrad TensorGrad::make_tensor_grad(Tensor& tensor,
                                               std::function<void(const Tensor&, std::vector<intrusive_ptr<TensorGrad>>&)> back_func,
                                               const TensorGrad& parent, Args&&... parents){
    TensorGrad out(tensor, true);
    out.Node->ensure_backward_initialization(false);
    out.Node->backwardFunc->set(back_func);
    out.track_tensors(parent, parents...);
    return std::move(out);
}


}

#endif //NT_TENSOR_GRAD_HPP__
