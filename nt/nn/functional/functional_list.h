#ifndef NT_TENSORGAD_FUNCTIONAL_LIST_H__
#define NT_TENSORGAD_FUNCTIONAL_LIST_H__

#include "../TensorGrad.h"
#include <type_traits>
#include "../../utils/utils.h"
#include "../../utils/always_inline_macro.h"

namespace nt{
namespace functional{

namespace details{

template<typename T,
         std::enable_if_t<std::is_same_v<std::decay_t<T>, TensorGrad>, bool> = true>
NT_ALWAYS_INLINE bool all_not_tracking_grad(T&& first){
    return first.track_grad() == false;
}


template<typename T, typename... Args,
         std::enable_if_t<std::is_same_v<std::decay_t<T>, TensorGrad>, bool> = true>
NT_ALWAYS_INLINE bool all_not_tracking_grad(T&& first, Args&&... args){
    return (first.track_grad() == false) && all_not_tracking_grad(std::forward<Args&&>(args)...);
}

template<typename T,
         std::enable_if_t<std::is_same_v<std::decay_t<T>, TensorGrad>, bool> = true>
NT_ALWAYS_INLINE void set_grad_if_tracking(Tensor* begin, T&& first){
    if(std::forward<T>(first).track_grad())
        *begin = std::forward<T>(first).grad();
}

template<typename T, typename... Args,
         std::enable_if_t<std::is_same_v<std::decay_t<T>, TensorGrad>, bool> = true>
NT_ALWAYS_INLINE void set_grad_if_tracking(Tensor* begin, T&& first, Args&&... args){
    if(std::forward<T>(first).track_grad())
        *begin = std::forward<T>(first).grad();
    ++begin;
    set_grad_if_tracking(begin, std::forward<Args>(args)...);
}

}

template<typename T, typename... Args,
         std::enable_if_t<std::is_same_v<std::decay_t<T>, TensorGrad>, int>>
inline TensorGrad list(T&& first, Args&&... rest){
	static_assert(utils::is_all_same_v<std::decay_t<T>, std::decay_t<Args>...>, 
                  "Expected to make a list of all TensorGrads");

    // create the result TensorGrad
    bool all_not_tracking = details::all_not_tracking_grad(std::forward<T>(first), std::forward<Args>(rest)...);
    if(all_not_tracking){
        return TensorGrad(list(std::forward<T>(first).detach(), (std::forward<Args>(rest).detach())...), false);
    }

    TensorGrad result(list(std::forward<T>(first).detach(), (std::forward<Args>(rest).detach()) ...),
                        true);
    // From here on out
    // If there is a tensor who's gradient is not being tracked
    // the result.grad()[i].item<Tensor>().is_null() == true
    if(!result.Node->grad) result.Node->grad = make_intrusive<tensor_holder>(Tensor::makeNullTensorArray(sizeof...(Args) + 1));
    else result.grad() = Tensor::makeNullTensorArray(sizeof...(Args) + 1);
    // for fast access:
    Tensor* grad_begin = reinterpret_cast<Tensor*>(result.grad().data_ptr());
    Tensor* grad_end = reinterpret_cast<Tensor*>(result.grad().data_ptr_end());
    if(std::forward<T>(first).track_grad()) std::forward<T>(first).Node->ensure_gradient_init();

    ((std::forward<Args>(rest).track_grad() ? (std::forward<Args>(rest).Node->ensure_gradient_init(), void()) : void()), ...);
    
    details::set_grad_if_tracking(grad_begin, std::forward<T>(first), std::forward<Args>(rest)...); 

    ////create the gradient for the result
    //result.grad = make_intrusive<tensor_holder>(
    //    list(first.grad->tensor, (rest.grad->tensor) ...));
  
    // Only track if the gradient is supposed to be tracked
    // There will not be a backward function anyways
    if(std::forward<T>(first).track_grad()) result.track_tensors(std::forward<T>(first));
    
    ((std::forward<Args>(rest).track_grad() ? (result.track_tensors(std::forward<Args>(rest)), void()) : void()), ...);

    return std::move(result);
}

}
}

#endif
