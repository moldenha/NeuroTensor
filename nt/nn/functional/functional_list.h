#ifndef _NT_TENSORGAD_FUNCTIONAL_LIST_H_
#define _NT_TENSORGAD_FUNCTIONAL_LIST_H_

#include "../TensorGrad.h"
#include <type_traits>
#include "../../utils/utils.h"

namespace nt{
namespace functional{

template<typename T, typename... Args,
         std::enable_if_t<std::is_same_v<std::decay_t<T>, TensorGrad>, int>>
inline TensorGrad list(T&& first, Args&&... rest){
	static_assert(utils::is_all_same_v<std::decay_t<T>, std::decay_t<Args>...>, 
                  "Expected to make a list of all TensorGrads");

    // create the result TensorGrad
    TensorGrad result(list(first.tensor, (rest.tensor) ...),
                      first.grad_required);

    bool track_grad = first.do_track_grad;
    bool require_grad = first.grad_required;

    // ensure consistency for tracking and requiring gradients
    ((utils::throw_exception(std::forward<Args>(rest).do_track_grad ==
                                 track_grad,
                             "Expected consistent track_grad values")),
     ...);
    ((utils::throw_exception(std::forward<Args>(rest).grad_required ==
                                 require_grad,
                             "Expected consistent grad_required values")),
     ...);

    // update track_grad and grad_required flags
    if (!require_grad) {
        track_grad = false;
    }
    if (!track_grad) {
        result.do_track_grad = false;
        return result; // return directly if tracking is not needed
    }



    // initialize grads if not already set
    if (first.grad == nullptr) {
        first.grad =
            make_intrusive<tensor_holder>(functional::zeros_like(first.tensor));
    }
    ((rest.grad = (rest.grad == nullptr)
                      ? make_intrusive<tensor_holder>(
                            functional::zeros_like(rest.tensor))
                      : rest.grad),
     ...);
    //create the gradient for the result
    result.grad = make_intrusive<tensor_holder>(
        list(first.grad->tensor, (rest.grad->tensor) ...));
    
    // set up parent references
    // this also automatically tracks children
    result.track_tensors(first, rest...);

    return result;
}
}
}

#endif
