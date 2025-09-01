#include "../../functional/functional.h"
#include "../../Tensor.h"
#include "../../intrusive_ptr/intrusive_ptr.hpp"
#include "../TensorGrad.h"
#include "../functional_class.h"
#include "../../utils/macros.h"

namespace nt{
namespace functional{

TensorGrad TensorGrad_Functional_Class::flip(const TensorGrad& x, utils::optional_list list){
    TensorGrad result(::nt::functional::flip(x.detach(), list), x.track_grad());
    if (!x.track_grad()) {
        result.track_grad_(false);
        return std::move(result);
    }
    result.track_tensors(x);
    if(!list){
        result.create_backward_function(
            [](const Tensor &grad,
                  std::vector<intrusive_ptr<TensorGrad>> &parents) {
                parents[0]->accumulate_gradient(::nt::functional::flip(grad));
            });
    }else{
        result.create_backward_function(
            [list](const Tensor &grad,
                  std::vector<intrusive_ptr<TensorGrad>> &parents) {
                parents[0]->accumulate_gradient(::nt::functional::flip(grad, list));
            });

    }
    return std::move(result);
}



TensorGrad TensorGrad_Functional_Class::flip_view(const TensorGrad& x, utils::optional_list list){
    TensorGrad result(::nt::functional::flip_view(x.detach(), list), x.track_grad());
    result.track_grad(x, [list](Tensor& grad){
        return ::nt::functional::flip_view(grad, list);
    });
    return std::move(result);

}


}
}
