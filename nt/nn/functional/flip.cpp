#include "../../functional/functional.h"
#include "../../Tensor.h"
#include "../../intrusive_ptr/intrusive_ptr.hpp"
#include "../TensorGrad.h"
#include "../functional_class.h"
#include "../../utils/macros.h"
#include <algorithm>

namespace nt{
namespace functional{

TensorGrad TensorGrad_Functional_Class::flip(const TensorGrad& x, utils::optional_list list){
    if(!x.track_grad()) return TensorGrad(::nt::functional::flip(x.detach(), list), false);

    TensorGrad result = TensorGrad::create_read_node(::nt::functional::flip(x.detach(), list), x);

    if(!list){
        result.create_read_backward_function(
            [](const Tensor &grad,
                  std::vector<intrusive_ptr<TensorGrad>> &parents) {
                parents[0]->accumulate_gradient(::nt::functional::flip(grad));
            });
    }else{
        result.create_read_backward_function(
            [list = std::move(list)](const Tensor &grad,
                  std::vector<intrusive_ptr<TensorGrad>> &parents) {
                if(list.is_scalar())
                    parents[0]->accumulate_gradient(::nt::functional::flip(grad, list));
                std::reverse(const_cast<int64_t*>(list.begin()), const_cast<int64_t*>(list.end()));
                parents[0]->accumulate_gradient(::nt::functional::flip(grad, list));
            });

    }
    return std::move(result);
}



TensorGrad TensorGrad_Functional_Class::flip_view(const TensorGrad& x, utils::optional_list list){
    if(!x.track_grad()) return TensorGrad(::nt::functional::flip_view(x.detach(), list), false);
    TensorGrad result = x.create_view_node(::nt::functional::flip_view(x.detach(), list));
    result.create_view_backward_function(x, [list](Tensor& grad){
        return ::nt::functional::flip_view(grad, list);
    });
    return std::move(result);

}


}
}

