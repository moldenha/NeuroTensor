#include "../../functional/functional.h"
#include "../../Tensor.h"
#include "../../intrusive_ptr/intrusive_ptr.hpp"
#include "../TensorGrad.h"
#include "../functional_class.h"
#include "../../utils/macros.h"

namespace nt{
namespace functional{

TensorGrad TensorGrad_Functional_Class::pad(const TensorGrad& input, std::vector<Tensor::size_value_t> padding, const char* mode, Scalar value){
    TensorGrad result(::nt::functional::pad(input.detach(), padding, mode, value), input.track_grad());
    if(!input.track_grad()){
        result.track_grad_(false);
        return std::move(result);
    }

    result.track_tensors(input);
    result.create_backward_function(
    [&padding](const Tensor& grad,
              std::vector<intrusive_ptr<TensorGrad>> &parents) {
        parents[0]->accumulate_gradient(::nt::functional::unpad(grad, padding, true)); //faster than copying memory
    });
    return std::move(result);
}


TensorGrad TensorGrad_Functional_Class::unpad(const TensorGrad& input, std::vector<Tensor::size_value_t> padding, bool no_contiguous){
    TensorGrad result(::nt::functional::unpad(input.detach(), padding, no_contiguous), input.track_grad());
    if(!input.track_grad()){
        result.track_grad_(false);
        return std::move(result);
    }

    result.track_grad(input,
        [&padding](const Tensor& grad){
        return ::nt::functional::unpad(grad, padding, true); // making it non-contiguous so the gradient can just be passed
    });
    return std::move(result);

}


}
}
