#include "../../functional/functional.h"
#include "../../Tensor.h"
#include "../../intrusive_ptr/intrusive_ptr.hpp"
#include "../TensorGrad.h"
#include "../functional_class.h"
#include "../../utils/macros.h"

namespace nt{
namespace functional{

TensorGrad TensorGrad_Functional_Class::round(const TensorGrad& input){
    TensorGrad result(::nt::functional::round(input.detach()), input.track_grad());
    if(!input.track_grad()){
        result.track_grad_(false);
        return std::move(result);
    }
    result.track_tensors(input);
    result.create_backward_function(
        [](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents){
            // commented out because it would just be 0 + 0 at all points
            // parents[0]->accumulate_gradient(::nt::functional::zeros_like(grad));  
        }
    );
    return std::move(result);
}
TensorGrad TensorGrad_Functional_Class::trunc(const TensorGrad& input){
    TensorGrad result(::nt::functional::trunc(input.detach()), input.track_grad());
    if(!input.track_grad()){
        result.track_grad_(false);
        return std::move(result);
    }
    result.track_tensors(input);
    result.create_backward_function(
        [](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents){
            // commented out because it would just be 0 + 0 at all points
            // parents[0]->accumulate_gradient(::nt::functional::zeros_like(grad));  
        }
    );
    return std::move(result);
}


TensorGrad TensorGrad_Functional_Class::floor(const TensorGrad& input){
    TensorGrad result(::nt::functional::floor(input.detach()), input.track_grad());
    if(!input.track_grad()){
        result.track_grad_(false);
        return std::move(result);
    }
    result.track_tensors(input);
    result.create_backward_function(
        [](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents){
            // commented out because it would just be 0 + 0 at all points
            // parents[0]->accumulate_gradient(::nt::functional::zeros_like(grad));  
        }
    );
    return std::move(result);
}


TensorGrad TensorGrad_Functional_Class::ceil(const TensorGrad& input){
    TensorGrad result(::nt::functional::ceil(input.detach()), input.track_grad());
    if(!input.track_grad()){
        result.track_grad_(false);
        return std::move(result);
    }
    result.track_tensors(input);
    result.create_backward_function(
        [](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents){
            // commented out because it would just be 0 + 0 at all points
            // parents[0]->accumulate_gradient(::nt::functional::zeros_like(grad));  
        }
    );
    return std::move(result);
}

}
}
