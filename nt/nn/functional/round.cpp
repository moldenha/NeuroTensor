#include "../../functional/functional.h"
#include "../../Tensor.h"
#include "../../intrusive_ptr/intrusive_ptr.hpp"
#include "../TensorGrad.h"
#include "../functional_class.h"
#include "../../utils/macros.h"

namespace nt{
namespace functional{

TensorGrad TensorGrad_Functional_Class::round(const TensorGrad& input){
    if(!input.track_grad()) return TensorGrad(::nt::functional::round(input.detach()), false);
    TensorGrad result = TensorGrad::create_read_node(::nt::functional::round(input.detach()), input);

    result.create_read_backward_function(
        [](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents){
            // commented out because it would just be 0 + 0 at all points
            // parents[0]->accumulate_gradient(::nt::functional::zeros_like(grad));  
            parents[0]->accumulate_gradient(0);
        }
    );
    return std::move(result);
}
TensorGrad TensorGrad_Functional_Class::trunc(const TensorGrad& input){
    if(!input.track_grad()) return TensorGrad(::nt::functional::trunc(input.detach()), false);
    TensorGrad result = TensorGrad::create_read_node(::nt::functional::trunc(input.detach()), input);

    result.create_read_backward_function(
        [](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents){
            // commented out because it would just be 0 + 0 at all points
            // parents[0]->accumulate_gradient(::nt::functional::zeros_like(grad));  
            parents[0]->accumulate_gradient(0);
        }
    );
    return std::move(result);
}


TensorGrad TensorGrad_Functional_Class::floor(const TensorGrad& input){
    if(!input.track_grad()) return TensorGrad(::nt::functional::floor(input.detach()), false);
    TensorGrad result = TensorGrad::create_read_node(::nt::functional::floor(input.detach()), input);
     
    result.create_read_backward_function(
        [](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents){
            // commented out because it would just be 0 + 0 at all points
            // parents[0]->accumulate_gradient(::nt::functional::zeros_like(grad));  
            parents[0]->accumulate_gradient(0);
        }
    );
    return std::move(result);
}


TensorGrad TensorGrad_Functional_Class::ceil(const TensorGrad& input){
    if(!input.track_grad()) return TensorGrad(::nt::functional::ceil(input.detach()), false);
    TensorGrad result = TensorGrad::create_read_node(::nt::functional::ceil(input.detach()), input);

    result.create_read_backward_function(
        [](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents){
            // commented out because it would just be 0 + 0 at all points
            // parents[0]->accumulate_gradient(::nt::functional::zeros_like(grad));  
            parents[0]->accumulate_gradient(0);
        }
    );
    return std::move(result);
}

}
}
