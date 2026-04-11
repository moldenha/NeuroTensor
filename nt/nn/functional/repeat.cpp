#include "../../functional/functional.h"
#include "../../Tensor.h"
#include "../../intrusive_ptr/intrusive_ptr.hpp"
#include "../TensorGrad.h"
#include "../functional_class.h"
#include "../../utils/macros.h"

namespace nt{
namespace functional{

TensorGrad TensorGrad_Functional_Class::repeat_(const TensorGrad& input, Tensor::size_value_t dim, Tensor::size_value_t amt){
    if(!input.track_grad()) return TensorGrad( ::nt::functional::repeat_(input.detach(), dim, amt),  false );
    TensorGrad result = input.create_view_node(::nt::functional::repeat_(input.detach(), dim, amt));
    result.create_view_backward_function(input,
        [&dim, &amt](const Tensor& grad){
        return ::nt::functional::repeat_(grad, dim, amt); 
    });
    return std::move(result);
}

TensorGrad TensorGrad_Functional_Class::repeat_(const TensorGrad& input, Tensor::size_value_t amt){
    if(!input.track_grad()) return TensorGrad(::nt::functional::repeat_(input.detach(), amt), false);

    TensorGrad result = input.create_view_node(::nt::functional::repeat_(input.detach(), amt));
    result.create_view_backward_function(input,
        [&amt](const Tensor& grad){
        return ::nt::functional::repeat_(grad, amt);
    });
    return std::move(result);
}


TensorGrad TensorGrad_Functional_Class::expand(const TensorGrad& input, SizeRef size){
    if(!input.track_grad()) return TensorGrad(::nt::functional::expand(input.detach(), size), false);

    TensorGrad result = input.create_view_node(::nt::functional::expand(input.detach(), size));
    result.create_view_backward_function(input,
        [&size](const Tensor& grad){
        return ::nt::functional::expand(grad, size);
    });
    return std::move(result);
 
}
TensorGrad TensorGrad_Functional_Class::expand_as(const TensorGrad& a, const TensorGrad& b){
    return expand(a, b.shape());
}
TensorGrad TensorGrad_Functional_Class::expand_as(const TensorGrad& a, const Tensor& b){
    return expand(a, b.shape());
}
Tensor TensorGrad_Functional_Class::expand_as(const Tensor& a, const TensorGrad& b){
    return ::nt::functional::expand(a, b.shape());
}



}
}
