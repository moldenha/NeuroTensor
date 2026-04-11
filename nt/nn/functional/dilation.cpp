#include "../../functional/functional.h"
#include "../../Tensor.h"
#include "../../intrusive_ptr/intrusive_ptr.hpp"
#include "../TensorGrad.h"
#include "../functional_class.h"
#include "../../utils/macros.h"

namespace nt{
namespace functional{






TensorGrad TensorGrad_Functional_Class::dilate(const TensorGrad& x, std::vector<Tensor::size_value_t> dils, bool test){
    if(!x.track_grad()) return TensorGrad(::nt::functional::dilate(x.detach(), dils, test), false);

    
    TensorGrad result = TensorGrad::create_read_node(::nt::functional::dilate(x.detach(), dils, test), x);

    result.create_read_backward_function(
        [test, dils = std::move(dils)](const Tensor &grad,
              std::vector<intrusive_ptr<TensorGrad>> &parents) {
            parents[0]->accumulate_gradient( ::nt::functional::undilate_(grad, std::move(dils), test) );
        });
    return std::move(result);
}



TensorGrad TensorGrad_Functional_Class::undilate(const TensorGrad& x, std::vector<Tensor::size_value_t> dils){
    if(!x.track_grad()) return TensorGrad(::nt::functional::undilate(x.detach(), dils), false);
    
    TensorGrad result = TensorGrad::create_read_node(::nt::functional::undilate(x.detach(), dils), x);

    result.create_read_backward_function(
        [dils = std::move(dils)](const Tensor &grad,
              std::vector<intrusive_ptr<TensorGrad>> &parents) {
            parents[0]->accumulate_gradient( ::nt::functional::dilate(grad, std::move(dils)) );
        });
    return std::move(result);
}

TensorGrad TensorGrad_Functional_Class::undilate_(const TensorGrad& x, std::vector<Tensor::size_value_t> dils, bool test){
    if(!x.track_grad()) return TensorGrad(::nt::functional::undilate_(x.detach(), dils, test), false);
    TensorGrad result = x.create_view_node(::nt::functional::undilate_(x.detach(), dils, test));
    result.create_view_backward_function(x, [&dils, &test](Tensor& grad){
        return ::nt::functional::undilate_(grad, dils, test);
    });
    return std::move(result);
}

}
}

