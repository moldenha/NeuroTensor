#include "../../functional/functional.h"
#include "../../Tensor.h"
#include "../../intrusive_ptr/intrusive_ptr.hpp"
#include "../TensorGrad.h"
#include "../functional_class.h"
#include "../../utils/macros.h"

namespace nt{
namespace functional{

TensorGrad TensorGrad_Functional_Class::zeros_like(const TensorGrad& tg){
    TensorGrad result(::nt::functional::zeros_like(tg.detach()), tg.track_grad());
    return std::move(result);
}

TensorGrad TensorGrad_Functional_Class::ones_like(const TensorGrad& tg){
    TensorGrad result(::nt::functional::ones_like(tg.detach()), tg.track_grad());
    return std::move(result);
}

TensorGrad TensorGrad_Functional_Class::nums_like(const TensorGrad& tg, Scalar s){
    TensorGrad result(::nt::functional::nums_like(tg.detach(), s), tg.track_grad());
    return std::move(result);
}

TensorGrad& TensorGrad_Functional_Class::fill_diagonal_(TensorGrad& tg, Scalar s){
    ::nt::functional::fill_diagonal_(tg.detach(), s);
    tg.track_self_mod_tensors(
        [](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents) {
            Tensor grad_ = grad.clone();
            ::nt::functional::fill_diagonal_(grad_, 0);
            parents[0]->accumulate_gradient(grad_);
    }, "fill_diagonal_");
    return tg;
}

TensorGrad& TensorGrad_Functional_Class::fill_(TensorGrad& tg, Scalar s){
    ::nt::functional::fill_(tg.detach(), s);
    std::cout << "fill forward called" << std::endl;
    tg.track_self_mod_tensors(
        [](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents) {
        std::cout << "fill backward called" << std::endl;
        std::cout << "fill backward with grad shape" << grad.shape() << std::endl;
        parents[0]->accumulate_gradient(0);
        // gradient not propogated
        // ::nt::functional::fill_(parents[0]->grad->tensor, 0);
    }, "fill_");
    return tg;
}

TensorGrad& TensorGrad_Functional_Class::set_(TensorGrad& tg, const Tensor& t){
    ::nt::functional::set_(tg.detach(), t);
    tg.track_self_mod_tensors(
        [](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents) {
        parents[0]->accumulate_gradient(0);
        // gradient not propogated
        // ::nt::functional::fill_(parents[0]->grad->tensor, 0);
    }, "set_");
    return tg;
}

TensorGrad& TensorGrad_Functional_Class::set_(TensorGrad& tg, const TensorGrad& t){
    ::nt::functional::set_(tg.detach(), t.detach());
    tg.track_self_mod_tensors(
        [](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents) {
        parents[0]->accumulate_gradient(0);
        // gradient not propogated
        // ::nt::functional::fill_(parents[0]->grad->tensor, 0);
    }, "set_");
    return tg;
}


}
}
