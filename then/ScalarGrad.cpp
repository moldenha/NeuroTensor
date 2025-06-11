#include "ScalarGrad.h"
#include "../dtype/DType.h"
#include "../functional/functional.h"

namespace nt{

ScalarGrad::ScalarGrad(Scalar _item, Tensor _grad, intrusive_ptr<TensorGrad> _parent)
    :item(_item), parent(_parent), grad(_grad)
    {}

ScalarGrad::ScalarGrad(Scalar _item, Tensor _grad, TensorGrad _parent)
    :item(_item), parent(make_intrusive<TensorGrad>(_parent)), grad(_grad)
    {}

ScalarGrad ScalarGrad::to(DType dt){
    if(DTypeFuncs::is_complex(dt)){
        return ScalarGrad(item.toComplex(), grad.to(dt), parent);
    }else if(DTypeFuncs::is_integer(dt)){
        return ScalarGrad(item.toIntegral(), grad.to(dt), parent);
    }else if (DTypeFuncs::is_floating(dt)){
        return ScalarGrad(item.toFloatingPoint(), grad.to(dt), parent);
    }else{
        return ScalarGrad(item, grad.to(dt), parent);
    }
}

void ScalarGrad::backward(){
    if(this->parent == nullptr){return;}
    utils::throw_exception(!this->grad.is_null(),
                           "Cannod run backward with a null gradient");
    if(this->isNan()){
        utils::throw_exception(!functional::any(this->grad == ::nt::nan), "Cannot run backward with gradient containing NaN values");
    }
    /* RUN BACKWARD FUNCTION */
    if(!this->parent->grad){
        this->parent->grad = nt::make_intrusive<tensor_holder>(
            nt::functional::zeros_like(this->parent->tensor)
        );
    }
    this->parent->grad->tensor = grad;
    /* RAN BACKWARD FUNCTION */

    /* RUN PARENT BACKWARD */
    if(this->parent->backwardFunc == nullptr){
        this->parent = nullptr;
        return;
    }
    if(this->parent->backwardFunc->used()){
        this->parent = nullptr;
        return;
    }
    this->parent->run_backward(weak_intrusive_ptr<TensorGrad>(parent));

    this->parent = nullptr;
    return;
}

std::ostream& operator<<(std::ostream& os, const ScalarGrad& s){return os << s.item;}


}
