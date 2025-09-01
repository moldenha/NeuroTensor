#include "../../functional/functional.h"
#include "../../Tensor.h"
#include "../../intrusive_ptr/intrusive_ptr.hpp"
#include "../TensorGrad.h"
#include "../functional_class.h"
#include "../../utils/macros.h"

namespace nt{
namespace functional{



TensorGrad TensorGrad_Functional_Class::multiply(const TensorGrad& input, const TensorGrad& other){
    if(!input.track_grad() || !other.track_grad()){
        if(!input.track_grad()) return TensorGrad_Functional_Class::multiply(input.detach(), other); 
        if(!other.track_grad()) return TensorGrad_Functional_Class::multiply(input, other.detach()); 
    }
    TensorGrad result(::nt::functional::multiply(input.detach(), other.detach()), input.track_grad());
    result.track_tensors(input, other);
    result.create_backward_function(
    [](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents,
       intrusive_ptr<tensor_holder> a, intrusive_ptr<tensor_holder> b) {
        parents[0]->accumulate_gradient(::nt::functional::multiply(b->tensor, grad));
        parents[1]->accumulate_gradient(::nt::functional::multiply(a->tensor, grad));
    },
    input, other);
    return std::move(result);
}

TensorGrad TensorGrad_Functional_Class::multiply(const TensorGrad& input, const Tensor& other){
    if(!input.track_grad()){
        TensorGrad result(::nt::functional::multiply(input.detach(), other), input.track_grad()); 
        result.track_grad_(false);
        return std::move(result);
    }
    TensorGrad result(::nt::functional::multiply(input.detach(), other), input.track_grad());
    result.track_tensors(input);
    result.create_backward_function(
    [](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents,
       intrusive_ptr<tensor_holder> b) {
        parents[0]->accumulate_gradient(::nt::functional::multiply(b->tensor, grad));
    },
    other);
    return std::move(result);
}

TensorGrad TensorGrad_Functional_Class::multiply(const Tensor& input, const TensorGrad& other){
    if(!other.track_grad()){
        TensorGrad result(::nt::functional::multiply(input, other.detach()), other.track_grad()); 
        result.track_grad_(false);
        return std::move(result);
    }
    TensorGrad result(::nt::functional::multiply(input, other.detach()), other.track_grad());
    result.track_tensors(other);
    result.create_backward_function(
    [](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents,
       intrusive_ptr<tensor_holder> a) {
        parents[0]->accumulate_gradient(::nt::functional::multiply(a->tensor, grad));
    },
    input);
    return std::move(result);
}

TensorGrad TensorGrad_Functional_Class::multiply(const TensorGrad& input, const Scalar& other){
    if(!input.track_grad()){
        TensorGrad result(::nt::functional::multiply(input.detach(), other), input.track_grad()); 
        result.track_grad_(false);
        return std::move(result);
    }
    TensorGrad result(::nt::functional::multiply(input.detach(), other), input.track_grad());
    result.track_tensors(input);
    result.create_backward_function(
    [other](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents) {
        parents[0]->accumulate_gradient(::nt::functional::multiply(grad, other));

    });
    return std::move(result);
}

TensorGrad TensorGrad_Functional_Class::multiply(const Scalar& other, const TensorGrad& input){
    if(!input.track_grad()){
        TensorGrad result(::nt::functional::multiply(other, input.detach()), input.track_grad()); 
        result.track_grad_(false);
        return std::move(result);
    }
    TensorGrad result(::nt::functional::multiply(other, input.detach()), input.track_grad());
    result.track_tensors(input);
    result.create_backward_function(
    [other](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents) {
        parents[0]->accumulate_gradient(::nt::functional::multiply(other, grad));
    });
    return std::move(result);
}



TensorGrad TensorGrad_Functional_Class::add(const TensorGrad& input, const TensorGrad& other){
    if(!input.track_grad() || !other.track_grad()){
        if(!input.track_grad()) return TensorGrad_Functional_Class::add(input.detach(), other); 
        if(!other.track_grad()) return TensorGrad_Functional_Class::add(input, other.detach()); 
    }
    TensorGrad result(::nt::functional::add(input.detach(), other.detach()), input.track_grad());
    result.track_tensors(input, other);
    result.create_backward_function(
    [](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents) {
        parents[0]->accumulate_gradient(grad);
        parents[1]->accumulate_gradient(grad);
    });
    return std::move(result);
}

TensorGrad TensorGrad_Functional_Class::add(const TensorGrad& input, const Tensor& other){
    if(!input.track_grad()){
        TensorGrad result(::nt::functional::add(input.detach(), other), input.track_grad()); 
        result.track_grad_(false);
        return std::move(result);
    }
    TensorGrad result(::nt::functional::add(input.detach(), other), input.track_grad());
    result.track_tensors(input);
    result.create_backward_function(
    [](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents) {
        parents[0]->accumulate_gradient(grad);
    });
    return std::move(result);
}

TensorGrad TensorGrad_Functional_Class::add(const Tensor& input, const TensorGrad& other){
    if(!other.track_grad()){
        TensorGrad result(::nt::functional::add(input, other.detach()), other.track_grad()); 
        result.track_grad_(false);
        return std::move(result);
    }
    TensorGrad result(::nt::functional::add(input, other.detach()), other.track_grad());
    result.track_tensors(other);
    result.create_backward_function(
    [](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents) {
        parents[0]->accumulate_gradient(grad);
    });
    return std::move(result);
}


TensorGrad TensorGrad_Functional_Class::add(const TensorGrad& input, const Scalar& other){
    if(!input.track_grad()){
        TensorGrad result(::nt::functional::add(input.detach(), other), input.track_grad()); 
        result.track_grad_(false);
        return std::move(result);
    }
    TensorGrad result(::nt::functional::add(input.detach(), other), input.track_grad());
    result.track_tensors(input);
    result.create_backward_function(
    [](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents) {
        parents[0]->accumulate_gradient(grad);
    });
    return std::move(result);
}

TensorGrad TensorGrad_Functional_Class::add(const Scalar& other, const TensorGrad& input){
    if(!input.track_grad()){
        TensorGrad result(::nt::functional::add(other, input.detach()), input.track_grad()); 
        result.track_grad_(false);
        return std::move(result);
    }
    TensorGrad result(::nt::functional::add(other, input.detach()), input.track_grad());
    result.track_tensors(input);
    result.create_backward_function(
    [](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents) {
        parents[0]->accumulate_gradient(grad);
    });
    return std::move(result);
}


NT_ALWAYS_INLINE Tensor div_grad_a(const Tensor& grad, const intrusive_ptr<tensor_holder>& b){return ::nt::functional::divide(grad, b->tensor);}
NT_ALWAYS_INLINE Tensor div_grad_a(const Tensor& grad, const Scalar& b){return ::nt::functional::divide(grad, b);}
NT_ALWAYS_INLINE Tensor div_grad_b(const Tensor& grad, const intrusive_ptr<tensor_holder>& a, const intrusive_ptr<tensor_holder>& b){
    Tensor pa = ::nt::functional::multiply(a->tensor, grad);
    Tensor pb = ::nt::functional::pow(b->tensor, 2);
    return ::nt::functional::divide(pa, pb);
}

NT_ALWAYS_INLINE Tensor div_grad_b(const Tensor& grad, const Scalar& a, const intrusive_ptr<tensor_holder>& b){
    Tensor pa = ::nt::functional::multiply(a, grad);
    Tensor pb = ::nt::functional::pow(b->tensor, 2);
    return ::nt::functional::divide(pa, pb);
}


TensorGrad TensorGrad_Functional_Class::divide(const TensorGrad& input, const TensorGrad& other){
    if(!input.track_grad() || !other.track_grad()){
        if(!input.track_grad()) return TensorGrad_Functional_Class::divide(input.detach(), other); 
        if(!other.track_grad()) return TensorGrad_Functional_Class::divide(input, other.detach()); 
    }
    TensorGrad result(::nt::functional::divide(input.detach(), other.detach()), input.track_grad());
    result.track_tensors(input, other);
    result.create_backward_function(
    [](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents,
       intrusive_ptr<tensor_holder> a, intrusive_ptr<tensor_holder> b) {
        parents[0]->accumulate_gradient(div_grad_a(grad, b));
        parents[1]->accumulate_gradient(div_grad_b(grad, a, b));
    },
    input, other);
    return std::move(result);
}

TensorGrad TensorGrad_Functional_Class::divide(const TensorGrad& input, const Tensor& other){
    if(!input.track_grad()){
        TensorGrad result(::nt::functional::divide(input.detach(), other), input.track_grad()); 
        result.track_grad_(false);
        return std::move(result);
    }
    TensorGrad result(::nt::functional::divide(input.detach(), other), input.track_grad());
    result.track_tensors(input);
    result.create_backward_function(
    [](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents,
       intrusive_ptr<tensor_holder> b) {
        parents[0]->accumulate_gradient(div_grad_a(grad, b));
    },
    other);
    return std::move(result);
}

TensorGrad TensorGrad_Functional_Class::divide(const Tensor& input, const TensorGrad& other){
    if(!other.track_grad()){
        TensorGrad result(::nt::functional::divide(input, other.detach()), other.track_grad()); 
        result.track_grad_(false);
        return std::move(result);
    }
    TensorGrad result(::nt::functional::divide(input, other.detach()), other.track_grad());
    result.track_tensors(other);
    result.create_backward_function(
    [](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents,
       intrusive_ptr<tensor_holder> a, intrusive_ptr<tensor_holder> b) {
        parents[0]->accumulate_gradient(div_grad_b(grad, a, b));
    },
    input, other);
    return std::move(result);
}


TensorGrad TensorGrad_Functional_Class::divide(const TensorGrad& input, const Scalar& other){
    if(!input.track_grad()){
        TensorGrad result(::nt::functional::divide(input.detach(), other), input.track_grad()); 
        result.track_grad_(false);
        return std::move(result);
    }
    TensorGrad result(::nt::functional::divide(input.detach(), other), input.track_grad());
    result.track_tensors(input);
    result.create_backward_function(
    [other](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents) {
        parents[0]->accumulate_gradient(div_grad_a(grad, other));
    });
    return std::move(result);
}

TensorGrad TensorGrad_Functional_Class::divide(const Scalar& other, const TensorGrad& input){
    if(!input.track_grad()){
        TensorGrad result(::nt::functional::divide(other, input.detach()), input.track_grad()); 
        result.track_grad_(false);
        return std::move(result);
    }
    TensorGrad result(::nt::functional::divide(other, input.detach()), input.track_grad());
    result.track_tensors(input);
    result.create_backward_function(
    [other](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents,
       intrusive_ptr<tensor_holder> b) {
        parents[0]->accumulate_gradient(div_grad_b(grad, other, b));
    }, input);
    return std::move(result);
}


TensorGrad TensorGrad_Functional_Class::subtract(const TensorGrad& input, const TensorGrad& other){
    if(!input.track_grad() || !other.track_grad()){
        if(!input.track_grad()) return TensorGrad_Functional_Class::subtract(input.detach(), other); 
        if(!other.track_grad()) return TensorGrad_Functional_Class::subtract(input, other.detach()); 
    }
    TensorGrad result(::nt::functional::subtract(input.detach(), other.detach()), input.track_grad());
    result.track_tensors(input, other);
    result.create_backward_function(
    [](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents) {
        parents[0]->accumulate_gradient(grad);
        parents[1]->accumulate_gradient(-grad);
        // parents[1]->accumulate_gradient(-grad);
    });
    return std::move(result);
}

TensorGrad TensorGrad_Functional_Class::subtract(const TensorGrad& input, const Tensor& other){
    if(!input.track_grad()){
        TensorGrad result(::nt::functional::subtract(input.detach(), other), input.track_grad()); 
        result.track_grad_(false);
        return std::move(result);
    }
    TensorGrad result(::nt::functional::subtract(input.detach(), other), input.track_grad());
    result.track_tensors(input);
    result.create_backward_function(
    [](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents) {
        parents[0]->accumulate_gradient(grad);
    });
    return std::move(result);
}

TensorGrad TensorGrad_Functional_Class::subtract(const Tensor& input, const TensorGrad& other){
    if(!other.track_grad()){
        TensorGrad result(::nt::functional::subtract(input, other.detach()), other.track_grad()); 
        result.track_grad_(false);
        return std::move(result);
    }
    TensorGrad result(::nt::functional::subtract(input, other.detach()), other.track_grad());
    result.track_tensors(other);
    result.create_backward_function(
    [](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents) {
        parents[0]->accumulate_gradient(-grad);
        // parents[0]->accumulate_gradient(-grad);
    });
    return std::move(result);
}


TensorGrad TensorGrad_Functional_Class::subtract(const TensorGrad& input, const Scalar& other){
    if(!input.track_grad()){
        TensorGrad result(::nt::functional::subtract(input.detach(), other), input.track_grad()); 
        result.track_grad_(false);
        return std::move(result);
    }
    TensorGrad result(::nt::functional::subtract(input.detach(), other), input.track_grad());
    result.track_tensors(input);
    result.create_backward_function(
    [](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents) {
        parents[0]->accumulate_gradient(grad);
    });
    return std::move(result);
}

TensorGrad TensorGrad_Functional_Class::subtract(const Scalar& other, const TensorGrad& input){
    if(!input.track_grad()){
        TensorGrad result(::nt::functional::subtract(other, input.detach()), input.track_grad()); 
        result.track_grad_(false);
        return std::move(result);
    }
    TensorGrad result(::nt::functional::subtract(other, input.detach()), input.track_grad());
    result.track_tensors(input);
    result.create_backward_function(
    [](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents) {
        parents[0]->accumulate_gradient(-grad);
        // parents[0]->accumulate_gradient(-grad);
    });
    return std::move(result);
}



TensorGrad& TensorGrad_Functional_Class::multiply_(TensorGrad& input, const TensorGrad& other){
    if(!input.track_grad() || !other.track_grad()){
        if(!input.track_grad()){
            ::nt::functional::multiply_(input.detach(), other.detach());
            return input;
        }
        if(!other.track_grad()) return TensorGrad_Functional_Class::multiply_(input, other.detach()); 
    }
    intrusive_ptr<tensor_holder> this_clone =
        make_intrusive<tensor_holder>(input.detach().conditional_mutate_clone());
    intrusive_ptr<tensor_holder> other_clone =
        make_intrusive<tensor_holder>(other.detach().conditional_mutate_clone());
    ::nt::functional::multiply_(input.detach(), other.detach());
    input.track_self_mod_tensors(
    [a = std::move(this_clone), b = std::move(other_clone)](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents) {
        parents[0]->accumulate_gradient(::nt::functional::multiply(b->tensor, grad));
        parents[1]->accumulate_gradient(::nt::functional::multiply(a->tensor, grad));
    },
    "Multiply_", other);
    return input;
}


TensorGrad& TensorGrad_Functional_Class::multiply_(TensorGrad& input, const Tensor& other){
    if(!input.track_grad() ){
        ::nt::functional::multiply_(input.detach(), other);
        return input;
    }
    intrusive_ptr<tensor_holder> other_clone =
        make_intrusive<tensor_holder>(other.conditional_mutate_clone());
    ::nt::functional::multiply_(input.detach(), other);
    input.track_self_mod_tensors(
    [b = std::move(other_clone)](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents) {
        parents[0]->accumulate_gradient(::nt::functional::multiply(b->tensor, grad));
    },
    "Multiply_");
    return input;
}


Tensor& TensorGrad_Functional_Class::multiply_(Tensor& input, const TensorGrad& other){
    return ::nt::functional::multiply_(input, other.detach());
}


TensorGrad& TensorGrad_Functional_Class::multiply_(TensorGrad& input, const Scalar& other){
    if(!input.track_grad()){
        ::nt::functional::multiply_(input.detach(), other);
        return input;
    }
    ::nt::functional::multiply_(input.detach(), other);
    input.track_self_mod_tensors(
    [other](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents) {
        parents[0]->accumulate_gradient(::nt::functional::multiply(other, grad));
    }, "Multiply_");
    return input;
}

TensorGrad& TensorGrad_Functional_Class::add_(TensorGrad& input, const TensorGrad& other){
    if(!input.track_grad() || !other.track_grad()){
        if(!input.track_grad()){
            ::nt::functional::add_(input.detach(), other.detach());
            return input;
        }
        if(!other.track_grad()) return TensorGrad_Functional_Class::add_(input, other.detach()); 
    }
    ::nt::functional::add_(input.detach(), other.detach());
    input.track_self_mod_tensors(
    [](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents) {
        parents[0]->accumulate_gradient(grad);
        parents[1]->accumulate_gradient(grad);
    }, "Add_", other);
    return input;
}


TensorGrad& TensorGrad_Functional_Class::add_(TensorGrad& input, const Tensor& other){
    if(!input.track_grad() ){
        ::nt::functional::add_(input.detach(), other);
        return input;
    }
    ::nt::functional::add_(input.detach(), other);
    input.track_self_mod_tensors(
    [](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents) {
        parents[0]->accumulate_gradient(grad);
    }, "Add_");
    return input;
}


Tensor& TensorGrad_Functional_Class::add_(Tensor& input, const TensorGrad& other){
    return ::nt::functional::add_(input, other.detach());
}


TensorGrad& TensorGrad_Functional_Class::add_(TensorGrad& input, const Scalar& other){
    if(!input.track_grad()){
        ::nt::functional::add_(input.detach(), other);
        return input;
    }
    ::nt::functional::add_(input.detach(), other);
    input.track_self_mod_tensors(
    [other](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents) {
        parents[0]->accumulate_gradient(grad);
    }, "Add_");
    return input;
}



TensorGrad& TensorGrad_Functional_Class::divide_(TensorGrad& input, const TensorGrad& other){
    if(!input.track_grad() || !other.track_grad()){
        if(!input.track_grad()){
            ::nt::functional::divide_(input.detach(), other.detach());
            return input;
        }
        if(!other.track_grad()) return TensorGrad_Functional_Class::divide_(input, other.detach()); 
    }
    intrusive_ptr<tensor_holder> this_clone =
        make_intrusive<tensor_holder>(input.detach().conditional_mutate_clone());
    intrusive_ptr<tensor_holder> other_clone =
        make_intrusive<tensor_holder>(other.detach().conditional_mutate_clone());
    ::nt::functional::divide_(input.detach(), other.detach());
    input.track_self_mod_tensors(
    [a = std::move(this_clone), b = std::move(other_clone)](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents) {
        parents[0]->accumulate_gradient(div_grad_a(grad, b));
        parents[1]->accumulate_gradient(div_grad_b(grad, a, b));
    }, "Divide_", other);
    return input;
}


TensorGrad& TensorGrad_Functional_Class::divide_(TensorGrad& input, const Tensor& other){
    if(!input.track_grad() ){
        ::nt::functional::divide_(input.detach(), other);
        return input;
    }
    intrusive_ptr<tensor_holder> other_clone =
        make_intrusive<tensor_holder>(other.conditional_mutate_clone());
    ::nt::functional::divide_(input.detach(), other);
    input.track_self_mod_tensors(
    [b = std::move(other_clone)](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents) {
        parents[0]->accumulate_gradient(div_grad_a(grad, b));
    },
    __func__);
    return input;
}


Tensor& TensorGrad_Functional_Class::divide_(Tensor& input, const TensorGrad& other){
    return ::nt::functional::divide_(input, other.detach());
}


TensorGrad& TensorGrad_Functional_Class::divide_(TensorGrad& input, const Scalar& other){
    if(!input.track_grad()){
        ::nt::functional::divide_(input.detach(), other);
        return input;
    }
    ::nt::functional::divide_(input.detach(), other);
    input.track_self_mod_tensors(
    [other](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents) {
        parents[0]->accumulate_gradient(div_grad_a(grad, other));
    }, __func__);
    return input;
}

TensorGrad& TensorGrad_Functional_Class::subtract_(TensorGrad& input, const TensorGrad& other){
    if(!input.track_grad() || !other.track_grad()){
        if(!input.track_grad()){
            ::nt::functional::subtract_(input.detach(), other.detach());
            return input;
        }
        if(!other.track_grad()) return TensorGrad_Functional_Class::subtract_(input, other.detach()); 
    }
    ::nt::functional::subtract_(input.detach(), other.detach());
    input.track_self_mod_tensors(
    [](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents) {
        parents[0]->accumulate_gradient(grad);
        parents[1]->accumulate_gradient(-grad);
    }, __func__, other);
    return input;
}




TensorGrad& TensorGrad_Functional_Class::subtract_(TensorGrad& input, const Tensor& other){
    if(!input.track_grad() ){
        ::nt::functional::subtract_(input.detach(), other);
        return input;
    }
    ::nt::functional::subtract_(input.detach(), other);
    input.track_self_mod_tensors(
    [](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents) {
        parents[0]->accumulate_gradient(grad);
    }, __func__);
    return input;
}


Tensor& TensorGrad_Functional_Class::subtract_(Tensor& input, const TensorGrad& other){
    return ::nt::functional::subtract_(input, other.detach());
}


TensorGrad& TensorGrad_Functional_Class::subtract_(TensorGrad& input, const Scalar& other){
    if(!input.track_grad()){
        ::nt::functional::subtract_(input.detach(), other);
        return input;
    }
    ::nt::functional::subtract_(input.detach(), other);
    input.track_self_mod_tensors(
    [other](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents) {
        parents[0]->accumulate_gradient(grad);
    }, __func__);
    return input;
}


TensorGrad TensorGrad_Functional_Class::fmod(const TensorGrad& input, const TensorGrad& other){
    if(!input.track_grad()){
        return TensorGrad_Functional_Class::fmod(input.detach(), other);
    }
    if(!other.track_grad()){
        return TensorGrad_Functional_Class::fmod(input, other.detach());
    }
    TensorGrad result(::nt::functional::fmod(input.detach(), other.detach()), input.track_grad());
    result.track_tensors(input, other);
    result.create_backward_function(
    [](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents,
       intrusive_ptr<tensor_holder> a, intrusive_ptr<tensor_holder> b) {
        parents[0]->accumulate_gradient(grad);
        parents[1]->accumulate_gradient(::nt::functional::fmod_b_backward(a->tensor, b->tensor, grad));
    }, input, other);
    return std::move(result);
 
}
TensorGrad TensorGrad_Functional_Class::fmod(const TensorGrad& input, const Tensor& other){
    if(!input.track_grad()){
        TensorGrad result(::nt::functional::fmod(input.detach(), other), input.track_grad());
        result.track_grad_(false);
        return std::move(result);
    }
    TensorGrad result(::nt::functional::fmod(input.detach(), other), input.track_grad());
    result.track_tensors(input);
    result.create_backward_function(
    [](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents) {
        parents[0]->accumulate_gradient(grad);
    });
    return std::move(result);

}
TensorGrad TensorGrad_Functional_Class::fmod(const Tensor& input, const TensorGrad& other){
    if(!other.track_grad()){
        TensorGrad result(::nt::functional::fmod(input, other.detach()), other.track_grad());
        result.track_grad_(false);
        return std::move(result);
    }
    TensorGrad result(::nt::functional::fmod(input, other.detach()), other.track_grad());
    result.track_tensors(other);
    result.create_backward_function(
    [](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents,
       intrusive_ptr<tensor_holder> a, intrusive_ptr<tensor_holder> b) {
        parents[0]->accumulate_gradient(::nt::functional::fmod_b_backward(a->tensor, b->tensor, grad));
    }, input, other);
    return std::move(result);

}
TensorGrad TensorGrad_Functional_Class::fmod(const TensorGrad& input, const Scalar& other){
    if(!input.track_grad()){
        TensorGrad result(::nt::functional::fmod(input.detach(), other), input.track_grad());
        result.track_grad_(false);
        return std::move(result);
    }
    TensorGrad result(::nt::functional::fmod(input.detach(), other), input.track_grad());
    result.track_tensors(input);
    result.create_backward_function(
    [](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents) {
        parents[0]->accumulate_gradient(grad);
    });
    return std::move(result);

}
TensorGrad TensorGrad_Functional_Class::fmod(const Scalar& input, const TensorGrad& other){
    if(!other.track_grad()){
        TensorGrad result(::nt::functional::fmod(input, other.detach()), other.track_grad());
        result.track_grad_(false);
        return std::move(result);
    }
    TensorGrad result(::nt::functional::fmod(input, other.detach()), other.track_grad());
    result.track_tensors(other);
    result.create_backward_function(
    [input](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents,
       intrusive_ptr<tensor_holder> b) {
        parents[0]->accumulate_gradient(::nt::functional::fmod_b_backward(input, b->tensor, grad));
    }, other);
    return std::move(result);

}





TensorGrad TensorGrad_Functional_Class::remainder(const TensorGrad& input, const TensorGrad& other){
    if(!input.track_grad()){
        return TensorGrad_Functional_Class::remainder(input.detach(), other);
    }
    if(!other.track_grad()){
        return TensorGrad_Functional_Class::remainder(input, other.detach());
    }
    TensorGrad result(::nt::functional::remainder(input.detach(), other.detach()), input.track_grad());
    result.track_tensors(input, other);
    result.create_backward_function(
    [](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents,
       intrusive_ptr<tensor_holder> a, intrusive_ptr<tensor_holder> b) {
        parents[0]->accumulate_gradient(grad);
        parents[1]->accumulate_gradient(::nt::functional::remainder_b_backward(a->tensor, b->tensor, grad));
    }, input, other);
    return std::move(result);
 
}
TensorGrad TensorGrad_Functional_Class::remainder(const TensorGrad& input, const Tensor& other){
    if(!input.track_grad()){
        TensorGrad result(::nt::functional::remainder(input.detach(), other), input.track_grad());
        result.track_grad_(false);
        return std::move(result);
    }
    TensorGrad result(::nt::functional::remainder(input.detach(), other), input.track_grad());
    result.track_tensors(input);
    result.create_backward_function(
    [](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents) {
        parents[0]->accumulate_gradient(grad);
    });
    return std::move(result);

}
TensorGrad TensorGrad_Functional_Class::remainder(const Tensor& input, const TensorGrad& other){
    if(!other.track_grad()){
        TensorGrad result(::nt::functional::remainder(input, other.detach()), other.track_grad());
        result.track_grad_(false);
        return std::move(result);
    }
    TensorGrad result(::nt::functional::remainder(input, other.detach()), other.track_grad());
    result.track_tensors(other);
    result.create_backward_function(
    [](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents,
       intrusive_ptr<tensor_holder> a, intrusive_ptr<tensor_holder> b) {
        parents[0]->accumulate_gradient(::nt::functional::remainder_b_backward(a->tensor, b->tensor, grad));
    }, input, other);
    return std::move(result);

}
TensorGrad TensorGrad_Functional_Class::remainder(const TensorGrad& input, const Scalar& other){
    if(!input.track_grad()){
        TensorGrad result(::nt::functional::remainder(input.detach(), other), input.track_grad());
        result.track_grad_(false);
        return std::move(result);
    }
    TensorGrad result(::nt::functional::remainder(input.detach(), other), input.track_grad());
    result.track_tensors(input);
    result.create_backward_function(
    [](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents) {
        parents[0]->accumulate_gradient(grad);
    });
    return std::move(result);

}
TensorGrad TensorGrad_Functional_Class::remainder(const Scalar& input, const TensorGrad& other){
    if(!other.track_grad()){
        TensorGrad result(::nt::functional::remainder(input, other.detach()), other.track_grad());
        result.track_grad_(false);
        return std::move(result);
    }
    TensorGrad result(::nt::functional::remainder(input, other.detach()), other.track_grad());
    result.track_tensors(other);
    result.create_backward_function(
    [input](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents,
       intrusive_ptr<tensor_holder> b) {
        parents[0]->accumulate_gradient(::nt::functional::remainder_b_backward(input, b->tensor, grad));
    }, other);
    return std::move(result);

}


TensorGrad TensorGrad_Functional_Class::inverse(const TensorGrad& input){
    if(!input.track_grad()){
        TensorGrad result(::nt::functional::inverse(input.detach()), input.track_grad());
        result.track_grad_(false);
        return std::move(result);
    }
    TensorGrad result(::nt::functional::inverse(input.detach()), input.track_grad());

    result.create_backward_function(
        [](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents,
           intrusive_ptr<tensor_holder> a) {
            parents[0]->accumulate_gradient(-grad / a->tensor);
        },
        make_intrusive<tensor_holder>(::nt::functional::pow(input.detach(), 2)));

    return std::move(result);
}

TensorGrad& TensorGrad_Functional_Class::inverse_(TensorGrad& input){
    if(!input.track_grad()){
        ::nt::functional::inverse_(input.detach());
        return input;
    }
    intrusive_ptr<tensor_holder> this_clone =
        make_intrusive<tensor_holder>(::nt::functional::pow(input.detach(), 2));
    ::nt::functional::inverse_(input.detach());
    input.track_self_mod_tensors(
        [a = std::move(this_clone)](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents) {
            parents[0]->accumulate_gradient(-grad / a->tensor);
        }, __func__);

    return input;
}

}
}
