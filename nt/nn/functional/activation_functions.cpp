#include "../../functional/functional.h"
#include "../../Tensor.h"
#include "../../intrusive_ptr/intrusive_ptr.hpp"
#include "../TensorGrad.h"
#include "../functional_class.h"
#include "../../utils/macros.h"

namespace nt{
namespace functional{

//this is for simple non-linear functions where they only take a single argument and return a single argument
//just an easy way to put it all into one cpp file

#define ADD_UNDERSCORE(name) name##_
#define ADD_DOUBLE_UNDERSCORE(name) _##name##_



#define NT_MAKE_NONLINEAR_DEFINED_FUNCTION_(func_name)\
TensorGrad TensorGrad_Functional_Class::func_name(const TensorGrad &x) { \
    TensorGrad result(::nt::functional::func_name(x.detach()), x.track_grad()); \
    if (!x.track_grad()) { \
        result.track_grad_(false); \
        return std::move(result); \
    } \
    result.track_tensors(x); \
    result.create_backward_function( \
            [](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents, \
                                                    intrusive_ptr<tensor_holder> saved_x) { \
        parents[0]->accumulate_gradient(grad * _NT_GLUE_(::nt::functional::d, func_name)(saved_x->tensor)); \
    }, \
    make_intrusive<tensor_holder>(x.detach().conditional_mutate_clone())); \
    return std::move(result); \
} \
\
TensorGrad& TensorGrad_Functional_Class::ADD_UNDERSCORE(func_name)(TensorGrad &x){\
    if(!x.track_grad()){\
        ::nt::functional::ADD_UNDERSCORE(func_name)(x.detach());\
        return x;\
    }\
    intrusive_ptr<tensor_holder> this_clone = \
        make_intrusive<tensor_holder>(x.detach().conditional_mutate_clone()); \
    ::nt::functional::ADD_UNDERSCORE(func_name)(x.detach());\
    x.track_self_mod_tensors( \
        [this_clone](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>> &parents) { \
        parents[0]->accumulate_gradient(grad * _NT_GLUE_(::nt::functional::d, func_name)(this_clone->tensor)); \
    }, #func_name \
        "_"); \
    return x;\
}\


NT_MAKE_NONLINEAR_DEFINED_FUNCTION_(sqrt);
NT_MAKE_NONLINEAR_DEFINED_FUNCTION_(invsqrt);
// NT_MAKE_NONLINEAR_DEFINED_FUNCTION_(log);
NT_MAKE_NONLINEAR_DEFINED_FUNCTION_(silu);
NT_MAKE_NONLINEAR_DEFINED_FUNCTION_(gelu);
// NT_MAKE_NONLINEAR_DEFINED_FUNCTION_(sigmoid);

#undef NT_MAKE_NONLINEAR_DEFINED_FUNCTION_ 
#undef ADD_UNDERSCORE 
#undef ADD_DOUBLE_UNDERSCORE


TensorGrad TensorGrad_Functional_Class::abs(const TensorGrad &x){
    Tensor a = ::nt::functional::abs(x.detach());
    TensorGrad result(std::move(a), x.track_grad());
    if (!x.track_grad()) {
        result.track_grad_(false);
        return std::move(result);
    }


    intrusive_ptr<tensor_holder> saved_x =
            make_intrusive<tensor_holder>(x.detach().conditional_mutate_clone());
    
    result.track_tensors(x);


    result.create_backward_function(
            [](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents,
               intrusive_ptr<tensor_holder> saved_x) {
                //compute the gradient using the saved input tensor
                Tensor sign_grad = ((saved_x->tensor > 0).to(DType::Float32) -
                                   (saved_x->tensor < 0).to(DType::Float32)).to(grad.dtype()); //compute sign
                // Tensor sign_grad = ((saved_x->tensor < 0).to(DType::Float32) -
                //                    (saved_x->tensor > 0).to(DType::Float32))
                //                     .to(grad.dtype()); //compute sign
                parents[0]->accumulate_gradient( grad * sign_grad );
            },
            saved_x);

    return std::move(result);
}

TensorGrad& TensorGrad_Functional_Class::abs_(TensorGrad &x){
    if(!x.track_grad()){
        ::nt::functional::abs_(x.detach());
        return x;
    }

    intrusive_ptr<tensor_holder> saved_x =
            make_intrusive<tensor_holder>(x.detach().conditional_mutate_clone());
    
    ::nt::functional::abs_(x.detach());
    x.track_self_mod_tensors(
            [saved_x](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents) {
                //compute the gradient using the saved input tensor
                // Tensor sign_grad = (saved_x->tensor > 0).to(DType::Float32) -
                //                    (saved_x->tensor < 0).to(DType::Float32); //compute sign
                Tensor sign_grad = ((saved_x->tensor < 0).to(DType::Float32) -
                                   (saved_x->tensor > 0).to(DType::Float32))
                                    .to(grad.dtype()); //compute sign
                parents[0]->accumulate_gradient( grad * sign_grad );
            },"Abs_");

    return x;
}




TensorGrad TensorGrad_Functional_Class::softplus(const TensorGrad &x,
                                                     Scalar beta,
                                                     Scalar threshold) {
    Tensor softplus_x = x.detach() * beta;

    Tensor where = softplus_x < threshold;
    if (!::nt::functional::any(where)) {
        return x;
    }
    bool all_below = ::nt::functional::all(where);
    if(all_below){
        softplus_x.set_(::nt::functional::log(1 + std::exp(softplus_x)).divide_(beta));
    }else{
        softplus_x[where].set_(::nt::functional::log(1 + std::exp(softplus_x[where])).divide_(beta));
    }
    TensorGrad result(std::move(softplus_x), x.track_grad());
    if (!x.track_grad()) {
        result.track_grad_(false);
        return std::move(result);
    }
    intrusive_ptr<tensor_holder> sx_c = make_intrusive<tensor_holder>(x.detach().conditional_mutate_clone());
    intrusive_ptr<tensor_holder> wx_c = make_intrusive<tensor_holder>(where);
    if(all_below) wx_c.reset();
    result.track_tensors(x);
    result.create_backward_function(
            [beta, all_below](
                    const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents,
                    intrusive_ptr<tensor_holder> x, intrusive_ptr<tensor_holder> where) {
                if(all_below){
                    Tensor x_w = x->tensor * beta;
                    Tensor grad_w = grad * ::nt::functional::sigmoid(x_w);
                    parents[0]->accumulate_gradient( grad_w );
                }else{
                    Tensor x_w = x->tensor[where->tensor] * beta;
                    Tensor grad_w = grad.clone();
                    grad_w[where->tensor] *= ::nt::functional::sigmoid(x_w);
                    parents[0]->accumulate_gradient( grad_w );
                }
            },
            sx_c, wx_c);
    return std::move(result);
}

TensorGrad& TensorGrad_Functional_Class::softplus_(TensorGrad &x,
                                                     Scalar beta,
                                                     Scalar threshold) {
    Tensor softplus_x = x.detach() * beta;

    Tensor where = softplus_x < threshold;
    if (!::nt::functional::any(where)) {
        return x;
    }
    bool all_below = ::nt::functional::all(where);

    intrusive_ptr<tensor_holder> sx_c = make_intrusive<tensor_holder>(x.detach().conditional_mutate_clone());
    intrusive_ptr<tensor_holder> wx_c = make_intrusive<tensor_holder>(where);
    if(all_below) wx_c.reset();


    if(all_below){
        x.detach().set_(::nt::functional::log(1 + std::exp(x.detach())).divide_(beta));
    }else{
        x.detach()[where].set_(::nt::functional::log(1 + std::exp(x.detach()[where])).divide_(beta));
    }
    if (!x.track_grad()) {
        return x;
    }


    x.track_self_mod_tensors(
            [beta, sx_c, wx_c, all_below](
                    const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents) {
                if(all_below){
                    Tensor x_w = sx_c->tensor * beta;
                    Tensor grad_w = grad * ::nt::functional::sigmoid(x_w);
                    parents[0]->accumulate_gradient( grad_w );
                }else{
                    Tensor x_w = sx_c->tensor[wx_c->tensor] * beta;
                    Tensor grad_w = grad.clone();
                    grad_w[wx_c->tensor] *= ::nt::functional::sigmoid(x_w);
                    parents[0]->accumulate_gradient( grad_w );
                }

            }, "Softplus_");
    return x;
}

TensorGrad TensorGrad_Functional_Class::relu(const TensorGrad &x) {
    return clamp(x, 0, std::nullopt);
}

TensorGrad& TensorGrad_Functional_Class::relu_(TensorGrad &x) {
    return clamp_(x, 0, std::nullopt);
}


TensorGrad TensorGrad_Functional_Class::pow(const TensorGrad &x, Scalar exponent){
    Tensor a = ::nt::functional::pow(x.detach(), exponent);
    TensorGrad result(std::move(a), x.track_grad());
    if (!x.track_grad()) {
        result.track_grad_(false);
        return std::move(result);
    }


    intrusive_ptr<tensor_holder> saved_x =
            make_intrusive<tensor_holder>(x.detach().conditional_mutate_clone());
    
    result.track_tensors(x);


    result.create_backward_function(
            [exponent](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents,
               intrusive_ptr<tensor_holder> saved_x) {
                Scalar one(complex_64(1,1));
                const Tensor& x = saved_x->tensor;
                parents[0]->accumulate_gradient(exponent * saved_x->tensor.pow(exponent - one) * grad);
            },
            saved_x);

    return std::move(result);
}

TensorGrad& TensorGrad_Functional_Class::pow_(TensorGrad &x, Scalar exponent){
    if(!x.track_grad()){
        ::nt::functional::pow_(x.detach(), exponent);
        return x;
    }

    intrusive_ptr<tensor_holder> saved_x =
            make_intrusive<tensor_holder>(x.detach().conditional_mutate_clone());
    
    ::nt::functional::pow_(x.detach(), exponent);
    x.track_self_mod_tensors(
            [exponent, saved_x](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents) {
                Scalar one(complex_64(1,1));
                parents[0]->accumulate_gradient(exponent * saved_x->tensor.pow(exponent - one) * grad);
            }, "Pow_");

    return x;
}


TensorGrad TensorGrad_Functional_Class::sigmoid(const TensorGrad &x) {
    Tensor a = ::nt::functional::sigmoid(x.detach());
    if(!x.track_grad()){
        return TensorGrad(a, x.track_grad());
    }
    intrusive_ptr<tensor_holder> sigmoid_x =
            make_intrusive<tensor_holder>(a.conditional_mutate_clone());
    TensorGrad result(std::move(a), x.track_grad());
    result.track_tensors(x);
    result.create_backward_function(
            [](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents,
                 intrusive_ptr<tensor_holder> x) {
                parents[0]->accumulate_gradient( grad * ::nt::functional::dsigmoid(x->tensor, false) );
            },
            sigmoid_x, "Sigmoid");
    return std::move(result);
}

TensorGrad& TensorGrad_Functional_Class::sigmoid_(TensorGrad &x) {
    ::nt::functional::sigmoid_(x.detach());
    if(!x.track_grad()){
        return x;
    }
    intrusive_ptr<tensor_holder> sigmoid_x =
            make_intrusive<tensor_holder>(x.detach().conditional_mutate_clone());
    x.track_self_mod_tensors(
            [sigmoid_x](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents) {
                parents[0]->accumulate_gradient(grad * ::nt::functional::dsigmoid(sigmoid_x->tensor, false));
            },
            "Sigmoid_");
    return x;
}


}
}
