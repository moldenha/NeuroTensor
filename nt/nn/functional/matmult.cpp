#include "../../functional/functional.h"
#include "../../Tensor.h"
#include "../../intrusive_ptr/intrusive_ptr.hpp"
#include "../TensorGrad.h"
#include "../functional_class.h"

namespace nt{
namespace functional{

TensorGrad TensorGrad_Functional_Class::matmult(const TensorGrad &a, const TensorGrad &b, bool transpose_a, bool transpose_b) {
    if (!a.track_grad()) {
        if (!b.track_grad()) {
            Tensor out = ::nt::functional::matmult(a.detach(), b.detach(), transpose_a, transpose_b);
            TensorGrad result(std::move(out), false);
            result.track_grad_(false);
            return std::move(result);
        }
        return matmult(a.detach(), b, transpose_a, transpose_b);
    }
    if (!b.track_grad()) {
        return matmult(a, b.detach(), transpose_a, transpose_b);
    }
    // a and b are going to have to be cloned anyways so:

    // intrusive_ptr<tensor_holder> a_c =
    //         make_intrusive<tensor_holder>(transpose_a ? a.detach().transpose(-1,-2).conditional_mutate_clone() : a.detach().conditional_mutate_clone());
    // intrusive_ptr<tensor_holder> b_c =
    //         make_intrusive<tensor_holder>(transpose_b ? b.detach().transpose(-1,-2).conditional_mutate_clone() : b.detach().conditional_mutate_clone());
    intrusive_ptr<tensor_holder> a_c =
            make_intrusive<tensor_holder>(a.detach().conditional_mutate_clone());
    intrusive_ptr<tensor_holder> b_c =
            make_intrusive<tensor_holder>(b.detach().conditional_mutate_clone());
    TensorGrad result(::nt::functional::matmult(a_c->tensor, b_c->tensor, transpose_a, transpose_b), true);
    result.track_tensors(a, b);

    // Define backward function
    result.create_backward_function(
            [transpose_a, transpose_b](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents,
                 intrusive_ptr<tensor_holder> a, intrusive_ptr<tensor_holder> b) {
                if(transpose_a){
                    //this is what I am doing below:
                    //parents[0]->accumulate_gradient(
                    //    ::nt::functional::matmult(grad, b->tensor, false, !transpose_b).transpose(-1, -2));
                    //if mult(A, B) = C
                    //C transpose = mult(B.T, A.T)
                    //the below is just faster to calculate without an extra transpose
                    parents[0]->accumulate_gradient( 
                        ::nt::functional::matmult(b->tensor, grad, transpose_b, true));
                }else{
                    parents[0]->accumulate_gradient(::nt::functional::matmult(grad, b->tensor, false, !transpose_b));
                }
                
                if(transpose_b){
                    //parents[1]->accumulate_gradient(
                    //    ::nt::functional::matmult(a->tensor, grad, !transpose_a, false).transpose(-1, -2));
                    parents[1]->accumulate_gradient(
                        ::nt::functional::matmult(grad, a->tensor, true, transpose_a));
                }else{
                    parents[1]->accumulate_gradient(
                        ::nt::functional::matmult(a->tensor, grad, !transpose_a, false));
                }            
        },
            a_c, b_c);
    return result;
}

TensorGrad TensorGrad_Functional_Class::matmult(const Tensor &a, const TensorGrad &b, bool transpose_a, bool transpose_b) {
    if (!b.track_grad()) {
        Tensor out = ::nt::functional::matmult(a, b.detach(), transpose_a, transpose_b);
        TensorGrad result(std::move(out), b.track_grad());
        result.track_grad_(false);
        return std::move(result);
    }
    intrusive_ptr<tensor_holder> a_c = make_intrusive<tensor_holder>(a.conditional_mutate_clone());
    TensorGrad result(::nt::functional::matmult(a_c->tensor, b.detach(), transpose_a, transpose_b), b.track_grad());
    result.track_tensors(b);

    result.create_backward_function(
            [transpose_a, transpose_b](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents,
                 intrusive_ptr<tensor_holder> a) {
                if(transpose_b){
                    parents[0]->accumulate_gradient(
                        ::nt::functional::matmult(grad, a->tensor, true, transpose_a));
                }else{
                    parents[0]->accumulate_gradient(
                        ::nt::functional::matmult(a->tensor, grad, !transpose_a, false));
                }
            },
            a_c);
    return result;
}

TensorGrad TensorGrad_Functional_Class::matmult(const TensorGrad &a, const Tensor &b, bool transpose_a, bool transpose_b) {
    if (!a.track_grad()) {
        Tensor out = ::nt::functional::matmult(a.detach(), b, transpose_a, transpose_b);
        TensorGrad result(std::move(out), a.track_grad());
        result.track_grad_(false);
        return std::move(result);
    }
    intrusive_ptr<tensor_holder> b_c = make_intrusive<tensor_holder>(b.conditional_mutate_clone());
    TensorGrad result(::nt::functional::matmult(a.detach(), b_c->tensor, transpose_a, transpose_b), a.track_grad());
    result.track_tensors(a);

    result.create_backward_function(
            [transpose_a, transpose_b](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents,
                 intrusive_ptr<tensor_holder> b) {
                if(transpose_a){
                    //this is what I am doing below:
                    //parents[0]->accumulate_gradient(
                    //    ::nt::functional::matmult(grad, b->tensor, false, !transpose_b).transpose(-1, -2));
                    //if mult(A, B) = C
                    //C transpose = mult(B.T, A.T)
                    //the below is just faster to calculate without an extra transpose
                    parents[0]-> accumulate_gradient(
                        ::nt::functional::matmult(b->tensor, grad, transpose_b, true));
                }else{
                    parents[0]->accumulate_gradient(
                        ::nt::functional::matmult(grad, b->tensor, false, !transpose_b));
                }

            },
            b_c);
    return result;
}


TensorGrad& TensorGrad_Functional_Class::matmult(const TensorGrad &a, const TensorGrad &b, TensorGrad& out, bool transpose_a, bool transpose_b) {
    if (!a.track_grad()) {
        if (!b.track_grad()) {
            out.track_grad_(false);
            ::nt::functional::matmult(a.detach(), b.detach(), out.detach(), transpose_a, transpose_b);
            return out;
        }
        return matmult(a.detach(), b, out, transpose_a, transpose_b);
    }
    if (!b.track_grad()) {
        return matmult(a, b.detach(), out, transpose_a, transpose_b);
    }

    // intrusive_ptr<tensor_holder> a_c =
    //         make_intrusive<tensor_holder>(transpose_a ? a.detach().transpose(-1,-2).conditional_mutate_clone() : a.detach().conditional_mutate_clone());
    // intrusive_ptr<tensor_holder> b_c =
    //         make_intrusive<tensor_holder>(transpose_b ? b.detach().transpose(-1,-2).conditional_mutate_clone() : b.detach().conditional_mutate_clone());
    intrusive_ptr<tensor_holder> a_c =
            make_intrusive<tensor_holder>(a.detach().conditional_mutate_clone());
    intrusive_ptr<tensor_holder> b_c =
            make_intrusive<tensor_holder>(b.detach().conditional_mutate_clone());
    ::nt::functional::matmult(a_c->tensor, b_c->tensor, out.detach(), transpose_a, transpose_b);

    // Define backward function
    out.track_self_mod_tensors(
            [transpose_a, transpose_b, a = std::move(a_c), b = std::move(b_c)](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents) {
                if(transpose_a){
                    //this is what I am doing below:
                    //parents[0]->accumulate_gradient(
                    //    ::nt::functional::matmult(grad, b->tensor, false, !transpose_b).transpose(-1, -2));
                    //if mult(A, B) = C
                    //C transpose = mult(B.T, A.T)
                    //the below is just faster to calculate without an extra transpose
                    parents[0]-> accumulate_gradient(
                        ::nt::functional::matmult(b->tensor, grad, transpose_b, true));
                }else{
                    parents[0]->accumulate_gradient(
                        ::nt::functional::matmult(grad, b->tensor, false, !transpose_b));
                }
                
                if(transpose_b){
                    //parents[1]->accumulate_gradient(
                    //    ::nt::functional::matmult(a->tensor, grad, !transpose_a, false).transpose(-1, -2));
                    parents[1]->accumulate_gradient(
                        ::nt::functional::matmult(grad, a->tensor, true, transpose_a));
                }else{
                    parents[1]->accumulate_gradient(
                        ::nt::functional::matmult(a->tensor, grad, !transpose_a, false));
                }            
        }, "Matmult", a, b );
    return out;
}

TensorGrad& TensorGrad_Functional_Class::matmult(const Tensor &a, const TensorGrad &b, TensorGrad& out, bool transpose_a, bool transpose_b) {
    if (!b.track_grad()) {
        out.track_grad_(false);
        ::nt::functional::matmult(a, b.detach(), out.detach(), transpose_a, transpose_b);

        return out;
    }
    intrusive_ptr<tensor_holder> a_c = make_intrusive<tensor_holder>(a.conditional_mutate_clone());
    ::nt::functional::matmult(a_c->tensor, b.detach(), out.detach(), transpose_a, transpose_b);

    out.track_self_mod_tensors(
            [transpose_a, transpose_b, a = std::move(a_c)](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents) {
                if(transpose_b){
                    parents[0]->accumulate_gradient(
                        ::nt::functional::matmult(grad, a->tensor, true, transpose_a));
                }else{
                    parents[0]->accumulate_gradient(
                        ::nt::functional::matmult(a->tensor, grad, !transpose_a, false));
                }
            },
            "Matmult", b);
    return out;
}

TensorGrad& TensorGrad_Functional_Class::matmult(const TensorGrad &a, const Tensor &b, TensorGrad& out, bool transpose_a, bool transpose_b) {
    if (!a.track_grad()) {
        out.track_grad_(false);
        ::nt::functional::matmult(a.detach(), b, out.detach(), transpose_a, transpose_b);
        return out;

    }
    intrusive_ptr<tensor_holder> b_c = make_intrusive<tensor_holder>(b.conditional_mutate_clone());
    ::nt::functional::matmult(a.detach(), b_c->tensor, out.detach(), transpose_a, transpose_b);

    out.track_self_mod_tensors(
            [transpose_a, transpose_b, b = std::move(b_c)](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents) {
                if(transpose_a){
                    //this is what I am doing below:
                    //parents[0]->accumulate_gradient(
                    //    ::nt::functional::matmult(grad, b->tensor, false, !transpose_b).transpose(-1, -2));
                    //if mult(A, B) = C
                    //C transpose = mult(B.T, A.T)
                    //the below is just faster to calculate without an extra transpose
                    parents[0]-> accumulate_gradient(
                        ::nt::functional::matmult(b->tensor, grad, transpose_b, true));
                }else{
                    parents[0]->accumulate_gradient(
                        ::nt::functional::matmult(grad, b->tensor, false, !transpose_b));
                }

            },
            "Matmult", a);
    return out;
}



TensorGrad TensorGrad_Functional_Class::linear(const TensorGrad& x, const TensorGrad& w, const TensorGrad& b, bool trans_a, bool trans_b){
    if(b.is_null()){
        utils::throw_exception(!x.is_null() && !w.is_null(), "Error got x or w null tensors for linear!");
        return matmult(x, w, trans_a, trans_b);
    }
    if(!x.track_grad() || !w.track_grad() || !b.track_grad()){
        if(!x.track_grad()) return linear(x.detach(), w, b, trans_a, trans_b);
        if(!w.track_grad()) return linear(x, w.detach(), b, trans_a, trans_b);
        return linear(x, w, b.detach(), trans_a, trans_b);
    }
    intrusive_ptr<tensor_holder> x_tensor = make_intrusive<tensor_holder>(x.detach().conditional_mutate_clone());
    intrusive_ptr<tensor_holder> w_tensor = make_intrusive<tensor_holder>(w.detach().conditional_mutate_clone());
    
    TensorGrad result(::nt::functional::linear(x_tensor->tensor, w_tensor->tensor, b.detach(), trans_a, trans_b), /*track_grad = */true);
    result.track_tensors(x, w, b);
    
        result.create_backward_function(
        [trans_a, trans_b](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
            intrusive_ptr<tensor_holder> x, intrusive_ptr<tensor_holder> w)
        {
            // grad w.r.t. x
            if (trans_a) {
                parents[0]-> accumulate_gradient(::nt::functional::matmult(w->tensor, grad, trans_b, true));
            } else {
                parents[0]-> accumulate_gradient(::nt::functional::matmult(grad, w->tensor, false, !trans_b));
            }

            // grad w.r.t. w
            if (trans_b) {
                parents[1]-> accumulate_gradient(::nt::functional::matmult(grad, x->tensor, true, trans_a));
            } else {
                parents[1]-> accumulate_gradient(::nt::functional::matmult(x->tensor, grad, !trans_a, false));
            }

            // grad w.r.t. b
            parents[2]-> accumulate_gradient(grad);  // gradient is broadcasted back to bias shape
        },
        x_tensor, w_tensor
    );

    return std::move(result);

}

TensorGrad TensorGrad_Functional_Class::linear(const Tensor& x, const TensorGrad& w, const TensorGrad& b, bool trans_a, bool trans_b){
    if(b.is_null()){
        utils::throw_exception(!x.is_null() && !w.is_null(), "Error got x or w null tensors for linear!");
        return matmult(x, w, trans_a, trans_b);
    }
    if(!w.track_grad() || !b.track_grad()){
        if(!w.track_grad()) return linear(x, w.detach(), b, trans_a, trans_b);
        return linear(x, w, b.detach(), trans_a, trans_b);
    }
    intrusive_ptr<tensor_holder> x_tensor = make_intrusive<tensor_holder>(x.conditional_mutate_clone());
    
    TensorGrad result(::nt::functional::linear(x_tensor->tensor, w.detach(), b.detach(), trans_a, trans_b), /*track_grad = */true);
    result.track_tensors(w, b);
    
        result.create_backward_function(
        [trans_a, trans_b](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
            intrusive_ptr<tensor_holder> x)
        {
            // grad w.r.t. w
            if (trans_b) {
                parents[0]-> accumulate_gradient(::nt::functional::matmult(grad, x->tensor, true, trans_a));
            } else {
                parents[0]-> accumulate_gradient(::nt::functional::matmult(x->tensor, grad, !trans_a, false));
            }

            // grad w.r.t. b
            parents[1]-> accumulate_gradient(grad);  // gradient is broadcasted back to bias shape
        },
        x_tensor
    );

    return std::move(result);

}

TensorGrad TensorGrad_Functional_Class::linear(const TensorGrad& x, const Tensor& w, const TensorGrad& b, bool trans_a, bool trans_b){
    if(b.is_null()){
        utils::throw_exception(!x.is_null() && !w.is_null(), "Error got x or w null tensors for linear!");
        return matmult(x, w, trans_a, trans_b);
    }
    if(!x.track_grad() || !b.track_grad()){
        if(!x.track_grad()) return linear(x.detach(), w, b, trans_a, trans_b);
        return linear(x, w, b.detach(), trans_a, trans_b);
    }
    intrusive_ptr<tensor_holder> w_tensor = make_intrusive<tensor_holder>(w.conditional_mutate_clone());
    
    TensorGrad result(::nt::functional::linear(x.detach(), w_tensor->tensor, b.detach(), trans_a, trans_b), /*track_grad = */true);
    result.track_tensors(x, b);
    
        result.create_backward_function(
        [trans_a, trans_b](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
            intrusive_ptr<tensor_holder> w)
        {
            // grad w.r.t. x
            if (trans_a) {
                parents[0]-> accumulate_gradient(::nt::functional::matmult(w->tensor, grad, trans_b, true));
            } else {
                parents[0]-> accumulate_gradient(::nt::functional::matmult(grad, w->tensor, false, !trans_b));
            }

            // grad w.r.t. b
            parents[1]-> accumulate_gradient(grad);  // gradient is broadcasted back to bias shape
        },
        w_tensor
    );

    return std::move(result);
}

TensorGrad TensorGrad_Functional_Class::linear(const TensorGrad& x, const TensorGrad& w, const Tensor& b, bool trans_a, bool trans_b){
    if(b.is_null()){
        utils::throw_exception(!x.is_null() && !w.is_null(), "Error got x or w null tensors for linear!");
        return matmult(x, w, trans_a, trans_b);
    }
    if(!x.track_grad() || !w.track_grad() ){
        if(!x.track_grad()) return linear(x.detach(), w, b, trans_a, trans_b);
        return linear(x, w.detach(), b, trans_a, trans_b);
    }
    intrusive_ptr<tensor_holder> x_tensor = make_intrusive<tensor_holder>(x.detach().conditional_mutate_clone());
    intrusive_ptr<tensor_holder> w_tensor = make_intrusive<tensor_holder>(w.detach().conditional_mutate_clone());
    
    TensorGrad result(::nt::functional::linear(x_tensor->tensor, w_tensor->tensor, b, trans_a, trans_b), /*track_grad = */true);
    result.track_tensors(x, w);
    
        result.create_backward_function(
        [trans_a, trans_b](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
            intrusive_ptr<tensor_holder> x, intrusive_ptr<tensor_holder> w)
        {
            // grad w.r.t. x
            if (trans_a) {
                parents[0]-> accumulate_gradient(::nt::functional::matmult(w->tensor, grad, trans_b, true));
            } else {
                parents[0]-> accumulate_gradient(::nt::functional::matmult(grad, w->tensor, false, !trans_b));
            }

            // grad w.r.t. w
            if (trans_b) {
                parents[1]-> accumulate_gradient(::nt::functional::matmult(grad, x->tensor, true, trans_a));
            } else {
                parents[1]-> accumulate_gradient(::nt::functional::matmult(x->tensor, grad, !trans_a, false));
            }
        },
        x_tensor, w_tensor
    );

    return std::move(result);

}

TensorGrad TensorGrad_Functional_Class::linear(const TensorGrad& x, const Tensor& w, const Tensor& b, bool trans_a, bool trans_b){
    if(b.is_null()){
        utils::throw_exception(!x.is_null() && !w.is_null(), "Error got x or w null tensors for linear!");
        return matmult(x, w, trans_a, trans_b);
    }

    if(!x.track_grad()){
        Tensor output = ::nt::functional::linear(x.detach(), w, b, trans_a, trans_b);
        TensorGrad result(output, x.track_grad());
        result.track_grad_(false);
        return std::move(result);
    }

    intrusive_ptr<tensor_holder> w_tensor = make_intrusive<tensor_holder>(w.conditional_mutate_clone());
    
    TensorGrad result(::nt::functional::linear(x.detach(), w_tensor->tensor, b, trans_a, trans_b), /*track_grad = */true);
    result.track_tensors(x);
    
        result.create_backward_function(
        [trans_a, trans_b](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
            intrusive_ptr<tensor_holder> w)
        {
            // grad w.r.t. x
            if (trans_a) {
                parents[0]-> accumulate_gradient(::nt::functional::matmult(w->tensor, grad, trans_b, true));
            } else {
                parents[0]-> accumulate_gradient(::nt::functional::matmult(grad, w->tensor, false, !trans_b));
            }

        },
        w_tensor
    );

    return std::move(result);
}

TensorGrad TensorGrad_Functional_Class::linear(const Tensor& x, const TensorGrad& w, const Tensor& b, bool trans_a, bool trans_b){
    if(b.is_null()){
        utils::throw_exception(!x.is_null() && !w.is_null(), "Error got x or w null tensors for linear!");
        return matmult(x, w, trans_a, trans_b);
    }
    if(!w.track_grad()){
        Tensor output = ::nt::functional::linear(x, w.detach(), b, trans_a, trans_b);
        TensorGrad result(output, w.track_grad());
        result.track_grad_(false);
        return std::move(result);
    }

    intrusive_ptr<tensor_holder> x_tensor = make_intrusive<tensor_holder>(x.conditional_mutate_clone());
    
    TensorGrad result(::nt::functional::linear(x_tensor->tensor, w.detach(), b, trans_a, trans_b), /*track_grad = */true);
    result.track_tensors(w);
    
        result.create_backward_function(
        [trans_a, trans_b](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
            intrusive_ptr<tensor_holder> x)
        {

            // grad w.r.t. w
            if (trans_b) {
                parents[0]-> accumulate_gradient(::nt::functional::matmult(grad, x->tensor, true, trans_a));
            } else {
                parents[0]-> accumulate_gradient(::nt::functional::matmult(x->tensor, grad, !trans_a, false));
            }
        },
        x_tensor
    );

    return std::move(result);
}

TensorGrad TensorGrad_Functional_Class::linear(const Tensor& x, const Tensor& w, const TensorGrad& b, bool trans_a, bool trans_b){
    if(b.is_null()){
        utils::throw_exception(!x.is_null() && !w.is_null(), "Error got x or w null tensors for linear!");
        Tensor out = ::nt::functional::matmult(x, w, trans_a, trans_b);
        TensorGrad result(std::move(out), b.track_grad());
        result.track_grad_(false);
        return std::move(result);
    }
    if(!b.track_grad()){
        Tensor output = ::nt::functional::linear(x, w, b.detach(), trans_a, trans_b);
        TensorGrad result(output, b.track_grad());
        result.track_grad_(false);
        return std::move(result);
    }
    
    TensorGrad result(::nt::functional::linear(x, w, b.detach(), trans_a, trans_b), /*track_grad = */true);
    result.track_tensors(b);
    
        result.create_backward_function(
        [trans_a, trans_b](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents)
        {
            // grad w.r.t. b
            parents[2]-> accumulate_gradient(grad);  // gradient is broadcasted back to bias shape
        }
    );

    return std::move(result);
}


}
}
