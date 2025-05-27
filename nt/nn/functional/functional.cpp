#include "../../functional/functional.h"
#include "../../Tensor.h"
#include "../../intrusive_ptr/intrusive_ptr.hpp"
#include "../TensorGrad.h"
#include "../functional_class.h"
#include "../../dtype/ArrayVoid.hpp"
// #include "functional.h"

namespace nt {
namespace functional {

TensorGrad TensorGrad_Functional_Class::matmult(const TensorGrad &a, const TensorGrad &b, bool transpose_a, bool transpose_b) {
    if (!a.do_track_grad) {
        if (!b.do_track_grad) {
            Tensor out = ::nt::functional::matmult(a.tensor, b.tensor, transpose_a, transpose_b);
            TensorGrad result(std::move(out), false);
            result.do_track_grad = false;
            return std::move(result);
        }
        return matmult(a.tensor, b, transpose_a, transpose_b);
    }
    if (!b.do_track_grad) {
        return matmult(a, b.tensor, transpose_a, transpose_b);
    }
    // a and b are going to have to be cloned anyways so:

    // intrusive_ptr<tensor_holder> a_c =
    //         make_intrusive<tensor_holder>(transpose_a ? a.tensor.transpose(-1,-2).conditional_mutate_clone() : a.tensor.conditional_mutate_clone());
    // intrusive_ptr<tensor_holder> b_c =
    //         make_intrusive<tensor_holder>(transpose_b ? b.tensor.transpose(-1,-2).conditional_mutate_clone() : b.tensor.conditional_mutate_clone());
    intrusive_ptr<tensor_holder> a_c =
            make_intrusive<tensor_holder>(a.tensor.conditional_mutate_clone());
    intrusive_ptr<tensor_holder> b_c =
            make_intrusive<tensor_holder>(b.tensor.conditional_mutate_clone());
    TensorGrad result(::nt::functional::matmult(a_c->tensor, b_c->tensor, transpose_a, transpose_b), true);
    result.track_tensors(a, b);

    // Define backward function
    result.create_backward_function(
            [transpose_a, transpose_b](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents,
                 intrusive_ptr<tensor_holder> a, intrusive_ptr<tensor_holder> b) {
                if(transpose_a){
                    //this is what I am doing below:
                    //parents[0]->grad->tensor +=
                    //    ::nt::functional::matmult(grad, b->tensor, false, !transpose_b).transpose(-1, -2);
                    //if mult(A, B) = C
                    //C transpose = mult(B.T, A.T)
                    //the below is just faster to calculate without an extra transpose
                    parents[0]->grad->tensor += 
                        ::nt::functional::matmult(b->tensor, grad, transpose_b, true);
                }else{
                    parents[0]->grad->tensor +=
                        ::nt::functional::matmult(grad, b->tensor, false, !transpose_b);
                }
                
                if(transpose_b){
                    //parents[1]->grad->tensor +=
                    //    ::nt::functional::matmult(a->tensor, grad, !transpose_a, false).transpose(-1, -2);
                    parents[1]->grad->tensor +=
                        ::nt::functional::matmult(grad, a->tensor, true, transpose_a);
                }else{
                    parents[1]->grad->tensor +=
                        ::nt::functional::matmult(a->tensor, grad, !transpose_a, false);
                }            
        },
            a_c, b_c);
    return result;
}

TensorGrad TensorGrad_Functional_Class::matmult(const Tensor &a, const TensorGrad &b, bool transpose_a, bool transpose_b) {
    if (!b.do_track_grad) {
        Tensor out = ::nt::functional::matmult(a, b.tensor, transpose_a, transpose_b);
        TensorGrad result(std::move(out), b.grad_required);
        result.do_track_grad = false;
        return std::move(result);
    }
    intrusive_ptr<tensor_holder> a_c = make_intrusive<tensor_holder>(a.conditional_mutate_clone());
    TensorGrad result(::nt::functional::matmult(a_c->tensor, b.tensor, transpose_a, transpose_b), b.grad_required);
    result.track_tensors(b);

    result.create_backward_function(
            [transpose_a, transpose_b](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents,
                 intrusive_ptr<tensor_holder> a) {
                if(transpose_b){
                    parents[0]->grad->tensor +=
                        ::nt::functional::matmult(grad, a->tensor, true, transpose_a);
                }else{
                    parents[0]->grad->tensor +=
                        ::nt::functional::matmult(a->tensor, grad, !transpose_a, false);
                }
            },
            a_c);
    return result;
}

TensorGrad TensorGrad_Functional_Class::matmult(const TensorGrad &a, const Tensor &b, bool transpose_a, bool transpose_b) {
    if (!a.do_track_grad) {
        Tensor out = ::nt::functional::matmult(a.tensor, b, transpose_a, transpose_b);
        TensorGrad result(std::move(out), a.grad_required);
        result.do_track_grad = false;
        return std::move(result);
    }
    intrusive_ptr<tensor_holder> b_c = make_intrusive<tensor_holder>(b.conditional_mutate_clone());
    TensorGrad result(::nt::functional::matmult(a.tensor, b_c->tensor, transpose_a, transpose_b), a.grad_required);
    result.track_tensors(a);

    result.create_backward_function(
            [transpose_a, transpose_b](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents,
                 intrusive_ptr<tensor_holder> b) {
                if(transpose_a){
                    //this is what I am doing below:
                    //parents[0]->grad->tensor +=
                    //    ::nt::functional::matmult(grad, b->tensor, false, !transpose_b).transpose(-1, -2);
                    //if mult(A, B) = C
                    //C transpose = mult(B.T, A.T)
                    //the below is just faster to calculate without an extra transpose
                    parents[0]->grad->tensor += 
                        ::nt::functional::matmult(b->tensor, grad, transpose_b, true);
                }else{
                    parents[0]->grad->tensor +=
                        ::nt::functional::matmult(grad, b->tensor, false, !transpose_b);
                }

            },
            b_c);
    return result;
}


TensorGrad TensorGrad_Functional_Class::linear(const TensorGrad& x, const TensorGrad& w, const TensorGrad& b, bool trans_a, bool trans_b){
    if(b.is_null()){
        utils::throw_exception(!x.is_null() && !w.is_null(), "Error got x or w null tensors for linear!");
        return matmult(x, w, trans_a, trans_b);
    }
    if(!x.do_track_grad || !w.do_track_grad || !b.do_track_grad){
        if(!x.do_track_grad) return linear(x.tensor, w, b, trans_a, trans_b);
        if(!w.do_track_grad) return linear(x, w.tensor, b, trans_a, trans_b);
        return linear(x, w, b.tensor, trans_a, trans_b);
    }
    intrusive_ptr<tensor_holder> x_tensor = make_intrusive<tensor_holder>(x.tensor.conditional_mutate_clone());
    intrusive_ptr<tensor_holder> w_tensor = make_intrusive<tensor_holder>(w.tensor.conditional_mutate_clone());
    
    TensorGrad result(::nt::functional::linear(x_tensor->tensor, w_tensor->tensor, b.tensor, trans_a, trans_b), /*track_grad = */true);
    result.track_tensors(x, w, b);
    
        result.create_backward_function(
        [trans_a, trans_b](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
            intrusive_ptr<tensor_holder> x, intrusive_ptr<tensor_holder> w)
        {
            // grad w.r.t. x
            if (trans_a) {
                parents[0]->grad->tensor += ::nt::functional::matmult(w->tensor, grad, trans_b, true);
            } else {
                parents[0]->grad->tensor += ::nt::functional::matmult(grad, w->tensor, false, !trans_b);
            }

            // grad w.r.t. w
            if (trans_b) {
                parents[1]->grad->tensor += ::nt::functional::matmult(grad, x->tensor, true, trans_a);
            } else {
                parents[1]->grad->tensor += ::nt::functional::matmult(x->tensor, grad, !trans_a, false);
            }

            // grad w.r.t. b
            parents[2]->grad->tensor += grad;  // gradient is broadcasted back to bias shape
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
    if(!w.do_track_grad || !b.do_track_grad){
        if(!w.do_track_grad) return linear(x, w.tensor, b, trans_a, trans_b);
        return linear(x, w, b.tensor, trans_a, trans_b);
    }
    intrusive_ptr<tensor_holder> x_tensor = make_intrusive<tensor_holder>(x.conditional_mutate_clone());
    
    TensorGrad result(::nt::functional::linear(x_tensor->tensor, w.tensor, b.tensor, trans_a, trans_b), /*track_grad = */true);
    result.track_tensors(w, b);
    
        result.create_backward_function(
        [trans_a, trans_b](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
            intrusive_ptr<tensor_holder> x)
        {
            // grad w.r.t. w
            if (trans_b) {
                parents[0]->grad->tensor += ::nt::functional::matmult(grad, x->tensor, true, trans_a);
            } else {
                parents[0]->grad->tensor += ::nt::functional::matmult(x->tensor, grad, !trans_a, false);
            }

            // grad w.r.t. b
            parents[1]->grad->tensor += grad;  // gradient is broadcasted back to bias shape
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
    if(!x.do_track_grad || !b.do_track_grad){
        if(!x.do_track_grad) return linear(x.tensor, w, b, trans_a, trans_b);
        return linear(x, w, b.tensor, trans_a, trans_b);
    }
    intrusive_ptr<tensor_holder> w_tensor = make_intrusive<tensor_holder>(w.conditional_mutate_clone());
    
    TensorGrad result(::nt::functional::linear(x.tensor, w_tensor->tensor, b.tensor, trans_a, trans_b), /*track_grad = */true);
    result.track_tensors(x, b);
    
        result.create_backward_function(
        [trans_a, trans_b](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
            intrusive_ptr<tensor_holder> w)
        {
            // grad w.r.t. x
            if (trans_a) {
                parents[0]->grad->tensor += ::nt::functional::matmult(w->tensor, grad, trans_b, true);
            } else {
                parents[0]->grad->tensor += ::nt::functional::matmult(grad, w->tensor, false, !trans_b);
            }

            // grad w.r.t. b
            parents[1]->grad->tensor += grad;  // gradient is broadcasted back to bias shape
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
    if(!x.do_track_grad || !w.do_track_grad ){
        if(!x.do_track_grad) return linear(x.tensor, w, b, trans_a, trans_b);
        return linear(x, w.tensor, b, trans_a, trans_b);
    }
    intrusive_ptr<tensor_holder> x_tensor = make_intrusive<tensor_holder>(x.tensor.conditional_mutate_clone());
    intrusive_ptr<tensor_holder> w_tensor = make_intrusive<tensor_holder>(w.tensor.conditional_mutate_clone());
    
    TensorGrad result(::nt::functional::linear(x_tensor->tensor, w_tensor->tensor, b, trans_a, trans_b), /*track_grad = */true);
    result.track_tensors(x, w);
    
        result.create_backward_function(
        [trans_a, trans_b](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
            intrusive_ptr<tensor_holder> x, intrusive_ptr<tensor_holder> w)
        {
            // grad w.r.t. x
            if (trans_a) {
                parents[0]->grad->tensor += ::nt::functional::matmult(w->tensor, grad, trans_b, true);
            } else {
                parents[0]->grad->tensor += ::nt::functional::matmult(grad, w->tensor, false, !trans_b);
            }

            // grad w.r.t. w
            if (trans_b) {
                parents[1]->grad->tensor += ::nt::functional::matmult(grad, x->tensor, true, trans_a);
            } else {
                parents[1]->grad->tensor += ::nt::functional::matmult(x->tensor, grad, !trans_a, false);
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

    if(!x.do_track_grad){
        Tensor output = ::nt::functional::linear(x.tensor, w, b, trans_a, trans_b);
        TensorGrad result(output, x.grad_required);
        result.do_track_grad = false;
        return std::move(result);
    }

    intrusive_ptr<tensor_holder> w_tensor = make_intrusive<tensor_holder>(w.conditional_mutate_clone());
    
    TensorGrad result(::nt::functional::linear(x.tensor, w_tensor->tensor, b, trans_a, trans_b), /*track_grad = */true);
    result.track_tensors(x);
    
        result.create_backward_function(
        [trans_a, trans_b](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
            intrusive_ptr<tensor_holder> w)
        {
            // grad w.r.t. x
            if (trans_a) {
                parents[0]->grad->tensor += ::nt::functional::matmult(w->tensor, grad, trans_b, true);
            } else {
                parents[0]->grad->tensor += ::nt::functional::matmult(grad, w->tensor, false, !trans_b);
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
    if(!w.do_track_grad){
        Tensor output = ::nt::functional::linear(x, w.tensor, b, trans_a, trans_b);
        TensorGrad result(output, w.grad_required);
        result.do_track_grad = false;
        return std::move(result);
    }

    intrusive_ptr<tensor_holder> x_tensor = make_intrusive<tensor_holder>(x.conditional_mutate_clone());
    
    TensorGrad result(::nt::functional::linear(x_tensor->tensor, w.tensor, b, trans_a, trans_b), /*track_grad = */true);
    result.track_tensors(w);
    
        result.create_backward_function(
        [trans_a, trans_b](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
            intrusive_ptr<tensor_holder> x)
        {

            // grad w.r.t. w
            if (trans_b) {
                parents[0]->grad->tensor += ::nt::functional::matmult(grad, x->tensor, true, trans_a);
            } else {
                parents[0]->grad->tensor += ::nt::functional::matmult(x->tensor, grad, !trans_a, false);
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
        TensorGrad result(std::move(out), b.grad_required);
        result.do_track_grad = false;
        return std::move(result);
    }
    if(!b.do_track_grad){
        Tensor output = ::nt::functional::linear(x, w, b.tensor, trans_a, trans_b);
        TensorGrad result(output, b.grad_required);
        result.do_track_grad = false;
        return std::move(result);
    }
    
    TensorGrad result(::nt::functional::linear(x, w, b.tensor, trans_a, trans_b), /*track_grad = */true);
    result.track_tensors(b);
    
        result.create_backward_function(
        [trans_a, trans_b](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents)
        {
            // grad w.r.t. b
            parents[2]->grad->tensor += grad;  // gradient is broadcasted back to bias shape
        }
    );

    return std::move(result);
}

//
// W2 @ (W1 @ input)
// +
// (input @ W1.T) @ W2.T
TensorGrad TensorGrad_Functional_Class::symmetric_bilinear(const TensorGrad& input, const TensorGrad& W1, const TensorGrad& W2){
    utils::THROW_EXCEPTION(!input.is_null(),
                           "Got null tensor for symmetric attention");
    utils::THROW_EXCEPTION(!W1.is_null(),
                           "Got null tensor for symmetric attention");
    utils::THROW_EXCEPTION(!W2.is_null(),
                           "Got null tensor for symmetric attention");
    
    if(!input.do_track_grad || !W1.do_track_grad || !W2.do_track_grad){
        if(!input.do_track_grad) return symmetric_bilinear(input.tensor, W1, W2);
        if(!W1.do_track_grad) return symmetric_bilinear(input, W1.tensor, W2);
        return symmetric_bilinear(input, W1, W2.tensor);
    }

    auto input_saved = make_intrusive<tensor_holder>(input.tensor.conditional_mutate_clone());
    auto w1_saved = make_intrusive<tensor_holder>(W1.tensor.conditional_mutate_clone());
    auto w2_saved = make_intrusive<tensor_holder>(W2.tensor.conditional_mutate_clone());
    
    Tensor left = functional::matmult(w1_saved->tensor, input_saved->tensor);
    Tensor left_out = functional::matmult(w2_saved->tensor, left);
    
    Tensor right = functional::matmult(input_saved->tensor, w1_saved->tensor, false, true);
    Tensor right_out = functional::matmult(right, w2_saved->tensor, false, true);
    
    TensorGrad result(right_out + left_out, true);
    result.track_tensors(input, W1, W2);
    
    auto right_saved = make_intrusive<tensor_holder>(right);
    auto left_saved = make_intrusive<tensor_holder>(left);

    // Define backward function
    result.create_backward_function(
        [w2_saved](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents,
           intrusive_ptr<tensor_holder> input_saved,
           intrusive_ptr<tensor_holder> weight1_saved,
           intrusive_ptr<tensor_holder> right_saved,
           intrusive_ptr<tensor_holder> left_saved) {
            
            const Tensor& X = input_saved->tensor;
            const Tensor& W1 = weight1_saved->tensor;
            const Tensor& W2 = w2_saved->tensor;
            const Tensor& right = right_saved->tensor;
            const Tensor& left = left_saved->tensor;

            Tensor dRightMid = ::nt::functional::matmult(grad, W2, false, false);
            Tensor dWeight2Right = ::nt::functional::matmult(grad, right, true, false);
            Tensor dWeight1Right = ::nt::functional::matmult(dRightMid, X, true, false);
            Tensor dInputRight = ::nt::functional::matmult(dRightMid, W1, false, false);

            Tensor dLeftMid = ::nt::functional::matmult(W2, grad, true, false);
            Tensor dWeight2Left = ::nt::functional::matmult(grad, left, false, true);
            Tensor dWeight1Left = ::nt::functional::matmult(dLeftMid, X, false, true);
            Tensor dInputLeft = ::nt::functional::matmult(W1, dLeftMid, true, false);

            // Accumulate gradients
            parents[0]->grad->tensor += dInputRight + dInputLeft;
            parents[1]->grad->tensor += dWeight1Right + dWeight1Left;
            parents[2]->grad->tensor += dWeight2Right + dWeight2Left;
        },
        input_saved, w1_saved, right_saved, left_saved, "SymmetricBilinear"
    );
    return std::move(result);
}

TensorGrad TensorGrad_Functional_Class::symmetric_bilinear(const Tensor& input, const TensorGrad& W1, const TensorGrad& W2){
    utils::THROW_EXCEPTION(!input.is_null(),
                           "Got null tensor for symmetric attention");
    utils::THROW_EXCEPTION(!W1.is_null(),
                           "Got null tensor for symmetric attention");
    utils::THROW_EXCEPTION(!W2.is_null(),
                           "Got null tensor for symmetric attention");
    if(!W2.do_track_grad || !W1.do_track_grad){
        if(!W2.do_track_grad) return symmetric_bilinear(input, W1, W2.tensor);
        return symmetric_bilinear(input, W1.tensor, W2);
    } 
    auto input_saved = make_intrusive<tensor_holder>(input.conditional_mutate_clone());
    // auto w1_saved = make_intrusive<tensor_holder>(W1.tensor.conditional_mutate_clone());
    auto w2_saved = make_intrusive<tensor_holder>(W2.tensor.conditional_mutate_clone());
    
    Tensor left = functional::matmult(W1.tensor, input_saved->tensor);
    Tensor left_out = functional::matmult(w2_saved->tensor, left);
    
    Tensor right = functional::matmult(input_saved->tensor, W1.tensor, false, true);
    Tensor right_out = functional::matmult(right, w2_saved->tensor, false, true);
    
    TensorGrad result(right_out + left_out, true);
    result.track_tensors(W1, W2);
    
    auto right_saved = make_intrusive<tensor_holder>(right);
    auto left_saved = make_intrusive<tensor_holder>(left);

    // Define backward function
    result.create_backward_function(
        [](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents,
           intrusive_ptr<tensor_holder> input_saved,
           intrusive_ptr<tensor_holder> w2_saved,
           intrusive_ptr<tensor_holder> right_saved,
           intrusive_ptr<tensor_holder> left_saved) {
            
            const Tensor& X = input_saved->tensor;
            // const Tensor& W1 = weight1_saved->tensor;
            const Tensor& W2 = w2_saved->tensor;
            const Tensor& right = right_saved->tensor;
            const Tensor& left = left_saved->tensor;

            Tensor dRightMid = ::nt::functional::matmult(grad, W2, false, false);
            Tensor dWeight2Right = ::nt::functional::matmult(grad, right, true, false);
            Tensor dWeight1Right = ::nt::functional::matmult(dRightMid, X, true, false);
            // Tensor dInputRight = ::nt::functional::matmult(dRightMid, W1, false, false);

            Tensor dLeftMid = ::nt::functional::matmult(W2, grad, true, false);
            Tensor dWeight2Left = ::nt::functional::matmult(grad, left, false, true);
            Tensor dWeight1Left = ::nt::functional::matmult(dLeftMid, X, false, true);
            // Tensor dInputLeft = ::nt::functional::matmult(W1, dLeftMid, true, false);

            // Accumulate gradients
            // parents[0]->grad->tensor += dInputRight + dInputLeft;
            parents[0]->grad->tensor += dWeight1Right + dWeight1Left;
            parents[1]->grad->tensor += dWeight2Right + dWeight2Left;
        },
        input_saved, w2_saved, right_saved, left_saved, "SymmetricBilinear"
    );
    return std::move(result);
}

TensorGrad TensorGrad_Functional_Class::symmetric_bilinear(const TensorGrad& input, const Tensor& W1, const TensorGrad& W2){
    utils::THROW_EXCEPTION(!input.is_null(),
                           "Got null tensor for symmetric attention");
    utils::THROW_EXCEPTION(!W1.is_null(),
                           "Got null tensor for symmetric attention");
    utils::THROW_EXCEPTION(!W2.is_null(),
                           "Got null tensor for symmetric attention");
    
    // auto input_saved = make_intrusive<tensor_holder>(input.tensor.conditional_mutate_clone());
    auto w1_saved = make_intrusive<tensor_holder>(W1.conditional_mutate_clone());
    auto w2_saved = make_intrusive<tensor_holder>(W2.tensor.conditional_mutate_clone());
    
    Tensor left = functional::matmult(w1_saved->tensor, input.tensor);
    Tensor left_out = functional::matmult(w2_saved->tensor, left);
    
    Tensor right = functional::matmult(input.tensor, w1_saved->tensor, false, true);
    Tensor right_out = functional::matmult(right, w2_saved->tensor, false, true);
    
    TensorGrad result(right_out + left_out, true);
    result.track_tensors(input, W2);
    
    auto right_saved = make_intrusive<tensor_holder>(right);
    auto left_saved = make_intrusive<tensor_holder>(left);

    // Define backward function
    result.create_backward_function(
        [](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents,
           intrusive_ptr<tensor_holder> w2_saved,
           intrusive_ptr<tensor_holder> w1_saved,
           intrusive_ptr<tensor_holder> right_saved,
           intrusive_ptr<tensor_holder> left_saved) {
            
            // const Tensor& X = input_saved->tensor;
            const Tensor& W1 = w1_saved->tensor;
            const Tensor& W2 = w2_saved->tensor;
            const Tensor& right = right_saved->tensor;
            const Tensor& left = left_saved->tensor;

            Tensor dRightMid = ::nt::functional::matmult(grad, W2, false, false);
            Tensor dWeight2Right = ::nt::functional::matmult(grad, right, true, false);
            // Tensor dWeight1Right = ::nt::functional::matmult(dRightMid, X, true, false);
            Tensor dInputRight = ::nt::functional::matmult(dRightMid, W1, false, false);

            Tensor dLeftMid = ::nt::functional::matmult(W2, grad, true, false);
            Tensor dWeight2Left = ::nt::functional::matmult(grad, left, false, true);
            // Tensor dWeight1Left = ::nt::functional::matmult(dLeftMid, X, false, true);
            Tensor dInputLeft = ::nt::functional::matmult(W1, dLeftMid, true, false);

            // Accumulate gradients
            parents[0]->grad->tensor += dInputRight + dInputLeft;
            // parents[1]->grad->tensor += dWeight1Right + dWeight1Left;
            parents[1]->grad->tensor += dWeight2Right + dWeight2Left;
        },
        w2_saved, w1_saved, right_saved, left_saved, "SymmetricBilinear"
    );


    return std::move(result);
}



//
// W2 @ (W1 @ input)
// +
// (input @ W1.T) @ W2.T
TensorGrad TensorGrad_Functional_Class::symmetric_bilinear(const TensorGrad& input, const TensorGrad& W1, const Tensor& W2){
    utils::THROW_EXCEPTION(!input.is_null(),
                           "Got null tensor for symmetric attention");
    utils::THROW_EXCEPTION(!W1.is_null(),
                           "Got null tensor for symmetric attention");
    utils::THROW_EXCEPTION(!W2.is_null(),
                           "Got null tensor for symmetric attention");
    
    if(!input.do_track_grad || !W1.do_track_grad){
        if(!input.do_track_grad) return symmetric_bilinear(input.tensor, W1, W2);
        return symmetric_bilinear(input, W1.tensor, W2);
    }

    auto input_saved = make_intrusive<tensor_holder>(input.tensor.conditional_mutate_clone());
    auto w1_saved = make_intrusive<tensor_holder>(W1.tensor.conditional_mutate_clone());
    auto w2_saved = make_intrusive<tensor_holder>(W2.conditional_mutate_clone());
    
    Tensor left = functional::matmult(w1_saved->tensor, input_saved->tensor);
    Tensor left_out = functional::matmult(w2_saved->tensor, left);
    
    Tensor right = functional::matmult(input_saved->tensor, w1_saved->tensor, false, true);
    Tensor right_out = functional::matmult(right, w2_saved->tensor, false, true);
    
    TensorGrad result(right_out + left_out, true);
    result.track_tensors(input, W1);
    
    // auto right_saved = make_intrusive<tensor_holder>(right);
    // auto left_saved = make_intrusive<tensor_holder>(left);

    // Define backward function
    result.create_backward_function(
        [](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents,
           intrusive_ptr<tensor_holder> input_saved,
           intrusive_ptr<tensor_holder> weight1_saved,
           intrusive_ptr<tensor_holder> w2_saved) {
            
            const Tensor& X = input_saved->tensor;
            const Tensor& W1 = weight1_saved->tensor;
            const Tensor& W2 = w2_saved->tensor;
            // const Tensor& right = right_saved->tensor;
            // const Tensor& left = left_saved->tensor;

            Tensor dRightMid = ::nt::functional::matmult(grad, W2, false, false);
            // Tensor dWeight2Right = ::nt::functional::matmult(grad, right, true, false);
            Tensor dWeight1Right = ::nt::functional::matmult(dRightMid, X, true, false);
            Tensor dInputRight = ::nt::functional::matmult(dRightMid, W1, false, false);

            Tensor dLeftMid = ::nt::functional::matmult(W2, grad, true, false);
            // Tensor dWeight2Left = ::nt::functional::matmult(grad, left, false, true);
            Tensor dWeight1Left = ::nt::functional::matmult(dLeftMid, X, false, true);
            Tensor dInputLeft = ::nt::functional::matmult(W1, dLeftMid, true, false);

            // Accumulate gradients
            parents[0]->grad->tensor += dInputRight + dInputLeft;
            parents[1]->grad->tensor += dWeight1Right + dWeight1Left;
            // parents[2]->grad->tensor += dWeight2Right + dWeight2Left;
        },
        input_saved, w1_saved, w2_saved, "SymmetricBilinear"
    );
    return std::move(result);
}


inline Tensor __nt__symmetric_bilinear__(const Tensor& input, const Tensor& W1, const Tensor& W2){
    Tensor O1 = ::nt::functional::matmult(W1, input);
    Tensor O2 = ::nt::functional::matmult(input, W1, false, true);
    Tensor O4 = ::nt::functional::matmult(W2, O1);
    Tensor O5 = ::nt::functional::matmult(O2, W2, false, true);
    return O5 + O4;

}

TensorGrad TensorGrad_Functional_Class::symmetric_bilinear(const Tensor& input, const Tensor& W1, const TensorGrad& W2){
    utils::THROW_EXCEPTION(!input.is_null(),
                           "Got null tensor for symmetric attention");
    utils::THROW_EXCEPTION(!W1.is_null(),
                           "Got null tensor for symmetric attention");
    utils::THROW_EXCEPTION(!W2.is_null(),
                           "Got null tensor for symmetric attention");
    if(!W2.do_track_grad){
        TensorGrad result(__nt__symmetric_bilinear__(input, W1, W2.tensor), W2.grad_required);
        result.do_track_grad = false;
        return std::move(result);
    } 
    // auto input_saved = make_intrusive<tensor_holder>(input.conditional_mutate_clone());
    // auto w1_saved = make_intrusive<tensor_holder>(W1.tensor.conditional_mutate_clone());
    // auto w2_saved = make_intrusive<tensor_holder>(W2.tensor.conditional_mutate_clone());
    
    Tensor left = functional::matmult(W1, input);
    Tensor left_out = functional::matmult(W2.tensor, left);
    
    Tensor right = functional::matmult(input, W1, false, true);
    Tensor right_out = functional::matmult(right, W2.tensor, false, true);
    
    TensorGrad result(right_out + left_out, true);
    result.track_tensors(W2);
    
    auto right_saved = make_intrusive<tensor_holder>(right);
    auto left_saved = make_intrusive<tensor_holder>(left);

    // Define backward function
    result.create_backward_function(
        [](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents,
           intrusive_ptr<tensor_holder> right_saved,
           intrusive_ptr<tensor_holder> left_saved) {
            
            // const Tensor& X = input_saved->tensor;
            // const Tensor& W1 = weight1_saved->tensor;
            // const Tensor& W2 = w2_saved->tensor;
            const Tensor& right = right_saved->tensor;
            const Tensor& left = left_saved->tensor;

            // Tensor dRightMid = ::nt::functional::matmult(grad, W2, false, false);
            Tensor dWeight2Right = ::nt::functional::matmult(grad, right, true, false);
            // Tensor dWeight1Right = ::nt::functional::matmult(dRightMid, X, true, false);
            // Tensor dInputRight = ::nt::functional::matmult(dRightMid, W1, false, false);

            // Tensor dLeftMid = ::nt::functional::matmult(W2, grad, true, false);
            Tensor dWeight2Left = ::nt::functional::matmult(grad, left, false, true);
            // Tensor dWeight1Left = ::nt::functional::matmult(dLeftMid, X, false, true);
            // Tensor dInputLeft = ::nt::functional::matmult(W1, dLeftMid, true, false);

            // Accumulate gradients
            // parents[0]->grad->tensor += dInputRight + dInputLeft;
            // parents[0]->grad->tensor += dWeight1Right + dWeight1Left;
            parents[0]->grad->tensor += dWeight2Right + dWeight2Left;
        },
        right_saved, left_saved, "SymmetricBilinear"
    );
    return std::move(result);
}



TensorGrad TensorGrad_Functional_Class::symmetric_bilinear(const Tensor& input, const TensorGrad& W1, const Tensor& W2){
    utils::THROW_EXCEPTION(!input.is_null(),
                           "Got null tensor for symmetric attention");
    utils::THROW_EXCEPTION(!W1.is_null(),
                           "Got null tensor for symmetric attention");
    utils::THROW_EXCEPTION(!W2.is_null(),
                           "Got null tensor for symmetric attention");
    
    if(!W1.do_track_grad){
        TensorGrad result(__nt__symmetric_bilinear__(input, W1.tensor, W2), W1.grad_required);
        result.do_track_grad = false;
        return std::move(result);
    } 

    auto input_saved = make_intrusive<tensor_holder>(input.conditional_mutate_clone());
    auto w2_saved = make_intrusive<tensor_holder>(W2.conditional_mutate_clone());
    
    Tensor left = functional::matmult(W1.tensor, input_saved->tensor);
    Tensor left_out = functional::matmult(w2_saved->tensor, left);
    
    Tensor right = functional::matmult(input_saved->tensor, W1.tensor, false, true);
    Tensor right_out = functional::matmult(right, w2_saved->tensor, false, true);
    
    TensorGrad result(right_out + left_out, true);
    result.track_tensors(W1);
    
    // auto right_saved = make_intrusive<tensor_holder>(right);
    // auto left_saved = make_intrusive<tensor_holder>(left);

    // Define backward function
    result.create_backward_function(
        [](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents,
           intrusive_ptr<tensor_holder> input_saved,
           intrusive_ptr<tensor_holder> w2_saved) {
            
            const Tensor& X = input_saved->tensor;
            // const Tensor& W1 = weight1_saved->tensor;
            const Tensor& W2 = w2_saved->tensor;
            // const Tensor& right = right_saved->tensor;
            // const Tensor& left = left_saved->tensor;

            Tensor dRightMid = ::nt::functional::matmult(grad, W2, false, false);
            // Tensor dWeight2Right = ::nt::functional::matmult(grad, right, true, false);
            Tensor dWeight1Right = ::nt::functional::matmult(dRightMid, X, true, false);
            // Tensor dInputRight = ::nt::functional::matmult(dRightMid, W1, false, false);

            Tensor dLeftMid = ::nt::functional::matmult(W2, grad, true, false);
            // Tensor dWeight2Left = ::nt::functional::matmult(grad, left, false, true);
            Tensor dWeight1Left = ::nt::functional::matmult(dLeftMid, X, false, true);
            // Tensor dInputLeft = ::nt::functional::matmult(W1, dLeftMid, true, false);

            // Accumulate gradients
            // parents[0]->grad->tensor += dInputRight + dInputLeft;
            parents[0]->grad->tensor += dWeight1Right + dWeight1Left;
            // parents[2]->grad->tensor += dWeight2Right + dWeight2Left;
        },
        input_saved, w2_saved, "SymmetricBilinear"
    );
    return std::move(result);
}



TensorGrad TensorGrad_Functional_Class::symmetric_bilinear(const TensorGrad& input, const Tensor& W1, const Tensor& W2){
    utils::THROW_EXCEPTION(!input.is_null(),
                           "Got null tensor for symmetric attention");
    utils::THROW_EXCEPTION(!W1.is_null(),
                           "Got null tensor for symmetric attention");
    utils::THROW_EXCEPTION(!W2.is_null(),
                           "Got null tensor for symmetric attention");
    
    if(!input.do_track_grad){
        TensorGrad result(__nt__symmetric_bilinear__(input.tensor, W1, W2), input.grad_required);
        result.do_track_grad = false;
        return std::move(result);
    }

    // auto input_saved = make_intrusive<tensor_holder>(input.tensor.conditional_mutate_clone());
    auto w1_saved = make_intrusive<tensor_holder>(W1.conditional_mutate_clone());
    auto w2_saved = make_intrusive<tensor_holder>(W2.conditional_mutate_clone());
    
    Tensor left = functional::matmult(w1_saved->tensor, input.tensor);
    Tensor left_out = functional::matmult(w2_saved->tensor, left);
    
    Tensor right = functional::matmult(input.tensor, w1_saved->tensor, false, true);
    Tensor right_out = functional::matmult(right, w2_saved->tensor, false, true);
    
    TensorGrad result(right_out + left_out, true);
    result.track_tensors(input);
    
    // auto right_saved = make_intrusive<tensor_holder>(right);
    // auto left_saved = make_intrusive<tensor_holder>(left);

    // Define backward function
    result.create_backward_function(
        [](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents,
           // intrusive_ptr<tensor_holder> input_saved,
           intrusive_ptr<tensor_holder> weight1_saved,
           intrusive_ptr<tensor_holder> w2_saved) {
            
            // const Tensor& X = input_saved->tensor;
            const Tensor& W1 = weight1_saved->tensor;
            const Tensor& W2 = w2_saved->tensor;
            // const Tensor& right = right_saved->tensor;
            // const Tensor& left = left_saved->tensor;

            Tensor dRightMid = ::nt::functional::matmult(grad, W2, false, false);
            // Tensor dWeight2Right = ::nt::functional::matmult(grad, right, true, false);
            // Tensor dWeight1Right = ::nt::functional::matmult(dRightMid, X, true, false);
            Tensor dInputRight = ::nt::functional::matmult(dRightMid, W1, false, false);

            Tensor dLeftMid = ::nt::functional::matmult(W2, grad, true, false);
            // Tensor dWeight2Left = ::nt::functional::matmult(grad, left, false, true);
            // Tensor dWeight1Left = ::nt::functional::matmult(dLeftMid, X, false, true);
            Tensor dInputLeft = ::nt::functional::matmult(W1, dLeftMid, true, false);

            // Accumulate gradients
            parents[0]->grad->tensor += dInputRight + dInputLeft;
            // parents[1]->grad->tensor += dWeight1Right + dWeight1Left;
            // parents[2]->grad->tensor += dWeight2Right + dWeight2Left;
        },
        w1_saved, w2_saved, "SymmetricBilinear"
    );
    return std::move(result);
}

////(1/2)[WA(W^T) + (W^T)(A)(W)]
//TensorGrad TensorGrad_Functional_Class::symmetric_bilinear(const TensorGrad& input, const TensorGrad& weight){
//    if(!input.do_track_grad || !weight.do_track_grad){
//        if(!input.do_track_grad){
//            return symmetric_bilinear(input.tensor, weight);
//        }
//        return symmetric_bilinear(input, weight.tensor);
//    }
    
//    // Clone inputs for backward
//    auto input_saved = make_intrusive<tensor_holder>(input.tensor.conditional_mutate_clone());
//    auto weight_saved = make_intrusive<tensor_holder>(weight.tensor.conditional_mutate_clone());


//    Tensor left = ::nt::functional::matmult(weight_saved->tensor, input_saved->tensor);
//    Tensor left_out = ::nt::functional::matmult(left, weight_saved->tensor, false, true);

//    Tensor right = ::nt::functional::matmult(weight_saved->tensor, input_saved->tensor, true, false);
//    Tensor right_out = ::nt::functional::matmult(right, weight_saved->tensor);

//    Tensor symmetric_out = (0.5) * (left_out + right_out);


//    auto right_saved = make_intrusive<tensor_holder>(right);
//    auto left_saved = make_intrusive<tensor_holder>(left);

//    TensorGrad result(symmetric_out, true);
//    result.track_tensors(input, weight);

//    // Define backward function
//    result.create_backward_function(
//        [](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents,
//           intrusive_ptr<tensor_holder> input_saved,
//           intrusive_ptr<tensor_holder> weight_saved,
//           intrusive_ptr<tensor_holder> right_saved,
//           intrusive_ptr<tensor_holder> left_saved) {
            
//            const Tensor& X = input_saved->tensor;
//            const Tensor& W = weight_saved->tensor;
//            const Tensor& right = right_saved->tensor;
//            const Tensor& left = left_saved->tensor;

//            Tensor grad_sym = grad * 0.5;

//            // Right path: W^T X W
//            Tensor dRightMid = ::nt::functional::matmult(grad_sym, W, false, true);             // dL/d(right)
//            Tensor dWeightRightOut = ::nt::functional::matmult(right, grad_sym, true, false);   // dL/d(W) from second mult
//            Tensor dWeightRightFirst = ::nt::functional::matmult(X, dRightMid, false, true);    // dL/d(W) from first mult
//            Tensor dInputRight = ::nt::functional::matmult(W, dRightMid, false, false);          // dL/d(X) from right path

//            // Left path: W X W^T
//            Tensor dLeftMid = ::nt::functional::matmult(grad_sym, W, false, true);              // dL/d(left)
//            Tensor dWeightLeftOut = ::nt::functional::matmult(grad_sym, left, false, false);    // dL/d(W) from second mult
//            Tensor dWeightLeftFirst = ::nt::functional::matmult(dLeftMid, X, false, true);      // dL/d(W) from first mult
//            Tensor dInputLeft = ::nt::functional::matmult(W, grad_sym, true, false);            // dL/d(X) from left path

//            // Accumulate gradients
//            parents[0]->grad->tensor += dInputRight + dInputLeft;
//            parents[1]->grad->tensor += dWeightRightOut + dWeightRightFirst + dWeightLeftOut + dWeightLeftFirst;
//        },
//        input_saved, weight_saved, right_saved, left_saved, "SymmetricBilinear"
//    );
//    return std::move(result);
//}

//inline Tensor __nt__symmetric_bilinear__(const Tensor& input, const Tensor& weight){
//    return (0.5) * (::nt::functional::matmult(::nt::functional::matmult(weight, input), weight, false, true)
//                    + ::nt::functional::matmult(::nt::functional::matmult(weight, input, true, false), weight));
//}

//TensorGrad TensorGrad_Functional_Class::symmetric_bilinear(const TensorGrad& input, const Tensor& weight){
//    if (!input.do_track_grad) {
//        TensorGrad result(__nt__symmetric_bilinear__(input.tensor, weight), input.grad_required);
//        result.do_track_grad = false;
//        return std::move(result);
//    }
//    // Clone inputs for backward
//    auto input_saved = make_intrusive<tensor_holder>(input.tensor.conditional_mutate_clone());
//    auto weight_saved = make_intrusive<tensor_holder>(weight.conditional_mutate_clone());


//    Tensor left = ::nt::functional::matmult(weight_saved->tensor, input_saved->tensor);
//    Tensor left_out = ::nt::functional::matmult(left, weight_saved->tensor, false, true);

//    Tensor right = ::nt::functional::matmult(weight_saved->tensor, input_saved->tensor, true, false);
//    Tensor right_out = ::nt::functional::matmult(right, weight_saved->tensor);

//    Tensor symmetric_out = (0.5) * (left_out + right_out);


//    auto right_saved = make_intrusive<tensor_holder>(right);
//    auto left_saved = make_intrusive<tensor_holder>(left);

//    TensorGrad result(symmetric_out, true);
//    result.track_tensors(input);

//    // Define backward function
//    result.create_backward_function(
//        [](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents,
//           /*intrusive_ptr<tensor_holder> input_saved,*/
//           intrusive_ptr<tensor_holder> weight_saved,
//           intrusive_ptr<tensor_holder> right_saved,
//           intrusive_ptr<tensor_holder> left_saved) {
            
//            //const Tensor& X = input_saved->tensor;
//            const Tensor& W = weight_saved->tensor;
//            const Tensor& right = right_saved->tensor;
//            const Tensor& left = left_saved->tensor;

//            Tensor grad_sym = grad * 0.5;

//            // Right path: W^T X W
//            Tensor dRightMid = ::nt::functional::matmult(grad_sym, W, false, true);             // dL/d(right)
//            //Tensor dWeightRightOut = ::nt::functional::matmult(right, grad_sym, true, false);   // dL/d(W) from second mult
//            //Tensor dWeightRightFirst = ::nt::functional::matmult(X, dRightMid, false, true);    // dL/d(W) from first mult
//            Tensor dInputRight = ::nt::functional::matmult(W, dRightMid, false, false);          // dL/d(X) from right path

//            // Left path: W X W^T
//            Tensor dLeftMid = ::nt::functional::matmult(grad_sym, W, false, true);              // dL/d(left)
//            //Tensor dWeightLeftOut = ::nt::functional::matmult(grad_sym, left, false, false);    // dL/d(W) from second mult
//            //Tensor dWeightLeftFirst = ::nt::functional::matmult(dLeftMid, X, false, true);      // dL/d(W) from first mult
//            Tensor dInputLeft = ::nt::functional::matmult(W, grad_sym, true, false);            // dL/d(X) from left path

//            // Accumulate gradients
//            parents[0]->grad->tensor += dInputRight + dInputLeft;
//            //parents[1]->grad->tensor += dWeightRightOut + dWeightRightFirst + dWeightLeftOut + dWeightLeftFirst;
//        },
//        /*input_saved,*/ weight_saved, right_saved, left_saved, "SymmetricBilinear"
//    );
//    return std::move(result);
//}

//TensorGrad TensorGrad_Functional_Class::symmetric_bilinear(const Tensor& input, const TensorGrad& weight){
//   if (!weight.do_track_grad) {
//        TensorGrad result(__nt__symmetric_bilinear__(input, weight.tensor), weight.grad_required);
//        result.do_track_grad = false;
//        return std::move(result);
//    }

//    // Clone inputs for backward
//    auto input_saved = make_intrusive<tensor_holder>(input.conditional_mutate_clone());
//    auto weight_saved = make_intrusive<tensor_holder>(weight.tensor.conditional_mutate_clone());


//    Tensor left = ::nt::functional::matmult(weight_saved->tensor, input_saved->tensor);
//    Tensor left_out = ::nt::functional::matmult(left, weight_saved->tensor, false, true);

//    Tensor right = ::nt::functional::matmult(weight_saved->tensor, input_saved->tensor, true, false);
//    Tensor right_out = ::nt::functional::matmult(right, weight_saved->tensor);

//    Tensor symmetric_out = (0.5) * (left_out + right_out);


//    auto right_saved = make_intrusive<tensor_holder>(right);
//    auto left_saved = make_intrusive<tensor_holder>(left);

//    TensorGrad result(symmetric_out, true);
//    result.track_tensors(weight);

//    // Define backward function
//    result.create_backward_function(
//        [](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents,
//           intrusive_ptr<tensor_holder> input_saved,
//           intrusive_ptr<tensor_holder> weight_saved,
//           intrusive_ptr<tensor_holder> right_saved,
//           intrusive_ptr<tensor_holder> left_saved) {
            
//            const Tensor& X = input_saved->tensor;
//            const Tensor& W = weight_saved->tensor;
//            const Tensor& right = right_saved->tensor;
//            const Tensor& left = left_saved->tensor;

//            Tensor grad_sym = grad * 0.5;

//            // Right path: W^T X W
//            Tensor dRightMid = ::nt::functional::matmult(grad_sym, W, false, true);             // dL/d(right)
//            Tensor dWeightRightOut = ::nt::functional::matmult(right, grad_sym, true, false);   // dL/d(W) from second mult
//            Tensor dWeightRightFirst = ::nt::functional::matmult(X, dRightMid, false, true);    // dL/d(W) from first mult
//            //Tensor dInputRight = ::nt::functional::matmult(W, dRightMid, false, false);          // dL/d(X) from right path

//            // Left path: W X W^T
//            Tensor dLeftMid = ::nt::functional::matmult(grad_sym, W, false, true);              // dL/d(left)
//            Tensor dWeightLeftOut = ::nt::functional::matmult(grad_sym, left, false, false);    // dL/d(W) from second mult
//            Tensor dWeightLeftFirst = ::nt::functional::matmult(dLeftMid, X, false, true);      // dL/d(W) from first mult
//            //Tensor dInputLeft = ::nt::functional::matmult(W, grad_sym, true, false);            // dL/d(X) from left path

//            // Accumulate gradients
//            //parents[0]->grad->tensor += dInputRight + dInputLeft;
//            parents[0]->grad->tensor += dWeightRightOut + dWeightRightFirst + dWeightLeftOut + dWeightLeftFirst;
//        },
//        input_saved, weight_saved, right_saved, left_saved, "SymmetricBilinear"
//    );
//    return std::move(result);

//}

////(1/2)[WA(W^T) + (W)(A^T)(W^T)] + B
//TensorGrad TensorGrad_Functional_Class::symmetric_bilinear_bias(const TensorGrad& input, const TensorGrad& weight, const TensorGrad& bias){
//    utils::throw_exception(bias.dims() == 1,
//                           "Expected bias for symmetric bilinear bias to only have 1 dim got shape $", bias.shape());
//    utils::throw_exception(input.dims() >= 2,
//                           "Expected input for symmetric bilinear bias to have at least 2 dimensions got shape for symmetric bilinear",
//                           input.shape());
//    utils::throw_exception(input.shape()[-1] == input.shape()[-2],
//                           "Expected input to be a square matrix but got shape $ for symmetric bilinear", input.shape());
//    utils::throw_exception(weight.dims() >= 2,
//                           "Expected weight for symmetric bilinear bias to have at least 2 dimensions got shape for symmetric bilinear", 
//                           weight.shape());
//    utils::throw_exception(weight.shape()[-1] == input.shape()[-2],
//                           "Expected weight to be a square matrix but got shape $ for symmetric bilinear", weight.shape());
//    utils::throw_exception(input.shape()[-1] == weight.shape()[-1] && input.shape()[-2] == weight.shape()[-2],
//                           "Expected input shape ($) at last 2 dimensions to equal weight shape ($) at last 2 dimensions for symmetric bilinear", 
//                           input.shape(), weight.shape());
//    utils::throw_exception(bias.shape()[0] == input.shape()[-1],
//                           "Expected the number of elements in bias ($) to be equal to the number of rows and columns in the input ($) for "
//                           "symmetric bilinear", bias.shape(), input.shape());
//    TensorGrad B = bias.view(-1, 1) * bias.view(1, -1);
//    return symmetric_bilinear(input, weight) + B; 
//}


TensorGrad TensorGrad_Functional_Class::unfold3d(
        const TensorGrad &x, utils::my_n_tuple<3> kernel_size,
        utils::my_n_tuple<3> dilation, utils::my_n_tuple<3> padding,
        utils::my_n_tuple<3> stride, bool transpose_out) {
    TensorGrad result(::nt::functional::unfold3d(x.tensor, kernel_size, dilation, padding, stride,
                                                         transpose_out), x.grad_required);
    result.track_tensors(x);
    result.create_backward_function(
            [kernel_size, dilation, padding, stride, transpose_out](
                    const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents) {
                utils::my_n_tuple<3> output_size(parents[0]->grad->tensor.shape()[-3],
                                                                                 parents[0]->grad->tensor.shape()[-2],
                                                                                 parents[0]->grad->tensor.shape()[-1]);
                ::nt::functional::unfold3d_backward(grad, parents[0]->grad->tensor, output_size,
                                                    kernel_size, dilation, padding, stride,
                                                    transpose_out);
            });
    return std::move(result);
}

TensorGrad TensorGrad_Functional_Class::unfold1d(
        const TensorGrad &x, Tensor::size_value_t kernel_size,
        Tensor::size_value_t dilation, Tensor::size_value_t padding,
        Tensor::size_value_t stride, bool transpose_out) {
    TensorGrad result(::nt::functional::unfold1d(x.tensor, kernel_size, dilation, padding, stride,
                                                         transpose_out), x.grad_required);
    result.track_tensors(x);
    result.create_backward_function(
            [kernel_size, dilation, padding, stride, transpose_out](
                    const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents) {
                Tensor::size_value_t output_size = parents[0]->grad->tensor.shape()[-1];
                ::nt::functional::unfold1d_backward(grad, parents[0]->grad->tensor, output_size,
                                                    kernel_size, dilation, padding, stride,
                                                    transpose_out);
            });
    return std::move(result);
}

TensorGrad TensorGrad_Functional_Class::fold(const TensorGrad &x,
                                             utils::my_tuple output_size,
                                             utils::my_tuple kernel_size,
                                             utils::my_tuple dilation,
                                             utils::my_tuple padding,
                                             utils::my_tuple stride) {
    TensorGrad result(
            ::nt::functional::fold(x.tensor, output_size, kernel_size, dilation, padding, stride), x.grad_required);
    result.track_tensors(x);
    // it is coppied because the backward pass will go out of scope of this
    // function and so I dont want that memory to try to be referenced
    result.create_backward_function(
            [output_size, kernel_size, dilation, padding, stride](
                    const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents) {
                ::nt::functional::fold_backward(grad, parents[0]->grad->tensor, output_size, kernel_size,
                                            dilation, padding, stride);
            });
    return std::move(result);
}

TensorGrad TensorGrad_Functional_Class::unfold(
        const TensorGrad &x, utils::my_tuple kernel_size, utils::my_tuple dilation,
        utils::my_tuple padding, utils::my_tuple stride, bool transpose_out) {
    TensorGrad result(
            ::nt::functional::unfold(x.tensor, kernel_size, dilation, padding, stride, transpose_out), x.grad_required);
    result.track_tensors(x);
    result.create_backward_function(
            [kernel_size, dilation, padding, stride, transpose_out](
                    const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents) {
                utils::my_tuple output_size(parents[0]->grad->tensor.shape()[-2],
                                                                        parents[0]->grad->tensor.shape()[-1]);
                ::nt::functional::unfold_backward(grad, parents[0]->grad->tensor, output_size,
                                                kernel_size, dilation, padding, stride, transpose_out);
            });
    return std::move(result);
}

TensorGrad TensorGrad_Functional_Class::sigmoid(const TensorGrad &x) {
    Tensor a = ::nt::functional::sigmoid(x.tensor);
    intrusive_ptr<tensor_holder> sigmoid_x =
            make_intrusive<tensor_holder>(a.conditional_mutate_clone());
    TensorGrad result(std::move(a), x.grad_required);
    result.track_tensors(x);
    result.create_backward_function(
            [](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents,
                 intrusive_ptr<tensor_holder> x) {
                parents[0]->grad->tensor += grad * ::nt::functional::dsigmoid(x->tensor, false);
            },
            sigmoid_x);
    return std::move(result);
}





TensorGrad TensorGrad_Functional_Class::clamp(const TensorGrad &x,
                                                std::optional<int64_t> min,
                                                std::optional<int64_t> max) {
    TensorGrad out = x.clone();
    if (min && max) {
        out[out < min.value() && out > max.value()] = 0;
        return std::move(out);
    } else if (min)
        out[out < min.value()] = 0;
    else if (max)
        out[out > max.value()] = 0;
    return std::move(out);
}

TensorGrad TensorGrad_Functional_Class::relu(const TensorGrad &x) {
    return clamp(x, 0, std::nullopt);
}

TensorGrad TensorGrad_Functional_Class::var(const TensorGrad &x,
                                            utils::optional_list dim,
                                            int64_t correction, bool keepdim) {
    if (!x.do_track_grad) {
        Tensor out = ::nt::functional::var(x.tensor, dim, correction, keepdim);
        TensorGrad result(std::move(out), x.grad_required);
        result.do_track_grad = false;
        return std::move(result);
    }
    intrusive_ptr<tensor_holder> x_c =
            make_intrusive<tensor_holder>(x.tensor.conditional_mutate_clone());
    TensorGrad result(
            ::nt::functional::var(x_c->tensor, dim, correction, keepdim), x.grad_required);
    result.track_tensors(x);
    result.create_backward_function(
            [dim, correction](const Tensor &grad,
                                                std::vector<intrusive_ptr<TensorGrad>> &parents,
                                                intrusive_ptr<tensor_holder> x) {
                parents[0]->grad->tensor += ::nt::functional::dvar(grad, x->tensor, dim, correction);
            },
            x_c);
    return std::move(result);
}


Tensor cat_vec(std::vector<TensorGrad> &tgs) {
    const typename SizeRef::value_type &num = tgs.size();
    auto begin = tgs.begin();
    auto end = tgs.end();
    const SizeRef sh = begin->shape();
    const SizeRef sh_smaller = sh.pop_front();
    int64_t n_dim_size = sh[0];
    auto begin_cpy = begin;
    ++begin;
    for (; begin != end; ++begin) {
        n_dim_size += begin->shape()[0];
        utils::THROW_EXCEPTION(begin->shape().pop_front() == sh_smaller,
                                                     "Expected all shapes in concatenate to be the "
                                                     "same, but got $ and $",
                                                     begin->shape().pop_front(), sh_smaller);
    }
    std::vector<typename SizeRef::value_type> vec = sh.Vec();
    vec[0] = n_dim_size;
    std::vector<std::reference_wrapper<const ArrayVoid>> arrVds;
    arrVds.reserve(num); // okay because it is allocating a reference wrapper,
                                             // putting a number there would cause an allocation error
    begin = begin_cpy;
    typename SizeRef::value_type i = 0;
    for (typename SizeRef::value_type i = 0; begin != end; ++begin, ++i) {
        arrVds.push_back(std::cref(begin->tensor.arr_void()));
    }
    return Tensor(ArrayVoid::cat(arrVds), SizeRef(std::move(vec)));
}

Tensor cat_vec(std::vector<TensorGrad> &tgs, int64_t dim) {

    if (dim == 0) {
        return cat_vec(tgs);
    }
    const typename SizeRef::value_type &num = tgs.size();
    auto begin = tgs.begin();
    auto end = tgs.end();
    const SizeRef sh = begin->shape().transpose(0, dim);
    int64_t n_dim_size = sh[0];
    const SizeRef sh_smaller = sh.pop_front();
    auto begin_cpy = begin;
    ++begin;
    for (; begin != end; ++begin) {
        n_dim_size += begin->shape()[dim];
        utils::THROW_EXCEPTION(begin->shape().transpose(0, dim).pop_front() ==
                                                             sh_smaller,
                                                     "Expected all shapes in concatenate to be the "
                                                     "same, but got $ and $",
                                                     begin->shape(), sh);
    }
    std::vector<typename SizeRef::value_type> vec = sh.Vec();
    vec[0] = n_dim_size;
    std::vector<ArrayVoid> arrVds;
    //arrVds.reserve(num); // okay because it is allocating a reference wrapper,
    // putting a number there would cause an allocation error
    begin = begin_cpy;
    typename SizeRef::value_type i = 0;
    for (typename SizeRef::value_type i = 0; begin != end; ++begin, ++i) {
        arrVds.push_back(begin->tensor.transpose(0, dim).arr_void());
    }
    SizeRef shape(std::move(vec));
    return Tensor(ArrayVoid::cat(arrVds), std::move(shape)).transpose(0, dim);
}

Tensor cat_vec_grad(std::vector<intrusive_ptr<TensorGrad>> &tgs) {
    const typename SizeRef::value_type &num = tgs.size();
    auto begin = tgs.begin();
    auto end = tgs.end();
    const SizeRef sh = (*begin)->shape();
    const SizeRef sh_smaller = sh.pop_front();
    int64_t n_dim_size = sh[0];
    auto begin_cpy = begin;
    ++begin;
    for (; begin != end; ++begin) {
        n_dim_size += (*begin)->shape()[0];
        utils::THROW_EXCEPTION((*begin)->shape().pop_front() == sh_smaller,
                                                     "Expected all shapes in concatenate to be the "
                                                     "same, but got $ and $",
                                                     (*begin)->shape().pop_front(), sh_smaller);
    }
    std::vector<typename SizeRef::value_type> vec = sh.Vec();
    vec[0] = n_dim_size;
    std::vector<std::reference_wrapper<const ArrayVoid>> arrVds;
    arrVds.reserve(num); // okay because it is allocating a reference wrapper,
                                             // putting a number there would cause an allocation error
    begin = begin_cpy;
    typename SizeRef::value_type i = 0;
    for (typename SizeRef::value_type i = 0; begin != end; ++begin, ++i) {
        arrVds.push_back(std::cref((*begin)->grad->tensor.arr_void()));
    }
    return Tensor(ArrayVoid::cat(arrVds), SizeRef(std::move(vec)));
}

Tensor cat_vec_grad(std::vector<intrusive_ptr<TensorGrad>> &tgs, int64_t dim) {
    if (dim == 0) {
        return cat_vec_grad(tgs);
    }
    const typename SizeRef::value_type &num = tgs.size();
    auto begin = tgs.begin();
    auto end = tgs.end();
    const SizeRef sh = (*begin)->shape().transpose(0, dim);
    int64_t n_dim_size = sh[0];
    const SizeRef sh_smaller = sh.pop_front();
    auto begin_cpy = begin;
    ++begin;
    for (; begin != end; ++begin) {
        n_dim_size += (*begin)->shape()[dim];
        utils::THROW_EXCEPTION(
                (*begin)->shape().transpose(0, dim).pop_front() == sh_smaller,
                "Expected all shapes in concatenate to be the same, but got $ and "
                "$",
                (*begin)->shape(), sh);
    }
    std::vector<typename SizeRef::value_type> vec = sh.Vec();
    vec[0] = n_dim_size;
    std::vector<ArrayVoid> arrVds;
    //arrVds.reserve(num); // okay because it is allocating a reference wrapper,
                                             // putting a number there would cause an allocation error
    begin = begin_cpy;
    // typename SizeRef::value_type i = 0;
    for (;begin != end; ++begin) {
        arrVds.push_back((*begin)->grad->tensor.transpose(0, dim).arr_void());
    }
    SizeRef shape(std::move(vec));
    return Tensor(ArrayVoid::cat(arrVds), std::move(shape)).transpose(0, dim);
}

TensorGrad TensorGrad_Functional_Class::cat(std::vector<TensorGrad> tgs, int64_t dim) {
    bool track_grad = tgs[0].do_track_grad;
    bool require_grad = tgs[0].grad_required;
    for (const auto &tg : tgs) {
        utils::throw_exception(tg.do_track_grad == track_grad,
                                                     "Cannot concatenate tensors that are both tracking "
                                                     "the gradient and are not");
        utils::throw_exception(tg.grad_required == require_grad,
                                                     "Cannot concatenate tensors that are both tracking "
                                                     "the gradient and are not");
        utils::throw_exception(!tg.is_null(), "Cannot concatenate null tensors");
    }
    if (!require_grad) {
        track_grad = false;
    }
    TensorGrad result(cat_vec(tgs, dim), require_grad);
    if (!track_grad) {
        result.do_track_grad = false;
        return std::move(result);
    }

    // tracking the gradient itself
    // rather than tracking each parent individually
    for (const auto &tg : tgs) {
        if (tg.grad == nullptr) {
            tg.grad =
                    make_intrusive<tensor_holder>(::nt::functional::zeros_like(tg.tensor));
        }
    }
    result.track_tensors(tgs);
    result.grad = make_intrusive<tensor_holder>(cat_vec_grad(result.parents->get(), dim));
    return std::move(result);
}

// inline std::vector<Tensor> vectorize(Tensor& t){
//     utils::throw_exception(t.dtype == DType::TensorObj,
//                            "can only vectorize tensor of tensors");
//     return t.arr_void().cexecute_function<WRAP_DTYPES<DTypeEnum<DType::TensorObj > > >
//         ([](auto begin, auto end) -> std::vector<Tensor> {return std::vector<Tensor>(begin, end);});

// }

TensorGrad TensorGrad_Functional_Class::cat(TensorGrad tgs, int64_t dim) {
    // std::cout << tgs << std::endl;
    // std::cout << "on dim "<<dim<<std::endl;
    // std::vector<std::reference_wrapper<Tensor>> first_cat;
    // first_cat.reserve(tgs.numel());
    // Tensor* begin = reinterpret_cast<Tensor*>(tgs.tensor.data_ptr());
    // Tensor* end = begin + tgs.numel();
    // for(;begin != end; ++begin)
    //     first_cat.push_back(std::ref(*begin));

    // for(int64_t i = 0; i < tgs.numel(); ++i)
    //     first_cat[i] = tgs[i].tensor;
    TensorGrad result(::nt::functional::cat(tgs.tensor, dim), tgs.grad_required);

    // if(tgs.grad == nullptr){
    //     std::cout << tgs.tensor.dtype << std::endl;
    //     Tensor zeros = ::nt::functional::zeros_like(tgs.tensor);
    //     std::cout << "zeros: "<<zeros<<std::endl;
    // }
    // else{
    //     std::cout << "tgs.grad is not nullptr"<<std::endl;
    //     std::cout << tgs.grad->tensor << std::endl;
    //     std::cout << ::nt::functional::zeros_like(tgs.tensor);
    // }
    result.track_grad(tgs, [dim](Tensor &grad) {return ::nt::functional::cat(grad, dim); });
    return std::move(result);
}


TensorGrad TensorGrad_Functional_Class::stack(std::vector<TensorGrad> tgs, int64_t dim) {
    bool track_grad = tgs[0].do_track_grad;
    bool require_grad = tgs[0].grad_required;
    for (const auto &tg : tgs) {
        utils::throw_exception(tg.do_track_grad == track_grad,
                                                     "Cannot concatenate tensors that are both tracking "
                                                     "the gradient and are not");
        utils::throw_exception(tg.grad_required == require_grad,
                                                     "Cannot concatenate tensors that are both tracking "
                                                     "the gradient and are not");
        utils::throw_exception(!tg.is_null(), "Cannot concatenate null tensors");
    }
    if (!require_grad) {
        track_grad = false;
    }
    std::vector<std::reference_wrapper<Tensor>> tgs_data_ref;
    tgs_data_ref.reserve(tgs.size());
    for (int64_t i = 0; i < tgs.size(); ++i) {
        tgs_data_ref.push_back(std::ref(tgs[i].tensor));
    }

    TensorGrad result(::nt::functional::stack(tgs_data_ref, dim), require_grad);
    if (!track_grad) {
        result.do_track_grad = false;
        return std::move(result);
    }
    std::vector<std::reference_wrapper<Tensor>> tgs_grad_ref;
    tgs_grad_ref.reserve(tgs.size());
    for (const auto &tg : tgs) {
        if (tg.grad == nullptr) {
            tg.grad =
                    make_intrusive<tensor_holder>(functional::zeros_like(tg.tensor));
        }
        // result.parents.push_back(make_intrusive<TensorGrad>(tg));
        tgs_grad_ref.push_back(std::ref(tg.grad->tensor));
    }
    result.grad = make_intrusive<tensor_holder>(::nt::functional::stack(tgs_grad_ref, dim));
    result.track_tensors(tgs);
    return std::move(result);
}

TensorGrad TensorGrad_Functional_Class::stack(TensorGrad tgs, int64_t dim) {
    TensorGrad result(::nt::functional::stack(tgs.tensor, dim), tgs.grad_required);
    result.track_grad(tgs, [dim](Tensor &grad) { return ::nt::functional::stack(grad, dim); });
    return std::move(result);
}



TensorGrad TensorGrad_Functional_Class::split(
        TensorGrad input, std::vector<typename Tensor::size_value_t> splits, int64_t dim) {
    TensorGrad result(::nt::functional::split(input.tensor, splits, dim), input.grad_required);
    result.track_grad(
            input, [splits, dim](Tensor &grad) { return ::nt::functional::split(grad, splits, dim); });
    return std::move(result);
}

TensorGrad TensorGrad_Functional_Class::split(
        TensorGrad input, typename Tensor::size_value_t splits, int64_t dim) {
    TensorGrad result(::nt::functional::split(input.tensor, splits, dim), input.grad_required);
    result.track_grad(
            input, [splits, dim](Tensor &grad) { return ::nt::functional::split(grad, splits, dim); });
    return std::move(result);
}

TensorGrad TensorGrad_Functional_Class::chunk(
        TensorGrad input, typename Tensor::size_value_t chunks, int64_t dim) {
    TensorGrad result(::nt::functional::chunk(input.tensor, chunks, dim), input.grad_required);
    result.track_grad(
            input, [chunks, dim](Tensor &grad) { return ::nt::functional::chunk(grad, chunks, dim); });
    return std::move(result);
}

TensorGrad TensorGrad_Functional_Class::logsumexp(const TensorGrad& x, utils::optional_list list, bool keepdim){
    TensorGrad result(::nt::functional::logsumexp(x.tensor), x.grad_required);
    if (!x.do_track_grad) {
        result.do_track_grad = false;
        return std::move(result);
    }
    
    result.track_tensors(x);
    result.create_backward_function(
            [list](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents,
                 intrusive_ptr<tensor_holder> saved_x) {
                parents[0]->grad->tensor += ::nt::functional::dlogsumexp(grad, saved_x->tensor, list);
            },
            make_intrusive<tensor_holder>(x.tensor.conditional_mutate_clone()));
    return std::move(result);
}

TensorGrad TensorGrad_Functional_Class::softplus(const TensorGrad &x,
                                                     Scalar beta,
                                                     Scalar threshold) {
    Tensor softplus_x = x.tensor * beta;

    Tensor where = softplus_x < threshold;
    if (!::nt::functional::any(where)) {
        return x;
    }

    softplus_x[where].set_(::nt::functional::log(1 + std::exp(softplus_x[where])).divide_(beta));
    TensorGrad result(std::move(softplus_x), x.grad_required);
    if (!x.do_track_grad) {
        result.do_track_grad = false;
        return std::move(result);
    }
    intrusive_ptr<tensor_holder> sx_c = make_intrusive<tensor_holder>(x.tensor.conditional_mutate_clone());
    intrusive_ptr<tensor_holder> wx_c = make_intrusive<tensor_holder>(where);

    result.track_tensors(x);
    result.create_backward_function(
            [beta](
                    const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents,
                    intrusive_ptr<tensor_holder> x, intrusive_ptr<tensor_holder> where) {
                Tensor x_w = x->tensor[where];
                Tensor grad_w = grad[where];
                x_w *= -beta;
                x_w.exp_();
                x_w += 1;
                x_w.inverse_();
                grad_w *= x_w;
                parents[0]->grad->tensor += grad;
            },
            sx_c, wx_c);
    return std::move(result);
}


TensorGrad TensorGrad_Functional_Class::softmax(const TensorGrad& inputs, bool stable){
    Tensor softmax_x = stable ? ::nt::functional::softmax_stable(inputs.tensor) : ::nt::functional::softmax(inputs.tensor);
    intrusive_ptr<tensor_holder> sx_c = make_intrusive<tensor_holder>(softmax_x.conditional_mutate_clone());
    TensorGrad result(std::move(softmax_x));
    result.track_tensors(inputs);
    result.create_backward_function(
        [](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
           intrusive_ptr<tensor_holder> sx){
            parents[0]->grad->tensor = ::nt::functional::dsoftmax(grad, sx->tensor); 
        },
    sx_c);
    return std::move(result);

}

TensorGrad TensorGrad_Functional_Class::softmax(const TensorGrad& inputs, typename SizeRef::value_type dim, bool stable){
    Tensor softmax_x = stable ? ::nt::functional::softmax_stable(inputs.tensor, dim) : 
        ::nt::functional::softmax(inputs.tensor, dim);
    intrusive_ptr<tensor_holder> sx_c = make_intrusive<tensor_holder>(softmax_x.conditional_mutate_clone());
    TensorGrad result(std::move(softmax_x));
    result.track_tensors(inputs);
    result.create_backward_function(
        [dim](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
           intrusive_ptr<tensor_holder> sx){
            parents[0]->grad->tensor = ::nt::functional::dsoftmax(grad, sx->tensor, dim); 
        },
    sx_c);
    return std::move(result);

}

inline Tensor sample_gumbel_noise(const TensorGrad& logits){
    Tensor u = ::nt::functional::rand(0, 1, logits.shape(), logits.dtype); // Uniform (0, 1)
    // return u;
    return -::nt::functional::log(-::nt::functional::log(u + 1e-10)+1e-10);      // Gumbel (0, 1)
}

TensorGrad TensorGrad_Functional_Class::gumbel_softmax(const TensorGrad & logits, Scalar tau, bool hard, bool stable){
    Tensor gumbel_noise = sample_gumbel_noise(logits).clip_(-3, 3);
    // std::cout << "max: "<<gumbel_noise.max().values.toScalar() << std::endl;
    // TensorGrad y = ((logits + gumbel_noise) / tau);
    // Tensor gumbel_noise = ::nt::functional::zeros_like(logits.tensor) + 3;
    TensorGrad y = (logits + gumbel_noise) / tau;
    // std::cout << y << std::endl;
    y = softmax(y, -1, stable);
    if(hard){
        // Straight-through: make y_hard one-hot
        Tensor y_hard = ::nt::functional::one_hot(::nt::functional::argmax(y.tensor, -1), y.shape()[-1]).to(y.dtype);
        // Use straight-through estimator
        return (y_hard - y.tensor) + y;
    }
    return std::move(y);
}


TensorGrad TensorGrad_Functional_Class::dropout(const TensorGrad &inputs, double p) {
    Tensor bools = ::nt::functional::randbools(inputs.shape(), p);
    Tensor out = inputs.tensor.conditional_mutate_clone();
    out[bools] = 0;
    TensorGrad result(out, inputs.grad_required);
    if (!inputs.do_track_grad) {
        result.do_track_grad = false;
        return std::move(result);
    }
    result.track_tensors(inputs);
    result.create_backward_function(
            [](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents,
                 intrusive_ptr<tensor_holder> saved_bools) {
                parents[0]->grad->tensor += grad;
                parents[0]->grad->tensor[saved_bools->tensor] = 0;
            },
            make_intrusive<tensor_holder>(bools));
    return std::move(result);
}

TensorGrad TensorGrad_Functional_Class::abs(const TensorGrad &x){
    Tensor a = ::nt::functional::abs(x.tensor);
    TensorGrad result(std::move(a), x.grad_required);
    if (!x.do_track_grad) {
        result.do_track_grad = false;
        return std::move(result);
    }


    intrusive_ptr<tensor_holder> saved_x =
            make_intrusive<tensor_holder>(x.tensor.conditional_mutate_clone());
    
    result.track_tensors(x);


    result.create_backward_function(
            [](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents,
               intrusive_ptr<tensor_holder> saved_x) {
                //compute the gradient using the saved input tensor
                Tensor sign_grad = (saved_x->tensor > 0).to(DType::Float32) -
                                   (saved_x->tensor < 0).to(DType::Float32); //compute sign
                parents[0]->grad->tensor += grad * sign_grad;
            },
            saved_x);

    return std::move(result);


}




TensorGrad  TensorGrad_Functional_Class::conv1d(const Tensor& image, const TensorGrad& kernel, int64_t stride, int64_t padding, int64_t dilation, int64_t groups){
    if(kernel.grad_required == false || kernel.do_track_grad == false ){
        return TensorGrad(::nt::functional::conv1d(image, kernel.tensor, stride, padding, dilation, groups), false);
    }
    intrusive_ptr<tensor_holder> original_x = make_intrusive<tensor_holder>(Tensor::Null());
    const SizeRef image_shape = image.shape().clone();
    const SizeRef kernel_shape = kernel.shape().clone();
    TensorGrad result(::nt::functional::conv1d(image, kernel.tensor, stride, padding, dilation, groups, original_x));
    result.track_tensors(kernel);
    result.create_backward_function([image_shape, kernel_shape, stride, padding, dilation, groups](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
						intrusive_ptr<tensor_holder> img){
        ::nt::functional::conv_dkernel(grad, img->tensor, parents[0]->grad->tensor, {image_shape[-1]}, groups);
    }, original_x);
    return std::move(result);
}

TensorGrad  TensorGrad_Functional_Class::conv1d(const TensorGrad& image, const Tensor& kernel, int64_t stride, int64_t padding, int64_t dilation, int64_t groups){
	if(image.grad_required == false || image.do_track_grad == false ){
        TensorGrad result(::nt::functional::conv1d(image.tensor, kernel, stride, padding, dilation, groups), false);
        return std::move(result);
    }
    intrusive_ptr<tensor_holder> original_w = make_intrusive<tensor_holder>(Tensor::Null());
    const SizeRef image_shape = image.shape().clone();
    const SizeRef kernel_shape = kernel.shape().clone();
    TensorGrad result(::nt::functional::conv1d(image.tensor, kernel, stride, padding, dilation, groups, nullptr, original_w));
    result.track_tensors(image);
    result.create_backward_function([image_shape, kernel_shape, stride, padding, dilation, groups](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
						intrusive_ptr<tensor_holder> kern){
        ::nt::functional::conv_dimage(grad, kern->tensor, parents[0]->grad->tensor, 
                                     {kernel_shape[-1]},
                                     {stride},
                                     {padding},
                                     {dilation},
                                     groups);
    }, original_w);
    return std::move(result);
}

TensorGrad  TensorGrad_Functional_Class::conv1d(const TensorGrad& image, const TensorGrad& kernel, int64_t stride, int64_t padding, int64_t dilation, int64_t groups){
	if(image.grad_required == false || image.do_track_grad == false ){
        return conv1d(image.tensor, kernel, stride, padding, dilation, groups);
    }
    if(kernel.grad_required == false || kernel.do_track_grad == false ){
        return conv1d(image, kernel.tensor, stride, padding, dilation, groups);
    }
    intrusive_ptr<tensor_holder> original_x = make_intrusive<tensor_holder>(Tensor::Null());
    intrusive_ptr<tensor_holder> original_w = make_intrusive<tensor_holder>(Tensor::Null());
    const SizeRef image_shape = image.shape().clone();
    const SizeRef kernel_shape = kernel.shape().clone();
    TensorGrad result(::nt::functional::conv1d(image.tensor, kernel.tensor, stride, padding, dilation, groups, original_x, original_w));
    result.track_tensors(image, kernel);
    result.create_backward_function([image_shape, kernel_shape, stride, padding, dilation, groups](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
						intrusive_ptr<tensor_holder> img, intrusive_ptr<tensor_holder> kern){
        ::nt::functional::conv_dimage(grad, kern->tensor, parents[0]->grad->tensor, 
                                     {kernel_shape[-1]},
                                     {stride},
                                     {padding},
                                     {dilation},
                                     groups);
        ::nt::functional::conv_dkernel(grad, img->tensor, parents[1]->grad->tensor, {image_shape[-1]}, groups);
    }, original_x, original_w);
    return std::move(result);
}

TensorGrad  TensorGrad_Functional_Class::conv2d(const Tensor& image, const TensorGrad& kernel, utils::my_tuple stride, utils::my_tuple padding, utils::my_tuple dilation, int64_t groups){
	if(kernel.grad_required == false || kernel.do_track_grad == false){
        //if the kernel isn't tracking the gradient, then the gradient for neither is tracked
        TensorGrad result(::nt::functional::conv2d(image, kernel.tensor, stride, padding, dilation, groups), false);
        return std::move(result);
    }
    intrusive_ptr<tensor_holder> original_x = make_intrusive<tensor_holder>(Tensor::Null());
    const SizeRef image_shape = image.shape().clone();
    const SizeRef kernel_shape = kernel.shape().clone();
    TensorGrad result(::nt::functional::conv2d(image, kernel.tensor, stride, padding, dilation, groups, original_x));
    result.track_tensors( kernel);
    result.create_backward_function([image_shape, kernel_shape, stride, padding, dilation, groups](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
						intrusive_ptr<tensor_holder> img){
          ::nt::functional::conv_dkernel(grad, img->tensor, parents[0]->grad->tensor, {image_shape[-2], image_shape[-3]}, groups);
    }, original_x);
    return std::move(result);
}

TensorGrad  TensorGrad_Functional_Class::conv2d(const TensorGrad& image, const Tensor& kernel, utils::my_tuple stride, utils::my_tuple padding, utils::my_tuple dilation, int64_t groups){
	if(image.grad_required == false || image.do_track_grad == false){
        //if one of the tensors isn't tracking the gradient, then the gradient for neither is tracked
        TensorGrad result(::nt::functional::conv2d(image.tensor, kernel, stride, padding, dilation, groups), false);
        return std::move(result);
    }
    intrusive_ptr<tensor_holder> original_w = make_intrusive<tensor_holder>(Tensor::Null());
    const SizeRef image_shape = image.shape().clone();
    const SizeRef kernel_shape = kernel.shape().clone();
    TensorGrad result(::nt::functional::conv2d(image.tensor, kernel, stride, padding, dilation, groups, nullptr, original_w));
    result.track_tensors(image);
    result.create_backward_function([image_shape, kernel_shape, stride, padding, dilation, groups](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
						intrusive_ptr<tensor_holder> kern){
        ::nt::functional::conv_dimage(grad, kern->tensor, parents[0]->grad->tensor, 
                                     {kernel_shape[-2], kernel_shape[-1]},
                                     {stride[0], stride[1]},
                                     {padding[0], padding[1]},
                                     {dilation[0], dilation[1]},
                                     groups);
    }, original_w);
    return std::move(result);
}


TensorGrad  TensorGrad_Functional_Class::conv2d(const TensorGrad& image, const TensorGrad& kernel, utils::my_tuple stride, utils::my_tuple padding, utils::my_tuple dilation, int64_t groups){
	if(image.grad_required == false || image.do_track_grad == false ){
        return conv2d(image.tensor, kernel, stride, padding, dilation, groups);
    }
    if(kernel.grad_required == false || kernel.do_track_grad == false ){
        return conv2d(image, kernel.tensor, stride, padding, dilation, groups);
    }

    intrusive_ptr<tensor_holder> original_x = make_intrusive<tensor_holder>(Tensor::Null());
    intrusive_ptr<tensor_holder> original_w = make_intrusive<tensor_holder>(Tensor::Null());
    const SizeRef image_shape = image.shape().clone();
    const SizeRef kernel_shape = kernel.shape().clone();
    TensorGrad result(::nt::functional::conv2d(image.tensor, kernel.tensor, stride, padding, dilation, groups, original_x, original_w));
    result.track_tensors(image, kernel);
    result.create_backward_function([image_shape, kernel_shape, stride, padding, dilation, groups](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
						intrusive_ptr<tensor_holder> img, intrusive_ptr<tensor_holder> kern){
        ::nt::functional::conv_dimage(grad, kern->tensor, parents[0]->grad->tensor, 
                                     {kernel_shape[-2], kernel_shape[-1]},
                                     {stride[0], stride[1]},
                                     {padding[0], padding[1]},
                                     {dilation[0], dilation[1]},
                                     groups);
        ::nt::functional::conv_dkernel(grad, img->tensor, parents[1]->grad->tensor, {image_shape[-2], image_shape[-3]}, groups);
    }, original_x, original_w);
    return std::move(result);
}



TensorGrad  TensorGrad_Functional_Class::conv3d(const Tensor& image, const TensorGrad& kernel, utils::my_n_tuple<3> stride, utils::my_n_tuple<3> padding, utils::my_n_tuple<3> dilation, int64_t groups){
    if(kernel.grad_required == false || kernel.do_track_grad == false ){
        return TensorGrad(::nt::functional::conv3d(image, kernel.tensor, stride, padding, dilation, groups), false);
    }

    intrusive_ptr<tensor_holder> original_x = make_intrusive<tensor_holder>(Tensor::Null());
    const SizeRef image_shape = image.shape().clone();
    const SizeRef kernel_shape = kernel.shape().clone();
    TensorGrad result(::nt::functional::conv3d(image, kernel.tensor, stride, padding, dilation, groups, original_x));
    result.track_tensors(kernel);
    result.create_backward_function([image_shape, kernel_shape, stride, padding, dilation, groups](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
						intrusive_ptr<tensor_holder> img){
        ::nt::functional::conv_dkernel(grad, img->tensor, parents[0]->grad->tensor, {image_shape[-3], image_shape[-2], image_shape[-3]}, groups);
    }, original_x);
    return std::move(result);

}

TensorGrad  TensorGrad_Functional_Class::conv3d(const TensorGrad& image, const Tensor& kernel, utils::my_n_tuple<3> stride, utils::my_n_tuple<3> padding, utils::my_n_tuple<3> dilation, int64_t groups){
    if(image.grad_required == false || image.do_track_grad == false ){
        return TensorGrad(::nt::functional::conv3d(image.tensor, kernel, stride, padding, dilation, groups), false);
    }

    intrusive_ptr<tensor_holder> original_w = make_intrusive<tensor_holder>(Tensor::Null());
    const SizeRef image_shape = image.shape().clone();
    const SizeRef kernel_shape = kernel.shape().clone();
    TensorGrad result(::nt::functional::conv3d(image.tensor, kernel, stride, padding, dilation, groups, nullptr, original_w));
    result.track_tensors(image);
    result.create_backward_function([image_shape, kernel_shape, stride, padding, dilation, groups](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
						intrusive_ptr<tensor_holder> kern){
        ::nt::functional::conv_dimage(grad, kern->tensor, parents[0]->grad->tensor, 
                                     {kernel_shape[-3], kernel_shape[-2], kernel_shape[-1]},
                                     {stride[0], stride[1], stride[2]},
                                     {padding[0], padding[1], padding[2]},
                                     {dilation[0], dilation[1], dilation[2]},
                                     groups);
    }, original_w);
    return std::move(result);
}

TensorGrad  TensorGrad_Functional_Class::conv3d(const TensorGrad& image, const TensorGrad& kernel, utils::my_n_tuple<3> stride, utils::my_n_tuple<3> padding, utils::my_n_tuple<3> dilation, int64_t groups){
	if(image.grad_required == false || image.do_track_grad == false ){
        return conv3d(image.tensor, kernel, stride, padding, dilation, groups);
    }
    if(kernel.grad_required == false || kernel.do_track_grad == false ){
        return conv3d(image, kernel.tensor, stride, padding, dilation, groups);
    }
    intrusive_ptr<tensor_holder> original_x = make_intrusive<tensor_holder>(Tensor::Null());
    intrusive_ptr<tensor_holder> original_w = make_intrusive<tensor_holder>(Tensor::Null());
    const SizeRef image_shape = image.shape().clone();
    const SizeRef kernel_shape = kernel.shape().clone();
    TensorGrad result(::nt::functional::conv3d(image.tensor, kernel.tensor, stride, padding, dilation, groups, original_x, original_w));
    result.track_tensors(image, kernel);
    result.create_backward_function([image_shape, kernel_shape, stride, padding, dilation, groups](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
						intrusive_ptr<tensor_holder> img, intrusive_ptr<tensor_holder> kern){
        ::nt::functional::conv_dimage(grad, kern->tensor, parents[0]->grad->tensor, 
                                     {kernel_shape[-3], kernel_shape[-2], kernel_shape[-1]},
                                     {stride[0], stride[1], stride[2]},
                                     {padding[0], padding[1], padding[2]},
                                     {dilation[0], dilation[1], dilation[2]},
                                     groups);
        ::nt::functional::conv_dkernel(grad, img->tensor, parents[1]->grad->tensor, {image_shape[-3], image_shape[-2], image_shape[-3]}, groups);
    }, original_x, original_w);
    return std::move(result);
}

TensorGrad  TensorGrad_Functional_Class::conv_transpose1d(const Tensor& image, const TensorGrad& kernel, int64_t stride, int64_t padding, int64_t output_padding, int64_t dilation, int64_t groups){
    if(kernel.grad_required == false || kernel.do_track_grad == false ){
        return TensorGrad(::nt::functional::conv_transpose1d(image, kernel.tensor, stride, padding, output_padding, dilation, groups), false);
    }
    intrusive_ptr<tensor_holder> original_x = make_intrusive<tensor_holder>(Tensor::Null());
    const SizeRef image_shape = image.shape().clone();
    const SizeRef kernel_shape = kernel.shape().clone();
    TensorGrad result(::nt::functional::conv_transpose1d(image, kernel.tensor, stride, padding, output_padding, dilation, groups, original_x));
    result.track_tensors(kernel);
    result.create_backward_function(
        [image_shape, kernel_shape, stride, padding, output_padding, dilation, groups](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
						intrusive_ptr<tensor_holder> img){
        ::nt::functional::convt_dkernel(grad, img->tensor, parents[0]->grad->tensor, {padding}, {image_shape[-1]}, groups);
    }, original_x);
    return std::move(result);
}

TensorGrad  TensorGrad_Functional_Class::conv_transpose1d(const TensorGrad& image, const Tensor& kernel, int64_t stride, int64_t padding, int64_t output_padding, int64_t dilation, int64_t groups){
	if(image.grad_required == false || image.do_track_grad == false ){
        TensorGrad result(::nt::functional::conv_transpose1d(image.tensor, kernel, stride, padding, output_padding, dilation, groups), false);
        return std::move(result);
    }
    intrusive_ptr<tensor_holder> original_w = make_intrusive<tensor_holder>(Tensor::Null());
    const SizeRef image_shape = image.shape().clone();
    const SizeRef kernel_shape = kernel.shape().clone();
    TensorGrad result(::nt::functional::conv_transpose1d(image.tensor, kernel, stride, padding, output_padding, dilation, groups, nullptr, original_w));
    result.track_tensors(image);
    result.create_backward_function(
        [image_shape, kernel_shape, stride, padding, output_padding, dilation, groups](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
						intrusive_ptr<tensor_holder> kern){
        ::nt::functional::convt_dimage(grad, kern->tensor, parents[0]->grad->tensor, 
                                     {kernel_shape[-1]},
                                     {stride},
                                     {padding},
                                     {output_padding},
                                     {dilation},
                                     groups);
    }, original_w);
    return std::move(result);
}

TensorGrad  TensorGrad_Functional_Class::conv_transpose1d(const TensorGrad& image, const TensorGrad& kernel, int64_t stride, int64_t padding, int64_t output_padding, int64_t dilation, int64_t groups){
	if(image.grad_required == false || image.do_track_grad == false ){
        return conv_transpose1d(image.tensor, kernel, stride, padding, output_padding, dilation, groups);
    }
    if(kernel.grad_required == false || kernel.do_track_grad == false ){
        return conv_transpose1d(image, kernel.tensor, stride, padding, output_padding, dilation, groups);
    }
    intrusive_ptr<tensor_holder> original_x = make_intrusive<tensor_holder>(Tensor::Null());
    intrusive_ptr<tensor_holder> original_w = make_intrusive<tensor_holder>(Tensor::Null());
    const SizeRef image_shape = image.shape().clone();
    const SizeRef kernel_shape = kernel.shape().clone();
    TensorGrad result(::nt::functional::conv_transpose1d(image.tensor, kernel.tensor, stride, padding, output_padding, dilation, groups, original_x, original_w));
    result.track_tensors(image, kernel);
    result.create_backward_function([image_shape, kernel_shape, stride, padding, output_padding, dilation, groups](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
						intrusive_ptr<tensor_holder> img, intrusive_ptr<tensor_holder> kern){
        ::nt::functional::convt_dimage(grad, kern->tensor, parents[0]->grad->tensor, 
                                     {kernel_shape[-1]},
                                     {stride},
                                     {padding},
                                     {output_padding},
                                     {dilation},
                                     groups);
        ::nt::functional::convt_dkernel(grad, img->tensor, parents[1]->grad->tensor, {padding}, {image_shape[-1]}, groups);
    }, original_x, original_w);
    return std::move(result);
}

TensorGrad  TensorGrad_Functional_Class::conv_transpose2d(const Tensor& image, const TensorGrad& kernel, utils::my_tuple stride, utils::my_tuple padding, utils::my_tuple output_padding, utils::my_tuple dilation, int64_t groups){
	if(kernel.grad_required == false || kernel.do_track_grad == false){
        //if the kernel isn't tracking the gradient, then the gradient for neither is tracked
        TensorGrad result(::nt::functional::conv_transpose2d(image, kernel.tensor, stride, padding, output_padding, dilation, groups), false);
        return std::move(result);
    }
    intrusive_ptr<tensor_holder> original_x = make_intrusive<tensor_holder>(Tensor::Null());
    const SizeRef image_shape = image.shape().clone();
    const SizeRef kernel_shape = kernel.shape().clone();
    TensorGrad result(::nt::functional::conv_transpose2d(image, kernel.tensor, stride, padding, output_padding, dilation, groups, original_x));
    result.track_tensors( kernel);
    result.create_backward_function([image_shape, kernel_shape, stride, padding, output_padding, dilation, groups](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
						intrusive_ptr<tensor_holder> img){
          ::nt::functional::convt_dkernel(grad, img->tensor, parents[0]->grad->tensor, {padding[0], padding[1]}, {image_shape[-2], image_shape[-3]}, groups);
    }, original_x);
    return std::move(result);
}

TensorGrad  TensorGrad_Functional_Class::conv_transpose2d(const TensorGrad& image, const Tensor& kernel, utils::my_tuple stride, utils::my_tuple padding, utils::my_tuple output_padding, utils::my_tuple dilation, int64_t groups){
	if(image.grad_required == false || image.do_track_grad == false){
        //if one of the tensors isn't tracking the gradient, then the gradient for neither is tracked
        TensorGrad result(::nt::functional::conv_transpose2d(image.tensor, kernel, stride, padding, output_padding, dilation, groups), false);
        return std::move(result);
    }
    intrusive_ptr<tensor_holder> original_w = make_intrusive<tensor_holder>(Tensor::Null());
    const SizeRef image_shape = image.shape().clone();
    const SizeRef kernel_shape = kernel.shape().clone();
    TensorGrad result(::nt::functional::conv_transpose2d(image.tensor, kernel, stride, padding, output_padding, dilation, groups, nullptr, original_w));
    result.track_tensors(image);
    result.create_backward_function([image_shape, kernel_shape, stride, padding, output_padding, dilation, groups](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
						intrusive_ptr<tensor_holder> kern){
        ::nt::functional::convt_dimage(grad, kern->tensor, parents[0]->grad->tensor, 
                                     {kernel_shape[-2], kernel_shape[-1]},
                                     {stride[0], stride[1]},
                                     {padding[0], padding[1]},
                                     {output_padding[0], output_padding[1]},
                                     {dilation[0], dilation[1]},
                                     groups);
    }, original_w);
    return std::move(result);
}


TensorGrad  TensorGrad_Functional_Class::conv_transpose2d(const TensorGrad& image, const TensorGrad& kernel, utils::my_tuple stride, utils::my_tuple padding, utils::my_tuple output_padding, utils::my_tuple dilation, int64_t groups){
	if(image.grad_required == false || image.do_track_grad == false ){
        return conv_transpose2d(image.tensor, kernel, stride, padding, output_padding, dilation, groups);
    }
    if(kernel.grad_required == false || kernel.do_track_grad == false ){
        return conv_transpose2d(image, kernel.tensor, stride, padding, output_padding, dilation, groups);
    }

    intrusive_ptr<tensor_holder> original_x = make_intrusive<tensor_holder>(Tensor::Null());
    intrusive_ptr<tensor_holder> original_w = make_intrusive<tensor_holder>(Tensor::Null());
    const SizeRef image_shape = image.shape().clone();
    const SizeRef kernel_shape = kernel.shape().clone();
    TensorGrad result(::nt::functional::conv_transpose2d(image.tensor, kernel.tensor, stride, padding, output_padding, dilation, groups, original_x, original_w));
    result.track_tensors(image, kernel);
    result.create_backward_function([image_shape, kernel_shape, stride, padding, output_padding, dilation, groups](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
						intrusive_ptr<tensor_holder> img, intrusive_ptr<tensor_holder> kern){
        ::nt::functional::convt_dimage(grad, kern->tensor, parents[0]->grad->tensor, 
                                     {kernel_shape[-2], kernel_shape[-1]},
                                     {stride[0], stride[1]},
                                     {padding[0], padding[1]},
                                     {output_padding[0], output_padding[1]},
                                     {dilation[0], dilation[1]},
                                     groups);
        ::nt::functional::convt_dkernel(grad, img->tensor, parents[1]->grad->tensor, {padding[0], padding[1]}, {image_shape[-2], image_shape[-3]}, groups);
    }, original_x, original_w);
    return std::move(result);
}



TensorGrad  TensorGrad_Functional_Class::conv_transpose3d(const Tensor& image, const TensorGrad& kernel, utils::my_n_tuple<3> stride, utils::my_n_tuple<3> padding, utils::my_n_tuple<3> output_padding, utils::my_n_tuple<3> dilation, int64_t groups){
    if(kernel.grad_required == false || kernel.do_track_grad == false ){
        return TensorGrad(::nt::functional::conv_transpose3d(image, kernel.tensor, stride, padding, output_padding, dilation, groups), false);
    }

    intrusive_ptr<tensor_holder> original_x = make_intrusive<tensor_holder>(Tensor::Null());
    const SizeRef image_shape = image.shape().clone();
    const SizeRef kernel_shape = kernel.shape().clone();
    TensorGrad result(::nt::functional::conv_transpose3d(image, kernel.tensor, stride, padding, output_padding, dilation, groups, original_x));
    result.track_tensors(kernel);
    result.create_backward_function([image_shape, kernel_shape, stride, padding, output_padding, dilation, groups](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
						intrusive_ptr<tensor_holder> img){
        ::nt::functional::convt_dkernel(grad, img->tensor, parents[0]->grad->tensor, {padding[0], padding[1], padding[2]}, {image_shape[-3], image_shape[-2], image_shape[-3]}, groups);
    }, original_x);
    return std::move(result);

}

TensorGrad  TensorGrad_Functional_Class::conv_transpose3d(const TensorGrad& image, const Tensor& kernel, utils::my_n_tuple<3> stride, utils::my_n_tuple<3> padding, utils::my_n_tuple<3> output_padding, utils::my_n_tuple<3> dilation, int64_t groups){
    if(image.grad_required == false || image.do_track_grad == false ){
        return TensorGrad(::nt::functional::conv_transpose3d(image.tensor, kernel, stride, padding, output_padding, dilation, groups), false);
    }

    intrusive_ptr<tensor_holder> original_w = make_intrusive<tensor_holder>(Tensor::Null());
    const SizeRef image_shape = image.shape().clone();
    const SizeRef kernel_shape = kernel.shape().clone();
    TensorGrad result(::nt::functional::conv_transpose3d(image.tensor, kernel, stride, padding, output_padding, dilation, groups, nullptr, original_w));
    result.track_tensors(image);
    result.create_backward_function([image_shape, kernel_shape, stride, padding, output_padding, dilation, groups](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
						intrusive_ptr<tensor_holder> kern){
        ::nt::functional::convt_dimage(grad, kern->tensor, parents[0]->grad->tensor, 
                                     {kernel_shape[-3], kernel_shape[-2], kernel_shape[-1]},
                                     {stride[0], stride[1], stride[2]},
                                     {padding[0], padding[1], padding[2]},
                                     {output_padding[0], output_padding[1], output_padding[2]},
                                     {dilation[0], dilation[1], dilation[2]},
                                     groups);
    }, original_w);
    return std::move(result);
}

TensorGrad  TensorGrad_Functional_Class::conv_transpose3d(const TensorGrad& image, const TensorGrad& kernel, utils::my_n_tuple<3> stride, utils::my_n_tuple<3> padding, utils::my_n_tuple<3> output_padding, utils::my_n_tuple<3> dilation, int64_t groups){
	if(image.grad_required == false || image.do_track_grad == false ){
        return conv_transpose3d(image.tensor, kernel, stride, padding, output_padding, dilation, groups);
    }
    if(kernel.grad_required == false || kernel.do_track_grad == false ){
        return conv_transpose3d(image, kernel.tensor, stride, padding, output_padding, dilation, groups);
    }
    intrusive_ptr<tensor_holder> original_x = make_intrusive<tensor_holder>(Tensor::Null());
    intrusive_ptr<tensor_holder> original_w = make_intrusive<tensor_holder>(Tensor::Null());
    const SizeRef image_shape = image.shape().clone();
    const SizeRef kernel_shape = kernel.shape().clone();
    TensorGrad result(::nt::functional::conv_transpose3d(image.tensor, kernel.tensor, stride, padding, output_padding, dilation, groups, original_x, original_w));
    result.track_tensors(image, kernel);
    result.create_backward_function([image_shape, kernel_shape, stride, padding, output_padding, dilation, groups](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
						intrusive_ptr<tensor_holder> img, intrusive_ptr<tensor_holder> kern){
        ::nt::functional::convt_dimage(grad, kern->tensor, parents[0]->grad->tensor, 
                                     {kernel_shape[-3], kernel_shape[-2], kernel_shape[-1]},
                                     {stride[0], stride[1], stride[2]},
                                     {padding[0], padding[1], padding[2]},
                                     {output_padding[0], output_padding[1], output_padding[2]},
                                     {dilation[0], dilation[1], dilation[2]},
                                     groups);
        ::nt::functional::convt_dkernel(grad, img->tensor, parents[1]->grad->tensor, {padding[0], padding[1], padding[2]}, {image_shape[-3], image_shape[-2], image_shape[-3]}, groups);
    }, original_x, original_w);
    return std::move(result);
}



} // namespace functional
} // namespace nt
