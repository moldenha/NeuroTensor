#include "../../functional/functional.h"
#include "../../Tensor.h"
#include "../../intrusive_ptr/intrusive_ptr.hpp"
#include "../TensorGrad.h"
#include "../functional_class.h"
#include "../../dtype/ArrayVoid.hpp"
// #include "functional.h"

namespace nt {
namespace functional {


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
    
    if(!input.track_grad() || !W1.track_grad() || !W2.track_grad()){
        if(!input.track_grad()) return symmetric_bilinear(input.detach(), W1, W2);
        if(!W1.track_grad()) return symmetric_bilinear(input, W1.detach(), W2);
        return symmetric_bilinear(input, W1, W2.detach());
    }

    auto input_saved = make_intrusive<tensor_holder>(input.detach().conditional_mutate_clone());
    auto w1_saved = make_intrusive<tensor_holder>(W1.detach().conditional_mutate_clone());
    auto w2_saved = make_intrusive<tensor_holder>(W2.detach().conditional_mutate_clone());
    
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
            parents[0]->accumulate_gradient(dInputRight + dInputLeft);
            parents[1]->accumulate_gradient(dWeight1Right + dWeight1Left);
            parents[2]->accumulate_gradient(dWeight2Right + dWeight2Left);
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
    if(!W2.track_grad() || !W1.track_grad()){
        if(!W2.track_grad()) return symmetric_bilinear(input, W1, W2.detach());
        return symmetric_bilinear(input, W1.detach(), W2);
    } 
    auto input_saved = make_intrusive<tensor_holder>(input.conditional_mutate_clone());
    // auto w1_saved = make_intrusive<tensor_holder>(W1.detach().conditional_mutate_clone());
    auto w2_saved = make_intrusive<tensor_holder>(W2.detach().conditional_mutate_clone());
    
    Tensor left = functional::matmult(W1.detach(), input_saved->tensor);
    Tensor left_out = functional::matmult(w2_saved->tensor, left);
    
    Tensor right = functional::matmult(input_saved->tensor, W1.detach(), false, true);
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
            // parents[0]->grad() += dInputRight + dInputLeft;
            parents[0]->accumulate_gradient(dWeight1Right + dWeight1Left);
            parents[1]->accumulate_gradient(dWeight2Right + dWeight2Left);
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
    
    // auto input_saved = make_intrusive<tensor_holder>(input.detach().conditional_mutate_clone());
    auto w1_saved = make_intrusive<tensor_holder>(W1.conditional_mutate_clone());
    auto w2_saved = make_intrusive<tensor_holder>(W2.detach().conditional_mutate_clone());
    
    Tensor left = functional::matmult(w1_saved->tensor, input.detach());
    Tensor left_out = functional::matmult(w2_saved->tensor, left);
    
    Tensor right = functional::matmult(input.detach(), w1_saved->tensor, false, true);
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
            parents[0]->accumulate_gradient(dInputRight + dInputLeft);
            // parents[1]->grad() += dWeight1Right + dWeight1Left;
            parents[1]->accumulate_gradient(dWeight2Right + dWeight2Left);
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
    
    if(!input.track_grad() || !W1.track_grad()){
        if(!input.track_grad()) return symmetric_bilinear(input.detach(), W1, W2);
        return symmetric_bilinear(input, W1.detach(), W2);
    }

    auto input_saved = make_intrusive<tensor_holder>(input.detach().conditional_mutate_clone());
    auto w1_saved = make_intrusive<tensor_holder>(W1.detach().conditional_mutate_clone());
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
            parents[0]->accumulate_gradient(dInputRight + dInputLeft);
            parents[1]->accumulate_gradient(dWeight1Right + dWeight1Left);
            // parents[2]->grad() += dWeight2Right + dWeight2Left;
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
    if(!W2.track_grad()){
        TensorGrad result(__nt__symmetric_bilinear__(input, W1, W2.detach()), W2.track_grad());
        result.track_grad_(false);
        return std::move(result);
    } 
    // auto input_saved = make_intrusive<tensor_holder>(input.conditional_mutate_clone());
    // auto w1_saved = make_intrusive<tensor_holder>(W1.detach().conditional_mutate_clone());
    // auto w2_saved = make_intrusive<tensor_holder>(W2.detach().conditional_mutate_clone());
    
    Tensor left = functional::matmult(W1, input);
    Tensor left_out = functional::matmult(W2.detach(), left);
    
    Tensor right = functional::matmult(input, W1, false, true);
    Tensor right_out = functional::matmult(right, W2.detach(), false, true);
    
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
            // parents[0]->grad() += dInputRight + dInputLeft;
            // parents[0]->grad() += dWeight1Right + dWeight1Left;
            parents[0]->accumulate_gradient(dWeight2Right + dWeight2Left);
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
    
    if(!W1.track_grad()){
        TensorGrad result(__nt__symmetric_bilinear__(input, W1.detach(), W2), W1.track_grad());
        result.track_grad_(false);
        return std::move(result);
    } 

    auto input_saved = make_intrusive<tensor_holder>(input.conditional_mutate_clone());
    auto w2_saved = make_intrusive<tensor_holder>(W2.conditional_mutate_clone());
    
    Tensor left = functional::matmult(W1.detach(), input_saved->tensor);
    Tensor left_out = functional::matmult(w2_saved->tensor, left);
    
    Tensor right = functional::matmult(input_saved->tensor, W1.detach(), false, true);
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
            // parents[0]->grad() += dInputRight + dInputLeft;
            parents[0]->accumulate_gradient(dWeight1Right + dWeight1Left);
            // parents[2]->grad() += dWeight2Right + dWeight2Left;
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
    
    if(!input.track_grad()){
        TensorGrad result(__nt__symmetric_bilinear__(input.detach(), W1, W2), input.track_grad());
        result.track_grad_(false);
        return std::move(result);
    }

    // auto input_saved = make_intrusive<tensor_holder>(input.detach().conditional_mutate_clone());
    auto w1_saved = make_intrusive<tensor_holder>(W1.conditional_mutate_clone());
    auto w2_saved = make_intrusive<tensor_holder>(W2.conditional_mutate_clone());
    
    Tensor left = functional::matmult(w1_saved->tensor, input.detach());
    Tensor left_out = functional::matmult(w2_saved->tensor, left);
    
    Tensor right = functional::matmult(input.detach(), w1_saved->tensor, false, true);
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
            parents[0]->accumulate_gradient(dInputRight + dInputLeft);
            // parents[1]->grad() += dWeight1Right + dWeight1Left;
            // parents[2]->grad() += dWeight2Right + dWeight2Left;
        },
        w1_saved, w2_saved, "SymmetricBilinear"
    );
    return std::move(result);
}

////(1/2)[WA(W^T) + (W^T)(A)(W)]
//TensorGrad TensorGrad_Functional_Class::symmetric_bilinear(const TensorGrad& input, const TensorGrad& weight){
//    if(!input.track_grad() || !weight.track_grad()){
//        if(!input.track_grad()){
//            return symmetric_bilinear(input.detach(), weight);
//        }
//        return symmetric_bilinear(input, weight.detach());
//    }
    
//    // Clone inputs for backward
//    auto input_saved = make_intrusive<tensor_holder>(input.detach().conditional_mutate_clone());
//    auto weight_saved = make_intrusive<tensor_holder>(weight.detach().conditional_mutate_clone());


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
//            parents[0]->grad() += dInputRight + dInputLeft;
//            parents[1]->grad() += dWeightRightOut + dWeightRightFirst + dWeightLeftOut + dWeightLeftFirst;
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
//    if (!input.track_grad()) {
//        TensorGrad result(__nt__symmetric_bilinear__(input.detach(), weight), input.track_grad());
//        result.track_grad_(false);
//        return std::move(result);
//    }
//    // Clone inputs for backward
//    auto input_saved = make_intrusive<tensor_holder>(input.detach().conditional_mutate_clone());
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
//            parents[0]->grad() += dInputRight + dInputLeft;
//            //parents[1]->grad() += dWeightRightOut + dWeightRightFirst + dWeightLeftOut + dWeightLeftFirst;
//        },
//        /*input_saved,*/ weight_saved, right_saved, left_saved, "SymmetricBilinear"
//    );
//    return std::move(result);
//}

//TensorGrad TensorGrad_Functional_Class::symmetric_bilinear(const Tensor& input, const TensorGrad& weight){
//   if (!weight.track_grad()) {
//        TensorGrad result(__nt__symmetric_bilinear__(input, weight.detach()), weight.track_grad());
//        result.track_grad_(false);
//        return std::move(result);
//    }

//    // Clone inputs for backward
//    auto input_saved = make_intrusive<tensor_holder>(input.conditional_mutate_clone());
//    auto weight_saved = make_intrusive<tensor_holder>(weight.detach().conditional_mutate_clone());


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
//            //parents[0]->grad() += dInputRight + dInputLeft;
//            parents[0]->grad() += dWeightRightOut + dWeightRightFirst + dWeightLeftOut + dWeightLeftFirst;
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
        arrVds.push_back(std::cref(begin->detach().arr_void()));
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
        arrVds.push_back(begin->detach().transpose(0, dim).arr_void());
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
        arrVds.push_back(std::cref((*begin)->grad().arr_void()));
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
        arrVds.push_back((*begin)->grad().transpose(0, dim).arr_void());
    }
    SizeRef shape(std::move(vec));
    return Tensor(ArrayVoid::cat(arrVds), std::move(shape)).transpose(0, dim);
}


TensorGrad TensorGrad_Functional_Class::cat(std::vector<TensorGrad> tgs, int64_t dim) {
    bool track_grad = tgs[0].track_grad();
    for (const auto &tg : tgs) {
        utils::throw_exception(tg.track_grad() == track_grad,
                                                     "Cannot concatenate tensors that are both tracking "
                                                     "the gradient and are not");
        utils::throw_exception(!tg.is_null(), "Cannot concatenate null tensors");
    }
    TensorGrad result(cat_vec(tgs, dim), track_grad);
    if (!track_grad) {
        result.track_grad_(false);
        return std::move(result);
    }

    // tracking the gradient itself
    // rather than tracking each parent individually
    for (const auto &tg : tgs) {
        tg.Node->ensure_gradient_init();
    }
    result.track_tensors(tgs);
    result.Node->ensure_backward_initialization();
    std::vector<intrusive_ptr<TensorGrad> > parents;
    parents.reserve(tgs.size());
    for(const auto& tg : result.Node->parents){
        parents.emplace_back(make_intrusive<TensorGrad>(tg.lock()));
    }
    result.grad() = cat_vec_grad(parents, dim);
    return std::move(result);
}

// inline std::vector<Tensor> vectorize(Tensor& t){
//     utils::throw_exception(t.dtype() == DType::TensorObj,
//                            "can only vectorize tensor of tensors");
//     return t.arr_void().cexecute_function<WRAP_DTYPES<DTypeEnum<DType::TensorObj > > >
//         ([](auto begin, auto end) -> std::vector<Tensor> {return std::vector<Tensor>(begin, end);});

// }


TensorGrad TensorGrad_Functional_Class::cat(TensorGrad tgs, int64_t dim) {
    // std::cout << tgs << std::endl;
    // std::cout << "on dim "<<dim<<std::endl;
    // std::vector<std::reference_wrapper<Tensor>> first_cat;
    // first_cat.reserve(tgs.numel());
    // Tensor* begin = reinterpret_cast<Tensor*>(tgs.detach().data_ptr());
    // Tensor* end = begin + tgs.numel();
    // for(;begin != end; ++begin)
    //     first_cat.push_back(std::ref(*begin));

    // for(int64_t i = 0; i < tgs.numel(); ++i)
    //     first_cat[i] = tgs[i].detach();
    TensorGrad result(::nt::functional::cat(tgs.detach(), dim), tgs.track_grad());

    // if(tgs.grad == nullptr){
    //     std::cout << tgs.detach().dtype() << std::endl;
    //     Tensor zeros = ::nt::functional::zeros_like(tgs.detach());
    //     std::cout << "zeros: "<<zeros<<std::endl;
    // }
    // else{
    //     std::cout << "tgs.grad is not nullptr"<<std::endl;
    //     std::cout << tgs.grad->tensor << std::endl;
    //     std::cout << ::nt::functional::zeros_like(tgs.detach());
    // }
    
    result.track_grad(tgs, [dim](Tensor &grad) {return ::nt::functional::cat(grad, dim); });
    return std::move(result);
}


TensorGrad TensorGrad_Functional_Class::stack(std::vector<TensorGrad> tgs, int64_t dim) {
    bool track_grad = tgs[0].track_grad();
    for (const auto &tg : tgs) {
        utils::throw_exception(tg.track_grad() == track_grad,
                                                     "Cannot concatenate tensors that are both tracking "
                                                     "the gradient and are not");
        utils::throw_exception(!tg.is_null(), "Cannot concatenate null tensors");
    }
    std::vector<std::reference_wrapper<Tensor>> tgs_data_ref;
    tgs_data_ref.reserve(tgs.size());
    for (int64_t i = 0; i < tgs.size(); ++i) {
        tgs_data_ref.push_back(std::ref(tgs[i].detach()));
    }

    TensorGrad result(::nt::functional::stack(tgs_data_ref, dim), track_grad);
    if (!track_grad) {
        result.track_grad_(false);
        return std::move(result);
    }
    std::vector<std::reference_wrapper<Tensor>> tgs_grad_ref;
    tgs_grad_ref.reserve(tgs.size());
    for (auto &tg : tgs) {
        tg.Node->ensure_gradient_init();
        tgs_grad_ref.push_back(std::ref(tg.grad()));
    }
    result.grad() = ::nt::functional::stack(tgs_grad_ref, dim);
    result.track_tensors(tgs);
    return std::move(result);
}

TensorGrad TensorGrad_Functional_Class::stack(TensorGrad tgs, int64_t dim) {
    TensorGrad result(::nt::functional::stack(tgs.detach(), dim), tgs.track_grad());
    result.track_grad(tgs, [dim](Tensor &grad) { return ::nt::functional::stack(grad, dim); });
    return std::move(result);
}



TensorGrad TensorGrad_Functional_Class::real(const TensorGrad& tg){
    TensorGrad result(::nt::functional::real(tg.detach()), tg.track_grad());
    if(!tg.track_grad()){
        result.track_grad_(false);
        return std::move(result);
    }
    DType original = tg.dtype();
    result.track_tensors(tg);
    result.create_backward_function(
            [original](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents) {
                parents[0]->accumulate_gradient(::nt::functional::to_complex_from_real(grad).to(original));
            });
    return std::move(result);
}

TensorGrad TensorGrad_Functional_Class::to_complex_from_real(const TensorGrad& tg){
    TensorGrad result(::nt::functional::to_complex_from_real(tg.detach()), tg.track_grad());
    if(!tg.track_grad()){
        result.track_grad_(false);
        return std::move(result);
    }
    DType original = tg.dtype();
    result.track_tensors(tg);
    result.create_backward_function(
            [original](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents) {
                parents[0]->accumulate_gradient(::nt::functional::real(grad).to(original));
            });
    return std::move(result);
}

TensorGrad TensorGrad_Functional_Class::imag(const TensorGrad& tg){
    TensorGrad result(::nt::functional::imag(tg.detach()), tg.track_grad());
    if(!tg.track_grad()){
        result.track_grad_(false);
        return std::move(result);
    }
    DType original = tg.dtype();
    result.track_tensors(tg);
    result.create_backward_function(
            [original](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents) {
                parents[0]->accumulate_gradient(::nt::functional::to_complex_from_imag(grad).to(original));
            });
    return std::move(result);
}

TensorGrad TensorGrad_Functional_Class::to_complex_from_imag(const TensorGrad& tg){
    TensorGrad result(::nt::functional::to_complex_from_imag(tg.detach()), tg.track_grad());
    if(!tg.track_grad()){
        result.track_grad_(false);
        return std::move(result);
    }
    DType original = tg.dtype();
    result.track_tensors(tg);
    result.create_backward_function(
            [original](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents) {
                parents[0]->accumulate_gradient(::nt::functional::imag(grad).to(original));
            });
    return std::move(result);
}







TensorGrad TensorGrad_Functional_Class::softmax(const TensorGrad& inputs, bool stable){
    Tensor softmax_x = stable ? ::nt::functional::softmax_stable(inputs.detach()) : ::nt::functional::softmax(inputs.detach());
    intrusive_ptr<tensor_holder> sx_c = make_intrusive<tensor_holder>(softmax_x.conditional_mutate_clone());
    TensorGrad result(std::move(softmax_x), inputs.track_grad());
    if(!inputs.track_grad())
        return std::move(result);
    result.track_tensors(inputs);
    result.create_backward_function(
        [](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
           intrusive_ptr<tensor_holder> sx){
            parents[0]->accumulate_gradient(::nt::functional::dsoftmax(grad, sx->tensor)); 
        },
    sx_c);
    return std::move(result);

}

TensorGrad TensorGrad_Functional_Class::softmax(const TensorGrad& inputs, typename SizeRef::value_type dim, bool stable){
    Tensor softmax_x = stable ? ::nt::functional::softmax_stable(inputs.detach(), dim) : 
        ::nt::functional::softmax(inputs.detach(), dim);
    intrusive_ptr<tensor_holder> sx_c = make_intrusive<tensor_holder>(softmax_x.conditional_mutate_clone());
    TensorGrad result(std::move(softmax_x), inputs.track_grad());
    if(!inputs.track_grad())
        return std::move(result);

    result.track_tensors(inputs);
    result.create_backward_function(
        [dim](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
           intrusive_ptr<tensor_holder> sx){
            parents[0]->accumulate_gradient(::nt::functional::dsoftmax(grad, sx->tensor, dim)); 
        },
    sx_c);
    return std::move(result);

}

inline Tensor sample_gumbel_noise(const TensorGrad& logits){
    Tensor u = ::nt::functional::rand(0, 1, logits.shape(), logits.dtype()); // Uniform (0, 1)
    // return u;
    return -::nt::functional::log(-::nt::functional::log(u + 1e-10)+1e-10);      // Gumbel (0, 1)
}

TensorGrad TensorGrad_Functional_Class::gumbel_softmax(const TensorGrad & logits, Scalar tau, bool hard, int64_t dim, bool stable){
    Tensor gumbel_noise = sample_gumbel_noise(logits).clip_(-3, 3);
    // std::cout << "max: "<<gumbel_noise.max().values.toScalar() << std::endl;
    // TensorGrad y = ((logits + gumbel_noise) / tau);
    // Tensor gumbel_noise = ::nt::functional::zeros_like(logits.detach()) + 3;
    TensorGrad y = (logits + gumbel_noise) / tau;
    // std::cout << y << std::endl;
    y = softmax(y, dim, stable);
    if(hard){
        // Straight-through: make y_hard one-hot
        Tensor y_hard = ::nt::functional::one_hot(::nt::functional::argmax(y.detach(), dim), y.shape()[dim]).to(y.dtype());
        // Use straight-through estimator
        return (y_hard - y.detach()) + y;
    }
    return std::move(y);
}


TensorGrad TensorGrad_Functional_Class::dropout(const TensorGrad &inputs, double p) {
    Tensor bools = ::nt::functional::randbools(inputs.shape(), p);
    Tensor out = inputs.detach().clone();
    out[bools] = 0;
    TensorGrad result(out, inputs.track_grad());
    if (!inputs.track_grad()) {
        result.track_grad_(false);
        return std::move(result);
    }
    result.track_tensors(inputs);
    result.create_backward_function(
            [](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents,
                 intrusive_ptr<tensor_holder> saved_bools) {
                Tensor accumulation = grad.clone();
                accumulation[saved_bools->tensor].fill_(0);
                parents[0]->accumulate_gradient(accumulation);
            },
            make_intrusive<tensor_holder>(bools));
    return std::move(result);
}


TensorGrad TensorGrad_Functional_Class::dropout2d(const TensorGrad &inputs, double p) {
    Tensor out = inputs.detach().clone();
    Tensor split = out.split_axis(-2);
    Tensor bools = randbools(split.shape(), p);
    split[bools] = 0;
    TensorGrad result(out, inputs.track_grad());
    if (!inputs.track_grad()) {
        result.track_grad_(false);
        return std::move(result);
    }
    result.track_tensors(inputs);
    result.create_backward_function(
            [](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents,
                 intrusive_ptr<tensor_holder> saved_bools) {
                Tensor accumulation = grad.clone();
                accumulation.split_axis(-2)[saved_bools->tensor].fill_(0);
                parents[0]->accumulate_gradient(accumulation);
            },
            make_intrusive<tensor_holder>(bools));
    return std::move(result);
}

TensorGrad TensorGrad_Functional_Class::dropout3d(const TensorGrad &inputs, double p) {
    Tensor out = inputs.detach().clone();
    Tensor split = out.split_axis(-3);
    Tensor bools = randbools(split.shape(), p);
    split[bools] = 0;
    TensorGrad result(out, inputs.track_grad());
    if (!inputs.track_grad()) {
        result.track_grad_(false);
        return std::move(result);
    }
    result.track_tensors(inputs);
    result.create_backward_function(
            [](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents,
                 intrusive_ptr<tensor_holder> saved_bools) {
                Tensor accumulation = grad.clone();
                accumulation.split_axis(-3)[saved_bools->tensor].fill_(0);
                parents[0]->accumulate_gradient(accumulation);
            },
            make_intrusive<tensor_holder>(bools));
    return std::move(result);
}


TensorGrad TensorGrad_Functional_Class::to(const TensorGrad& x, DType dt){
    TensorGrad result(::nt::functional::to(x.detach(), dt), x.track_grad());
    if (!x.track_grad()) {
        result.track_grad_(false);
        return std::move(result);
    }
    DType original = x.dtype();
    result.track_tensors(x);
    result.create_backward_function(
        [original](const Tensor &grad,
           std::vector<intrusive_ptr<TensorGrad>> &parents) {
            parents[0]->accumulate_gradient( grad.to(original));
        });
    return std::move(result);
}

} // namespace functional
} // namespace nt
