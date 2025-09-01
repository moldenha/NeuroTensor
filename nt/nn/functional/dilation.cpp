#include "../../functional/functional.h"
#include "../../Tensor.h"
#include "../../intrusive_ptr/intrusive_ptr.hpp"
#include "../TensorGrad.h"
#include "../functional_class.h"
#include "../../utils/macros.h"

namespace nt{
namespace functional{

TensorGrad TensorGrad_Functional_Class::dilate(const TensorGrad& x, Tensor::size_value_t a){
    TensorGrad result(::nt::functional::dilate(x.detach(), a));
    if (!x.track_grad()) {
        result.track_grad_(false);
        return std::move(result);
    }
    result.track_tensors(x);
    result.create_backward_function(
        [a](const Tensor &grad,
              std::vector<intrusive_ptr<TensorGrad>> &parents) {
            parents[0]->accumulate_gradient( ::nt::functional::undilate_(grad, a) );
        });
    return std::move(result);
}

TensorGrad TensorGrad_Functional_Class::dilate(const TensorGrad& x, Tensor::size_value_t a, Tensor::size_value_t b){
    TensorGrad result(::nt::functional::dilate(x.detach(), a, b));
    if (!x.track_grad()) {
        result.track_grad_(false);
        return std::move(result);
    }
    result.track_tensors(x);
    result.create_backward_function(
        [a, b](const Tensor &grad,
              std::vector<intrusive_ptr<TensorGrad>> &parents) {
            parents[0]->accumulate_gradient( ::nt::functional::undilate_(grad, a, b) );
        });
    return std::move(result);
}


TensorGrad TensorGrad_Functional_Class::dilate(const TensorGrad& x, Tensor::size_value_t a, Tensor::size_value_t b, Tensor::size_value_t c){
    TensorGrad result(::nt::functional::dilate(x.detach(), a, b, c));
    if (!x.track_grad()) {
        result.track_grad_(false);
        return std::move(result);
    }
    result.track_tensors(x);
    result.create_backward_function(
        [a, b, c](const Tensor &grad,
              std::vector<intrusive_ptr<TensorGrad>> &parents) {
            parents[0]->accumulate_gradient( ::nt::functional::undilate_(grad, a, b, c) );
        });
    return std::move(result);
}

TensorGrad TensorGrad_Functional_Class::undilate(const TensorGrad& x, Tensor::size_value_t a){
    TensorGrad result(::nt::functional::undilate(x.detach(), a));
    if (!x.track_grad()) {
        result.track_grad_(false);
        return std::move(result);
    }
    result.track_tensors(x);
    result.create_backward_function(
        [a](const Tensor &grad,
              std::vector<intrusive_ptr<TensorGrad>> &parents) {
            parents[0]->accumulate_gradient( ::nt::functional::dilate(grad, a) );
        });
    return std::move(result);
}

TensorGrad TensorGrad_Functional_Class::undilate(const TensorGrad& x, Tensor::size_value_t a, Tensor::size_value_t b){
    TensorGrad result(::nt::functional::undilate(x.detach(), a, b));
    if (!x.track_grad()) {
        result.track_grad_(false);
        return std::move(result);
    }
    result.track_tensors(x);
    result.create_backward_function(
        [a, b](const Tensor &grad,
              std::vector<intrusive_ptr<TensorGrad>> &parents) {
            parents[0]->accumulate_gradient( ::nt::functional::dilate(grad, a, b) );
        });
    return std::move(result);
}


TensorGrad TensorGrad_Functional_Class::undilate(const TensorGrad& x, Tensor::size_value_t a, Tensor::size_value_t b, Tensor::size_value_t c){
    TensorGrad result(::nt::functional::undilate(x.detach(), a, b, c));
    if (!x.track_grad()) {
        result.track_grad_(false);
        return std::move(result);
    }
    result.track_tensors(x);
    result.create_backward_function(
        [a, b, c](const Tensor &grad,
              std::vector<intrusive_ptr<TensorGrad>> &parents) {
            parents[0]->accumulate_gradient( ::nt::functional::dilate(grad, a, b, c) );
        });
    return std::move(result);
}



TensorGrad TensorGrad_Functional_Class::undilate_(const TensorGrad& x, Tensor::size_value_t a){
    TensorGrad result(::nt::functional::undilate_(x.detach(), a));
    result.track_grad(x, [a](Tensor& grad){
        return ::nt::functional::undilate_(grad, a);
    });
    return std::move(result);
}
TensorGrad TensorGrad_Functional_Class::undilate_(const TensorGrad& x, Tensor::size_value_t a, Tensor::size_value_t b){
    TensorGrad result(::nt::functional::undilate_(x.detach(), a, b));
    result.track_grad(x, [a, b](Tensor& grad){
        return ::nt::functional::undilate_(grad, a, b);
    });
    return std::move(result);
}
TensorGrad TensorGrad_Functional_Class::undilate_(const TensorGrad& x, Tensor::size_value_t a, Tensor::size_value_t b, Tensor::size_value_t c){
    TensorGrad result(::nt::functional::undilate_(x.detach(), a, b, c));
    result.track_grad(x, [a, b, c](Tensor& grad){
        return ::nt::functional::undilate_(grad, a, b, c);
    });
    return std::move(result);
}


}
}
