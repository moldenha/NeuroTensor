#include "../../functional/functional.h"
#include "../../Tensor.h"
#include "../../intrusive_ptr/intrusive_ptr.hpp"
#include "../TensorGrad.h"
#include "../functional_class.h"
#include "../../utils/macros.h"

namespace nt{
namespace functional{

TensorGrad TensorGrad_Functional_Class::transpose(const TensorGrad& input, Tensor::size_value_t a, Tensor::size_value_t b){
    TensorGrad result(::nt::functional::transpose(input.detach(), a, b), input.track_grad());
    result.track_grad(input, [a, b](const Tensor& grad){
        return ::nt::functional::transpose(grad, a, b);
    });
    return std::move(result);
}

TensorGrad TensorGrad_Functional_Class::permute(const TensorGrad& input, std::vector<Tensor::size_value_t> permutations){
    TensorGrad result(::nt::functional::permute(input.detach(), permutations), input.track_grad());
    result.track_grad(input, [&permutations](const Tensor& grad){
        return ::nt::functional::permute(grad, permutations);
    });
    return std::move(result);
}

TensorGrad& TensorGrad_Functional_Class::row_col_swap_(TensorGrad& input){
    ::nt::functional::row_col_swap_(input.detach());
    if(!input.track_grad()){
        return input;
    }
    input.track_self_mod_tensors(
        [](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents) {
            parents[0]->accumulate_gradient(::nt::functional::transpose(grad, -1, -2));
        }, __func__);
    return input;
}


}
}
