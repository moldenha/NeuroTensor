#include "../../functional/functional.h"
#include "../../Tensor.h"
#include "../../intrusive_ptr/intrusive_ptr.hpp"
#include "../TensorGrad.h"
#include "../functional_class.h"
#include "../../utils/macros.h"
#include <set>

namespace nt{
namespace functional{

TensorGrad TensorGrad_Functional_Class::logsumexp(const TensorGrad& x, utils::optional_list list, bool keepdim){
    if (!x.track_grad()) return TensorGrad(::nt::functional::logsumexp(x.detach(), list, keepdim), false); 
    
    TensorGrad result = TensorGrad::create_read_node(::nt::functional::logsumexp(x.detach(), list, keepdim), x);
    result.create_read_backward_function(
            [list](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents,
                 intrusive_ptr<tensor_holder> saved_x) {
                parents[0]->accumulate_gradient(::nt::functional::dlogsumexp(grad, saved_x->tensor, list).view(parents[0]->shape()) );
            },
            make_intrusive<tensor_holder>(x.detach().conditional_mutate_clone()));
    return std::move(result);
}

TensorGrad TensorGrad_Functional_Class::log(const TensorGrad& x){
    if (!x.track_grad()) TensorGrad(::nt::functional::log(x.detach()), false); 
    
    TensorGrad result = TensorGrad::create_read_node(::nt::functional::log(x.detach()), x);
    result.create_read_backward_function(
            [](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents,
                 intrusive_ptr<tensor_holder> saved_x) {
                parents[0]->accumulate_gradient(grad * ::nt::functional::dlog(saved_x->tensor));
            },
            make_intrusive<tensor_holder>(x.detach().conditional_mutate_clone()));
    return std::move(result);
}

TensorGrad& TensorGrad_Functional_Class::log_(TensorGrad& x){
    if(!x.track_grad()){
        ::nt::functional::log_(x.detach());
        return x;
    }
    // should be mutable if this function is called
    intrusive_ptr<tensor_holder> this_clone =
        make_intrusive<tensor_holder>(x.detach().conditional_mutate_clone());
    ::nt::functional::log_(x.detach());
    x.create_write_node(
        [a = std::move(this_clone)](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents) {
            parents[0]->accumulate_gradient(::nt::functional::multiply(::nt::functional::dlog(a->tensor), grad));
        },
        __func__);
    return x;
}


TensorGrad TensorGrad_Functional_Class::exp(const TensorGrad& x){
    if (!x.track_grad()) return TensorGrad(::nt::functional::exp(x.detach()), false); 
    
    TensorGrad result = TensorGrad::create_read_node(::nt::functional::exp(x.detach()), x);
    result.create_read_backward_function(
            [](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents,
                 intrusive_ptr<tensor_holder> saved_x) {
                parents[0]->accumulate_gradient(grad * saved_x->tensor);
            },
            make_intrusive<tensor_holder>(result.detach().conditional_mutate_clone()));
    return std::move(result);
}


TensorGrad& TensorGrad_Functional_Class::exp_(TensorGrad& x){
    ::nt::functional::exp_(x.detach());
    if(!x.track_grad()){
        return x;
    }

    // clone the tensor with the exp(tensor) function already applied
    // this will save computational time on the way backward
    // because the gradient of exp(x) is exp(x)
    intrusive_ptr<tensor_holder> this_clone =
        make_intrusive<tensor_holder>(x.detach().conditional_mutate_clone());
    x.create_write_node(
        [a = std::move(this_clone)](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents) {
            parents[0]->accumulate_gradient(::nt::functional::multiply(a->tensor, grad));
        },
        __func__);
    return x;
}


TensorGrad TensorGrad_Functional_Class::sum(const TensorGrad& input, utils::optional_list list, bool keepdim){
    Tensor summed = ::nt::functional::sum(input.detach(), list, true);
    std::vector<int64_t> dims = summed.shape().Vec();
    SizeRef out_shape = summed.shape().clone();
    if(!keepdim){
        std::set<int64_t> remove_set(list.cbegin(), list.cend());
        std::vector<int64_t> n_shape;
        n_shape.reserve(input.dims() - list->size());
        const auto& shape = input.shape();
        for(int64_t i = 0; i < shape.size(); ++i){
            if(remove_set.find(i) == remove_set.end()) {
                n_shape.push_back(shape[i]);
            }
        }
        out_shape = SizeRef(n_shape);
    }
    if(!input.track_grad()) return TensorGrad(summed.view(out_shape), false);
    TensorGrad result = TensorGrad::create_read_node(summed.view(out_shape), input);
    SizeRef original_shape = input.shape().clone();
    // define the backward function
    result.create_read_backward_function(
        [dims, original_shape](const Tensor &grad,
               std::vector<intrusive_ptr<TensorGrad>> &parents) {
            // repeat the gradient along the summed dimension
            parents[0]->accumulate_gradient(grad.view(SizeRef(std::move(dims)))
                                        .expand(original_shape));
        });

    return std::move(result);

}


}
}
