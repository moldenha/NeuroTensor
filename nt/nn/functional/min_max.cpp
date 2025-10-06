#include "../../functional/functional.h"
#include "../../Tensor.h"
#include "../../intrusive_ptr/intrusive_ptr.hpp"
#include "../TensorGrad.h"
#include "../functional_class.h"
#include "../../utils/macros.h"
#include <algorithm>

namespace nt{
namespace functional{

result_types::max<TensorGrad, Tensor> TensorGrad_Functional_Class::max(const TensorGrad& input, utils::optional_list dim, bool keepdim){
    result_types::max<Tensor, Tensor> out = ::nt::functional::max(input.detach(), dim, keepdim);
    Tensor& indices = out.indices;
    Tensor& vals = out.values;
    TensorGrad result(vals, input.track_grad());
    if(!input.track_grad()){
        result.track_grad_(false);
        return result_types::max<TensorGrad, Tensor>(result, indices);
    }
    result.track_grad(input, [&indices](Tensor& grad){return grad[indices];});
    return result_types::max<TensorGrad, Tensor>(result, indices);
}


result_types::max<TensorGrad, Tensor> TensorGrad_Functional_Class::min(const TensorGrad& input, utils::optional_list dim, bool keepdim){
    result_types::max<Tensor, Tensor> out = ::nt::functional::min(input.detach(), dim, keepdim);
    Tensor& indices = out.indices;
    Tensor& vals = out.values;
    TensorGrad result(vals, input.track_grad());
    if(!input.track_grad()){
        result.track_grad_(false);
        return result_types::max<TensorGrad, Tensor>(result, indices);
    }
    result.track_grad(input, [&indices](Tensor& grad){return grad[indices];});
    return result_types::max<TensorGrad, Tensor>(result, indices);

}

TensorGrad TensorGrad_Functional_Class::clamp(const TensorGrad &x,
                                                std::optional<Scalar> min,
                                                std::optional<Scalar> max) {
    TensorGrad result(::nt::functional::clamp(x.detach(), min, max), x.track_grad());
    if(!x.track_grad()){
        result.track_grad_(false);
        return std::move(result);
    }
    intrusive_ptr<tensor_holder> before = make_intrusive<tensor_holder>(x.detach().conditional_mutate_clone());
    result.track_tensors(x);
    intrusive_ptr<tensor_holder> after = make_intrusive<tensor_holder>(result.detach().clone());

    result.create_backward_function(
        [](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>> &parents,
           intrusive_ptr<tensor_holder> before, intrusive_ptr<tensor_holder> after){
            Tensor where = (before->tensor != after->tensor);
            Tensor grad_ = grad.clone();
            grad_[where] = 0;
            parents[0]->accumulate_gradient(grad_);
        }
    , before, after);
    return std::move(result);
}


TensorGrad& TensorGrad_Functional_Class::clamp_(TensorGrad &x,
                                                std::optional<Scalar> min,
                                                std::optional<Scalar> max) {
    if(!x.track_grad()){
        ::nt::functional::clamp_(x.detach(), min, max);
        return x;
    }
    
    intrusive_ptr<tensor_holder> before = make_intrusive<tensor_holder>(x.detach().conditional_mutate_clone());
    ::nt::functional::clamp_(x.detach(), min, max);
    intrusive_ptr<tensor_holder> after = make_intrusive<tensor_holder>(x.detach().clone());
    x.track_self_mod_tensors( 
        [before, after](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>> &parents){
            Tensor where = (before->tensor != after->tensor);
            Tensor relu_grad = ::nt::functional::ones_like(grad);
            relu_grad[where] = 0;
            parents[0]->accumulate_gradient(grad * relu_grad);
    }, "Clamp_");
    return x;
}




TensorGrad TensorGrad_Functional_Class::maximum(const std::vector<TensorGrad>& tgs, const std::vector<Tensor>& ts, const std::vector<Scalar>& ss){
    if(tgs.empty() || tgs.size() == 0){
        if(ss.empty())
            return TensorGrad(::nt::functional::maximum(ts), false);
        if(ts.empty())
            return TensorGrad(::nt::functional::maximum(ss), false);
        return TensorGrad(::nt::functional::maximum(ts, ::nt::functional::maximum(ss)), false);
    }
    if(std::any_of(tgs.cbegin(), tgs.cend(), [](const auto& tg){return !tg.track_grad();})){
        std::vector<TensorGrad> n_tgs;
        n_tgs.reserve(tgs.size());
        std::vector<Tensor> n_ts;
        n_ts.reserve(ts.size() + 1);

        std::copy_if(tgs.begin(), tgs.end(), std::back_inserter(n_tgs),
                     [](const TensorGrad& tg){return tg.track_grad();});
        for(auto begin = tgs.begin(); begin != tgs.end(); ++begin){
            if(!begin->track_grad())
                n_ts.emplace_back(begin->detach());
        }
        std::copy(ts.begin(), ts.end(), std::back_inserter(n_ts));
        return maximum(n_tgs, n_ts, ss);
    }
    std::vector<Tensor> all_tensors(tgs.size() + ts.size(), Tensor::Null());
    int64_t cntr = 0;
    for(const auto& tg : tgs){
        all_tensors[cntr] = tg.detach();
        ++cntr;
    }
    for(const auto& te : ts){
        all_tensors[cntr] = te;
        ++cntr;
    }
    bool do_scalars = (ss.size() > 0);
    Tensor out = (do_scalars) ? ::nt::functional::maximum(all_tensors, ::nt::functional::maximum(std::move(ss))) 
                                : ::nt::functional::maximum(all_tensors);
    intrusive_ptr<tensor_holder> result_th = make_intrusive<tensor_holder>(out.conditional_mutate_clone());
    TensorGrad result(out, true);
    result.track_tensors(const_cast<std::vector<TensorGrad>&>(tgs));
    intrusive_ptr<tensor_holder> holding = make_intrusive<tensor_holder>(Tensor::makeNullTensorArray(tgs.size()));
    Tensor* begin_h = reinterpret_cast<Tensor*>(holding->tensor.data_ptr());
    // Tensor* end_h = reinterpret_cast<Tensor*>(holding.data_ptr_end());
    for(auto begin = tgs.cbegin(); begin != tgs.end(); ++begin, ++begin_h){
        *begin_h = begin->detach().conditional_mutate_clone();
    }
    //function requires non-const vector in order to make sure the gradients are tracked properly
    result.create_backward_function(
        [](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>> &parents,
           intrusive_ptr<tensor_holder> tensors, intrusive_ptr<tensor_holder> result){
        Tensor* begin_h = reinterpret_cast<Tensor*>(tensors->tensor.data_ptr());
        Tensor* end_h = begin_h + tensors->tensor.numel();
        utils::THROW_EXCEPTION(tensors->tensor.numel() == parents.size(),
                               "Error with maximum gradient tracking got mismatch sizes of $ and $", 
                               tensors->tensor.numel(), parents.size());
        auto begin = parents.begin();
        for(;begin_h != end_h; ++begin_h, ++begin){
            Tensor mask = ::nt::functional::equal(*begin_h, result->tensor).to(grad.dtype());
            (*begin)->accumulate_gradient(mask * grad);
        }
    }, holding, result_th);
    return std::move(result);
}


TensorGrad TensorGrad_Functional_Class::minimum(const std::vector<TensorGrad>& tgs, const std::vector<Tensor>& ts, const std::vector<Scalar>& ss){
    if(tgs.empty() || tgs.size() == 0){
        if(ss.empty())
            return TensorGrad(::nt::functional::minimum(ts), false);
        if(ts.empty())
            return TensorGrad(::nt::functional::minimum(ss), false);
        return TensorGrad(::nt::functional::minimum(ts, ::nt::functional::minimum(ss)), false);
    }
    if(std::any_of(tgs.cbegin(), tgs.cend(), [](const auto& tg){return !tg.track_grad();})){
        std::vector<TensorGrad> n_tgs;
        n_tgs.reserve(tgs.size());
        std::vector<Tensor> n_ts;
        n_ts.reserve(ts.size() + 1);
        std::copy_if(tgs.begin(), tgs.end(), std::back_inserter(n_tgs),
                     [](const TensorGrad& tg){return tg.track_grad();});
        for(auto begin = tgs.begin(); begin != tgs.end(); ++begin){
            if(!begin->track_grad())
                n_ts.emplace_back(begin->detach());
        }
        std::copy(ts.begin(), ts.end(), std::back_inserter(n_ts));
        return minimum(n_tgs, n_ts, ss);
    }
    std::vector<Tensor> all_tensors(tgs.size() + ts.size(), Tensor::Null());
    int64_t cntr = 0;
    for(const auto& tg : tgs){
        all_tensors[cntr] = tg.detach();
        ++cntr;
    }
    for(const auto& te : ts){
        all_tensors[cntr] = te;
        ++cntr;
    }
    bool do_scalars = (ss.size() > 0);
    Tensor out = (do_scalars) ? ::nt::functional::minimum(std::move(all_tensors), ::nt::functional::minimum(std::move(ss))) : ::nt::functional::minimum(std::move(all_tensors));

    intrusive_ptr<tensor_holder> result_th = make_intrusive<tensor_holder>(out.conditional_mutate_clone());
    TensorGrad result(out, true);
    result.track_tensors(const_cast<std::vector<TensorGrad>&>(tgs));
    intrusive_ptr<tensor_holder> holding = make_intrusive<tensor_holder>(Tensor::makeNullTensorArray(tgs.size()));
    Tensor* begin_h = reinterpret_cast<Tensor*>(holding->tensor.data_ptr());
    // Tensor* end_h = reinterpret_cast<Tensor*>(holding.data_ptr_end());
    for(auto begin = tgs.cbegin(); begin != tgs.end(); ++begin, ++begin_h){
        *begin_h = begin->detach().conditional_mutate_clone();
    }
    //function requires non-const vector in order to make sure the gradients are tracked properly
    result.create_backward_function(
        [](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>> &parents,
           intrusive_ptr<tensor_holder> tensors, intrusive_ptr<tensor_holder> result){
        Tensor* begin_h = reinterpret_cast<Tensor*>(tensors->tensor.data_ptr());
        Tensor* end_h = begin_h + tensors->tensor.numel();
        utils::THROW_EXCEPTION(tensors->tensor.numel() == parents.size(),
                               "Error with minimum gradient tracking got mismatch sizes of $ and $", 
                               tensors->tensor.numel(), parents.size());
        auto begin = parents.begin();
        for(;begin_h != end_h; ++begin_h, ++begin){
            Tensor mask = ::nt::functional::equal(*begin_h, result->tensor).to(grad.dtype());
            (*begin)->accumulate_gradient(mask * grad);
        }
    }, holding, result_th);
    return std::move(result);
}

}
}
