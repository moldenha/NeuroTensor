#include "../../functional/functional.h"
#include "../../Tensor.h"
#include "../../intrusive_ptr/intrusive_ptr.hpp"
#include "../TensorGrad.h"
#include "../functional_class.h"
#include "../../utils/macros.h"

namespace nt{
namespace functional{


TensorGrad TensorGrad_Functional_Class::sort(const TensorGrad& input, const Tensor::size_value_t dim,
        bool descending, bool return_sorted,
        bool return_indices){
    
    utils::throw_exception(return_sorted || return_indices, "Sort function must return indices or the sorted tensor");
    Tensor _indices = ::nt::functional::sort(input.detach(), dim, descending, /* return_sorted = */ false, /* return_indices = */ true);
    if(!return_sorted){
        return TensorGrad(_indices, false);
    }

    TensorGrad result(input.detach().flatten(0, -1)[_indices.flatten(0, -1)], input.track_grad());
    result.track_grad(input, [&_indices](const Tensor& grad){
        return grad.flatten(0, -1)[_indices.flatten(0, -1)];
    });
    if(!return_indices){
        return std::move(result);
    }
    return functional::list(result, TensorGrad(_indices, false));

}

TensorGrad TensorGrad_Functional_Class::coordsort(const TensorGrad& input, const Tensor::size_value_t dim, bool descending, 
                                            bool return_sorted, bool return_indices){
    utils::throw_exception(return_sorted || return_indices, "Sort function must return indices or the sorted tensor");
    const auto& shape = input.shape();
    int64_t per_dim = input.shape()[dim];
    if(!return_sorted){
        Tensor split = input.detach().split_axis(dim).view(-1, per_dim);
        return TensorGrad(::nt::functional::sort(split, -1, descending, false, true), false);
    }
    
    auto [sorted, _indices] = get<2>(::nt::functional::coordsort(input.detach(), dim, descending, /* return_sorted = */ true, /* return_indices = */ true));
    nt::Tensor& __indices = _indices;
    TensorGrad result(sorted, input.track_grad());
    result.track_grad(input, [&__indices, &per_dim, &dim](const Tensor& grad){
        Tensor split_grad = grad.split_axis(dim).view(-1, per_dim);
        Tensor sorted_grad = split_grad[__indices];
        return ::nt::functional::cat(std::move(sorted_grad));
    });
    if(!return_indices){
        return std::move(result);
    }
    return functional::list(result, TensorGrad(_indices, false));

}


}
}
