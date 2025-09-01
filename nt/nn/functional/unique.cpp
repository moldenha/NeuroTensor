#include "../../functional/functional.h"
#include "../../Tensor.h"
#include "../../intrusive_ptr/intrusive_ptr.hpp"
#include "../TensorGrad.h"
#include "../functional_class.h"
#include "../../utils/macros.h"

namespace nt{
namespace functional{

TensorGrad TensorGrad_Functional_Class::unique(const TensorGrad& input, std::optional<int64_t> dim, bool return_sorted, bool return_indices){
    utils::throw_exception(return_sorted || return_indices, "Error: Expected to return sorted or return indices");
    if(!return_sorted){
        Tensor indices = ::nt::functional::unique(input.detach(), dim, false, true);
        return TensorGrad(indices, false);
    }
    auto [out, indices] = ::nt::get<2>(::nt::functional::unique(input.detach(), dim, return_sorted, return_indices));
    if(!input.track_grad()){
        if(return_indices){
            return functional::list(TensorGrad(out, false), TensorGrad(indices, false));
        }
        return TensorGrad(out, false);
    }
    if(!dim.has_value()){
        if(!return_indices)
            return input[indices];
        return functional::list(input[indices], TensorGrad(indices));
    }
    TensorGrad result = TensorGrad(out, input.track_grad());
    const auto& out_shape = out.shape();
    intrusive_ptr<tensor_holder> idx = make_intrusive<tensor_holder>(indices);
    const Tensor& __indices=  indices;
    result.track_grad(input, [&__indices, &dim, &out_shape](const Tensor& grad){
        Tensor i = ::nt::functional::transpose(grad, dim.value(), -1);
        Tensor splits = i.split_axis(-2);
        const Tensor* s_begin = reinterpret_cast<const Tensor*>(splits.data_ptr());
        Tensor out_grad = Tensor::makeNullTensorArray(__indices.numel());
        Tensor* o_begin = reinterpret_cast<Tensor*>(out_grad.data_ptr());
        Tensor* o_end = reinterpret_cast<Tensor*>(out_grad.data_ptr_end());
        const int64_t* begin = reinterpret_cast<const int64_t*>(__indices.data_ptr());
        const int64_t* end = reinterpret_cast<const int64_t*>(__indices.data_ptr_end());
        for(;begin != end; ++begin, ++o_begin){
            *o_begin = s_begin[*begin];
        }
        return ::nt::functional::cat_unordered(out_grad).view(out_shape);
    });
    if(!return_indices)
        return std::move(result);
    return functional::list(result, TensorGrad(indices));
}

}
}
