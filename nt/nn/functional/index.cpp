#include "../../functional/functional.h"
#include "../../Tensor.h"
#include "../../intrusive_ptr/intrusive_ptr.hpp"
#include "../TensorGrad.h"
#include "../functional_class.h"

namespace nt{
namespace functional{

TensorGrad TensorGrad_Functional_Class::at(const TensorGrad& input, Tensor::size_value_t index){
    TensorGrad result(::nt::functional::at(input.detach(), index), input.track_grad());
    result.track_grad(input, [&index](Tensor& grad){
        return ::nt::functional::at(grad, index);
    });
    return std::move(result);
}
TensorGrad TensorGrad_Functional_Class::at(const TensorGrad& input, const Tensor& index){
    TensorGrad result(::nt::functional::at(input.detach(), index), input.track_grad());
    result.track_grad(input, [&index](Tensor& grad){
        return ::nt::functional::at(grad, index);
    });
    return std::move(result);
}
TensorGrad TensorGrad_Functional_Class::at(const TensorGrad& input, const TensorGrad& index){return TensorGrad_Functional_Class::at(input, index.detach());}
Tensor TensorGrad_Functional_Class::at(const Tensor& input, const TensorGrad& index){
    return ::nt::functional::at(input, index.detach());
}
TensorGrad TensorGrad_Functional_Class::at(const TensorGrad& input, std::vector<Tensor::size_value_t> index){
    TensorGrad result(::nt::functional::at(input.detach(), index));
    result.track_grad(input, [&index](Tensor& grad){
        return ::nt::functional::at(grad, index);
    });
    return std::move(result);
}
TensorGrad TensorGrad_Functional_Class::at_tensor_split(const TensorGrad & input, const TensorGrad & index, Tensor::size_value_t splitting){
    return TensorGrad_Functional_Class::at_tensor_split(input, index.detach(), splitting);
}
TensorGrad TensorGrad_Functional_Class::at_tensor_split(const TensorGrad & input, const Tensor & index, Tensor::size_value_t splitting){
    TensorGrad result(::nt::functional::at_tensor_split(input.detach(), index, splitting), input.track_grad());
    result.track_grad(input, [&index, &splitting](Tensor& grad){
        return ::nt::functional::at_tensor_split(grad, index, splitting);
    });
    return std::move(result);
}
TensorGrad& TensorGrad_Functional_Class::at_tensor_split(const TensorGrad & input, const TensorGrad & index, Tensor::size_value_t splitting,
                    TensorGrad& result){
    return TensorGrad_Functional_Class::at_tensor_split(input, index.detach(), splitting, result);
}

TensorGrad& TensorGrad_Functional_Class::at_tensor_split(const TensorGrad & input, const Tensor & index, Tensor::size_value_t splitting,
                    TensorGrad & result){
    utils::throw_exception(!result.is_null(), "Cannot perform operations on a null tensor");
    ::nt::functional::at_tensor_split(input.detach(), index, splitting, result.detach());
    result.track_self_mod_tensors(
        nullptr, "AtTensorSplit"
    );
    result.track_grad(input, [&index, &splitting](Tensor& grad){
        return ::nt::functional::at_tensor_split(grad, index, splitting);
    });
    return result;
}
TensorGrad TensorGrad_Functional_Class::index_except(const TensorGrad & input, int64_t dim, Tensor::size_value_t excluding_index){
    TensorGrad result(::nt::functional::index_except(input.detach(), dim, excluding_index), input.track_grad());
    result.track_grad(input, [&dim, &excluding_index](Tensor& grad){
        return ::nt::functional::index_except(grad, dim, excluding_index);
    });
    return std::move(result);
}
TensorGrad TensorGrad_Functional_Class::index_select(const TensorGrad & input, int64_t dim, const Tensor& index){
    TensorGrad result(::nt::functional::index_select(input.detach(), dim, index), input.track_grad());
    result.track_grad(input, [&dim, &index](Tensor& grad){
        return ::nt::functional::index_select(grad, dim, index);
    });
    return std::move(result);
}

TensorGrad TensorGrad_Functional_Class::index_select(const TensorGrad & input, int64_t dim, const TensorGrad& index){
    return TensorGrad_Functional_Class::index_select(input, dim, index.detach());
}
TensorGrad TensorGrad_Functional_Class::select(const TensorGrad& input, Tensor::size_value_t dim, Tensor::size_value_t index){
    TensorGrad result(::nt::functional::select(input.detach(), dim, index), input.track_grad());
    result.track_grad(input, [&dim, &index](Tensor& grad){
        return ::nt::functional::select(grad, dim, index);
    });
    return std::move(result);
}


}
}
