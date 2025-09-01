#include "../../functional/functional.h"
#include "../../Tensor.h"
#include "../../intrusive_ptr/intrusive_ptr.hpp"
#include "../TensorGrad.h"
#include "../functional_class.h"
#include "../../utils/macros.h"

namespace nt{
namespace functional{

TensorGrad TensorGrad_Functional_Class::split(const TensorGrad& input, int64_t dim, utils::optional_list splitting){
    TensorGrad result(::nt::functional::split(input.detach(), dim, splitting), input.track_grad());
    result.track_grad(input, [&dim, &splitting](const Tensor& grad){
        return ::nt::functional::split(grad, dim, splitting);
    });
    return std::move(result);
}

TensorGrad TensorGrad_Functional_Class::chunk(const TensorGrad& input, const Tensor::size_value_t chunks, int64_t dim){
 TensorGrad result(::nt::functional::chunk(input.detach(), chunks, dim), input.track_grad());
    result.track_grad(input, [&dim, &chunks](const Tensor& grad){
        return ::nt::functional::chunk(grad, chunks, dim);
    });
    return std::move(result);
}

}
}
