#include "../../functional/functional.h"
#include "../../Tensor.h"
#include "../../intrusive_ptr/intrusive_ptr.hpp"
#include "../TensorGrad.h"
#include "../functional_class.h"
#include "../../utils/macros.h"

namespace nt{
namespace functional{

TensorGrad TensorGrad_Functional_Class::diagonal(const TensorGrad& input, bool keep_dims){
    if(!input.track_grad()) return TensorGrad(::nt::functional::diagonal(input.detach(), keep_dims), false); 
    TensorGrad result = input.create_view_node(::nt::functional::diagonal(input.detach(), keep_dims));
    result.create_view_backward_function(input, [&keep_dims](const Tensor& grad){
        return ::nt::functional::diagonal(grad, keep_dims);
    });
    return std::move(result);
}
TensorGrad TensorGrad_Functional_Class::as_strided(const TensorGrad &input, const SizeRef n_size, SizeRef n_stride,
              const int64_t storage_offset, bool whole_tensor){
    if(!input.track_grad())
        return TensorGrad(::nt::functional::as_strided(input.detach(), n_size, n_stride, storage_offset, whole_tensor), false); 
    TensorGrad result = input.create_view_node(::nt::functional::as_strided(input.detach(), n_size, n_stride, storage_offset, whole_tensor));
    result.create_view_backward_function(input, [&n_size, &n_stride, &storage_offset, whole_tensor](const Tensor& grad){
        return ::nt::functional::as_strided(grad, n_size, n_stride, storage_offset, whole_tensor);
    });
    return std::move(result);

}

}
}
