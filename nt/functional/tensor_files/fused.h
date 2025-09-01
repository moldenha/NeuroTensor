#ifndef NT_FUNCTIONAL_TENSOR_FILES_FUSED_H__
#define NT_FUNCTIONAL_TENSOR_FILES_FUSED_H__

#include "../../Tensor.h"
#include "../../dtype/Scalar.h"

namespace nt{
namespace functional{

//returns c + (a * b);
NEUROTENSOR_API Tensor fused_multiply_add(const Tensor& c, const Tensor& a, const Tensor& b);
NEUROTENSOR_API Tensor fused_multiply_add(const Tensor& c, const Tensor& a, Scalar b);
//returns c += (a * b);
NEUROTENSOR_API Tensor& fused_multiply_add_(Tensor& c, const Tensor& a, const Tensor& b);
NEUROTENSOR_API Tensor& fused_multiply_add_(Tensor& c, const Tensor& a, Scalar b);
//returns c - (a * b);
NEUROTENSOR_API Tensor fused_multiply_subtract(const Tensor& c, const Tensor& a, const Tensor& b);
NEUROTENSOR_API Tensor fused_multiply_subtract(const Tensor& c, const Tensor& a, Scalar b);
//returns c -= (a * b);
NEUROTENSOR_API Tensor& fused_multiply_subtract_(Tensor& c, const Tensor& a, const Tensor& b);
NEUROTENSOR_API Tensor& fused_multiply_subtract_(Tensor& c, const Tensor& a, Scalar b);


}
}


#endif
