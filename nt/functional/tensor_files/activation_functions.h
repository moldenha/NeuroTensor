#ifndef NT_FUNCTIONAL_TENSOR_FILES_ACTIVATION_FUNCTIONS_H__
#define NT_FUNCTIONAL_TENSOR_FILES_ACTIVATION_FUNCTIONS_H__

#include "../../Tensor.h"
#include "../../dtype/Scalar.h"

namespace nt{
namespace functional{

NEUROTENSOR_API Tensor sigmoid(const Tensor&);
NEUROTENSOR_API Tensor& sigmoid_(Tensor&);
NEUROTENSOR_API Tensor dsigmoid(const Tensor&, bool apply_sigmoid = true);
NEUROTENSOR_API Tensor& dsigmoid_(Tensor&, bool apply_sigmoid = true);
NEUROTENSOR_API Tensor sqrt(const Tensor &);
NEUROTENSOR_API Tensor& sqrt_(Tensor &);
NEUROTENSOR_API Tensor dsqrt(const Tensor &);
NEUROTENSOR_API Tensor& dsqrt_(Tensor &);
NEUROTENSOR_API Tensor invsqrt(const Tensor &);  // 1 / sqrt(x);
NEUROTENSOR_API Tensor& invsqrt_(Tensor &);  // 1 / sqrt(x);
NEUROTENSOR_API Tensor dinvsqrt(const Tensor &); // derivative of invsqrt
NEUROTENSOR_API Tensor& dinvsqrt_(Tensor &); // derivative of invsqrt
NEUROTENSOR_API Tensor pow(const Tensor&, Scalar);
NEUROTENSOR_API Tensor& pow_(Tensor&, Scalar);
NEUROTENSOR_API Tensor abs(const Tensor& ); // absolte value
NEUROTENSOR_API Tensor& abs_(Tensor& ); // absolte value
NEUROTENSOR_API Tensor softplus(const Tensor &x, Scalar beta = 1.0, Scalar threshold = 20.0);
NEUROTENSOR_API Tensor& softplus_(Tensor &x, Scalar beta = 1.0, Scalar threshold = 20.0);

NEUROTENSOR_API Tensor relu(const Tensor &);
NEUROTENSOR_API Tensor& relu_(Tensor &);
NEUROTENSOR_API Tensor silu(const Tensor &);
NEUROTENSOR_API Tensor dsilu(const Tensor &);
NEUROTENSOR_API Tensor& silu_(Tensor &);
NEUROTENSOR_API Tensor& dsilu_(Tensor &);
NEUROTENSOR_API Tensor gelu(const Tensor &);
NEUROTENSOR_API Tensor dgelu(const Tensor &);
NEUROTENSOR_API Tensor& gelu_(Tensor &);
NEUROTENSOR_API Tensor& dgelu_(Tensor &);

}
}


#endif
