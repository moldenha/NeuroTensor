#ifndef __NT_FUNCTIONAL_TENSOR_FILES_ACTIVATION_FUNCTIONS_H__
#define __NT_FUNCTIONAL_TENSOR_FILES_ACTIVATION_FUNCTIONS_H__

#include "../../Tensor.h"
#include "../../dtype/Scalar.h"

namespace nt{
namespace functional{

Tensor sigmoid(const Tensor&);
Tensor& sigmoid_(Tensor&);
Tensor dsigmoid(const Tensor&, bool apply_sigmoid = true);
Tensor& dsigmoid_(Tensor&, bool apply_sigmoid = true);
Tensor sqrt(const Tensor &);
Tensor& sqrt_(Tensor &);
Tensor dsqrt(const Tensor &);
Tensor& dsqrt_(Tensor &);
Tensor invsqrt(const Tensor &);  // 1 / sqrt(x);
Tensor& invsqrt_(Tensor &);  // 1 / sqrt(x);
Tensor dinvsqrt(const Tensor &); // derivative of invsqrt
Tensor& dinvsqrt_(Tensor &); // derivative of invsqrt
Tensor pow(const Tensor&, Scalar);
Tensor& pow_(Tensor&, Scalar);
Tensor abs(const Tensor& ); // absolte value
Tensor& abs_(Tensor& ); // absolte value
Tensor softplus(const Tensor &x, Scalar beta = 1.0, Scalar threshold = 20.0);
Tensor& softplus(Tensor &x, Scalar beta = 1.0, Scalar threshold = 20.0);

Tensor relu(const Tensor &);
Tensor& relu_(Tensor &);
Tensor silu(const Tensor &);
Tensor dsilu(const Tensor &);
Tensor& silu_(Tensor &);
Tensor& dsilu_(Tensor &);
Tensor gelu(const Tensor &);
Tensor dgelu(const Tensor &);
Tensor& gelu_(Tensor &);
Tensor& dgelu_(Tensor &);

}
}


#endif
