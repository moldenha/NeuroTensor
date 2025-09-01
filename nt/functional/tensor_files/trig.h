#ifndef NT_FUNCTIONAL_TENSOR_FILES_TRIG_H__
#define NT_FUNCTIONAL_TENSOR_FILES_TRIG_H__

#include "../../Tensor.h"

namespace nt {
namespace functional {

NEUROTENSOR_API Tensor tan(const Tensor &);
NEUROTENSOR_API Tensor tanh(const Tensor &);
NEUROTENSOR_API Tensor atan(const Tensor &);
NEUROTENSOR_API Tensor atanh(const Tensor &);
NEUROTENSOR_API Tensor cotan(const Tensor &);
NEUROTENSOR_API Tensor cotanh(const Tensor &);

NEUROTENSOR_API Tensor sin(const Tensor &);
NEUROTENSOR_API Tensor sinh(const Tensor &);
NEUROTENSOR_API Tensor asin(const Tensor &);
NEUROTENSOR_API Tensor asinh(const Tensor &);
NEUROTENSOR_API Tensor csc(const Tensor &);
NEUROTENSOR_API Tensor csch(const Tensor &);

NEUROTENSOR_API Tensor cos(const Tensor &);
NEUROTENSOR_API Tensor cosh(const Tensor &);
NEUROTENSOR_API Tensor acos(const Tensor &);
NEUROTENSOR_API Tensor acosh(const Tensor &);
NEUROTENSOR_API Tensor sec(const Tensor &);
NEUROTENSOR_API Tensor sech(const Tensor &);


NEUROTENSOR_API Tensor& tan_(Tensor &);
NEUROTENSOR_API Tensor& tanh_(Tensor &);
NEUROTENSOR_API Tensor& atan_(Tensor &);
NEUROTENSOR_API Tensor& atanh_(Tensor &);
NEUROTENSOR_API Tensor& cotan_(Tensor &);
NEUROTENSOR_API Tensor& cotanh_(Tensor &);

NEUROTENSOR_API Tensor& sin_(Tensor &);
NEUROTENSOR_API Tensor& sinh_(Tensor &);
NEUROTENSOR_API Tensor& asin_(Tensor &);
NEUROTENSOR_API Tensor& asinh_(Tensor &);
NEUROTENSOR_API Tensor& csc_(Tensor &);
NEUROTENSOR_API Tensor& csch_(Tensor &);

NEUROTENSOR_API Tensor& cos_(Tensor &);
NEUROTENSOR_API Tensor& cosh_(Tensor &);
NEUROTENSOR_API Tensor& acos_(Tensor &);
NEUROTENSOR_API Tensor& acosh_(Tensor &);
NEUROTENSOR_API Tensor& sec_(Tensor &);
NEUROTENSOR_API Tensor& sech_(Tensor &);

NEUROTENSOR_API Tensor dtan(const Tensor &);  // derivative of tan
NEUROTENSOR_API Tensor dtanh(const Tensor &); // derivative of tanh
NEUROTENSOR_API Tensor datan(const Tensor &);
NEUROTENSOR_API Tensor datanh(const Tensor &);
NEUROTENSOR_API Tensor dcotan(const Tensor &);
NEUROTENSOR_API Tensor dcotanh(const Tensor &);

NEUROTENSOR_API Tensor dcos(const Tensor &);
NEUROTENSOR_API Tensor dcosh(const Tensor &);
NEUROTENSOR_API Tensor dacos(const Tensor &);
NEUROTENSOR_API Tensor dacosh(const Tensor &);
NEUROTENSOR_API Tensor dsec(const Tensor &);
NEUROTENSOR_API Tensor dsech(const Tensor &);

NEUROTENSOR_API Tensor dsin(const Tensor &);
NEUROTENSOR_API Tensor dsinh(const Tensor &);
NEUROTENSOR_API Tensor dasin(const Tensor &);
NEUROTENSOR_API Tensor dasinh(const Tensor &);
NEUROTENSOR_API Tensor dcsc(const Tensor &);
NEUROTENSOR_API Tensor dcsch(const Tensor &);

} // namespace functional
} // namespace nt

#endif
