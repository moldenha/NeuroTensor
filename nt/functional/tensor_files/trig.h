#ifndef __NT_FUNCTIONAL_TENSOR_FILES_TRIG_H__
#define __NT_FUNCTIONAL_TENSOR_FILES_TRIG_H__

#include "../../Tensor.h"

namespace nt{
namespace functional{

Tensor tan(const Tensor&);
Tensor tanh(const Tensor&);
Tensor atan(const Tensor&);
Tensor atanh(const Tensor&);
Tensor cotan(const Tensor&);
Tensor cotanh(const Tensor&);

Tensor sin(const Tensor&);
Tensor sinh(const Tensor&);
Tensor asin(const Tensor&);
Tensor asinh(const Tensor&);
Tensor csc(const Tensor&);
Tensor csch(const Tensor&);

Tensor cos(const Tensor&);
Tensor cosh(const Tensor&);
Tensor acos(const Tensor&);
Tensor acosh(const Tensor&);
Tensor sec(const Tensor&);
Tensor sech(const Tensor&);

Tensor dtan(const Tensor &);  // derivative of tan
Tensor dtanh(const Tensor &); // derivative of tanh
Tensor datan(const Tensor &);
Tensor datanh(const Tensor &);
Tensor dcotan(const Tensor &);
Tensor dcotanh(const Tensor &);

Tensor dcos(const Tensor &);
Tensor dcosh(const Tensor &);
Tensor dacos(const Tensor &);
Tensor dacosh(const Tensor &);
Tensor dsec(const Tensor &);
Tensor dsech(const Tensor &);

Tensor dsin(const Tensor &);
Tensor dsinh(const Tensor &);
Tensor dasin(const Tensor &);
Tensor dasinh(const Tensor &);
Tensor dcsc(const Tensor &);
Tensor dcsch(const Tensor &);

}
}

#endif
