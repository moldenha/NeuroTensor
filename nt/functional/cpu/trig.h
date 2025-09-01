#ifndef NT_FUNCTIONAL_CPU_TRIG_H__
#define NT_FUNCTIONAL_CPU_TRIG_H__

#include "../../dtype/ArrayVoid.h"
#include "../../utils/always_inline_macro.h"

namespace nt{
namespace functional{
namespace cpu{

NEUROTENSOR_API void _tan(const ArrayVoid&, ArrayVoid&);
NEUROTENSOR_API void _tanh(const ArrayVoid&, ArrayVoid&);
NEUROTENSOR_API void _atan(const ArrayVoid&, ArrayVoid&);
NEUROTENSOR_API void _atanh(const ArrayVoid&, ArrayVoid&);
NEUROTENSOR_API void _cotan(const ArrayVoid&, ArrayVoid&);
NEUROTENSOR_API void _cotanh(const ArrayVoid&, ArrayVoid&);

NEUROTENSOR_API void _sin(const ArrayVoid&, ArrayVoid&);
NEUROTENSOR_API void _sinh(const ArrayVoid&, ArrayVoid&);
NEUROTENSOR_API void _asin(const ArrayVoid&, ArrayVoid&);
NEUROTENSOR_API void _asinh(const ArrayVoid&, ArrayVoid&);
NEUROTENSOR_API void _csc(const ArrayVoid&, ArrayVoid&);
NEUROTENSOR_API void _csch(const ArrayVoid&, ArrayVoid&);

NEUROTENSOR_API void _cos(const ArrayVoid&, ArrayVoid&);
NEUROTENSOR_API void _cosh(const ArrayVoid&, ArrayVoid&);
NEUROTENSOR_API void _acos(const ArrayVoid&, ArrayVoid&);
NEUROTENSOR_API void _acosh(const ArrayVoid&, ArrayVoid&);
NEUROTENSOR_API void _sec(const ArrayVoid&, ArrayVoid&);
NEUROTENSOR_API void _sech(const ArrayVoid&, ArrayVoid&);


NEUROTENSOR_API void _tan_(ArrayVoid&);
NEUROTENSOR_API void _tanh_(ArrayVoid&);
NEUROTENSOR_API void _atan_(ArrayVoid&);
NEUROTENSOR_API void _atanh_(ArrayVoid&);
NEUROTENSOR_API void _cotan_(ArrayVoid&);
NEUROTENSOR_API void _cotanh_(ArrayVoid&);

NEUROTENSOR_API void _sin_(ArrayVoid&);
NEUROTENSOR_API void _sinh_(ArrayVoid&);
NEUROTENSOR_API void _asin_(ArrayVoid&);
NEUROTENSOR_API void _asinh_(ArrayVoid&);
NEUROTENSOR_API void _csc_(ArrayVoid&);
NEUROTENSOR_API void _csch_(ArrayVoid&);

NEUROTENSOR_API void _cos_(ArrayVoid&);
NEUROTENSOR_API void _cosh_(ArrayVoid&);
NEUROTENSOR_API void _acos_(ArrayVoid&);
NEUROTENSOR_API void _acosh_(ArrayVoid&);
NEUROTENSOR_API void _sec_(ArrayVoid&);
NEUROTENSOR_API void _sech_(ArrayVoid&);

NEUROTENSOR_API void _dtan(const ArrayVoid&, ArrayVoid&);
NEUROTENSOR_API void _dtan_(ArrayVoid&);
NEUROTENSOR_API void _dtanh(const ArrayVoid&, ArrayVoid&);
NEUROTENSOR_API void _dtanh_(ArrayVoid&);
NEUROTENSOR_API void _datan(const ArrayVoid&, ArrayVoid&);
NEUROTENSOR_API void _datan_(ArrayVoid&);
NEUROTENSOR_API void _datanh(const ArrayVoid&, ArrayVoid&);
NEUROTENSOR_API void _datanh_(ArrayVoid&);
NEUROTENSOR_API void _dcotan(const ArrayVoid&, ArrayVoid&);
NEUROTENSOR_API void _dcotan_(ArrayVoid&);
NEUROTENSOR_API void _dcotanh(const ArrayVoid&, ArrayVoid&);
NEUROTENSOR_API void _dcotanh_(ArrayVoid&);


NT_ALWAYS_INLINE void _dsin(const ArrayVoid& a, ArrayVoid& b){_cos(a, b);}
NT_ALWAYS_INLINE void _dsin_(ArrayVoid& a){_cos_(a);}
NT_ALWAYS_INLINE void _dsinh(const ArrayVoid& a, ArrayVoid& b){_cosh(a, b);}
NT_ALWAYS_INLINE void _dsinh_(ArrayVoid& a){_cosh_(a);}
NEUROTENSOR_API void _dasin(const ArrayVoid&, ArrayVoid&);
NEUROTENSOR_API void _dasin_(ArrayVoid&);
NEUROTENSOR_API void _dasinh(const ArrayVoid&, ArrayVoid&);
NEUROTENSOR_API void _dasinh_(ArrayVoid&);
NEUROTENSOR_API void _dcsc(const ArrayVoid&, ArrayVoid&);
NEUROTENSOR_API void _dcsc_(ArrayVoid&);
NEUROTENSOR_API void _dcsch(const ArrayVoid&, ArrayVoid&);
NEUROTENSOR_API void _dcsch_(ArrayVoid&);


NEUROTENSOR_API void _dcos(const ArrayVoid&, ArrayVoid&);
NEUROTENSOR_API void _dcos_(ArrayVoid&);
NT_ALWAYS_INLINE void _dcosh(const ArrayVoid& a, ArrayVoid& b){_sinh(a, b);}
NT_ALWAYS_INLINE void _dcosh_(ArrayVoid& a) {_sinh_(a);}
NEUROTENSOR_API void _dacos(const ArrayVoid&, ArrayVoid&);
NEUROTENSOR_API void _dacos_(ArrayVoid&);
NEUROTENSOR_API void _dacosh(const ArrayVoid&, ArrayVoid&);
NEUROTENSOR_API void _dacosh_(ArrayVoid&);
NEUROTENSOR_API void _dsec(const ArrayVoid&, ArrayVoid&);
NEUROTENSOR_API void _dsec_(ArrayVoid&);
NEUROTENSOR_API void _dsech(const ArrayVoid&, ArrayVoid&);
NEUROTENSOR_API void _dsech_(ArrayVoid&);


}
}
}

#endif
