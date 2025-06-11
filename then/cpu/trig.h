#ifndef __NT_FUNCTIONAL_CPU_TRIG_H__
#define __NT_FUNCTIONAL_CPU_TRIG_H__

#include "../../dtype/ArrayVoid.h"

namespace nt{
namespace functional{
namespace cpu{

void _tan(const ArrayVoid&, ArrayVoid&);
void _tanh(const ArrayVoid&, ArrayVoid&);
void _atan(const ArrayVoid&, ArrayVoid&);
void _atanh(const ArrayVoid&, ArrayVoid&);
void _cotan(const ArrayVoid&, ArrayVoid&);
void _cotanh(const ArrayVoid&, ArrayVoid&);

void _sin(const ArrayVoid&, ArrayVoid&);
void _sinh(const ArrayVoid&, ArrayVoid&);
void _asin(const ArrayVoid&, ArrayVoid&);
void _asinh(const ArrayVoid&, ArrayVoid&);
void _csc(const ArrayVoid&, ArrayVoid&);
void _csch(const ArrayVoid&, ArrayVoid&);

void _cos(const ArrayVoid&, ArrayVoid&);
void _cosh(const ArrayVoid&, ArrayVoid&);
void _acos(const ArrayVoid&, ArrayVoid&);
void _acosh(const ArrayVoid&, ArrayVoid&);
void _sec(const ArrayVoid&, ArrayVoid&);
void _sech(const ArrayVoid&, ArrayVoid&);


void _tan_(ArrayVoid&);
void _tanh_(ArrayVoid&);
void _atan_(ArrayVoid&);
void _atanh_(ArrayVoid&);
void _cotan_(ArrayVoid&);
void _cotanh_(ArrayVoid&);

void _sin_(ArrayVoid&);
void _sinh_(ArrayVoid&);
void _asin_(ArrayVoid&);
void _asinh_(ArrayVoid&);
void _csc_(ArrayVoid&);
void _csch_(ArrayVoid&);

void _cos_(ArrayVoid&);
void _cosh_(ArrayVoid&);
void _acos_(ArrayVoid&);
void _acosh_(ArrayVoid&);
void _sec_(ArrayVoid&);
void _sech_(ArrayVoid&);

void _dtan(const ArrayVoid&, ArrayVoid&);
void _dtan_(ArrayVoid&);
void _dtanh(const ArrayVoid&, ArrayVoid&);
void _dtanh_(ArrayVoid&);
void _datan(const ArrayVoid&, ArrayVoid&);
void _datan_(ArrayVoid&);
void _datanh(const ArrayVoid&, ArrayVoid&);
void _datanh_(ArrayVoid&);
void _dcotan(const ArrayVoid&, ArrayVoid&);
void _dcotan_(ArrayVoid&);
void _dcotanh(const ArrayVoid&, ArrayVoid&);
void _dcotanh_(ArrayVoid&);


inline void _dsin(const ArrayVoid& a, ArrayVoid& b){_cos(a, b);}
inline void _dsin_(ArrayVoid& a){_cos_(a);}
inline void _dsinh(const ArrayVoid& a, ArrayVoid& b){_cosh(a, b);}
inline void _dsinh_(ArrayVoid& a){_cosh_(a);}
void _dasin(const ArrayVoid&, ArrayVoid&);
void _dasin_(ArrayVoid&);
void _dasinh(const ArrayVoid&, ArrayVoid&);
void _dasinh_(ArrayVoid&);
void _dcsc(const ArrayVoid&, ArrayVoid&);
void _dcsc_(ArrayVoid&);
void _dcsch(const ArrayVoid&, ArrayVoid&);
void _dcsch_(ArrayVoid&);


void _dcos(const ArrayVoid&, ArrayVoid&);
void _dcos_(ArrayVoid&);
inline void _dcosh(const ArrayVoid& a, ArrayVoid& b){_sinh(a, b);}
inline void _dcosh_(ArrayVoid& a) {_cosh_(a);}
void _dacos(const ArrayVoid&, ArrayVoid&);
void _dacos_(ArrayVoid&);
void _dacosh(const ArrayVoid&, ArrayVoid&);
void _dacosh_(ArrayVoid&);
void _dsec(const ArrayVoid&, ArrayVoid&);
void _dsec_(ArrayVoid&);
void _dsech(const ArrayVoid&, ArrayVoid&);
void _dsech_(ArrayVoid&);


}
}
}

#endif
