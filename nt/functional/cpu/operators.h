#ifndef NT_FUNCTIONAL_CPU_OPERATORS_ARRAY_VOID_H__
#define NT_FUNCTIONAL_CPU_OPERATORS_ARRAY_VOID_H__
#include "../../dtype/ArrayVoid.h"
#include "../../dtype/Scalar.h"

namespace nt {
namespace functional {
namespace cpu {

NEUROTENSOR_API void _operator_mdsa(const ArrayVoid &a, const ArrayVoid &b, ArrayVoid &o,
                    int op);
NEUROTENSOR_API void _operator_mdsa_(ArrayVoid &a, const ArrayVoid &b, int op);
NEUROTENSOR_API void _operator_mdsa_scalar(const ArrayVoid &in, ArrayVoid &out, Scalar s,
                           int op);
NEUROTENSOR_API void _operator_mdsa_scalar_first(const ArrayVoid &in, ArrayVoid &out, Scalar s,
                           int op);
NEUROTENSOR_API void _operator_mdsa_scalar_(ArrayVoid &out, Scalar s, int op);

NEUROTENSOR_API void _inverse_(ArrayVoid &);
NEUROTENSOR_API ArrayVoid _inverse(const ArrayVoid &);

NEUROTENSOR_API void _fmod_(ArrayVoid &, Scalar);
NEUROTENSOR_API void _fmod_first_scalar_(ArrayVoid &, Scalar); 
NEUROTENSOR_API void _fmod_array_(ArrayVoid &, const ArrayVoid&);
NEUROTENSOR_API void _fmod_first_array_(ArrayVoid &, Scalar); 
NEUROTENSOR_API void _fmod_backward(const ArrayVoid& a, const ArrayVoid& b, const ArrayVoid& grad, ArrayVoid& out);
NEUROTENSOR_API void _fmod_backward(const Scalar& a, const ArrayVoid& b, const ArrayVoid& grad, ArrayVoid& out);


NEUROTENSOR_API void _remainder_(ArrayVoid &, Scalar);
NEUROTENSOR_API void _remainder_first_scalar_(ArrayVoid &, Scalar); 
NEUROTENSOR_API void _remainder_array_(ArrayVoid &, const ArrayVoid&);
NEUROTENSOR_API void _remainder_first_array_(ArrayVoid &, Scalar); 
NEUROTENSOR_API void _remainder_backward(const ArrayVoid& a, const ArrayVoid& b, const ArrayVoid& grad, ArrayVoid& out);
NEUROTENSOR_API void _remainder_backward(const Scalar& a, const ArrayVoid& b, const ArrayVoid& grad, ArrayVoid& out);

} // namespace cpu
} // namespace functional
} // namespace nt

#endif
