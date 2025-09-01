#ifndef NT_FUNCTIONAL_CPU_COMPARE_H__
#define NT_FUNCTIONAL_CPU_COMPARE_H__

#include "../../dtype/ArrayVoid.h"
#include "../../dtype/Scalar.h"

namespace nt{
namespace functional{
namespace cpu{

NEUROTENSOR_API void _equal(ArrayVoid& out, const ArrayVoid& a, const ArrayVoid& b);
NEUROTENSOR_API void _not_equal(ArrayVoid& out, const ArrayVoid& a, const ArrayVoid& b);
NEUROTENSOR_API void _less_than(ArrayVoid& out, const ArrayVoid& a, const ArrayVoid& b);
NEUROTENSOR_API void _greater_than(ArrayVoid& out, const ArrayVoid& a, const ArrayVoid& b);
NEUROTENSOR_API void _less_than_equal(ArrayVoid& out, const ArrayVoid& a, const ArrayVoid& b);
NEUROTENSOR_API void _greater_than_equal(ArrayVoid& out, const ArrayVoid& a, const ArrayVoid& b);
NEUROTENSOR_API void _and_op(ArrayVoid& out, const ArrayVoid& a, const ArrayVoid& b);
NEUROTENSOR_API void _or_op(ArrayVoid& out, const ArrayVoid& a, const ArrayVoid& b);
NEUROTENSOR_API void _isnan(ArrayVoid& out, const ArrayVoid& a);

NEUROTENSOR_API void _equal(ArrayVoid& out, const ArrayVoid& a, Scalar b);
NEUROTENSOR_API void _not_equal(ArrayVoid& out, const ArrayVoid& a, Scalar b);
NEUROTENSOR_API void _less_than(ArrayVoid& out, const ArrayVoid& a, Scalar b);
NEUROTENSOR_API void _greater_than(ArrayVoid& out, const ArrayVoid& a, Scalar b);
NEUROTENSOR_API void _less_than_equal(ArrayVoid& out, const ArrayVoid& a, Scalar b);
NEUROTENSOR_API void _greater_than_equal(ArrayVoid& out, const ArrayVoid& a, Scalar b);

NEUROTENSOR_API bool _all(const ArrayVoid& a);
NEUROTENSOR_API bool _any(const ArrayVoid& a);
NEUROTENSOR_API bool _none(const ArrayVoid& a);

NEUROTENSOR_API int64_t _amount_of(const ArrayVoid& a, Scalar s);
NEUROTENSOR_API int64_t _count(const ArrayVoid& a);
}
}
}

#endif
