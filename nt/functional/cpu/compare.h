#ifndef __NT_FUNCTIONAL_CPU_COMPARE_H__
#define __NT_FUNCTIONAL_CPU_COMPARE_H__

#include "../../dtype/ArrayVoid.h"
#include "../../dtype/Scalar.h"

namespace nt{
namespace functional{
namespace cpu{

void _equal(ArrayVoid& out, const ArrayVoid& a, const ArrayVoid& b);
void _not_equal(ArrayVoid& out, const ArrayVoid& a, const ArrayVoid& b);
void _less_than(ArrayVoid& out, const ArrayVoid& a, const ArrayVoid& b);
void _greater_than(ArrayVoid& out, const ArrayVoid& a, const ArrayVoid& b);
void _less_than_equal(ArrayVoid& out, const ArrayVoid& a, const ArrayVoid& b);
void _greater_than_equal(ArrayVoid& out, const ArrayVoid& a, const ArrayVoid& b);
void _and_op(ArrayVoid& out, const ArrayVoid& a, const ArrayVoid& b);
void _or_op(ArrayVoid& out, const ArrayVoid& a, const ArrayVoid& b);

void _equal(ArrayVoid& out, const ArrayVoid& a, Scalar b);
void _not_equal(ArrayVoid& out, const ArrayVoid& a, Scalar b);
void _less_than(ArrayVoid& out, const ArrayVoid& a, Scalar b);
void _greater_than(ArrayVoid& out, const ArrayVoid& a, Scalar b);
void _less_than_equal(ArrayVoid& out, const ArrayVoid& a, Scalar b);
void _greater_than_equal(ArrayVoid& out, const ArrayVoid& a, Scalar b);

bool _all(const ArrayVoid& a);
bool _any(const ArrayVoid& a);
bool _none(const ArrayVoid& a);

}
}
}

#endif
