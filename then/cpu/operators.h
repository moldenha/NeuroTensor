#ifndef __NT_FUNCTIONAL_CPU_OPERATORS_ARRAY_VOID_H__
#define __NT_FUNCTIONAL_CPU_OPERATORS_ARRAY_VOID_H__
#include "../../dtype/ArrayVoid.h"

namespace nt {
namespace functional {
namespace cpu {

void _operator_mdsa(const ArrayVoid &a, const ArrayVoid &b, ArrayVoid &o,
                    int op);
void _operator_mdsa_(ArrayVoid &a, const ArrayVoid &b, int op);
void _operator_mdsa_scalar(const ArrayVoid& in, ArrayVoid& out, Scalar s, int op);
void _operator_mdsa_scalar_(ArrayVoid& out, Scalar s, int op);

void _inverse_(ArrayVoid&);
ArrayVoid _inverse(const ArrayVoid&);

} // namespace cpu
} // namespace functional
} // namespace nt

#endif
