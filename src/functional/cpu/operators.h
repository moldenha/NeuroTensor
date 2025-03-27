#ifndef __NT_CPU_OPERATORS_ARRAY_VOID_H__
#define __NT_CPU_OPERATORS_ARRAY_VOID_H__
#include "../../dtype/ArrayVoid.h"

namespace nt {
namespace functional {
namespace cpu {

void operator_mdsa_(const ArrayVoid &a, const ArrayVoid &b, ArrayVoid &o,
                    int op);
void operator_mdsa_(ArrayVoid &a, const ArrayVoid &b, int op);

} // namespace cpu
} // namespace functional
} // namespace nt

#endif
