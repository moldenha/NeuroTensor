#ifndef __NT_FUNCTIONAL_CPU_COMPLEX_H__
#define __NT_FUNCTIONAL_CPU_COMPLEX_H__

#include "../../dtype/ArrayVoid.h"

namespace nt{
namespace functional{
namespace cpu{

void _to_complex_from_real(const ArrayVoid&, ArrayVoid&);
void _to_complex_from_imag(const ArrayVoid&, ArrayVoid&);

}
}
}

#endif
