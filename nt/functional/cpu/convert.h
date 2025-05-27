#ifndef __NT_FUNCTIONAL_CPU_CONVERT_H__
#define __NT_FUNCTIONAL_CPU_CONVERT_H__

#include "../../dtype/ArrayVoid.h"

namespace nt{
namespace functional{
namespace cpu{

void _convert(const ArrayVoid&, ArrayVoid&);
void _floating_(ArrayVoid&);
void _complex_(ArrayVoid&);
void _integer_(ArrayVoid&);
void _unsigned_(ArrayVoid&);

}
}
}

#endif
