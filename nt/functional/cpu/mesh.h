#ifndef __NT_FUNCTIONAL_CPU_MESH_H__
#define __NT_FUNCTIONAL_CPU_MESH_H__

#include "../../dtype/ArrayVoid.h"

namespace nt{
namespace functional{
namespace cpu{

void _meshgrid(const ArrayVoid& x, const ArrayVoid& y, ArrayVoid& outX, ArrayVoid& outY);


}
}
}

#endif
