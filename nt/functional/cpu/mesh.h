#ifndef NT_FUNCTIONAL_CPU_MESH_H__
#define NT_FUNCTIONAL_CPU_MESH_H__

#include "../../dtype/ArrayVoid.h"

namespace nt{
namespace functional{
namespace cpu{

NEUROTENSOR_API void _meshgrid(const ArrayVoid& x, const ArrayVoid& y, ArrayVoid& outX, ArrayVoid& outY);


}
}
}

#endif
