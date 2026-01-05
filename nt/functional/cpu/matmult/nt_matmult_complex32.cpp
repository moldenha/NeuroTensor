#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wvla-cxx-extension"
#endif

#include "nt_matmult_blocks.h"
#include "../../../types/Types.h"
#include "nt_matmult.hpp"
#include "nt_matmult_blocks.h"

namespace nt::functional::cpu{

_NT_MATMULT_DECLARE_STATIC_BLOCK_(complex_32)

} // nt::functional::cpu::

#if defined(__clang__)
#pragma clang diagnostic pop
#endif
