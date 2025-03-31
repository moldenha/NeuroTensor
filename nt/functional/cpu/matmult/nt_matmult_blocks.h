#ifndef _NT_MATMULT_BLOCKS_H_
#define _NT_MATMULT_BLOCKS_H_

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace nt{
namespace functional{
namespace cpu{
// Function declarations for accessing static blocks
template <typename T>
T* get_blockA_packed();

template <typename T>
T* get_blockB_packed();
}}} //nt::functional::cpu
#endif // _NT_MATMULT_BLOCKS_H_

