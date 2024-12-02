#include "nt_matmult_blocks.h"
#include "nt_matmult.hpp"

namespace nt{
namespace functional{
namespace std_functional{
_NT_MATMULT_DECLARE_STATIC_BLOCK_(uint8_t)
template <>
uint8_t* get_blockA_packed<uint8_t>() {
	return blockA_packed_uint8_t;
}

template void nt_matmult<uint8_t>(const uint8_t* A, const uint8_t* B, uint8_t* C, int64_t a_rows, int64_t a_cols, int64_t b_rows, int64_t b_cols, bool transpose_a, bool transpose_b);
template void nt_matmult_batch<uint8_t>(const uint8_t** A, const uint8_t** B, uint8_t** C, int64_t batches, int64_t a_rows, int64_t a_cols, int64_t b_rows, int64_t b_cols, bool transpose_a, bool transpose_b);


}}} // nt::functional::std_functional::
