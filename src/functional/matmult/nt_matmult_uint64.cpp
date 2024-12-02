#include "nt_matmult_blocks.h"
#include "nt_matmult.hpp"

namespace nt{
namespace functional{
namespace std_functional{
_NT_MATMULT_DECLARE_STATIC_BLOCK_(uint64_t)
template <>
uint64_t* get_blockA_packed<uint64_t>() {
	return blockA_packed_uint64_t;
}

template void nt_matmult<uint64_t>(const uint64_t* A, const uint64_t* B, uint64_t* C, int64_t a_rows, int64_t a_cols, int64_t b_rows, int64_t b_cols, bool transpose_a, bool transpose_b);
template void nt_matmult_batch<uint64_t>(const uint64_t** A, const uint64_t** B, uint64_t** C, int64_t batches, int64_t a_rows, int64_t a_cols, int64_t b_rows, int64_t b_cols, bool transpose_a, bool transpose_b);


}}} // nt::functional::std_functional::
