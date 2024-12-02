#include "nt_matmult_blocks.h"
#include "nt_matmult.hpp"

namespace nt{
namespace functional{
namespace std_functional{
_NT_MATMULT_DECLARE_STATIC_BLOCK_(int16_t)
template <>
int16_t* get_blockA_packed<int16_t>() {
	return blockA_packed_int16_t;
}

template void nt_matmult<int16_t>(const int16_t* A, const int16_t* B, int16_t* C, int64_t a_rows, int64_t a_cols, int64_t b_rows, int64_t b_cols, bool transpose_a, bool transpose_b);
template void nt_matmult_batch<int16_t>(const int16_t** A, const int16_t** B, int16_t** C, int64_t batches, int64_t a_rows, int64_t a_cols, int64_t b_rows, int64_t b_cols, bool transpose_a, bool transpose_b);


}}} // nt::functional::std_functional::
