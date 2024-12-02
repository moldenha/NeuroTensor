#include "nt_matmult_blocks.h"
#include "nt_matmult.hpp"

namespace nt{
namespace functional{
namespace std_functional{
_NT_MATMULT_DECLARE_STATIC_BLOCK_(int8_t)
template <>
int8_t* get_blockA_packed<int8_t>() {
	return blockA_packed_int8_t;
}

template void nt_matmult<int8_t>(const int8_t* A, const int8_t* B, int8_t* C, int64_t a_rows, int64_t a_cols, int64_t b_rows, int64_t b_cols, bool transpose_a, bool transpose_b);
template void nt_matmult_batch<int8_t>(const int8_t** A, const int8_t** B, int8_t** C, int64_t batches, int64_t a_rows, int64_t a_cols, int64_t b_rows, int64_t b_cols, bool transpose_a, bool transpose_b);


}}} // nt::functional::std_functional::
