#include "nt_matmult_blocks.h"
#include "nt_matmult.hpp"

namespace nt{
namespace functional{
namespace std_functional{
_NT_MATMULT_DECLARE_STATIC_BLOCK_(int64_t)
template <>
int64_t* get_blockA_packed<int64_t>() {
	return blockA_packed_int64_t;
}

template void nt_matmult<int64_t>(const int64_t* A, const int64_t* B, int64_t* C, int64_t a_rows, int64_t a_cols, int64_t b_rows, int64_t b_cols, bool transpose_a, bool transpose_b);
template void nt_matmult_batch<int64_t>(const int64_t** A, const int64_t** B, int64_t** C, int64_t batches, int64_t a_rows, int64_t a_cols, int64_t b_rows, int64_t b_cols, bool transpose_a, bool transpose_b);


}}} // nt::functional::std_functional::
