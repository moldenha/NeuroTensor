#include "nt_matmult.hpp"
#include "nt_matmult_blocks.h"
namespace nt{
namespace functional{
namespace std_functional{
_NT_MATMULT_DECLARE_STATIC_BLOCK_(float)
template <>
float* get_blockA_packed<float>() {
	return blockA_packed_float;
}

template void nt_matmult<float>(const float* A, const float* B, float* C, int64_t a_rows, int64_t a_cols, int64_t b_rows, int64_t b_cols, bool transpose_a, bool transpose_b);
template void nt_matmult_batch<float>(const float** A, const float** B, float** C, int64_t batches, int64_t a_rows, int64_t a_cols, int64_t b_rows, int64_t b_cols, bool transpose_a, bool transpose_b);


}}} // nt::functional::std_functional::
