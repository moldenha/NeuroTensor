#include "nt_matmult_blocks.h"
#include "../../types/Types.h"
#include "nt_matmult.hpp"
#include "nt_matmult_blocks.h"
#ifdef _HALF_FLOAT_SUPPORT_

namespace nt{
namespace functional{
namespace std_functional{
template <>
complex_32* get_blockA_packed<complex_32>() {
	return nullptr;
}

template void nt_matmult<complex_32>(const complex_32* A, const complex_32* B, complex_32* C, int64_t a_rows, int64_t a_cols, int64_t b_rows, int64_t b_cols, bool transpose_a, bool transpose_b);
template void nt_matmult_batch<complex_32>(const complex_32** A, const complex_32** B, complex_32** C, int64_t batches, int64_t a_rows, int64_t a_cols, int64_t b_rows, int64_t b_cols, bool transpose_a, bool transpose_b);


}}} // nt::functional::std_functional::


#endif //_HALF_FLOAT_SUPPORT_
