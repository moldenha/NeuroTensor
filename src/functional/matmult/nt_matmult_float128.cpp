#include "nt_matmult_blocks.h"
#include "../../types/Types.h"
#include "nt_matmult.hpp"
#ifdef _128_FLOAT_SUPPORT_

namespace nt{
namespace functional{
namespace std_functional{

template <>
float128_t* get_blockA_packed<float128_t>() {
	return nullptr;
}

template void nt_matmult<float128_t>(const float128_t* A, const float128_t* B, float128_t* C, int64_t a_rows, int64_t a_cols, int64_t b_rows, int64_t b_cols, bool transpose_a, bool transpose_b);
template void nt_matmult_batch<float128_t>(const float128_t** A, const float128_t** B, float128_t** C, int64_t batches, int64_t a_rows, int64_t a_cols, int64_t b_rows, int64_t b_cols, bool transpose_a, bool transpose_b);


}}} // nt::functional::std_functional::


#endif //_HALF_FLOAT_SUPPORT_
