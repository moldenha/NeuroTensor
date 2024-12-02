#include "nt_matmult_blocks.h"
#include "../../types/Types.h"
#include "nt_matmult.hpp"
#ifdef __SIZEOF_INT128__

namespace nt{
namespace functional{
namespace std_functional{
template <>
uint128_t* get_blockA_packed<uint128_t>() {
	return nullptr;
}

template void nt_matmult<uint128_t>(const uint128_t* A, const uint128_t* B, uint128_t* C, int64_t a_rows, int64_t a_cols, int64_t b_rows, int64_t b_cols, bool transpose_a, bool transpose_b);
template void nt_matmult_batch<uint128_t>(const uint128_t** A, const uint128_t** B, uint128_t** C, int64_t batches, int64_t a_rows, int64_t a_cols, int64_t b_rows, int64_t b_cols, bool transpose_a, bool transpose_b);


}}} // nt::functional::std_functional::


#endif //__SIZEOF_INT128__

