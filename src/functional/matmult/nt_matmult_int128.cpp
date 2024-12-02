#include "nt_matmult_blocks.h"
#include "../../types/Types.h"
#include "nt_matmult.hpp"
#ifdef __SIZEOF_INT128__

namespace nt{
namespace functional{
namespace std_functional{
template <>
int128_t* get_blockA_packed<int128_t>() {
	return nullptr;
}

template void nt_matmult<int128_t>(const int128_t* A, const int128_t* B, int128_t* C, int64_t a_rows, int64_t a_cols, int64_t b_rows, int64_t b_cols, bool transpose_a, bool transpose_b);
template void nt_matmult_batch<int128_t>(const int128_t** A, const int128_t** B, int128_t** C, int64_t batches, int64_t a_rows, int64_t a_cols, int64_t b_rows, int64_t b_cols, bool transpose_a, bool transpose_b);


}}} // nt::functional::std_functional::


#endif //__SIZEOF_INT128__

