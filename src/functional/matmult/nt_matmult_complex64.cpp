#include "nt_matmult_blocks.h"
#include "../../types/Types.h"
#include "nt_matmult.hpp"

namespace nt{
namespace functional{
namespace std_functional{
template <>
complex_64* get_blockA_packed<complex_64>() {
	return nullptr;
}

template void nt_matmult<complex_64>(const complex_64* A, const complex_64* B, complex_64* C, int64_t a_rows, int64_t a_cols, int64_t b_rows, int64_t b_cols, bool transpose_a, bool transpose_b);
template void nt_matmult_batch<complex_64>(const complex_64** A, const complex_64** B, complex_64** C, int64_t batches, int64_t a_rows, int64_t a_cols, int64_t b_rows, int64_t b_cols, bool transpose_a, bool transpose_b);


}}} // nt::functional::std_functional::


