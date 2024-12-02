#ifndef _NT_MATMULT_H_
#define _NT_MATMULT_H_
namespace nt{
namespace functional{
namespace std_functional{
template<typename T>
void nt_matmult(const T* A, const T* B, T* C, int64_t a_rows, int64_t a_cols, int64_t b_rows, int64_t b_cols, bool transpose_a, bool transpose_b);

template<typename T>
void nt_matmult_batch(const T** A, const T** B, T** C, int64_t batches, int64_t a_rows, int64_t a_cols, int64_t b_rows, int64_t b_cols, bool transpose_a, bool transpose_b);

}}} // nt::functional::std_functional

#endif //_NT_MATMULT_H_
