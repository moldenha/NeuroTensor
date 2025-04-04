#include "../../functional/functional.h"
#include "../../linalg/linalg.h"
#include "../../mp/Threading.h"
#include "../../mp/simde_traits.h"
#include "../../sparse/SparseMatrix.h"
#include <cstdlib> // For aligned memory allocation
#include <cstring> // For memcpy
#include <iostream>
#include <map>
#include <Eigen/Dense>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/blocked_range2d.h>
#include <tbb/blocked_range3d.h>

namespace nt {
namespace tda {
namespace cpu {

#ifdef SIMDE_ARCH_X86_AVX2
inline static constexpr auto testz = simde_mm256_testz_ps;
#elif defined(SIMDE_ARCH_X86_AVX)
inline static constexpr auto testz = simde_mm_testz_ps;
#endif


// template<typename T>
// static mp::SimdTraits<float>::Type simde_zero = mp::SimdTraits<float>::zero();

// inline bool is_zero(const mp::SimdTraits<float>::Type &vec) noexcept {
//     return testz(vec, vec);
// }

// inline bool is_zero(const mp::SimdTraits<float>::Type *begin,
//                     const mp::SimdTraits<float>::Type *end) {
//     return std::all_of(begin, end, [](const mp::SimdTraits<float>::Type &vec) -> bool {
//         return testz(vec, vec);
//     });
// }

// fmadd is used with a scalar
template <typename T>
inline void _reduced_fmadd(typename mp::SimdTraits<T>::Type *begin_o,
                           typename mp::SimdTraits<T>::Type *end_o,
                           const typename mp::SimdTraits<T>::Type *begin_s,
                           const typename mp::SimdTraits<T>::Type &scalar) {
    

    tbb::parallel_for(tbb::blocked_range<size_t>(0, end_o-begin_o),
    [&](const tbb::blocked_range<size_t>& c){
        for(size_t i = c.begin(); i < c.end(); ++i){
            mp::SimdTraits<T>::fmadd(begin_s[i], scalar, begin_o[i]);
        }
    });
    // threading::preferential_parallel_for(
    //     threading::block_ranges<1>(0, end_o - begin_o),
    //     [&](threading::blocked_range<1> block) {
    //         for (int64_t i = block.begin[0]; i < block.end[0]; ++i) {
    //             // if (is_zero(begin_s[i]))
    //             //     continue;
    //             mp::SimdTraits<T>::fmadd(begin_s[i], scalar, begin_o[i]);
    //         }
    //     });
}


template<typename T>
inline void _multiply_vector(typename mp::SimdTraits<T>::Type *begin,
                             typename mp::SimdTraits<T>::Type *end,
                             const typename mp::SimdTraits<T>::Type &scalar) {
    for(;begin != end; ++begin){
        *begin = mp::SimdTraits<T>::multiply(*begin, scalar);
    }
    // tbb::parallel_for(tbb::blocked_range<size_t>(0, end-begin),
    // [&](const tbb::blocked_range<size_t>& c){
    //     for(size_t i = c.begin(); i < c.end(); ++i){
    //         begin[i] = mp::SimdTraits<T>::multiply(begin[i], scalar);
    //     }
    // });
    // threading::preferential_parallel_for(
    //     threading::block_ranges<1>(0, end - begin),
    //     [&](threading::blocked_range<1> block) {
    //         for (int64_t i = block.begin[0]; i < block.end[0]; ++i) {
    //             // if (is_zero(begin[i]))
    //             //     continue;
    //             begin[i] = mp::SimdTraits<T>::multiply(begin[i], scalar);
    //         }
    //     });
}

template<typename T>
inline void _multiply_vector(typename mp::SimdTraits<T>::Type *begin,
                             typename mp::SimdTraits<T>::Type *end,
                             const T &_scalar) {
    if(_scalar == 1) return;
    typename mp::SimdTraits<T>::Type scalar = mp::SimdTraits<T>::set1(_scalar);
    // for(;begin != end; ++begin){
    //     *begin = mp::SimdTraits<float>::multiply(*begin, scalar);
    // }
    for(;begin != end; ++begin){
        *begin = mp::SimdTraits<T>::multiply(*begin, scalar);
    }

   // tbb::parallel_for(tbb::blocked_range<size_t>(0, end-begin),
   //  [&](const tbb::blocked_range<size_t>& c){
   //      for(size_t i = c.begin(); i < c.end(); ++i){
   //          begin[i] = mp::SimdTraits<T>::multiply(begin[i], scalar);
   //      }
   //  }); 
    // threading::preferential_parallel_for(
    //     threading::block_ranges<1>(0, end - begin),
    //     [&](threading::blocked_range<1> block) {
    //         for (int64_t i = block.begin[0]; i < block.end[0]; ++i) {
    //             // if (is_zero(begin[i]))
    //             //     continue;
    //             begin[i] = mp::SimdTraits<T>::multiply(begin[i], scalar);
    //         }
    //     });
}

template<typename T>
typename mp::SimdTraits<T>::Type **allocateSIMDeMatrix(const T *input, size_t rows,
                                              size_t cols) {
    constexpr size_t simd_size = mp::SimdTraits<T>::pack_size;
    size_t num_vectors = (cols + simd_size - 1) / simd_size; // Ceiling division

    // Process full SIMD blocks
    size_t full_blocks = cols / simd_size;
    size_t tail_size = cols % simd_size;

    typename mp::SimdTraits<T>::Type **simd_matrix = new typename mp::SimdTraits<T>::Type *[rows];
    for (size_t i = 0; i < rows; ++i) {
        // Allocate aligned memory for the __m256 array
        typename mp::SimdTraits<T>::Type *simd_array =
            (typename mp::SimdTraits<T>::Type *)std::aligned_alloc(
                32, num_vectors * sizeof(simde__m256));

        for (size_t j = 0; j < full_blocks; ++j) {
            simd_array[j] =
                mp::SimdTraits<T>::loadu(&input[(i * cols) + j * simd_size]);
        }

        // Handle the tail (masking)
        if (tail_size > 0) {
            T tail[simd_size] = {0.0f}; // Zero initialize
            std::memcpy(tail, &input[full_blocks * simd_size],
                        tail_size * sizeof(T));
            simd_array[full_blocks] = mp::SimdTraits<T>::loadu(tail);

            // Create the mask for the tail
            // __mmask8 mask = (1 << tail_size) - 1;  // E.g., for 3 remaining
            // elements: 0b00000111

            // // Load tail into a register with masking
            // simd_array[full_blocks] = _mm256_maskz_loadu_ps(mask, tail);
        }
        simd_matrix[i] = simd_array;
    }
    return simd_matrix;
}

template<typename T>
typename mp::SimdTraits<T>::Type **allocateSIMDeMatrix(size_t rows, size_t cols) {
    constexpr size_t simd_size = mp::SimdTraits<T>::pack_size;
    size_t num_vectors = (cols + simd_size - 1) / simd_size; // Ceiling division

    static const typename mp::SimdTraits<T>::Type simde_zero = mp::SimdTraits<T>::zero();
    typename mp::SimdTraits<T>::Type **simd_matrix = new typename mp::SimdTraits<T>::Type *[rows];

    for (size_t i = 0; i < rows; ++i) {
        // Allocate aligned memory for the SIMD vectors
        typename mp::SimdTraits<T>::Type *simd_array =
            (typename mp::SimdTraits<T>::Type *)std::aligned_alloc(
                32, num_vectors * sizeof(typename mp::SimdTraits<T>::Type));

        // Initialize everything to zero
        for (size_t j = 0; j < num_vectors; ++j) {
            simd_array[j] = simde_zero;
        }

        simd_matrix[i] = simd_array;
    }
    return simd_matrix;
}

template<typename T>
inline void freeSIMDeArray(typename mp::SimdTraits<T>::Type **ptr, int64_t num_rows) {
    for (int64_t i = 0; i < num_rows; ++i) {
        std::free(ptr[i]);
    }
    delete[] ptr;
}

template<typename T>
inline T** SIMDeMatrix_TemplateMatrix(typename mp::SimdTraits<T>::Type **ptr, int64_t num_rows){
    T** out = new T*[num_rows];
    for(int64_t i = 0; i < num_rows; ++i){
        out[i] = reinterpret_cast<T*>(ptr[i]);
    }
    return out;
}

template<typename T>
void loadSparseMatrix(T** arr, const SparseMatrix& spm, bool transpose){
    auto begin = spm.cmem_begin<int8_t>();
    auto end = spm.cmem_end<int8_t>();
    if(!transpose){
        for(;begin != end; ++begin){
            arr[begin.Row()][begin.Col()] = T(*begin);
        }
    }else{
        for(;begin != end; ++begin){
            arr[begin.Col()][begin.Row()] = T(*begin);
        }
    }
}

template<typename T>
inline void freeTemplateMatrix(T** ptr){delete[] ptr;}
////TODO make a store matrix


template<typename T>
void print_matrix(T** mat, int64_t rows, int64_t cols){
    std::cout << '[';
    for(int64_t r = 0; r < rows; ++r){
        std::cout << '[';
        for(int64_t c = 0; c < cols-1; ++c){
            std::cout << mat[r][c]<<',';
        }
        if(r != rows-1){
            std::cout << mat[r][cols-1] << "],"<<std::endl;
        }
        else{
            std::cout << mat[r][cols-1] << "]]"<<std::endl;
        }
    }
}

template<typename T>
void partialColReduce(typename mp::SimdTraits<T>::Type **&A_begin, T **&a_access,
                      int64_t end_rows,
                      int64_t end_cols, const int64_t &mat_rows,
                      const int64_t &mat_cols) {


    int64_t i = 0, j = 0;

    constexpr size_t simd_size = mp::SimdTraits<T>::pack_size;
    size_t num_vectors = (mat_cols + simd_size - 1) / simd_size;

    while (i < end_rows && j < end_cols) {
        // numpy version is i, j [without transpose]
        //  bool do_continue = false;
        //  bool was_zero = false;
        //  int64_t nonzeroCol = 0;
        if (a_access[j][i] == 0) {
            //if A at row i and column j is 0
            int64_t nonzeroCol = j+1;
            //go down that row until it is not zero
            while (nonzeroCol < end_cols && a_access[nonzeroCol][i] == 0) {
                ++nonzeroCol;
            }
            //if the entire row is 0, just skip this row
            if (nonzeroCol == end_cols) {
                ++i;
                continue;
            }
            // swap cols
            //

            std::swap(a_access[j], a_access[nonzeroCol]);
            std::swap(A_begin[j], A_begin[nonzeroCol]);
        }

        // without transpose: pivot = A[i, j]

        float pivot = 1.0 / float(a_access[j][i]);
        pivot = (int64_t)pivot;
        T n_pivot = T(pivot);

        _multiply_vector<T>(A_begin[j], A_begin[j] + num_vectors, n_pivot);

        // if(j == end_cols-1) break;
        // tbb::parallel_for(tbb::blocked_range2d<int64_t>(j+1, end_cols, 0, num_vectors),
        // [&](const tbb::blocked_range2d<int64_t>& range){
        //     for(int64_t otherCol = range.rows().begin(); otherCol < range.rows().end(); ++otherCol){
        //         T scaleAmt = a_access[otherCol][i];
        //         if(scaleAmt == 0) continue;
        //         scaleAmt *= -1;
        //         if(scaleAmt == -1){
        //             for(int64_t k = range.cols().begin(); k < range.cols().end(); ++k){
        //                 A_begin[otherCol][k] = mp::SimdTraits<T>::subtract(A_begin[otherCol][k], A_begin[j][k]);
        //             }
        //             continue;
        //         }
        //         if(scaleAmt == 1){
        //             for(int64_t k = range.cols().begin(); k < range.cols().end(); ++k){
        //                 A_begin[otherCol][k] = mp::SimdTraits<T>::add(A_begin[otherCol][k], A_begin[j][k]);
        //             }
        //             continue;
        //         }
        //         typename mp::SimdTraits<T>::Type scalar = mp::SimdTraits<T>::set1(scaleAmt);
        //         for(int64_t k = range.cols().begin(); k < range.cols().end(); ++k){
        //             mp::SimdTraits<T>::fmadd(A_begin[j][k], scalar, A_begin[otherCol][k]);
        //         }
        //     }
        // });
        for(int64_t otherCol = j+1; otherCol < end_cols; ++otherCol){
            // if(otherCol == j) continue;
            T scaleAmt = a_access[otherCol][i];
            if(scaleAmt == 0) continue;
            scaleAmt *= -1;
            if(scaleAmt == -1){
                for(int64_t k = 0; k < num_vectors; ++k){
                    A_begin[otherCol][k] = mp::SimdTraits<T>::subtract(A_begin[otherCol][k], A_begin[j][k]);
                }
                continue;
            }else if(scaleAmt == 1){
                for(int64_t k = 0; k < num_vectors; ++k){
                    A_begin[otherCol][k] = mp::SimdTraits<T>::add(A_begin[otherCol][k], A_begin[j][k]);
                }
                continue;
            }
            typename mp::SimdTraits<T>::Type scalar = mp::SimdTraits<T>::set1(scaleAmt);
            for(int64_t k = 0; k < num_vectors; ++k){
                mp::SimdTraits<T>::fmadd(A_begin[j][k], scalar, A_begin[otherCol][k]);
            }
        }
        
        ++i;
        ++j;
    }

}


template<typename T>
void partialRowReduce(typename mp::SimdTraits<T>::Type **&B_begin, T **&b_access,
                      int64_t end_rows,
                      int64_t end_cols, const int64_t &mat_rows,
                      const int64_t &mat_cols) {

    constexpr size_t simd_size = mp::SimdTraits<T>::pack_size;
    size_t num_vectors = (mat_cols + simd_size - 1) / simd_size;
    int64_t i = 0, j = 0;
    while (i < end_rows && j < end_cols) {
        if (b_access[i][j] == 0) {
            int64_t nonzeroRow = i + 1;
            while (nonzeroRow < end_rows && b_access[nonzeroRow][j] == 0) {
                ++nonzeroRow;
            }
            if (nonzeroRow == end_rows) {
                ++j;
                // if(j % simd_size == 0){
                //     storeCol(b_access, B_begin, j, mat_rows, mat_cols);
                // }
                continue;
            }
            // swap rows
            std::swap(b_access[i], b_access[nonzeroRow]);
            std::swap(B_begin[i], B_begin[nonzeroRow]);
        }

        float pivot = 1.0 / float(b_access[i][j]);
        pivot = (int64_t)pivot;
        T n_pivot = T(pivot);
        _multiply_vector<T>(B_begin[i], &B_begin[i][num_vectors], n_pivot);

        // if(i == end_rows-1) break;
        // tbb::parallel_for(tbb::blocked_range2d<int64_t>(i+1, end_rows, 0, num_vectors),
        // [&](const tbb::blocked_range2d<int64_t>& range){
        //     for(int64_t otherRow = range.rows().begin(); otherRow < range.rows().end(); ++otherRow){
        //         T scaleAmt = b_access[otherRow][j];
        //         if(scaleAmt == 0) continue;
        //         scaleAmt *= -1;
        //         if(scaleAmt == -1){
        //             for(int64_t k = range.cols().begin(); k < range.cols().end(); ++k){
        //                 B_begin[otherRow][k] = mp::SimdTraits<T>::subtract(B_begin[otherRow][k], B_begin[i][k]);
        //             }
        //             continue;
        //         }
        //         if(scaleAmt == 1){
        //             for(int64_t k = range.cols().begin(); k < range.cols().end(); ++k){
        //                 B_begin[otherRow][k] = mp::SimdTraits<T>::add(B_begin[otherRow][k], B_begin[i][k]);
        //             }
        //             continue;
        //         }
        //         typename mp::SimdTraits<T>::Type scalar = mp::SimdTraits<T>::set1(scaleAmt);
        //         for(int64_t k = range.cols().begin(); k < range.cols().end(); ++k){
        //             mp::SimdTraits<T>::fmadd(B_begin[i][k], scalar, B_begin[otherRow][k]);
        //         }
        //     }
        // });
        for(int64_t otherRow = i+1; otherRow < end_rows; ++otherRow){
            if(otherRow == i)
                continue;
            T scaleAmt = b_access[otherRow][j];
            if(scaleAmt == 0)
                continue;
            scaleAmt *= -1;
            if(scaleAmt == -1){
                for(int64_t k = 0; k < num_vectors; ++k){
                    B_begin[otherRow][k] = mp::SimdTraits<T>::subtract(B_begin[otherRow][k], B_begin[i][k]);
                }
                continue;
            }else if(scaleAmt == 1){
                for(int64_t k = 0; k < num_vectors; ++k){
                    B_begin[otherRow][k] = mp::SimdTraits<T>::add(B_begin[otherRow][k], B_begin[i][k]);
                }
                continue;
            }
            typename mp::SimdTraits<T>::Type scalar = mp::SimdTraits<T>::set1(scaleAmt);
            for(int64_t k = 0; k < num_vectors; ++k){
                mp::SimdTraits<T>::fmadd(B_begin[i][k], scalar, B_begin[otherRow][k]);
            }

        }
        ++i;
        ++j;
        // storeCol(b_access, B_begin, j, mat_rows, mat_cols);
    }
}

template<typename T>
int64_t numPivotRows(typename mp::SimdTraits<T>::Type **arr, T** access, int64_t rows, int64_t cols){
    // constexpr size_t simd_size = mp::SimdTraits<float>::pack_size;
    // size_t num_vectors = (cols + simd_size - 1) / simd_size;
    // int64_t count = 0;
    // // std::cout << "max rows: "<<rows<<std::endl;
    // if(cols < simd_size){
    //     for(int64_t i = 0; i < rows; ++i){
    //         if(std::all_of(access[i], access[i] + cols, [](const float& val){return val == 0;})) ++count;
    //     }
    //     // std::cout << "pivot rows: "<<count<<std::endl;
    //     return count;
    // }
    // if(cols % simd_size == 0){
    //     for(int64_t i = 0; i < rows; ++i){
    //         if(is_zero(arr[i], arr[i] + num_vectors)) ++count;
    //     }
    //     // std::cout << "pivot rows: "<<count<<std::endl;
    //     return count;
    // }
    // int64_t start = cols - (cols % simd_size);
    // for(int64_t i = 0; i < rows; ++i){
    //     if(std::all_of(access[i] + start, access[i] + cols,
    //                     [](const int64_t& val){return val == 0;})
    //     && is_zero(arr[i], arr[i] + (num_vectors-1))) ++ count;
    // }
    // // std::cout << "pivot rows: "<<count<<std::endl;
    // return count;    
    int64_t count = 0;
    for(int64_t i = 0; i < rows; ++i){
        if(std::all_of(access[i], access[i] + cols, [](const T& val){return val == 0;})) ++count;
    }
    return rows - count;
}

std::map<double, int64_t> getBettiNumbers(
    SparseMatrix &d_k, SparseMatrix &d_kplus1,
    std::map<double, std::tuple<int64_t, int64_t, int64_t>> radi_bounds,
    double max, bool add_zeros) {
    utils::throw_exception(
        d_k.dtype() == DType::int8 && d_kplus1.dtype() == DType::int8,
        "Expected boundary matrices to have dtype int8 but got $ and $",
        d_k.dtype(), d_kplus1.dtype());
    utils::throw_exception(
        d_k.Cols() == d_kplus1.Rows(),
        "Matrices have incompatible shapes: d_k is {$, $}, d_kplus1 is {$, $}",
        d_k.Rows(), d_k.Cols(), d_kplus1.Rows(), d_kplus1.Cols());


    using value_t = float;
    
    const int64_t& a_mat_rows = d_k.Cols();
    const int64_t& a_mat_cols = d_k.Rows();
    const int64_t& b_mat_rows = d_kplus1.Rows();
    const int64_t& b_mat_cols = d_kplus1.Cols();

    mp::SimdTraits<value_t>::Type **A_simde =
        allocateSIMDeMatrix<value_t>(d_k.Cols(), d_k.Rows()); // transpose is loaded in
    mp::SimdTraits<value_t>::Type **B_simde =
        allocateSIMDeMatrix<value_t>(d_kplus1.Rows(), d_kplus1.Cols());
    
    value_t** a_access = SIMDeMatrix_TemplateMatrix<value_t>(A_simde, d_k.Cols());
    value_t** b_access = SIMDeMatrix_TemplateMatrix<value_t>(B_simde, d_kplus1.Rows());

    loadSparseMatrix<value_t>(a_access, d_k, true);
    loadSparseMatrix<value_t>(b_access, d_kplus1, false);

    std::map<double, int64_t> out;
    for(const auto& correspond : radi_bounds){
        auto [km1_size, k_size, kp1_size] = correspond.second;
        if(km1_size > a_mat_cols) continue;
        if(k_size > a_mat_rows) continue;
        if(k_size > b_mat_rows) continue;
        if(kp1_size > b_mat_cols) continue;
        if(max > 0 && correspond.first > max) break;
        // std::cout << "running radius "<<correspond.first<<std::endl;
        partialColReduce<value_t>(A_simde, a_access,
                         km1_size, k_size,
                         a_mat_rows, a_mat_cols);

        //the following works which is nice
        partialRowReduce<value_t>(B_simde, b_access,
                        k_size, kp1_size,
                         b_mat_rows, b_mat_cols);
        // if(i == 0
        

        int64_t rank_k = numPivotRows<value_t>(A_simde, a_access, k_size, km1_size);
        int64_t rank_kp1 = numPivotRows<value_t>(B_simde, b_access, k_size, kp1_size);
        int64_t betti = (k_size - rank_k) - rank_kp1;

        if(betti > 0){ 
            out[correspond.first] = betti;
        }else if (add_zeros){
            out[correspond.first] = 0;
        }
    }


    freeTemplateMatrix<value_t>(a_access);
    freeTemplateMatrix<value_t>(b_access);
    freeSIMDeArray<value_t>(A_simde, d_k.Cols());
    freeSIMDeArray<value_t>(B_simde, d_kplus1.Rows());
    return std::move(out);
}



template<typename T>
std::vector<bool> isPivotCol(T** access, const int64_t& rows, const int64_t& cols){
    std::vector<bool> is_pivot_col(cols, false);
    int64_t last = -1;
    for(int64_t r = 0; r < rows; ++r){
        for(int64_t c = 0; c < cols; ++c){
            if(access[r][c] == 1) {is_pivot_col[c] = true; break;}
        }
    }
    return std::move(is_pivot_col);
}



std::pair<std::map<double, int64_t>, std::vector<std::pair<double, SparseMatrix>>> getBettiNumbersColSpace(
    SparseMatrix &d_k, SparseMatrix &d_kplus1,
    std::map<double, std::tuple<int64_t, int64_t, int64_t>> radi_bounds,
    double max, bool add_zeros) {
    utils::throw_exception(
        d_k.dtype() == DType::int8 && d_kplus1.dtype() == DType::int8,
        "Expected boundary matrices to have dtype int8 but got $ and $",
        d_k.dtype(), d_kplus1.dtype());
    utils::throw_exception(
        d_k.Cols() == d_kplus1.Rows(),
        "Matrices have incompatible shapes: d_k is {$, $}, d_kplus1 is {$, $}",
        d_k.Rows(), d_k.Cols(), d_kplus1.Rows(), d_kplus1.Cols());


    
    using value_t = int8_t;
    
    const int64_t& a_mat_rows = d_k.Cols();
    const int64_t& a_mat_cols = d_k.Rows();
    const int64_t& b_mat_rows = d_kplus1.Rows();
    const int64_t& b_mat_cols = d_kplus1.Cols();

    mp::SimdTraits<value_t>::Type **A_simde =
        allocateSIMDeMatrix<value_t>(d_k.Cols(), d_k.Rows()); // transpose is loaded in
    mp::SimdTraits<value_t>::Type **B_simde =
        allocateSIMDeMatrix<value_t>(d_kplus1.Rows(), d_kplus1.Cols());
    
    value_t** a_access = SIMDeMatrix_TemplateMatrix<value_t>(A_simde, d_k.Cols());
    value_t** b_access = SIMDeMatrix_TemplateMatrix<value_t>(B_simde, d_kplus1.Rows());

    loadSparseMatrix<value_t>(a_access, d_k, true);
    loadSparseMatrix<value_t>(b_access, d_kplus1, false);

    std::map<double, int64_t> out;
    std::vector<std::pair<double, SparseMatrix> > col_spaces;
    // int cntr = 0;
    for(const auto& correspond : radi_bounds){
        auto [km1_size, k_size, kp1_size] = correspond.second;
        if(km1_size > a_mat_cols) continue;
        if(k_size > a_mat_rows) continue;
        if(k_size > b_mat_rows) continue;
        if(kp1_size > b_mat_cols) continue;
        if(max > 0 && correspond.first > max) break;
        // std::cout << "running radius "<<correspond.first<<std::endl;
        partialColReduce<value_t>(A_simde, a_access,
                         km1_size, k_size,
                         a_mat_rows, a_mat_cols);

        //the following works which is nice
        partialRowReduce<value_t>(B_simde, b_access,
                        k_size, kp1_size,
                         b_mat_rows, b_mat_cols);
        // if(i == 0
        

        int64_t rank_k = numPivotRows<value_t>(A_simde, a_access, k_size, km1_size);
        int64_t rank_kp1 = numPivotRows<value_t>(B_simde, b_access, k_size, kp1_size);
        int64_t betti = (k_size - rank_k) - rank_kp1;

        if(betti > 0){ 
            out[correspond.first] = betti;
            std::vector<bool> is_pivot_col = isPivotCol<value_t>(b_access, k_size, kp1_size);
            // if(cntr == 0){
            //     std::cout << "pivot cols: ";
            //     for(const auto& val : is_pivot_col){
            //         std::cout << std::boolalpha << val<<std::noboolalpha << ' ';
            //     }
            //     std::cout << std::endl;
            //     d_kplus1.block(0, k_size, 0, kp1_size).print();
            //     ++cntr;
            // }
            col_spaces.emplace_back(correspond.first, d_kplus1.extract_cols(std::move(is_pivot_col), k_size));
        }else if (add_zeros){
            out[correspond.first] = 0;
        }
    }


    freeTemplateMatrix<value_t>(a_access);
    freeTemplateMatrix<value_t>(b_access);
    freeSIMDeArray<value_t>(A_simde, d_k.Cols());
    freeSIMDeArray<value_t>(B_simde, d_kplus1.Rows());
    return std::pair<std::map<double, int64_t>, std::vector<std::pair<double, SparseMatrix>>>{std::move(out), std::move(col_spaces)};
}


//Eigen::MatrixXf makeEigenFromSparse(SparseMatrix& spm, int64_t max_rows, int64_t max_cols){
//    auto begin = spm.cmem_begin<int8_t>();
//    auto end = spm.cmem_end<int8_t>();
//    Eigen::MatrixXf out = Eigen::MatrixXf::Zero(max_rows, max_cols);
//    for(;begin != end; ++begin){
//        if(begin.Row() > max_rows) break;
//        if(begin.Col() > max_cols) continue;
//        out[begin.Row()][begin.Col()] = *begin;
//    }
//    return std::move(out);
//}

//Eigen::MatrixXf getKernel(Eigen::MatrixXf mat){
//    Eigen::FullPivLU<MatType> lu(mat);
//    return lu.kernel();
//}


//template<typename T>
//inline T** SIMDeMatrix_TemplateMatrix(typename mp::SimdTraits<T>::Type **ptr, int64_t num_rows){
//    T** out = new T*[num_rows];
//    for(int64_t i = 0; i < num_rows; ++i){
//        out[i] = reinterpret_cast<T*>(ptr[i]);
//    }
//    return out;
//}

//template<typename T>
//void loadSparseMatrix(T** arr, const SparseMatrix& col_space, Eigen::MatrixXf& null_space){
//    auto begin = spm.cmem_begin<int8_t>();
//    auto end = spm.cmem_end<int8_t>();
//    for(;begin != end; ++begin){
//        arr[begin.Row()][begin.Col()] = T(*begin);
//    }
//    int64_t start = begin.Cols();
//    for(size_t r = 0; r < null_space.rows(); ++r){
//        for(size_t c = 0; c < null_space.cols(); ++c){
//            arr[c][r+start] = null_space(r, c); //transpose initialize
//        }
//    }

//}


////gets betti numbers and generators
//std::pair<std::map<double, int64_t>, std::vector<std::pair<double, SparseMatrix>>> getBettiNumbersGenerators(
//    SparseMatrix &d_k, SparseMatrix &d_kplus1,
//    std::map<double, std::tuple<int64_t, int64_t, int64_t>> radi_bounds,
//    double max, bool add_zeros) {
//    utils::throw_exception(
//        d_k.dtype() == DType::int8 && d_kplus1.dtype() == DType::int8,
//        "Expected boundary matrices to have dtype int8 but got $ and $",
//        d_k.dtype(), d_kplus1.dtype());
//    utils::throw_exception(
//        d_k.Cols() == d_kplus1.Rows(),
//        "Matrices have incompatible shapes: d_k is {$, $}, d_kplus1 is {$, $}",
//        d_k.Rows(), d_k.Cols(), d_kplus1.Rows(), d_kplus1.Cols());


    
//    using value_t = int8_t;
    
//    const int64_t& a_mat_rows = d_k.Cols();
//    const int64_t& a_mat_cols = d_k.Rows();
//    const int64_t& b_mat_rows = d_kplus1.Rows();
//    const int64_t& b_mat_cols = d_kplus1.Cols();

//    mp::SimdTraits<value_t>::Type **A_simde =
//        allocateSIMDeMatrix<value_t>(d_k.Cols(), d_k.Rows()); // transpose is loaded in
//    mp::SimdTraits<value_t>::Type **B_simde =
//        allocateSIMDeMatrix<value_t>(d_kplus1.Rows(), d_kplus1.Cols());
    
//    value_t** a_access = SIMDeMatrix_TemplateMatrix<value_t>(A_simde, d_k.Cols());
//    value_t** b_access = SIMDeMatrix_TemplateMatrix<value_t>(B_simde, d_kplus1.Rows());

//    loadSparseMatrix<value_t>(a_access, d_k, true);
//    loadSparseMatrix<value_t>(b_access, d_kplus1, false);

//    std::map<double, int64_t> out;
//    std::vector<std::pair<double, Tensor> > generators;
//    for(const auto& correspond : radi_bounds){
//        auto [km1_size, k_size, kp1_size] = correspond.second;
//        if(km1_size > a_mat_cols) continue;
//        if(k_size > a_mat_rows) continue;
//        if(k_size > b_mat_rows) continue;
//        if(kp1_size > b_mat_cols) continue;
//        if(max > 0 && correspond.first > max) break;
//        // std::cout << "running radius "<<correspond.first<<std::endl;
//        partialColReduce<value_t>(A_simde, a_access,
//                         km1_size, k_size,
//                         a_mat_rows, a_mat_cols);

//        //the following works which is nice
//        partialRowReduce<value_t>(B_simde, b_access,
//                        k_size, kp1_size,
//                         b_mat_rows, b_mat_cols);
//        // if(i == 0
        

//        int64_t rank_k = numPivotRows<value_t>(A_simde, a_access, k_size, km1_size);
//        int64_t rank_kp1 = numPivotRows<value_t>(B_simde, b_access, k_size, kp1_size);
//        int64_t betti = (k_size - rank_k) - rank_kp1;

//        if(betti > 0){ 
//            out[correspond.first] = betti;
//            std::vector<bool> is_pivot_col = isPivotCol<value_t>(b_access, k_size, kp1_size);
//            SparseMatrix col_space = d_kplus1.extract_cols(std::move(is_pivot_col), k_size);
//            Eigen::MatrixXf null_space = getKernel(makeEigenFromSparse(d_k, km1_size, k_size));
//            utils::THROW_EXCEPTION(null_space.cols() == col_space.Rows(), "INTERANAL LOGIC ERROR");
//            int64_t quotient_rows = col_space.Rows();
//            int64_t quotient_cols = col_space.COls() + null_space.rows();
//            mp::SimdTraits<float>::Type **Quotient_simde =
//                    allocateSIMDeMatrix<float>(col_space.Rows(), col_space.Cols()+null_space.rows());
//            float** quotient_access = SIMDeMatrix_TemplateMatrix<float>(Quotient_simde, col_space.Cols()+null_space.rows());
//            loadSparseMatrix<float>(quotient_access, col_space, null_space); //this is the same as cat(col_space, null_space, 1)
            
//        }else if (add_zeros){
//            out[correspond.first] = 0;
//        }
//    }


//    freeTemplateMatrix<value_t>(a_access);
//    freeTemplateMatrix<value_t>(b_access);
//    freeSIMDeArray<value_t>(A_simde, d_k.Cols());
//    freeSIMDeArray<value_t>(B_simde, d_kplus1.Rows());
//    return std::pair<std::map<double, int64_t>, std::vector<std::pair<double, SparseMatrix>>>{std::move(out), std::move(col_spaces)};
//}


} // namespace cpu
} // namespace tda
} // namespace nt



/*


they are equal up at (23,30), not at (25,30)


skipping 0
pivot is (0) 1
pivot is (1) 1
swapping 2 and 24
pivot is (2) -1
swapping 3 and 24
pivot is (3) 1
swapping 4 and 24
pivot is (4) 1
swapping 5 and 24
pivot is (5) 1
skipping 7
swapping 6 and 24
pivot is (6) 1
swapping 7 and 24
pivot is (7) 1
skipping 10
swapping 8 and 24
pivot is (8) 1
swapping 9 and 24
pivot is (9) 1
skipping 13
swapping 10 and 23
pivot is (10) 1
swapping 11 and 24
pivot is (11) -1
skipping 16
swapping 12 and 23
pivot is (12) 1
skipping 18
skipping 19
skipping 20
skipping 21
skipping 22
swapping 13 and 24
pivot is (13) 1
swapping 14 and 23
pivot is (14) 1
swapping 15 and 24
pivot is (15) 1
swapping 16 and 23
pivot is (16) 1
swapping 17 and 24
pivot is (17) 1
skipping 28
skipping 29


skipping 0
a pivot(0: 1
a pivot(1: 1
swapping 2 and 24
a pivot(2: -1
swapping 3 and 24
a pivot(3: 1
swapping 4 and 24
a pivot(4: 1
swapping 5 and 24
a pivot(5: 1
skipping 7
swapping 6 and 24
a pivot(6: 1
swapping 7 and 24
a pivot(7: 1
skipping 10
swapping 8 and 24
a pivot(8: 1
swapping 9 and 24
a pivot(9: 1
skipping 13
swapping 10 and 23
a pivot(10: 1
swapping 11 and 24
a pivot(11: -1
skipping 16
swapping 12 and 23
a pivot(12: 1
skipping 18
skipping 19
skipping 20
skipping 21
skipping 22
swapping 13 and 24
a pivot(13: 1
swapping 14 and 23
a pivot(14: 1
swapping 15 and 24
a pivot(15: 1
swapping 16 and 23
a pivot(16: 1
swapping 17 and 24
a pivot(17: 1
skipping 28
skipping 29

boundary k: {60,1770}
boundary k+1: {1770,34220}
starting new function
duration_a: 29843
duration_b: 69144
finished test with durations 29843,69144
made simplices
made sparse tensor
boundary k: {60,1770}
boundary k+1: {1770,34220}
starting new function

*/
