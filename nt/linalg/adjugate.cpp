// #include "../linalg.h"
#include "../utils/utils.h"
#include "../Tensor.h"
#include "../functional/functional.h"
#include <limits>
#include <tuple>
#include "../dtype/ArrayVoid.hpp"
#define _SILENCE_CXX17_C_HEADER_DEPRECATION_WARNING

#ifdef USE_PARALLEL
	#include <tbb/parallel_for_each.h>
	#include <tbb/parallel_for.h>
	#include <tbb/blocked_range.h>
	#include <tbb/blocked_range2d.h>
	#include <tbb/blocked_range3d.h>
	#include <tbb/parallel_reduce.h>
	#include <thread>
    #include <tbb/concurrent_vector.h>
	/* #include "../mp/MP.hpp" */
	/* #include "../mp/Pool.hpp" */
#endif

namespace nt{
namespace linalg{

template<typename value_t, typename iterator>
inline value_t determinant_2x2(iterator it){return (it[0] * it[3]) - (it[1] * it[2]);}
template<typename value_t, typename iterator>
inline value_t determinant_3x3(iterator it){
    return (it[0] * ((it[4] * it[8]) - (it[5] * it[7]))) - 
            (it[1] * ((it[3] * it[8]) - (it[6] * it[5]))) +
            (it[2] * ((it[3] * it[7]) - (it[4] * it[6])));
}

template<typename value_t, typename iterator>
value_t cofactor(iterator it, int64_t n, int64_t r, int64_t c);

template<typename value_t, typename iterator>
inline value_t _nt_sub_ptr_determinant_(iterator it, int64_t n){
    if(n == 1){return *it;}
    if(n == 2){return determinant_2x2<value_t, iterator>(it);}
    if(n == 3){return determinant_3x3<value_t, iterator>(it);}
    value_t det = 0;
    for(int64_t c = 0; c < n; ++c){
        det += it[c] * cofactor<value_t, iterator>(it, n, 0, c); 
    }
    return det;
}


template<typename value_t, typename iterator>
inline value_t cofactor(iterator mat, int64_t n, int64_t r, int64_t c){
    //alternate signs
    //row + col even -> positive
    //row + col odd -> negative
    char sign = ((r + c) & 1) == 0 ? 1 : -1;

    value_t* minor_mat = new value_t[(n-1) * (n-1)];
    
    int64_t minor_i = 0;
    int64_t minor_j = 0;
    for(int64_t i = 0; i < r; ++i){
        for(int64_t j = 0; j < c; ++j){
            minor_mat[minor_i * (n - 1) + minor_j] = mat[i * n + j];
            ++minor_j;
        }
        for(int64_t j = c+1; j < n; ++j){
            minor_mat[minor_i * (n - 1) + minor_j] = mat[i * n + j];
            ++minor_j;
        }
        minor_j = 0;
    }
    for(int64_t i = r+1; i < n; ++i){
        for(int64_t j = 0; j < c; ++j){
            minor_mat[minor_i * (n - 1) + minor_j] = mat[i * n + j];
            ++minor_j;
        }
        for(int64_t j = c+1; j < n; ++j){
            minor_mat[minor_i * (n - 1) + minor_j] = mat[i * n + j];
            ++minor_j;
        }
        minor_j = 0;
    }
    
    value_t minor_det = _nt_sub_ptr_determinant_<value_t, value_t*>(minor_mat, n-1);
    delete[] minor_mat;
    minor_det *= ((r + c) % 2 == 0 ? 1 : -1);
    return minor_det;
}



template<typename value_t, typename iterator>
void _nt_adjugate_square_matrix_(iterator it, value_t* out, int64_t n){
    for(int64_t r = 0; r < n; ++r){
        for(int64_t c = 0; c < n; ++c){
            out[r * n + c] = cofactor<value_t, iterator>(it, n, c + 1, r + 1); //cols and rows are switched
        }
    }
}

Tensor adjugate(const Tensor& mat){
    if(mat.dtype == DType::TensorObj){
        std::vector<Tensor> out;
        out.reserve(mat.numel());
        mat.arr_void().cexecute_function<WRAP_DTYPES<DTypeEnum<DType::TensorObj> > >([&out](auto begin, auto end){for(;begin != end; ++begin){out.emplace_back(adjugate(*begin));}});
        return ::nt::functional::vectorize(std::move(out));
    }
    utils::throw_exception(mat.dtype != DType::Bool, "Cannot take adjugation of tensor with type bool but got $", mat.dtype);
    utils::throw_exception(mat.dims() >= 2, "Can only take the adjugation of a tensor with dims greater than or equal to 2, but got $", mat.dims());
    utils::throw_exception(mat.shape()[-1] == mat.shape()[-2], "Can only take the adjugation of square matricies, but got last 2 dims to be {$, $}", mat.shape()[-2], mat.shape()[-1]);
    if(mat.shape()[-1] == 1){return mat;}
    if(mat.dims() == 2){
        Tensor out(mat.shape(), mat.dtype);
        int64_t n = mat.shape()[-1];
        mat.arr_void().cexecute_function<WRAP_DTYPES<NumberTypesL>>([&out, &n](auto begin, auto end){
            using value_t = utils::IteratorBaseType_t<decltype(begin)>;
            value_t* val = reinterpret_cast<value_t*>(out.data_ptr());
            _nt_adjugate_square_matrix_<value_t, decltype(begin)>(begin, val, n);
        });
        return std::move(out);
    }
    Tensor out(mat.shape(), mat.dtype);
    int64_t n = mat.shape().back();
    mat.arr_void().cexecute_function<WRAP_DTYPES<NumberTypesL>>([&out, &n](auto begin, auto end){
        using value_t = utils::IteratorBaseType_t<decltype(begin)>;
        int64_t mat_size = n * n;
        int64_t batches = out.numel() / mat_size;
        value_t* o_vals = reinterpret_cast<value_t*>(out.data_ptr());
#ifdef USE_PARALLEL
        tbb::parallel_for(
            utils::calculateGrainSize1D(batches),
            [&](const tbb::blocked_range<int64_t> &range){
                auto o_begin = o_vals + (range.begin() * mat_size);
                auto o_end = o_vals + (range.end() * mat_size);
                auto i_begin = begin + (range.begin() * mat_size);
                for(;o_begin != o_end; o_begin += mat_size, i_begin += mat_size){
                    _nt_adjugate_square_matrix_<value_t, decltype(begin)>(i_begin, o_begin, n);
                }
            });
#else
        value_t* o_end = o_vals + (batches * mat_size);
        for(;o_vals != o_end; o_vals += mat_size, begin += mat_size){
            _nt_adjugate_square_matrix_<value_t, decltype(beigin)>(i_begin, o_vals, n);
        }
#endif
    });
    return std::move(out);
}


}}
