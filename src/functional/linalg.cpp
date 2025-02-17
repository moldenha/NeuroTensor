#include "linalg.h"
#include "../utils/utils.h"
#include "../Tensor.h"
#include "functional.h"
#include <limits>
#include <tuple>
#include "../dtype/ArrayVoid.hpp"

#ifdef USE_PARALLEL
	#include <tbb/parallel_for_each.h>
	#include <tbb/parallel_for.h>
	#include <tbb/blocked_range.h>
	#include <tbb/blocked_range2d.h>
	#include <tbb/blocked_range3d.h>
	#include <tbb/parallel_reduce.h>
	#include <thread>
	/* #include "../mp/MP.hpp" */
	/* #include "../mp/Pool.hpp" */
#endif

namespace nt{
namespace linalg{


    /* m, n = A.shape */
    /* Q = np.zeros_like(A) */
    /* R = np.zeros((n, n)) */

    /* for j in range(n): */
    /*     # Calculate the j-th column of Q */
    /*     v = A[:, j] */
    /*     for i in range(j): */
    /*         R[i, j] = np.dot(Q[:, i], A[:, j]) */
    /*         v = v - R[i, j] * Q[:, i] */

    /*     R[j, j] = np.linalg.norm(v) */
    /*     Q[:, j] = v / R[j, j] */

    /* return Q, R */

/* std::tuple<Tensor, Tensor> qr_decomposition(const Tensor& A){ */
	/* utils::throw_exception(A.dims() >= 2, "Expected to decompose a matrix or higher dimension tensor but got dims of $", A.dims()); */
	/* // Q: Orthogonal matrix. */
	/* //R: Upper triangular matrix. */
	/* int64_t rows = A.shape()[-2]; */
	/* int64_t cols = A.shape()[-1]; */

	/* Tensor Q = functional::zeros_like(A); */
	/* auto vec = A.shape().Vec(); */
	/* vec[vec.shape()-2] = cols; */
	/* Tensor R = functional::zeros(SizeRef(std::move(vec)), A.dtype); */

	/* Tensor a_split = A.split_axis(-1); */
	/* Tensor q_split = A.split_axis(-1); */


/* } */


//TODO: add nuclear norm
Tensor norm(const Tensor& A, std::variant<std::nullptr_t, std::string, int64_t> ord, utils::optional_list dim, bool keepdim){
	if(ord.index() == 0){
		return functional::sqrt(std::pow(A, 2).sum(dim, keepdim));
	}
	else if(ord.index() == 1){
		std::string name = std::get<1>(ord);
		utils::throw_exception(name == "fro",
				"Order strings accepted are fro but got $", name);
		if(name == "fro"){
			return functional::sqrt(std::pow(A, 2).sum(dim, keepdim));
		}
	}
	int64_t ord_i = std::get<2>(ord);
	if(ord_i == std::numeric_limits<int64_t>::infinity()){
		return functional::abs(A).max(dim).values;
	}
	if(ord_i == -std::numeric_limits<int64_t>::infinity()){
		return functional::abs(A).min(dim).values;
	}
	utils::throw_exception(ord_i > 0, "Order must be fro or positive integer");
	Tensor a_out = std::pow(functional::abs(A), ord_i).sum(dim);
	return std::pow(a_out, (1.0 / (double)ord_i));
}


Tensor eye(int64_t n, int64_t b, DType dtype){
	if(b > 0){
		Tensor out = functional::zeros({b, n, n}, dtype);
		out.arr_void().execute_function<WRAP_DTYPES<NumberTypesL>>([&b, &n](auto begin, auto end){
			using value_t = utils::IteratorBaseType_t<decltype(begin)>;
			for(int64_t k = 0; k < b; ++k){
				for(int64_t i = 0; i < n; ++i){
                    begin += i;
                    *begin = value_t(1);
                    begin += (n-i);
				}	
			}


		});
		return std::move(out);
	}
	Tensor out = functional::zeros({n, n}, dtype);
	out.arr_void().execute_function<WRAP_DTYPES<NumberTypesL>>([&n](auto begin, auto end){
		using value_t = utils::IteratorBaseType_t<decltype(begin)>;
		for(int64_t i = 0; i < n; ++i){
            begin += i;
            *begin = value_t(1);
            begin += (n-i);
		}	

	});
	return std::move(out);
}


Tensor eye_like(const Tensor& t){
	utils::throw_exception(t.dims() >= 2, "Expected to make identity matrix with dimensions greater than or equal to 2 but got $", t.dims());
	if(t.dims() == 2){
		return eye(t.shape()[-1], 0, t.dtype);
	}
	auto vec = t.shape().Vec();
	vec[vec.size()-2] = vec.back();
	return eye(t.shape()[-1], t.shape().flatten(0, -3)[0], t.dtype).view(SizeRef(std::move(vec)));
}


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
value_t _nt_sub_ptr_determinant_(iterator it, int64_t n){
    if(n == 1){return *it;}
    if(n == 2){return determinant_2x2<value_t, iterator>(it);}
    if(n == 3){return determinant_3x3<value_t, iterator>(it);}
    value_t det = 0;
    for(int64_t c = 0; c < n; ++c){
        det += it[c] * cofactor<value_t, iterator>(it, n, 0, c); 
    }
    return det;
}

//one allocation of memory mat of nxn can happen
//and it will suffice for all of the cofactors and determinants
template<typename value_t, typename iterator>
value_t cofactor(iterator mat, int64_t n, int64_t r, int64_t c){
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
    return ((r + c) % 2 == 0 ? 1 : -1) * minor_det;

}


Tensor determinant(const Tensor& mat){
    if(mat.dtype == DType::TensorObj){
        std::vector<Tensor> out;
        out.reserve(mat.numel());
        mat.arr_void().cexecute_function<WRAP_DTYPES<DTypeEnum<DType::TensorObj> > >([&out](auto begin, auto end){for(;begin != end; ++begin){out.emplace_back(determinant(*begin));}});
        return ::nt::functional::vectorize(std::move(out));
    }
    utils::throw_exception(mat.dtype != DType::Bool, "Cannot take determinant of tensor with type bool but got $", mat.dtype);
    utils::throw_exception(mat.dims() >= 2, "Can only take the determinant of a tensor with dims greater than or equal to 2, but got $", mat.dims());
    utils::throw_exception(mat.shape()[-1] == mat.shape()[-2], "Can only take the determinant of square matricies, but got last 2 dims to be {$, $}", mat.shape()[-2], mat.shape()[-1]);
    if(mat.shape()[-1] == 1){return mat;}
    if(mat.dims() == 2){
        Tensor out({1}, mat.dtype);
        int64_t n = mat.shape()[-1];
        mat.arr_void().cexecute_function<WRAP_DTYPES<NumberTypesL>>([&out, &n](auto begin, auto end){
            using value_t = utils::IteratorBaseType_t<decltype(begin)>;
            value_t* val = reinterpret_cast<value_t*>(out.data_ptr());
            *val = _nt_sub_ptr_determinant_<value_t, decltype(begin)>(begin, n);
        });
        return std::move(out);
    }
    auto shape_out = mat.shape().Vec();
    shape_out.pop_back();
    shape_out.back() = 1;
    Tensor out(SizeRef(std::move(shape_out)), mat.dtype);
    int64_t n = mat.shape().back();
    mat.arr_void().cexecute_function<WRAP_DTYPES<NumberTypesL>>([&out, &n](auto begin, auto end){
        using value_t = utils::IteratorBaseType_t<decltype(begin)>;
        int64_t mat_size = n * n;
        int64_t batches = out.numel();
        value_t* o_vals = reinterpret_cast<value_t*>(out.data_ptr());
#ifdef USE_PARALLEL
        tbb::parallel_for(
            utils::calculateGrainSize1D(batches),
            [&](const tbb::blocked_range<int64_t> &range){
                auto o_begin = o_vals + range.begin();
                auto o_end = o_vals + range.end();
                auto i_begin = begin + (range.begin() * mat_size);
                for(;o_begin != o_end; ++o_begin, i_begin += mat_size){
                    *o_begin = _nt_sub_ptr_determinant_<value_t, decltype(i_begin)>(i_begin, n);
                }
            });
#else
        value_t* o_end = o_vals + batches;
        for(;o_vals != o_end; ++o_vals, begin += mat_size){
            *o_vals = _nt_sub_ptr_determinant_<value_t, decltype(begin)>(begin, n);
        }
#endif
    });
    return std::move(out);
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


// void inverse_adj_matrix(Tensor& mat){
    
// }
// Tensor inverse(const Tensor& mat) {
//     Tensor adj = adjugate(mat);

// }


}} //nt::linalg::
