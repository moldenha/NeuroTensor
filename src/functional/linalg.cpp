#include "linalg.h"
#include "../utils/utils.h"
#include "../Tensor.h"
#include "functional.h"
#include <limits>
#include <tuple>
#include "../dtype/ArrayVoid.hpp"

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

}} //nt::linalg::
