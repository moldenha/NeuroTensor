/* #include <immintrin.h> */

#include <iostream>
#include <sys/wait.h>
#include "../convert/Convert.h"
#include "../dtype/Scalar.h"
#include "functional.h"
/* #include "matmult_mkl.cpp" */
#include "matmult_std.cpp"

namespace nt{
namespace functional{


Tensor matmult(const Tensor& a, const Tensor& b, bool trans_a, bool trans_b){
	return std_functional::std_matmult(a, b, trans_a, trans_b);
}

/* Tensor matmult(const Tensor& a, const Tensor& b, bool trans_a, bool trans_b){ */
/* 	return mkl_functions::mkl_matmult(a, b, trans_a, trans_b); */
/* } */

Tensor matmult_cT(const Tensor& a, const Tensor& b){return matmult(a,b,false,true);}

}
}
