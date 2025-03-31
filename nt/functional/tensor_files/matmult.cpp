/* #include <immintrin.h> */

#include "../../dtype/Scalar.h"
#include "../functional.h"
#include "../cpu/matmult_cpu.cpp"

namespace nt{
namespace functional{


Tensor matmult(const Tensor& a, const Tensor& b, bool trans_a, bool trans_b){
	return cpu::matmult(a, b, trans_a, trans_b);
}

/* Tensor matmult(const Tensor& a, const Tensor& b, bool trans_a, bool trans_b){ */
/* 	return mkl_functions::mkl_matmult(a, b, trans_a, trans_b); */
/* } */

Tensor matmult_cT(const Tensor& a, const Tensor& b){return matmult(a,b,false,true);}

}
}
