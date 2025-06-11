/* #include <immintrin.h> */

#include "../../dtype/Scalar.h"
#include "../functional.h"
#include "../cpu/matmult_cpu.cpp"

namespace nt{
namespace functional{


Tensor matmult(const Tensor& a, const Tensor& b, bool trans_a, bool trans_b){
    utils::throw_exception(!a.is_null() && !b.is_null(),
                           "Got null tensors for matrix multiplication");
	return cpu::matmult(a, b, trans_a, trans_b);
}

Tensor& matmult(const Tensor& a, const Tensor& b, Tensor& c, bool trans_a, bool trans_b){
    utils::throw_exception(c.is_mutable(), "output from matmult must be mutable");
    utils::throw_exception(!a.is_null() && !b.is_null() && !c.is_null(),
                           "Got null tensors for matrix multiplication");
	return cpu::matmult(a, b, c, trans_a, trans_b);
}

Tensor linear(const Tensor& input, const Tensor& weight, const Tensor& bias, bool trans_input, bool trans_weight){
    if(bias.is_null()) return matmult(input, weight, trans_input, trans_weight);
    utils::throw_exception(!input.is_null() && !weight.is_null() && !bias.is_null(),
                           "Got null tensors for matrix multiplication");
    return cpu::linear(input, weight, bias, trans_input, trans_weight);
}

/* Tensor matmult(const Tensor& a, const Tensor& b, bool trans_a, bool trans_b){ */
/* 	return mkl_functions::mkl_matmult(a, b, trans_a, trans_b); */
/* } */

Tensor matmult_cT(const Tensor& a, const Tensor& b){return matmult(a,b,false,true);}

}
}
