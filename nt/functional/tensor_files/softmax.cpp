#include "softmax.h"
#include "../../Tensor.h"
#include "../cpu/softmax.h"
#include "../../dtype/ArrayVoid.hpp"
#include "exceptions.hpp"
#include "rand.h"
#include "mesh.h"
#include "min_max.h"

namespace nt {
namespace functional {

//this applies a softmax function over the entire inputted tensor
void softmax_(Tensor& inp){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(inp);
    utils::throw_exception(inp.is_mutable(),
                           "Can only perform softmax function on self if the tensor is mutable");
    if(inp.dtype() == DType::TensorObj){
        inp.arr_void().execute_function<WRAP_DTYPES<DTypeEnum<DType::TensorObj> > >([](auto begin, auto end){
            for(;begin != end; ++begin)
                softmax_(*begin);
        });
        return;
    }
    cpu::_softmax(inp.arr_void(), inp.arr_void());
}

void softmax_(Tensor& inp, typename SizeRef::value_type dim){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(inp);
    utils::throw_exception(inp.is_mutable(),
                           "Can only perform softmax function on self if the tensor is mutable");
    if(inp.dims() == 1){
        softmax_(inp);
        return;
    }
    dim = dim < 0 ? dim + inp.dims() : dim;
    utils::throw_exception(dim >= 0 && dim < inp.dims(), "Got invalid dim $ to apply softmax for shape $", inp.shape());
    //if dim is the equivalent of -1 or -2 they need to be swapped:
    if(dim == (inp.dims()-1)){
        --dim;
    }else if (dim  == (inp.dims()-2)){
        ++dim;
    }
	Tensor tensors = inp.split_axis(dim);
    Tensor* begin = reinterpret_cast<Tensor*>(tensors.data_ptr());
    Tensor* end = reinterpret_cast<Tensor*>(tensors.data_ptr_end());
    for(;begin != end; ++begin)
        softmax_(*begin);

}

Tensor softmax(Tensor inp){
    //convert it to a double for stability
	Tensor outp = inp.dtype() == DType::Float64 ? inp.clone() : inp.to(DType::Float64);
	softmax_(outp);
    //convert back to the original dtype
	return outp.to(inp.dtype());
}

Tensor softmax(Tensor inp, typename SizeRef::value_type dim){
    //convert it to a double for stability
	Tensor outp = inp.dtype() == DType::Float64 ? inp.clone() : inp.to(DType::Float64);
    softmax_(outp, dim);
    //convert back to the original dtype
	return outp.to(inp.dtype());
}

void softmax_stable_(Tensor& inp){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(inp);
    utils::throw_exception(inp.is_mutable(),
                           "Can only perform softmax function on self if the tensor is mutable");
    if(inp.dtype() == DType::TensorObj){
        inp.arr_void().execute_function<WRAP_DTYPES<DTypeEnum<DType::TensorObj> > >([](auto begin, auto end){
            for(;begin != end; ++begin)
                softmax_stable_(*begin);
        });
        return;
    }
    Scalar max = inp.max().values.toScalar();
    cpu::_softmax_stable(inp.arr_void(), inp.arr_void(), max);
}

void softmax_stable_(Tensor& inp, typename SizeRef::value_type dim){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(inp);
    utils::throw_exception(inp.is_mutable(),
                           "Can only perform softmax function on self if the tensor is mutable");
    if(inp.dims() == 1){
        softmax_stable_(inp); 
        return;
    }
    dim = dim < 0 ? dim + inp.dims() : dim;
    utils::throw_exception(dim >= 0 && dim < inp.dims(), "Got invalid dim $ to apply softmax for shape $", inp.shape());
    //if dim is the equivalent of -1 or -2 they need to be swapped:
    if(dim == (inp.dims()-1)){
        --dim;
    }else if (dim  == (inp.dims()-2)){
        ++dim;
    }
	Tensor tensors = inp.split_axis(dim);
    softmax_stable_(tensors);
}

Tensor softmax_stable(Tensor inp){
    DType to_dtype = DTypeFuncs::is_complex(inp.dtype()) ? DType::Complex128 : DType::Float64;
	Tensor outp = inp.dtype() == to_dtype ? inp.clone() : inp.to(to_dtype);
    softmax_stable_(outp);
	return outp.to(inp.dtype());
}


Tensor softmax_stable(Tensor inp, typename SizeRef::value_type dim){
    DType to_dtype = DTypeFuncs::is_complex(inp.dtype()) ? DType::Complex128 : DType::Float64;
	Tensor outp = inp.dtype() == to_dtype ? inp.clone() : inp.to(to_dtype);
    softmax_stable_(outp, dim);
	return outp.to(inp.dtype());
}


Tensor gumbel_softmax(const Tensor& __logits, Scalar tau, bool hard, int64_t dim, bool stable){
    if(stable){
        DType to_dtype = DTypeFuncs::is_complex(__logits.dtype()) ? DType::Complex128 : DType::Float64;
        Tensor logits = __logits.to(to_dtype);
        Tensor u = rand(0, 1, logits.shape(), logits.dtype()); // Uniform(0, 1)
        Tensor y = (__logits.dtype() == to_dtype) ? logits.clone() : logits;
        cpu::_gumbel_algorithm_(y.arr_void(), u.arr_void(), tau);
        softmax_stable_(y, dim);
        if(!hard){
            return y.to(__logits.dtype());
        }
        return ::nt::functional::one_hot(::nt::functional::argmax(y, dim), y.shape()[dim]).to(__logits.dtype());
    }
    Tensor u = rand(0, 1, __logits.shape(), __logits.dtype()); // Uniform(0, 1)
    // this is the noise before applying the  gumbel algorithm (look at cpu)
    Tensor y = __logits.clone();
    cpu::_gumbel_algorithm_(y.arr_void(), u.arr_void(), tau);
    softmax_(y, dim);
    if(!hard) return std::move(y);
    return ::nt::functional::one_hot(::nt::functional::argmax(y, dim), y.shape()[dim]).to(__logits.dtype());
}



Tensor dsoftmax(const Tensor& dy, const Tensor& last_softmax){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(dy, last_softmax);
    utils::THROW_EXCEPTION(dy.dtype() == last_softmax.dtype(),
                           "Expected the softmax derivative ($) and last softmax ($) to have the same dtype" , 
                           dy.dtype(), last_softmax.dtype());
    utils::THROW_EXCEPTION(dy.shape() == last_softmax.shape(),
                           "Expected the softmax derivative ($) and last softmax ($) to have the same shape" , 
                           dy.shape(), last_softmax.shape());
    utils::THROW_EXCEPTION(dy.dtype() != DType::TensorObj,
                           "dsoftmax is not implemented for tensor objects");

    Tensor output(last_softmax.shape(), DType::Float64);
    Tensor _last_softmax = last_softmax.to(DType::Float64);
    Tensor _dy = dy.to(DType::Float64);
    cpu::_dsoftmax(_last_softmax.arr_void(), _dy.arr_void(), output.arr_void());
    return output.to(dy.dtype());
}

Tensor dsoftmax(const Tensor& dy, const Tensor& last_softmax, typename SizeRef::value_type dim){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(dy, last_softmax);
    utils::THROW_EXCEPTION(dy.dtype() == last_softmax.dtype(),
                           "Expected the softmax derivative ($) and last softmax ($) to have the same dtype" , 
                           dy.dtype(), last_softmax.dtype());
    utils::THROW_EXCEPTION(dy.shape() == last_softmax.shape(),
                           "Expected the softmax derivative ($) and last softmax ($) to have the same shape" , 
                           dy.shape(), last_softmax.shape());

    if(dy.dims() == 1){
        return dsoftmax(dy, last_softmax);
    }
    dim = dim < 0 ? dim + dy.dims() : dim;
    utils::throw_exception(dim >= 0 && dim < dy.dims(), "Got invalid dim $ to apply dsoftmax for shape $", dy.shape());
    //if dim is the equivalent of -1 or -2 they need to be swapped:
    if(dim == (dy.dims()-1)){
        --dim;
    }else if (dim  == (dy.dims()-2)){
        ++dim;
    }
    Tensor output(last_softmax.shape(), DType::Float64);
    Tensor output_split = output.split_axis(dim);
    Tensor _last_softmax = last_softmax.to(DType::Float64);
    Tensor _dy = dy.to(DType::Float64);
    Tensor dy_split = _dy.split_axis(dim);
    Tensor ls_split = _last_softmax.split_axis(dim);
    Tensor* o_begin = reinterpret_cast<Tensor*>(output_split.data_ptr());
    Tensor* dy_begin = reinterpret_cast<Tensor*>(dy_split.data_ptr());
    Tensor* dy_end = reinterpret_cast<Tensor*>(dy_split.data_ptr_end());
    Tensor* ls_begin = reinterpret_cast<Tensor*>(ls_split.data_ptr());
    for(;dy_begin != dy_end; ++dy_begin, ++o_begin, ++ls_begin){
        cpu::_dsoftmax(ls_begin->arr_void(), dy_begin->arr_void(), o_begin->arr_void());
    }
    return output.to(dy.dtype());

}


} // namespace functional
} // namespace nt


