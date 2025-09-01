#include "convert.h"
#include "../cpu/convert.h"
#include "exceptions.hpp"
#include "../../dtype/ArrayVoid.hpp"

namespace nt{
namespace functional{

Tensor to(const Tensor& t, DType dt){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(t);
    if(t.dtype() == dt) return t;
    if(dt == DType::TensorObj){
        Tensor buckets = t.arr_void().get_bucket().split<Tensor>(1); // super inefficient but possible
        Tensor* begin = reinterpret_cast<Tensor*>(buckets.data_ptr());
        Tensor* end = begin + buckets.numel();
        typedef typename SizeRef::ArrayRefInt::value_type m_size_t;
        for(;begin != end; ++begin){
            begin->arr_void().get_bucket().dtype = t.dtype();
            begin->set_mutability(t.is_mutable());
        }
        return std::move(buckets);
    }
    Tensor out(t.shape(), dt);
    cpu::_convert(t.arr_void(), out.arr_void());
    return std::move(out);
}

Tensor& floating_(Tensor& t){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(t);
    cpu::_floating_(t.arr_void());
    t.arr_void().get_bucket().dtype = DTypeFuncs::floating_size(DTypeFuncs::size_of_dtype(t.dtype()));
    return t;
}
Tensor& complex_(Tensor& t){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(t);
    cpu::_complex_(t.arr_void());
    t.arr_void().get_bucket().dtype = DTypeFuncs::complex_size(DTypeFuncs::size_of_dtype(t.dtype()));
    return t;

}
Tensor& integer_(Tensor& t){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(t);
    cpu::_integer_(t.arr_void());
    t.arr_void().get_bucket().dtype = DTypeFuncs::integer_size(DTypeFuncs::size_of_dtype(t.dtype()));
    return t;

}
Tensor& unsigned_(Tensor& t){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(t);
    cpu::_unsigned_(t.arr_void());
    t.arr_void().get_bucket().dtype = DTypeFuncs::unsigned_size(DTypeFuncs::size_of_dtype(t.dtype()));
    return t;

}

}
}
