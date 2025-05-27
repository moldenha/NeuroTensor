#include "convert.h"
#include "../cpu/convert.h"
#include "exceptions.hpp"
#include "../../dtype/ArrayVoid.hpp"

namespace nt{
namespace functional{

Tensor to(const Tensor& t, DType dt){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(t);
    if(t.dtype == dt) return t;
    if(dt == DType::TensorObj){
        Tensor buckets = t.arr_void().get_bucket().split<Tensor>(1); // super inefficient but possible
        Tensor* begin = reinterpret_cast<Tensor*>(buckets.data_ptr());
        Tensor* end = begin + buckets.numel();
        typedef typename SizeRef::ArrayRefInt::value_type m_size_t;
        for(;begin != end; ++begin){
            begin->dtype = t.dtype;
            begin->set_mutability(t.is_mutable());
        }
        return std::move(buckets);
    }else if(t.dtype == DType::TensorObj){
        Tensor out(t.shape(), dt);
        out.arr_void().execute_function<WRAP_DTYPES<NumberTypesL, DTypeEnum<DType::Bool> > >(
        [&t](auto begin, auto end){
            using value_t = utils::IteratorBaseType_t<decltype(begin)>;
            t.arr_void().cexecute_function<WRAP_DTYPES<DTypeEnum<DType::TensorObj> > >(
            [&](auto begin_t, auto end_t){
                for(;begin_t != end_t; ++begin_t, ++begin){
                    *begin = begin_t->toScalar().template to<value_t>();
                }
            });
        });
        return std::move(out);
    }
    Tensor out(t.shape(), dt);
    cpu::_convert(t.arr_void(), out.arr_void());
    return std::move(out);
}

Tensor& floating_(Tensor& t){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(t);
    cpu::_floating_(t.arr_void());
    return t;
}
Tensor& complex_(Tensor& t){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(t);
    cpu::_complex_(t.arr_void());
    return t;

}
Tensor& integer_(Tensor& t){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(t);
    cpu::_integer_(t.arr_void());
    return t;

}
Tensor& unsigned_(Tensor& t){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(t);
    cpu::_unsigned_(t.arr_void());
    return t;

}

}
}
