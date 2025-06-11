#include "../../Tensor.h"
#include "../../dtype/ArrayVoid.hpp"
#include "../cpu/trig.h"
#include "../../utils/numargs_macro.h"
#include "exceptions.hpp"

namespace nt{
namespace functional{

template<size_t N>
inline void check_dtypes(const DType& dt, const char(&s)[N]){
    utils::throw_exception(dt != DType::Bool && !DTypeFuncs::is_integer(dt), "Cannot perform $ on dtype $", dt);
}

#define ADD_UNDERSCORE(name) name##_
#define ADD_DOUBLE_UNDERSCORE(name) _##name##_

#define NT_MAKE_TRIG_FUNCTION_(func_name)\
Tensor func_name(const Tensor& x){\
    _NT_FUNCTIONAL_ALWAYS_CHECK_(x);\
    check_dtypes(x.dtype, #func_name);\
    if(x.dtype == DType::TensorObj){\
        Tensor out = Tensor::makeNullTensorArray(x.numel());\
        Tensor* o_begin = reinterpret_cast<Tensor*>(out.data_ptr());\
        x.arr_void().cexecute_function<WRAP_DTYPES<DTypeEnum<DType::TensorObj>>>([&o_begin](auto begin, auto end){\
            for(;begin != end; ++begin, ++o_begin){\
                *o_begin = func_name(*begin);\
            } \
        });\
        return std::move(out);\
    }\
    Tensor a = x.clone();\
    cpu::ADD_DOUBLE_UNDERSCORE(func_name)(a.arr_void());\
    return std::move(a);\
}
    

NT_MAKE_TRIG_FUNCTION_(tan);
NT_MAKE_TRIG_FUNCTION_(tanh);
NT_MAKE_TRIG_FUNCTION_(atan);
NT_MAKE_TRIG_FUNCTION_(atanh);
NT_MAKE_TRIG_FUNCTION_(cotan);
NT_MAKE_TRIG_FUNCTION_(cotanh);

NT_MAKE_TRIG_FUNCTION_(sin);
NT_MAKE_TRIG_FUNCTION_(sinh);
NT_MAKE_TRIG_FUNCTION_(asin);
NT_MAKE_TRIG_FUNCTION_(asinh);
NT_MAKE_TRIG_FUNCTION_(csc);
NT_MAKE_TRIG_FUNCTION_(csch);

NT_MAKE_TRIG_FUNCTION_(cos);
NT_MAKE_TRIG_FUNCTION_(cosh);
NT_MAKE_TRIG_FUNCTION_(acos);
NT_MAKE_TRIG_FUNCTION_(acosh);
NT_MAKE_TRIG_FUNCTION_(sec);
NT_MAKE_TRIG_FUNCTION_(sech);



NT_MAKE_TRIG_FUNCTION_(dtan);  // derivative of tan
NT_MAKE_TRIG_FUNCTION_(dtanh); // derivative of tanh
NT_MAKE_TRIG_FUNCTION_(datan);
NT_MAKE_TRIG_FUNCTION_(datanh);
NT_MAKE_TRIG_FUNCTION_(dcotan);
NT_MAKE_TRIG_FUNCTION_(dcotanh);

NT_MAKE_TRIG_FUNCTION_(dsin);
NT_MAKE_TRIG_FUNCTION_(dsinh);
NT_MAKE_TRIG_FUNCTION_(dasin);
NT_MAKE_TRIG_FUNCTION_(dasinh);
NT_MAKE_TRIG_FUNCTION_(dcsc);
NT_MAKE_TRIG_FUNCTION_(dcsch);

NT_MAKE_TRIG_FUNCTION_(dcos);
NT_MAKE_TRIG_FUNCTION_(dcosh);
NT_MAKE_TRIG_FUNCTION_(dacos);
NT_MAKE_TRIG_FUNCTION_(dacosh);
NT_MAKE_TRIG_FUNCTION_(dsec);
NT_MAKE_TRIG_FUNCTION_(dsech);



#undef NT_MAKE_TRIG_FUNCTION_ 
#undef ADD_UNDERSCORE 
#undef ADD_DOUBLE_UNDERSCORE 

}
}


