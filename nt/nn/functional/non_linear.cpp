#include "../../functional/functional.h"
#include "../../Tensor.h"
#include "../../intrusive_ptr/intrusive_ptr.hpp"
#include "../TensorGrad.h"
#include "../functional_class.h"
#include "../../utils/macros.h"

namespace nt{
namespace functional{

//this is for simple non-linear functions where they only take a single argument and return a single argument
//just an easy way to put it all into one cpp file

#define _NT_MAKE_NONLINEAR_DEFINED_FUNCTION_(func_name)\
TensorGrad TensorGrad_Functional_Class::func_name(const TensorGrad &x) { \
    TensorGrad result(::nt::functional::func_name(x.tensor), x.grad_required); \
    if (!x.do_track_grad) { \
        result.do_track_grad = false; \
        return std::move(result); \
    } \
    result.track_tensors(x); \
    result.create_backward_function( \
            [](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents, \
                                                    intrusive_ptr<tensor_holder> saved_x) { \
                parents[0]->grad->tensor += grad * _NT_GLUE_(::nt::functional::d, func_name)(saved_x->tensor); \
            }, \
            make_intrusive<tensor_holder>(x.tensor.conditional_mutate_clone())); \
    return std::move(result); \
} \


_NT_MAKE_NONLINEAR_DEFINED_FUNCTION_(sqrt);
_NT_MAKE_NONLINEAR_DEFINED_FUNCTION_(invsqrt);
_NT_MAKE_NONLINEAR_DEFINED_FUNCTION_(log);
_NT_MAKE_NONLINEAR_DEFINED_FUNCTION_(silu);
_NT_MAKE_NONLINEAR_DEFINED_FUNCTION_(gelu);

_NT_MAKE_NONLINEAR_DEFINED_FUNCTION_(tan);
_NT_MAKE_NONLINEAR_DEFINED_FUNCTION_(tanh);
_NT_MAKE_NONLINEAR_DEFINED_FUNCTION_(atan);
_NT_MAKE_NONLINEAR_DEFINED_FUNCTION_(atanh);
_NT_MAKE_NONLINEAR_DEFINED_FUNCTION_(cotan);
_NT_MAKE_NONLINEAR_DEFINED_FUNCTION_(cotanh);

_NT_MAKE_NONLINEAR_DEFINED_FUNCTION_(sin);
_NT_MAKE_NONLINEAR_DEFINED_FUNCTION_(sinh);
_NT_MAKE_NONLINEAR_DEFINED_FUNCTION_(asin);
_NT_MAKE_NONLINEAR_DEFINED_FUNCTION_(asinh);
_NT_MAKE_NONLINEAR_DEFINED_FUNCTION_(csc);
_NT_MAKE_NONLINEAR_DEFINED_FUNCTION_(csch);

_NT_MAKE_NONLINEAR_DEFINED_FUNCTION_(cos);
_NT_MAKE_NONLINEAR_DEFINED_FUNCTION_(cosh);
_NT_MAKE_NONLINEAR_DEFINED_FUNCTION_(acos);
_NT_MAKE_NONLINEAR_DEFINED_FUNCTION_(acosh);
_NT_MAKE_NONLINEAR_DEFINED_FUNCTION_(sec);
_NT_MAKE_NONLINEAR_DEFINED_FUNCTION_(sech);

#undef _NT_MAKE_NONLINEAR_DEFINED_FUNCTION_ 

}
}
