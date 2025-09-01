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

#define ADD_UNDERSCORE(name) name##_
#define ADD_DOUBLE_UNDERSCORE(name) _##name##_

#define NT_MAKE_TRIG_FUNCTION_(func_name)\
TensorGrad TensorGrad_Functional_Class::func_name(const TensorGrad &x) { \
    TensorGrad result(::nt::functional::func_name(x.detach()), x.track_grad()); \
    if (!x.track_grad()) { \
        result.track_grad_(false); \
        return std::move(result); \
    } \
    result.track_tensors(x); \
    result.create_backward_function( \
            [](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents, \
                                                    intrusive_ptr<tensor_holder> saved_x) { \
                parents[0]->accumulate_gradient(grad * _NT_GLUE_(::nt::functional::d, func_name)(saved_x->tensor)); \
            }, \
            make_intrusive<tensor_holder>(x.detach().conditional_mutate_clone())); \
    return std::move(result); \
} \
\
TensorGrad& TensorGrad_Functional_Class::ADD_UNDERSCORE(func_name)(TensorGrad &x){\
    if(!x.track_grad()){\
        ::nt::functional::ADD_UNDERSCORE(func_name)(x.detach());\
        return x;\
    }\
    intrusive_ptr<tensor_holder> this_clone = \
        make_intrusive<tensor_holder>(x.detach().conditional_mutate_clone()); \
    ::nt::functional::ADD_UNDERSCORE(func_name)(x.detach());\
    x.track_self_mod_tensors( \
    [saved_x = std::move(this_clone)](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>> &parents) { \
        parents[0]->accumulate_gradient(grad * _NT_GLUE_(::nt::functional::d, func_name)(saved_x->tensor)); \
    }, #func_name "_"); \
    return x;\
}\


    

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

#undef NT_MAKE_TRIG_FUNCTION_ 
#undef ADD_UNDERSCORE 
#undef ADD_DOUBLE_UNDERSCORE 

}
}
