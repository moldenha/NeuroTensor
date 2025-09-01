#include "activation_functions.h"
#include "../cpu/activation_functions.h"
#include "sum_exp_log.h"
#include "min_max.h"
#include "trig.h"
#include "../../utils/name_func_macro.h"
#include "exceptions.hpp"
#include "../../utils/macros.h"
#include "../../dtype/ArrayVoid.hpp"

namespace nt{
namespace functional{

//abs can happen on any but booleans
template<size_t N>
inline void check_dtypes(const DType& dt, const char(&s)[N]){
    utils::throw_exception(dt != DType::Bool, "Cannot perform $ on dtype $", dt);
}


#define ADD_UNDERSCORE(name) name##_
#define ADD_DOUBLE_UNDERSCORE(name) _##name##_

#define _NT_MAKE_ACTIVATION_FUNC_(func_name)\
Tensor& ADD_UNDERSCORE(func_name)(Tensor& x){\
    _NT_FUNCTIONAL_ALWAYS_CHECK_(x);\
    check_mutability(x);\
    if(x.dtype() == DType::TensorObj){\
        x.arr_void().execute_function<WRAP_DTYPES<DTypeEnum<DType::TensorObj> > >([](auto begin, auto end){\
            for(;begin != end; ++begin)\
                ADD_UNDERSCORE(func_name)(*begin);\
        });\
        return x;\
    }\
    cpu::ADD_DOUBLE_UNDERSCORE(func_name)(x.arr_void());\
    return x;\
}\
\
\
Tensor func_name(const Tensor& x){\
    _NT_FUNCTIONAL_ALWAYS_CHECK_(x);\
    Tensor a = x.clone();\
    ADD_UNDERSCORE(func_name)(a);\
    return std::move(a);\
}\

_NT_MAKE_ACTIVATION_FUNC_(sigmoid);




Tensor& dsigmoid_(Tensor & x, bool apply_sigmoid){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(x);
    check_mutability(x);
	if(x.dtype() == DType::TensorObj){
        x.arr_void().execute_function<WRAP_DTYPES<DTypeEnum<DType::TensorObj> > >([&apply_sigmoid](auto begin, auto end){
            for(;begin != end; ++begin)
                dsigmoid_(*begin, apply_sigmoid);
        });
        return x;
	}
    cpu::_dsigmoid_(x.arr_void(), apply_sigmoid);
	return x;
	
}

Tensor dsigmoid(const Tensor & x, bool apply_sigmoid){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(x);
    Tensor a = x.clone();
    dsigmoid_(a, apply_sigmoid);
    return std::move(a);	
}

Tensor& pow_(Tensor & x, Scalar p){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(x);
    check_mutability(x);
	if(x.dtype() == DType::TensorObj){
        x.arr_void().execute_function<WRAP_DTYPES<DTypeEnum<DType::TensorObj>>>([&p](auto begin, auto end){
            for(;begin != end; ++begin){
                pow_(*begin, p);
            } 
        });
        return x;
	}
    cpu::_pow_(x.arr_void(), p);
	return x;
	
}



Tensor pow(const Tensor & x, Scalar p){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(x);
	if(x.dtype() == DType::TensorObj){
        Tensor out = Tensor::makeNullTensorArray(x.numel());
        Tensor* o_begin = reinterpret_cast<Tensor*>(out.data_ptr());
        x.arr_void().cexecute_function<WRAP_DTYPES<DTypeEnum<DType::TensorObj>>>([o_begin, &p](auto begin, auto end){
            for(;begin != end; ++begin){
                *o_begin = pow(*begin, p);
            } 
        });
        return std::move(out);
	}
	Tensor a = x.clone();
    cpu::_pow_(a.arr_void(), p);
	return std::move(a);
	
}


_NT_MAKE_ACTIVATION_FUNC_(sqrt)
_NT_MAKE_ACTIVATION_FUNC_(dsqrt)
_NT_MAKE_ACTIVATION_FUNC_(invsqrt)
_NT_MAKE_ACTIVATION_FUNC_(dinvsqrt)
_NT_MAKE_ACTIVATION_FUNC_(abs)

Tensor softplus(const Tensor& x, Scalar beta, Scalar threshold){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(x);
	Tensor softplus_x = x * beta;
	Tensor where = x > threshold;
    Tensor softplus_where = softplus_x[where];
    if(softplus_where.is_null()) //nothing above threshold
        return softplus_x;
	softplus_where.set_(log(1 + exp(softplus_where)).divide_(beta));
	return std::move(softplus_x);
}

Tensor& softplus_(Tensor& x, Scalar beta, Scalar threshold){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(x);
    check_mutability(x);
	Tensor where = x > threshold;
    x *= beta;
    Tensor softplus_where = x[where];
    if(softplus_where.is_null()) //nothing above threshold
        return x;
	softplus_where.set_(log(1 + exp(softplus_where)).divide_(beta));
	return x;
}


Tensor relu(const Tensor& x){return clamp(x, 0);}

Tensor& relu_(Tensor& x){return clamp_(x, 0);}

_NT_MAKE_ACTIVATION_FUNC_(silu)
_NT_MAKE_ACTIVATION_FUNC_(dsilu)

// Tensor silu(const Tensor& x){
// 	return x * sigmoid(x);
// }

// Tensor dsilu(const Tensor& x){
// 	Tensor sigmoid_x = sigmoid(x);
// 	Tensor grad = sigmoid_x * (1 + x * (1 - sigmoid_x));
// 	return std::move(grad);
// }


_NT_MAKE_ACTIVATION_FUNC_(gelu)
_NT_MAKE_ACTIVATION_FUNC_(dgelu)

// Tensor gelu(const Tensor& x){
// 	Scalar sqrt_2_pi = std::sqrt(2.0 / M_PI);
// 	return 0.5 * x * (1.0 + tanh(sqrt_2_pi * (x + 0.044715 * std::pow(x, 3))));
// }

// Tensor dgelu(const Tensor& x) {
//     const Scalar sqrt_2_pi(std::sqrt(2.0 / M_PI));
//     const Scalar c(0.044715);

//     Tensor z = sqrt_2_pi * (x + c * std::pow(x, 3));
//     // Compute tanh(z) and its derivative
//     z = tanh(z);
//     Tensor tanh_derivative = 1 - (z * z);

//     // Gradient of z with respect to x
//     Tensor dz_dx = sqrt_2_pi * (1 + 3 * c.to<double>() * x * x);

//     // Final gradient
//     return 0.5 * (1 + z) + 0.5 * x * tanh_derivative * dz_dx;
// }


#undef _NT_MAKE_ACTIVATION_FUNCTION_ 
#undef ADD_UNDERSCORE
#undef ADD_DOUBLE_UNDERSCORE

}
}
