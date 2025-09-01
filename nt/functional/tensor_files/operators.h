#ifndef NT_FUNCTIONAL_TENSOR_FILES_OPERATORS_H__
#define NT_FUNCTIONAL_TENSOR_FILES_OPERATORS_H__

#include "../../Tensor.h"
#include "../../utils/always_inline_macro.h"

namespace nt{
namespace functional{
enum class functional_operator_num{
	Multiply = 0,
	Divide = 1,
	Subtract = 2,
	Add = 3
};

NEUROTENSOR_API Tensor functional_operator_out(const Tensor& a, const Tensor& b, const functional_operator_num op);
NEUROTENSOR_API void functional_operator_this(Tensor& a, const Tensor& b, const functional_operator_num op);
NEUROTENSOR_API Tensor hadamard_multiply(const Tensor&, const Tensor&);
NEUROTENSOR_API Tensor& hadamard_multiply_this(Tensor&, const Tensor&);
NT_ALWAYS_INLINE Tensor multiply(const Tensor& a, const Tensor& b){return hadamard_multiply(a, b);}
NT_ALWAYS_INLINE Tensor& multiply_(Tensor& a, const Tensor& b){return hadamard_multiply_this(a, b);}
NEUROTENSOR_API Tensor add(const Tensor&, const Tensor&);
NEUROTENSOR_API Tensor& add_(Tensor&, const Tensor&);
NEUROTENSOR_API Tensor subtract(const Tensor&, const Tensor&);
NEUROTENSOR_API Tensor& subtract_(Tensor&, const Tensor&);
NEUROTENSOR_API Tensor divide(const Tensor&, const Tensor&);
NEUROTENSOR_API Tensor& divide_(Tensor&, const Tensor&);
NEUROTENSOR_API Tensor dot(const Tensor&, const Tensor&,  utils::optional_list dim = nullptr, bool keepdim = false);


NEUROTENSOR_API Tensor multiply(const Tensor&, Scalar);
NEUROTENSOR_API Tensor multiply(Scalar s, const Tensor& t);
NEUROTENSOR_API Tensor& multiply_(Tensor&, Scalar);
NEUROTENSOR_API Tensor add(const Tensor&, Scalar);
NEUROTENSOR_API Tensor add(Scalar s, const Tensor& t);
NEUROTENSOR_API Tensor& add_(Tensor&, Scalar);
NEUROTENSOR_API Tensor subtract(const Tensor&, Scalar);
NEUROTENSOR_API Tensor subtract(Scalar, const Tensor&);
NEUROTENSOR_API Tensor& subtract_(Tensor&, Scalar);
NEUROTENSOR_API Tensor divide(const Tensor&, Scalar);
NEUROTENSOR_API Tensor divide(Scalar, const Tensor&);
NEUROTENSOR_API Tensor& divide_(Tensor&, Scalar);

NEUROTENSOR_API Tensor inverse(const Tensor&);
NEUROTENSOR_API Tensor& inverse_(Tensor&);

NEUROTENSOR_API Tensor fmod(const Tensor&, const Tensor&);
NEUROTENSOR_API Tensor fmod(const Tensor&, Scalar);
NEUROTENSOR_API Tensor fmod(Scalar, const Tensor&);
NEUROTENSOR_API Tensor& fmod_(Tensor&, Scalar);
NEUROTENSOR_API Tensor fmod_b_backward(const Tensor& a, const Tensor& b, const Tensor& grad);
NEUROTENSOR_API Tensor fmod_b_backward(const Scalar& a, const Tensor& b, const Tensor& grad);


NEUROTENSOR_API Tensor remainder(const Tensor&, const Tensor&);
NEUROTENSOR_API Tensor remainder(const Tensor&, Scalar);
NEUROTENSOR_API Tensor remainder(Scalar, const Tensor&);
NEUROTENSOR_API Tensor& remainder_(Tensor&, Scalar);
NEUROTENSOR_API Tensor remainder_b_backward(const Tensor& a, const Tensor& b, const Tensor& grad);
NEUROTENSOR_API Tensor remainder_b_backward(const Scalar& a, const Tensor& b, const Tensor& grad);


}
}

#endif
