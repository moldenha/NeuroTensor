#include "DType_operators.h"

namespace nt{
namespace DTypeFuncs{

template<DType dt>
Tensor& MultiplyThis<dt, std::enable_if_t<dt == DType::TensorObj, bool>>::operator()(Tensor& a, const Scalar& b){return a *= b;}
template<DType dt>
Tensor& MultiplyThis<dt, std::enable_if_t<dt == DType::TensorObj, bool>>::operator()(Tensor& a, const Tensor& b){return a *= b;}

template<DType dt>
Tensor& DivideThis<dt, std::enable_if_t<dt == DType::TensorObj, bool>>::operator()(Tensor& a, const Scalar& b){return a /= b;}
template<DType dt>
Tensor& DivideThis<dt, std::enable_if_t<dt == DType::TensorObj, bool>>::operator()(Tensor& a, const Tensor& b){return a /= b;}

template<DType dt>
Tensor& SubtractThis<dt, std::enable_if_t<dt == DType::TensorObj, bool>>::operator()(Tensor& a, const Scalar& b){return a -= b;}
template<DType dt>
Tensor& SubtractThis<dt, std::enable_if_t<dt == DType::TensorObj, bool>>::operator()(Tensor& a, const Tensor& b){return a -= b;}

template<DType dt>
Tensor& AddThis<dt, std::enable_if_t<dt == DType::TensorObj, bool>>::operator()(Tensor& a, const Scalar& b){return a += b;}
template<DType dt>
Tensor& AddThis<dt, std::enable_if_t<dt == DType::TensorObj, bool>>::operator()(Tensor& a, const Tensor& b){return a += b;}

template<DType dt>
Tensor Multiply<dt, std::enable_if_t<dt == DType::TensorObj, bool>>::operator()(const Tensor& a, const Scalar& b){return a * b;}
template<DType dt>
Tensor Multiply<dt, std::enable_if_t<dt == DType::TensorObj, bool>>::operator()(const Tensor& a, const Tensor& b){return a * b;}

template<DType dt>
Tensor Divide<dt, std::enable_if_t<dt == DType::TensorObj, bool>>::operator()(const Tensor& a, const Scalar& b){return a / b;}
template<DType dt>
Tensor Divide<dt, std::enable_if_t<dt == DType::TensorObj, bool>>::operator()(const Tensor& a, const Tensor& b){return a / b;}

template<DType dt>
Tensor Subtract<dt, std::enable_if_t<dt == DType::TensorObj, bool>>::operator()(const Tensor& a, const Scalar& b){return a - b;}
template<DType dt>
Tensor Subtract<dt, std::enable_if_t<dt == DType::TensorObj, bool>>::operator()(const Tensor& a, const Tensor& b){return a - b;}

template<DType dt>
Tensor Add<dt, std::enable_if_t<dt == DType::TensorObj, bool>>::operator()(const Tensor& a, const Scalar& b){return a + b;}
template<DType dt>
Tensor Add<dt, std::enable_if_t<dt == DType::TensorObj, bool>>::operator()(const Tensor& a, const Tensor& b){return a + b;}


}
}
