#ifndef NT_DTYPE_OPERATORS_H__
#define NT_DTYPE_OPERATORS_H__

#include "../Tensor.h"
#include "compatible/DType_compatible.h"
#include "../utils/utils.h"
#include "../types/Types.h"
#include "Scalar.h"
#include <complex>
#include <type_traits>
#include <stdlib.h>
#include "../convert/Convert.h"
#include "DType.h"


namespace nt{
namespace DTypeFuncs{

template<DType dt, class Enable=bool>
class MultiplyThis;
template<DType dt, class Enable=bool>
class DivideThis;
template<DType dt, class Enable=bool>
class SubtractThis;
template<DType dt, class Enable=bool>
class AddThis;


template<DType dt, class Enable=bool>
class Multiply;
template<DType dt, class Enable=bool>
class Divide;
template<DType dt, class Enable=bool>
class Add;
template<DType dt, class Enable=bool>
class Subtract;


template<DType dt> 
class MultiplyThis<dt, std::enable_if_t<is_dtype_num_v<dt>, bool>>{
	using type = dtype_to_type_t<dt>;
	type& operator()(type& a, const type& b){return a *= b;}
	template<typename A, std::enable_if_t<!std::is_same_v<A, type>, bool> = true>
	type& operator()(type& a, const A& b){return a *= convert::convert<dt>(b);}
};

template<DType dt> 
class DivideThis<dt, std::enable_if_t<is_dtype_num_v<dt>, bool>>{
	using type = dtype_to_type_t<dt>;
	type& operator()(type& a, const type& b){return a /= b;}
	template<typename A, std::enable_if_t<!std::is_same_v<A, type>, bool> = true>
	type& operator()(type& a, const A& b){return a /= convert::convert<dt>(b);}
};

template<DType dt> 
class AddThis<dt, std::enable_if_t<is_dtype_num_v<dt>, bool>>{
	using type = dtype_to_type_t<dt>;
	type& operator()(type& a, const type& b){return a += b;}
	template<typename A, std::enable_if_t<!std::is_same_v<A, type>, bool> = true>
	type& operator()(type& a, const A& b){return a += convert::convert<dt>(b);}};

template<DType dt> 
class SubtractThis<dt, std::enable_if_t<is_dtype_num_v<dt>, bool>>{
	using type = dtype_to_type_t<dt>;
	type& operator()(type& a, const type& b){return a -= b;}
	template<typename A, std::enable_if_t<!std::is_same_v<A, type>, bool> = true>
	type& operator()(type& a, const A& b){return a -= convert::convert<dt>(b);}};

template<DType dt>
class Multiply<dt, std::enable_if_t<is_dtype_num_v<dt>, bool>>{
	using type = dtype_to_type_t<dt>;
	type operator()(const type& a, const type& b){return a * b;}
	template<typename A>
	type operator()(const type& a, const A& b){return a * convert::convert<dt>(b);}
};

template<DType dt> 
class Divide<dt, std::enable_if_t<is_dtype_num_v<dt>, bool>>{
	using type = dtype_to_type_t<dt>;
	type operator()(const type& a, const type& b){return a / b;}
	template<typename A>
	type operator()(const type& a, const A& b){return a / convert::convert<dt>(b);}
};

template<DType dt> 
class Add<dt, std::enable_if_t<is_dtype_num_v<dt>, bool>>{
	using type = dtype_to_type_t<dt>;
	type operator()(const type& a, const type& b){return a + b;}
	template<typename A>
	type operator()(const type& a, const A& b){return a + convert::convert<dt>(b);}
};
template<DType dt> 
class Subtract<dt, std::enable_if_t<is_dtype_num_v<dt>, bool>>{
	using type = dtype_to_type_t<dt>;
	type operator()(const type& a, const type& b){return a - b;}
	template<typename A>
	type operator()(const type& a, const A& b){return a - convert::convert<dt>(b);}

};


template<DType dt> 
class MultiplyThis<dt, std::enable_if_t<dt == DType::TensorObj, bool>>{
	Tensor& operator()(Tensor& a, const Scalar& b);
	Tensor& operator()(Tensor& a, const Tensor& b);
};

template<DType dt> 
class DivideThis<dt, std::enable_if_t<dt == DType::TensorObj, bool>>{
	Tensor& operator()(Tensor& a, const Scalar& b);
	Tensor& operator()(Tensor& a, const Tensor& b);
};

template<DType dt> 
class SubtractThis<dt, std::enable_if_t<dt == DType::TensorObj, bool>>{
	Tensor& operator()(Tensor& a, const Scalar& b);
	Tensor& operator()(Tensor& a, const Tensor& b);
};

template<DType dt> 
class AddThis<dt, std::enable_if_t<dt == DType::TensorObj, bool>>{
	Tensor& operator()(Tensor& a, const Scalar& b);
	Tensor& operator()(Tensor& a, const Tensor& b);
};

template<DType dt> 
class Multiply<dt, std::enable_if_t<dt == DType::TensorObj, bool>>{
	Tensor operator()(const Tensor& a, const Scalar& b);
	Tensor operator()(const Tensor& a, const Tensor& b);
};

template<DType dt> 
class Divide<dt, std::enable_if_t<dt == DType::TensorObj, bool>>{
	Tensor operator()(const Tensor& a, const Scalar& b);
	Tensor operator()(const Tensor& a, const Tensor& b);
};

template<DType dt> 
class Subtract<dt, std::enable_if_t<dt == DType::TensorObj, bool>>{
	Tensor operator()(const Tensor& a, const Scalar& b);
	Tensor operator()(const Tensor& a, const Tensor& b);
};

template<DType dt> 
class Add<dt, std::enable_if_t<dt == DType::TensorObj, bool>>{
	Tensor operator()(const Tensor& a, const Scalar& b);
	Tensor operator()(const Tensor& a, const Tensor& b);
};


}
}

#endif //NT_DTYPE_OPERATORS_H__
