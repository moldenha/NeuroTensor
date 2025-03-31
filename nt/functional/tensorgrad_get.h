#ifndef _NT_TENSORGRAD_GET_H_
#define _NT_TENSORGRAD_GET_H_
//this is a specialization to split a tensor into a tuple of tensors

#include "../nn/TensorGrad.h"
#include "../utils/utils.h"
#include <tuple>
#include <utility> //std::index_sequence

//just in normal nt namespace
namespace nt{


//in tensorgrad both work the same way in this case
template<size_t... Indices>
inline utils::repeat_types_t<TensorGrad, sizeof...(Indices)> get_index_tensorgrad(TensorGrad& a,
		std::index_sequence<Indices...>){
	return std::make_tuple((a[Indices]) ...);
}

template<size_t N>
inline utils::repeat_types_t<TensorGrad, N> get(TensorGrad a){
	if(a.tensor.dtype == DType::TensorObj){
		utils::throw_exception(a.numel() == N,
				"Trying to get $ tensors with get function, but holds $ tensors",
				N, a.numel());
		return get_index_tensorgrad(a, std::make_index_sequence<N>{}); 
	}
	utils::throw_exception(a.shape()[0] == N,
				"Trying to get $ tensors with get function, but outter dim is $",
				N, a.shape()[0]);
	return get_index_tensorgrad(a, std::make_index_sequence<N>{}); 
}

} // nt::



#endif //_NT_TENSORGRAD_GET_H_
