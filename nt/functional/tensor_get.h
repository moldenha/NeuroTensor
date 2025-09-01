#ifndef NT_TENSOR_GET_H__
#define NT_TENSOR_GET_H__
//this is a specialization to split a tensor into a tuple of tensors

#include "../Tensor.h"
#include "../utils/utils.h"
#include <tuple>
#include <utility> //std::index_sequence

//just in normal nt namespace
namespace nt{


template<size_t... Indices>
inline utils::repeat_types_t<Tensor, sizeof...(Indices)> get_index_tensor_cast(Tensor& a,
		std::index_sequence<Indices...>){
	return std::make_tuple((a[Indices].item<Tensor>()) ...);
}

template<size_t... Indices>
inline utils::repeat_types_t<Tensor, sizeof...(Indices)> get_index_tensor_ncast(Tensor& a,
		std::index_sequence<Indices...>){
	return std::make_tuple((a[Indices]) ...);
}

template<size_t N>
inline utils::repeat_types_t<Tensor, N> get(Tensor a){
	if(a.dtype() == DType::TensorObj){
		utils::throw_exception(a.numel() == N,
				"Trying to get $ tensors with get function, but holds $ tensors",
				N, a.numel());
		return get_index_tensor_cast(a, std::make_index_sequence<N>{}); 
	}
	utils::throw_exception(a.shape()[0] == N,
				"Trying to get $ tensors with get function, but outter dim is $",
				N, a.shape()[0]);
	return get_index_tensor_ncast(a, std::make_index_sequence<N>{}); 
}

} // nt::



#endif //_NT_TENSOR_GET_H_
