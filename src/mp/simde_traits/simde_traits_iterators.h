//these are operations to just load and store elements from the iterators

#ifndef _NT_SIMDE_ITERATOR_H_
#define _NT_SIMDE_ITERATOR_H_
#include "../../memory/iterator.h"
#include "../simde_traits.h"
#include <type_traits>
#include <array>
#include <iostream>


namespace nt{
namespace mp{


//this is just a standard loadu
//for a contiguous iterator
template <typename T, 
          typename std::enable_if_t<utils::iterator_is_contiguous_v<T>, int> = 0>
inline constexpr simde_type<utils::IteratorBaseType_t<T>> it_loadu(const T iterator) noexcept {
	using base_type = utils::IteratorBaseType_t<T>; 
	static_assert(simde_supported_v<base_type>, "Expected to have a supported type");
	if constexpr (std::is_integral<base_type>::value || std::is_unsigned<base_type>::value){
		return SimdTraits<base_type>::loadu(reinterpret_cast<const simde_type<base_type>*>(iterator));
	}else{
		return SimdTraits<base_type>::loadu(iterator);
	}
}


template <typename T, std::size_t... Indexes>
inline constexpr simde_type<utils::IteratorBaseType_t<T>> load_indices_into_simd(T& vec, std::index_sequence<Indexes...>) noexcept {
	using base_type = utils::IteratorBaseType_t<T>; 
	return SimdTraits<base_type>::set((vec[Indexes])...);
}



//for a list it is going to have to be indiced at every step
template <typename T, 
          typename std::enable_if_t<utils::iterator_is_list_v<T>, int> = 0>
inline constexpr simde_type<utils::IteratorBaseType_t<T>> it_loadu(T& iterator) {
	using base_type = utils::IteratorBaseType_t<T>; 
	static_assert(simde_supported_v<base_type>, "Expected to have a supported type");
	return load_indices_into_simd(iterator, std::make_index_sequence<pack_size_v<base_type> >{});
}



template <typename T, 
          typename std::enable_if_t<utils::iterator_is_blocked_v<T>, int> = 0>
inline constexpr simde_type<utils::IteratorBaseType_t<T>> it_loadu(T& iterator) {
	using base_type = utils::IteratorBaseType_t<T>; 
	constexpr size_t pack_size = pack_size_v<base_type>;
	static_assert(simde_supported_v<base_type>, "Expected to have a supported type");
	//if the block size is big enough, load it into a vector
	//otherwise use the set function
	if constexpr (std::is_integral<base_type>::value || std::is_unsigned<base_type>::value){
		return iterator.template block_size_left<pack_size>() ? SimdTraits<base_type>::loadu(reinterpret_cast<const simde_type<base_type>*>((const base_type*)(iterator))) : 
			load_indices_into_simd(iterator, std::make_index_sequence<pack_size>{});
	}else{
		return iterator.template block_size_left<pack_size>() ? SimdTraits<base_type>::loadu((const base_type*)(iterator)) : 
			load_indices_into_simd(iterator, std::make_index_sequence<pack_size>{});
	
	}
}


//for the contiguous iterator it is a standard store
template <typename T, 
          typename std::enable_if_t<utils::iterator_is_contiguous_v<T>, int> = 0>
inline constexpr void it_storeu(T iterator, const simde_type<utils::IteratorBaseType_t<T>>& vector) noexcept {
	using base_type = utils::IteratorBaseType_t<T>; 
	static_assert(simde_supported_v<base_type>, "Expected to have a supported type");
	if constexpr (std::is_integral<base_type>::value || std::is_unsigned<base_type>::value){
		SimdTraits<base_type>::storeu(reinterpret_cast<simde_type<base_type>*>(iterator), vector);
	}else{
		SimdTraits<base_type>::storeu(iterator, vector);
	}
}

template <typename T, typename base_type, std::size_t... Indexes>
inline constexpr void store_indices_from_simd(base_type* arr, T& vec, std::index_sequence<Indexes...>) noexcept {
	((vec[Indexes] = arr[Indexes]), ...);
}

template <typename T, 
          typename std::enable_if_t<utils::iterator_is_list_v<T>, int> = 0>
inline constexpr void it_storeu(T& iterator, const simde_type<utils::IteratorBaseType_t<T>>& vector) noexcept {
	using base_type = utils::IteratorBaseType_t<T>; 
	static_assert(simde_supported_v<base_type>, "Expected to have a supported type");
	constexpr size_t pack_size = pack_size_v<base_type>;
	base_type arr[pack_size];
	if constexpr (std::is_integral<base_type>::value || std::is_unsigned<base_type>::value){
		SimdTraits<base_type>::storeu(reinterpret_cast<simde_type<base_type>*>(arr), vector);
	}else{
		SimdTraits<base_type>::storeu(arr, vector);
	}
	store_indices_from_simd(arr, iterator, std::make_index_sequence<pack_size>{});
}


template <typename T, 
          typename std::enable_if_t<utils::iterator_is_blocked_v<T>, int> = 0>
inline constexpr void it_storeu(T& iterator, const simde_type<utils::IteratorBaseType_t<T>>& vector) noexcept {
	using base_type = utils::IteratorBaseType_t<T>; 
	static_assert(simde_supported_v<base_type>, "Expected to have a supported type");
	constexpr size_t pack_size = pack_size_v<base_type>;
	if(iterator.template block_size_left<pack_size>()){
		if constexpr (std::is_integral<base_type>::value || std::is_unsigned<base_type>::value){
			SimdTraits<base_type>::storeu(reinterpret_cast<simde_type<base_type>*>((base_type*)(iterator)), vector);
		}else{
			SimdTraits<base_type>::storeu((base_type*)(iterator), vector);
		}
	}else{
		base_type arr[pack_size];
		if constexpr (std::is_integral<base_type>::value || std::is_unsigned<base_type>::value){
			SimdTraits<base_type>::storeu(reinterpret_cast<simde_type<base_type>*>(arr), vector);
		}else{
			SimdTraits<base_type>::storeu(arr, vector);
		}
		store_indices_from_simd(arr, iterator, std::make_index_sequence<pack_size>{});
	}
}


}} //nt::mp::


#endif //_NT_SIMDE_ITERATOR_H_ 
