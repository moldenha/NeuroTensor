#include "DType_list.h"
#include "DType.h"


#include <vector>

namespace nt{



template<typename T>
typename tdtype_list<T>::reference tdtype_list<T>::operator[](uint64_t i){
	if(current_bucket >= maxBS || (i + data_index < sizes[current_bucket]))
		return ptr_[i];
	i -= (sizes[current_bucket] - data_index);
	uint64_t cpyBKT = current_bucket + 1;
	while(i > sizes[cpyBKT] && cpyBKT < maxBS){i -= sizes[cpyBKT];++cpyBKT;}
	return reinterpret_cast<T*>(ptrs_[cpyBKT].get())[i];

}

template<typename T>
tdtype_list<T>& tdtype_list<T>::operator+=(uint64_t i) {
	if(current_bucket >= maxBS || (i + data_index < sizes[current_bucket])){
		ptr_ += i;
		data_index += i;
		return *this;
	}
	i -= (sizes[current_bucket] - data_index);
	++current_bucket;
	while(i > sizes[current_bucket] && current_bucket < maxBS){ i -= sizes[current_bucket];++current_bucket;}
	data_index = i;
	return *this;
}

template<typename T>
tdtype_list<T> tdtype_list<T>::operator+(uint64_t i){
	if(current_bucket >= maxBS || (i + data_index < sizes[current_bucket])){
		return tdtype_list<T>(ptrs_, sizes, current_bucket, data_index + i, maxBS);
	}
	i -= (sizes[current_bucket] - data_index);
	uint64_t cpyBKT = current_bucket + 1;
	while(i > sizes[cpyBKT] && cpyBKT < maxBS){i -= sizes[cpyBKT];++cpyBKT;}
	return tdtype_list<T>(ptrs_, sizes, cpyBKT, i, maxBS);
}

template<typename T>
std::ptrdiff_t tdtype_list<T>::operator-(const tdtype_list<T>& dt) const {
	if(current_bucket < dt.current_bucket){
		return dt - *this;
	}
	if(current_bucket == dt.current_bucket){
		return static_cast<std::ptrdiff_t>(std::abs(static_cast<int64_t>(data_index) - static_cast<int64_t>(dt.data_index)));
	}
	uint64_t current = dt.sizes[dt.current_bucket] - dt.data_index;
	uint64_t cpyBKT = dt.current_bucket + 1;
	while(cpyBKT < current_bucket){current += dt.sizes[cpyBKT];++cpyBKT;}
	return current + data_index;
}

//print_enum_special(enum_values,  half_enum_values,  f_128_enum_values, i_128_enum_values, 'template class tdtype_list<DTypeFuncs::dtype_to_type<', '> >;')

template class tdtype_list<float>;
template class tdtype_list<double>;
template class tdtype_list<complex_64>;
template class tdtype_list<complex_128>;
template class tdtype_list<uint32_t>;
template class tdtype_list<int32_t>;
template class tdtype_list<uint16_t>;
template class tdtype_list<int16_t>;
template class tdtype_list<uint8_t>;
template class tdtype_list<int8_t>;
template class tdtype_list<int64_t>;
template class tdtype_list<Tensor>;
template class tdtype_list<uint_bool_t>;
#ifdef _HALF_FLOAT_SUPPORT_
template class tdtype_list<float16_t>;
template class tdtype_list<complex_32>;
#endif
#ifdef __SIZEOF_INT128__
template class tdtype_list<int128_t>;
template class tdtype_list<uint128_t>;
#endif
#ifdef _128_FLOAT_SUPPORT_
template class tdtype_list<float128_t>;
#endif

template class tdtype_list<const float>;
template class tdtype_list<const double>;
template class tdtype_list<const complex_64>;
template class tdtype_list<const complex_128>;
template class tdtype_list<const uint32_t>;
template class tdtype_list<const int32_t>;
template class tdtype_list<const uint16_t>;
template class tdtype_list<const int16_t>;
template class tdtype_list<const uint8_t>;
template class tdtype_list<const int8_t>;
template class tdtype_list<const int64_t>;
template class tdtype_list<const Tensor>;
template class tdtype_list<const uint_bool_t>;
#ifdef _HALF_FLOAT_SUPPORT_
template class tdtype_list<const float16_t>;
template class tdtype_list<const complex_32>;
#endif
#ifdef __SIZEOF_INT128__
template class tdtype_list<const int128_t>;
template class tdtype_list<const uint128_t>;
#endif
#ifdef _128_FLOAT_SUPPORT_
template class tdtype_list<const float128_t>;
#endif

template class tdtype_list<DTypeFuncs::dtype_to_type<DType::Float> >;
template class tdtype_list<DTypeFuncs::dtype_to_type<DType::Double> >;
template class tdtype_list<DTypeFuncs::dtype_to_type<DType::Complex64> >;
template class tdtype_list<DTypeFuncs::dtype_to_type<DType::Complex128> >;
template class tdtype_list<DTypeFuncs::dtype_to_type<DType::uint8> >;
template class tdtype_list<DTypeFuncs::dtype_to_type<DType::int8> >;
template class tdtype_list<DTypeFuncs::dtype_to_type<DType::int16> >;
template class tdtype_list<DTypeFuncs::dtype_to_type<DType::uint16> >;
template class tdtype_list<DTypeFuncs::dtype_to_type<DType::Integer> >;
template class tdtype_list<DTypeFuncs::dtype_to_type<DType::Long> >;
template class tdtype_list<DTypeFuncs::dtype_to_type<DType::int64> >;
template class tdtype_list<DTypeFuncs::dtype_to_type<DType::Bool> >;
template class tdtype_list<DTypeFuncs::dtype_to_type<DType::TensorObj> >;
#ifdef _HALF_FLOAT_SUPPORT_
template class tdtype_list<DTypeFuncs::dtype_to_type<DType::Float16> >;
template class tdtype_list<DTypeFuncs::dtype_to_type<DType::Complex32> >;
#endif
#ifdef _128_FLOAT_SUPPORT_
template class tdtype_list<DTypeFuncs::dtype_to_type<DType::Float128> >;
#endif
#ifdef __SIZEOF_INT128__
template class tdtype_list<DTypeFuncs::dtype_to_type<DType::int128> >;
template class tdtype_list<DTypeFuncs::dtype_to_type<DType::uint128> >;
#endif

template class tdtype_list<DTypeFuncs::dtype_to_type<DType::Float> const>;
template class tdtype_list<DTypeFuncs::dtype_to_type<DType::Double> const>;
template class tdtype_list<DTypeFuncs::dtype_to_type<DType::Complex64> const>;
template class tdtype_list<DTypeFuncs::dtype_to_type<DType::Complex128> const>;
template class tdtype_list<DTypeFuncs::dtype_to_type<DType::uint8> const>;
template class tdtype_list<DTypeFuncs::dtype_to_type<DType::int8> const>;
template class tdtype_list<DTypeFuncs::dtype_to_type<DType::int16> const>;
template class tdtype_list<DTypeFuncs::dtype_to_type<DType::uint16> const>;
template class tdtype_list<DTypeFuncs::dtype_to_type<DType::Integer> const>;
template class tdtype_list<DTypeFuncs::dtype_to_type<DType::Long> const>;
template class tdtype_list<DTypeFuncs::dtype_to_type<DType::int64> const>;
template class tdtype_list<DTypeFuncs::dtype_to_type<DType::Bool> const>;
template class tdtype_list<DTypeFuncs::dtype_to_type<DType::TensorObj> const>;
#ifdef _HALF_FLOAT_SUPPORT_
template class tdtype_list<DTypeFuncs::dtype_to_type<DType::Float16> const>;
template class tdtype_list<DTypeFuncs::dtype_to_type<DType::Complex32> const>;
#endif
#ifdef _128_FLOAT_SUPPORT_
template class tdtype_list<DTypeFuncs::dtype_to_type<DType::Float128> const>;
#endif
#ifdef __SIZEOF_INT128__
template class tdtype_list<DTypeFuncs::dtype_to_type<DType::int128> const>;
template class tdtype_list<DTypeFuncs::dtype_to_type<DType::uint128> const>;
#endif


}
