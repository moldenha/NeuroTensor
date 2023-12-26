#include "DType_list.h"
#include "DType.h"
#include "DType_enum.h"
#include "Scalar.h"



#include <vector>

namespace nt{

template<typename T>
tdtype_list<T>::tdtype_list(void** _ptr)
	:m_ptr(_ptr)
	{}

template<typename T>
inline tdtype_list<T>& tdtype_list<T>::operator++(){
	++m_ptr;
	return *this;
}

template<typename T>
inline tdtype_list<T> tdtype_list<T>::operator++(int){
	tdtype_list<T> tmp = *this;
	++(*this);
	return tmp;
}

template<typename T>
inline const bool tdtype_list<T>::operator==(const tdtype_list<T>& b) const {return m_ptr == b.m_ptr;}
template<typename T>
inline const bool tdtype_list<T>::operator!=(const tdtype_list<T>& b) const {return m_ptr != b.m_ptr;}

template<typename T>
inline T& tdtype_list<T>::operator*() {
	return *static_cast<pointer>(*m_ptr);
}

template<typename T>
inline T& tdtype_list<T>::operator[](const uint32_t index){
	return *static_cast<pointer>(*(m_ptr + index));
}

template<typename T>
inline tdtype_list<T>& tdtype_list<T>::operator+=(const uint32_t index){m_ptr += index; return *this;}

template<typename T>
inline tdtype_list<T> tdtype_list<T>::operator+(const uint32_t index) const{
	tdtype_list<T> tmp = *this;
	tmp += index;
	return tmp;
}


const size_t dtype_list::get_val_add(DType _type){
	return DTypeFuncs::size_of_dtype(_type);
}

dtype_list::dtype_list(void** _p, DType _t)
	:m_ptr(_p),
	dtype(_t),
	add_val(get_val_add(_t))
{}

dtype_list& dtype_list::operator++(){
	++m_ptr;
	return *this;
}

dtype_list dtype_list::operator++(int){
	dtype_list tmp = *this;
	++(*this);
	return tmp;
}

dtype_list& dtype_list::operator+=(uint32_t index){
	m_ptr += index;
	return *this;
}

bool dtype_list::operator==(const dtype_list& b) const{
	return m_ptr == b.m_ptr;
}

bool dtype_list::operator!=(const dtype_list& b) const{
	return m_ptr != b.m_ptr;
}

/* void dtype_list::set_functions(){ */
/* 	switch(dtype){ */
/* 		case DType::Integer: */
/* 			ref_func = [](void** a) -> ScalarRef {return ScalarRef(*reinterpret_cast<int32_t*>(*a));}; */
/* 			break; */
/* 		case DType::Float: */
/* 			ref_func = [](void** a) -> ScalarRef {return ScalarRef(*reinterpret_cast<float*>(*a));}; */
/* 		case DType::Double: */
/* 			ref_func = [](void** a) -> ScalarRef {return ScalarRef(*reinterpret_cast<double*>(*a));}; */
/* 		case DType::Long: */
/* 			ref_func = [](void** a) -> ScalarRef {return ScalarRef(*reinterpret_cast<uint32_t*>(*a));}; */
/* 		case DType::Complex64: */
/* 			ref_func = [](void** a) -> ScalarRef {return ScalarRef(*reinterpret_cast<complex_64*>(*a));}; */
/* 		case DType::Complex128: */
/* 			ref_func = [](void** a) -> ScalarRef {return ScalarRef(*reinterpret_cast<complex_128*>(*a));}; */
/* 		case DType::uint8: */
/* 			ref_func = [](void** a) -> ScalarRef {return ScalarRef(*reinterpret_cast<uint8_t*>(*a));}; */
/* 		case DType::int8: */
/* 			ref_func = [](void** a) -> ScalarRef {return ScalarRef(*reinterpret_cast<int8_t*>(*a));}; */
/* 		case DType::int16: */
/* 			ref_func = [](void** a) -> ScalarRef {return ScalarRef(*reinterpret_cast<int16_t*>(*a));}; */
/* 		case DType::uint16: */
/* 			ref_func = [](void** a) -> ScalarRef {return ScalarRef(*reinterpret_cast<uint16_t*>(*a));}; */
/* 		case DType::LongLong: */
/* 			ref_func = [](void** a) -> ScalarRef {return ScalarRef(*reinterpret_cast<int64_t*>(*a));}; */
/* 		case DType::Bool: */
/* 			ref_func = [](void** a) -> ScalarRef {return ScalarRef(*reinterpret_cast<uint_bool_t*>(*a));}; */
/* 		case DType::TensorObj: */
/* 			ref_func = [](void** a) -> ScalarRef {return ScalarRef(*reinterpret_cast<Tensor*>(*a));}; */
/* #ifdef _HALF_FLOAT_SUPPORT_ */
/* 		case DType::Float16: */
/* 			ref_func = [](void** a) -> ScalarRef {return ScalarRef(*reinterpret_cast<float16_t*>(*a));}; */
/* 		case DType::Complex32: */
/* 			ref_func = [](void** a) -> ScalarRef {return ScalarRef(*reinterpret_cast<complex_32*>(*a));}; */
/* #endif */
/* #ifdef __SIZEOF_INT128__ */
/* 		case DType::uint128: */
/* 			ref_func = [](void** a) -> ScalarRef {return ScalarRef(*reinterpret_cast<uint128_t*>(*a));}; */
/* 		case DType::int128: */
/* 			ref_func = [](void** a) -> ScalarRef {return ScalarRef(*reinterpret_cast<int128_t*>(*a));}; */
/* #endif */
/* #ifdef _128_FLOAT_SUPPORT_ */
/* 		case DType::Float128: */
/* 			ref_func = [](void** a) -> ScalarRef {return ScalarRef(*reinterpret_cast<float128_t*>(*a));}; */
/* #endif */
/* 	} */
	
/* } */

ScalarRef dtype_list::operator*(){
	switch(dtype){
		case DType::Integer:
			return ScalarRef(*reinterpret_cast<int32_t*>(*m_ptr));
		case DType::Float:
			return ScalarRef(*reinterpret_cast<float*>(*m_ptr));
		case DType::Double:
			return ScalarRef(*reinterpret_cast<double*>(*m_ptr));
		case DType::Long:
			return ScalarRef(*reinterpret_cast<uint32_t*>(*m_ptr));
		case DType::Complex64:
			return ScalarRef(*reinterpret_cast<complex_64*>(*m_ptr));
		case DType::Complex128:
			return ScalarRef(*reinterpret_cast<complex_128*>(*m_ptr));
		case DType::uint8:
			return ScalarRef(*reinterpret_cast<uint8_t*>(*m_ptr));
		case DType::int8:
			return ScalarRef(*reinterpret_cast<int8_t*>(*m_ptr));
		case DType::int16:
			return ScalarRef(*reinterpret_cast<int16_t*>(*m_ptr));
		case DType::uint16:
			return ScalarRef(*reinterpret_cast<uint16_t*>(*m_ptr));
		case DType::LongLong:
			return ScalarRef(*reinterpret_cast<int64_t*>(*m_ptr));
		case DType::Bool:
			return ScalarRef(*reinterpret_cast<uint_bool_t*>(*m_ptr));
		case DType::TensorObj:
			return ScalarRef(*reinterpret_cast<Tensor*>(*m_ptr));
#ifdef _HALF_FLOAT_SUPPORT_
		case DType::Float16:
			return ScalarRef(*reinterpret_cast<float16_t*>(*m_ptr));
		case DType::Complex32:
			return ScalarRef(*reinterpret_cast<complex_32*>(*m_ptr));
#endif
#ifdef __SIZEOF_INT128__
		case DType::uint128:
			return ScalarRef(*reinterpret_cast<uint128_t*>(*m_ptr));
		case DType::int128:
			return ScalarRef(*reinterpret_cast<int128_t*>(*m_ptr));
#endif
#ifdef _128_FLOAT_SUPPORT_
		case DType::Float128:
			return ScalarRef(*reinterpret_cast<float128_t*>(*m_ptr));
#endif
	}
}

void dtype_list::set(const Scalar s){
	switch(dtype){
		case DType::Float:{
			using value_type = float;
			*reinterpret_cast<value_type*>(*m_ptr) = s.to<value_type>();
			return;
		}
		case DType::Double:{
			using value_type = double;
			*reinterpret_cast<value_type*>(*m_ptr) = s.to<value_type>();
			return;
		}
#ifdef _HALF_FLOAT_SUPPORT_
		case DType::Float16:{
			using value_type = float16_t;
			*reinterpret_cast<value_type*>(*m_ptr) = s.to<value_type>();
			return;
		}
		case DType::Complex32:{
			using value_type = complex_32;
			*reinterpret_cast<value_type*>(*m_ptr) = s.to<value_type>();
			return;
		}
#endif
#ifdef _128_FLOAT_SUPPORT_
		case DType::Float128:{
			using value_type = float128_t;
			*reinterpret_cast<value_type*>(*m_ptr) = s.to<value_type>();
			return;
		}
#endif
#ifdef __SIZEOF_INT128__
		case DType::int128:{
			using value_type = int128_t;
			*reinterpret_cast<value_type*>(*m_ptr) = s.to<value_type>();
			return;
		}
		case DType::uint128:{
			using value_type = uint128_t;
			*reinterpret_cast<value_type*>(*m_ptr) = s.to<value_type>();
			return;
		}
#endif
		case DType::Complex64:{
			using value_type = complex_64;
			*reinterpret_cast<value_type*>(*m_ptr) = s.to<value_type>();
			return;
		}
		case DType::Complex128:{
			using value_type = complex_128;
			*reinterpret_cast<value_type*>(*m_ptr) = s.to<value_type>();
			return;
		}
		case DType::int8:{
			using value_type = int8_t;
			*reinterpret_cast<value_type*>(*m_ptr) = s.to<value_type>();
			return;
		}
		case DType::uint8:{
			using value_type = uint8_t;
			*reinterpret_cast<value_type*>(*m_ptr) = s.to<value_type>();
			return;
		}
		case DType::int16:{
			using value_type = int16_t;
			*reinterpret_cast<value_type*>(*m_ptr) = s.to<value_type>();
			return;
		}
		case DType::uint16:{
			using value_type = uint16_t;
			*reinterpret_cast<value_type*>(*m_ptr) = s.to<value_type>();
			return;
		}
		case DType::int32:{
			using value_type = int32_t;
			*reinterpret_cast<value_type*>(*m_ptr) = s.to<value_type>();
			return;
		}
		case DType::uint32:{
			using value_type = uint32_t;
			*reinterpret_cast<value_type*>(*m_ptr) = s.to<value_type>();
			return;
		}
		case DType::int64:{
			using value_type = int64_t;
			*reinterpret_cast<value_type*>(*m_ptr) = s.to<value_type>();
			return;
		}
		case DType::Bool:{
			using value_type = uint_bool_t;
			*reinterpret_cast<value_type*>(*m_ptr) = s.to<value_type>();
			return;
		}
		case DType::TensorObj:{
			using value_type = Tensor;
			*reinterpret_cast<value_type*>(*m_ptr) = s;
			return;
		}
	}


}

const size_t const_dtype_list::get_val_add(DType _type){
	return DTypeFuncs::size_of_dtype(_type);
}

const_dtype_list::const_dtype_list(void** _p, DType _t)
	:m_ptr(_p),
	dtype(_t),
	add_val(get_val_add(_t))
{}

const_dtype_list& const_dtype_list::operator++(){
	++m_ptr;
	return *this;
}

const_dtype_list const_dtype_list::operator++(int){
	const_dtype_list tmp = *this;
	++(*this);
	return tmp;
}

const_dtype_list& const_dtype_list::operator+=(uint32_t index){
	m_ptr += index;
	return *this;
}

bool const_dtype_list::operator==(const const_dtype_list& b) const{
	return m_ptr == b.m_ptr;
}

bool const_dtype_list::operator!=(const const_dtype_list& b) const{
	return m_ptr != b.m_ptr;
}

ConstScalarRef const_dtype_list::operator*(){
	switch(dtype){
		case DType::Integer:
			return ConstScalarRef(*reinterpret_cast<const int32_t*>(*m_ptr));
		case DType::Float:
			return ConstScalarRef(*reinterpret_cast<const float*>(*m_ptr));
		case DType::Double:
			return ConstScalarRef(*reinterpret_cast<const double*>(*m_ptr));
		case DType::Long:
			return ConstScalarRef(*reinterpret_cast<const uint32_t*>(*m_ptr));
		case DType::Complex64:
			return ConstScalarRef(*reinterpret_cast<const complex_64*>(*m_ptr));
		case DType::Complex128:
			return ConstScalarRef(*reinterpret_cast<const complex_128*>(*m_ptr));
		case DType::uint8:
			return ConstScalarRef(*reinterpret_cast<const uint8_t*>(*m_ptr));
		case DType::int8:
			return ConstScalarRef(*reinterpret_cast<const int8_t*>(*m_ptr));
		case DType::int16:
			return ConstScalarRef(*reinterpret_cast<const int16_t*>(*m_ptr));
		case DType::uint16:
			return ConstScalarRef(*reinterpret_cast<const uint16_t*>(*m_ptr));
		case DType::LongLong:
			return ConstScalarRef(*reinterpret_cast<const int64_t*>(*m_ptr));
		case DType::Bool:
			return ConstScalarRef(*reinterpret_cast<const uint_bool_t*>(*m_ptr));
		case DType::TensorObj:
			return ConstScalarRef(*reinterpret_cast<const Tensor*>(*m_ptr));
#ifdef _HALF_FLOAT_SUPPORT_
		case DType::Float16:
			return ConstScalarRef(*reinterpret_cast<const float16_t*>(*m_ptr));
		case DType::Complex32:
			return ConstScalarRef(*reinterpret_cast<const complex_32*>(*m_ptr));
#endif
#ifdef __SIZEOF_INT128__
		case DType::uint128:
			return ConstScalarRef(*reinterpret_cast<const uint128_t*>(*m_ptr));
		case DType::int128:
			return ConstScalarRef(*reinterpret_cast<const int128_t*>(*m_ptr));
#endif
#ifdef _128_FLOAT_SUPPORT_
		case DType::Float128:
			return ConstScalarRef(*reinterpret_cast<const float128_t*>(*m_ptr));
#endif
	}
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
