#ifndef DTYPE_H
#define DTYPE_H
#include "DType_enum.h"
namespace nt{

template<DType... Rest>
struct VariadicArgCount{
	 static constexpr int value = 0;
};

template<DType dt, DType... Rest>
struct VariadicArgCount<dt, Rest...>{
	static constexpr int value = 1 + VariadicArgCount<Rest...>::value;
};

template<DType... Rest>
constexpr DType FirstVariadicDType = DType::Bool;


template<DType dt, DType... Rest>
constexpr DType FirstVariadicDType<dt, Rest...> = dt;

template <DType dt, DType... Rest>
struct DTypeEnum {
    using next_wrapper = DTypeEnum<Rest...>;
    static constexpr DType next = dt;
    static constexpr bool done = false;
};

template<DType T>
struct DTypeEnum<T>{
    using next_wrapper = DTypeEnum<T>;
    static constexpr DType next = T;
    static constexpr bool done = true;
};


//the input template is DTypeEnum or DTypeEnum
template <typename T, typename... Rest>
struct WRAP_DTYPES{
    static constexpr DType next = T::next;
    using next_wrapper = std::conditional_t<T::done, WRAP_DTYPES<Rest...>, WRAP_DTYPES<typename T::next_wrapper, Rest...> >;
    static constexpr bool done = false;
};



template<typename T>
struct WRAP_DTYPES<T>{
    static constexpr DType next  = T::next;
    using next_wrapper = std::conditional_t<T::done, WRAP_DTYPES<T>, WRAP_DTYPES<typename T::next_wrapper> >;
    static constexpr bool done = T::done;
};


namespace DTypeFuncs{
template<typename Test, template<typename...> class Ref>
struct is_specialization : std::false_type {};
template<template<typename...> class Ref, typename... Args>
struct is_specialization<Ref<Args...>, Ref>: std::true_type {};

template<class T>
struct is_wrapped_dtype{
	static constexpr bool value = is_specialization<T, WRAP_DTYPES>::value;
};

//universal declarations to any dtype
template<DType dt>
bool is_in(const DType inp);

template<DType dt, DType M, DType... Rest>
bool is_in(const DType inp){ return (inp == dt) ? true : is_in<M, Rest...>(inp); }

template<typename T, std::enable_if_t<DTypeFuncs::is_wrapped_dtype<T>::value, bool> = true>
bool is_in(const DType& inp){
	if(inp != T::next){
		if(T::done)
			return false;
		return is_in<typename T::next_wrapper>(inp);
	}
	return true;
}

template<DType dt, DType M, DType... Rest>
struct is_in_t{
	static constexpr bool value =  (dt == M) ? true : is_in_t<dt, Rest...>::value;
};

template<DType dt, DType M>
struct is_in_t<dt, M>{
	static constexpr bool value = (dt == M);
};

template<DType dt, DType M, DType... Rest>
inline constexpr bool is_in_v = is_in_t<dt, M, Rest...>::value;

template<DType dt, DType M>
inline constexpr bool is_in_v<dt, M> = is_in_t<dt, M>::value;

template<DType dt>
inline constexpr bool is_in_v<dt, DType::Bool> = false;

}
}
#include "../Tensor.h"
#include "compatible/DType_compatible.h"
#include "../utils/utils.h"
#include "../types/Types.h"
#include <complex>
#include <type_traits>
#include <stdlib.h>
#include <memory>

namespace nt{
std::ostream& operator<< (std::ostream &out, DType const& data);
std::ostream& operator<<(std::ostream &out, const uint_bool_t &data);

namespace DTypeFuncs{



template<DType dt>
std::ostream& print_dtypes(std::ostream& os){return os << dt << "}";}
template<DType dt, DType M, DType... Rest>
std::ostream& print_dtypes(std::ostream& os){
	os << dt << ",";
	return print_dtypes<M, Rest...>(os);
}

template<DType... Rest>
bool check_dtypes(const char* str, const DType dtype){
	bool outp = is_in<Rest...>(dtype);
	if(!outp){
		std::cout<< str<<"() was expected to support {";
		std::cout << print_dtypes<Rest...> <<" but instead got "<<dtype<<std::endl;
	}
	return outp;
}

template <typename...> struct all_dtype;

template <> struct all_dtype<> : std::true_type { };

template <typename T, typename ...Rest> struct all_dtype<T, Rest...> : std::integral_constant<bool, std::is_same_v<T, DType> && all_dtype<Rest...>::value>
{ };

template<class... DTs>
inline constexpr bool all_dtype_v = all_dtype<DTs...>::value;

template<class T>
void is_same(DType a, bool& outp, T b);

template<DType dt = DType::Integer>
std::size_t size_of_dtype_p(const DType& d);

template <class... DTs>
bool is_in(DType dt, DTs... dts){
	if constexpr(!all_dtype_v<DTs...>){
		throw std::runtime_error("expected only DType types");
	}
	bool outp = false;
	(is_same(dt, outp, dts), ...);
	return outp;
}


void convert_this_dtype_array(void* arr, const DType& from, const DType& to, const std::size_t& total);

template<DType F, DType T>
bool convert_this_typed_array(void* arr, void* arr2, const DType& from, const DType& to, const std::size_t& total);
void convert_to_dtype_array(void* arr, void* arr2, const DType& from, const DType& to, const std::size_t& total);

std::shared_ptr<void> make_shared_array(size_t size, const DType& dt);
template<DType dt = DType::Integer>
std::shared_ptr<void> share_part_ptr(const uint32_t& index, const DType& m_dt, const std::shared_ptr<void>& ptr);

std::size_t size_of_dtype(const DType&);
bool can_convert(const DType&, const DType&);

template<DType dt = DType::Integer>
void initialize_strides(void** ptrs, void* cast, const std::size_t& s, const DType& ds);
bool is_unsigned(const DType& dt);
bool is_integer(const DType& dt);
bool is_floating(const DType& dt);
bool is_complex(const DType& dt);
DType complex_size(const std::size_t& s);
DType floating_size(const std::size_t& s);
DType integer_size(const std::size_t& s);
DType unsigned_size(const std::size_t& s);

}
}
#endif
