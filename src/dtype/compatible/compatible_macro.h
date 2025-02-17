
//In all honesty, I am not a huge fan of using macros
//But, they definitely have their uses
//Below is a lot of macros that work
//basically, their main job is to create a way to easily register DTypes
//and then have all the necessary iterators, constexpr checks, and what not be pre-made once the macros are defined
//and it also has checks at the end to make sure everything was made properly and smoothly
//while the below isnt super pretty, if you go to the dtype_compatible.h file
//you can see how they make things really simple for potential future use of wanting to add extra dtypes
//this format will make things a lot easier, cleaner, and easier to understand
#ifndef __DTYPE_COMPATIBLE_MACRO_H__
#define __DTYPE_COMPATIBLE_MACRO_H__
#include "../../types/Types.h"
#include "../DType.h"
#include "../DType_enum.h"
#include "../../utils/utils.h"

/*

A way to acount for types of the same size: (like float16_t and bfloat16_t)

template<typename T>
struct TypeRegistry : std::false_type {};


#define REGISTER_TYPE(type)\
template<>\
struct TypeRegistry<type> : std::true_type {};\


 */


namespace nt{
namespace DTypeFuncs{

template<DType... dts>
inline constexpr bool integer_is_in_dtype_v = (is_in_v<DType::Integer, dts...>
					|| is_in_v<DType::Byte, dts...>
					|| is_in_v<DType::Short, dts...>
					|| is_in_v<DType::UnsignedShort, dts...>
					|| is_in_v<DType::LongLong, dts...>
					|| is_in_v<DType::Long, dts...>
#ifdef __SIZEOF_INT128__
					|| is_in_v<DType::Char, dts...>
					|| is_in_v<DType::int128, dts...>
#endif //__SIZEOF_INT128__
					|| is_in_v<DType::uint128, dts...>);


template<DType dt>
struct is_dtype_floating : std::false_type {};
template<DType dt>
struct is_dtype_complex : std::false_type {};
template<DType dt>
struct is_dtype_integer : std::false_type {};
template<DType dt>
struct is_dtype_signed : std::false_type {};
template<DType dt>
struct is_dtype_unsigned : std::false_type {};
template<DType dt>
struct is_dtype_real_num : std::false_type {};
template<DType dt>
struct is_dtype_num : std::false_type {};
template<DType dt>
struct is_dtype_other : std::false_type {};
template<int i>
struct is_registered_type : std::false_type {};


template<DType dt>
inline constexpr bool is_dtype_floating_v = is_dtype_floating<dt>::value;
template<DType dt>
inline constexpr bool is_dtype_complex_v = is_dtype_complex<dt>::value;
template<DType dt>
inline constexpr bool is_dtype_integer_v = is_dtype_integer<dt>::value;
template<DType dt>
inline constexpr bool is_dtype_signed_v = is_dtype_signed<dt>::value;
template<DType dt>
inline constexpr bool is_dtype_unsigned_v = is_dtype_unsigned<dt>::value;
template<DType dt>
inline constexpr bool is_dtype_real_num_v = is_dtype_real_num<dt>::value;
template<DType dt>
inline constexpr bool is_dtype_num_v = is_dtype_num<dt>::value;
template<DType dt>
inline constexpr bool is_dtype_other_v = is_dtype_other<dt>::value;
template<int i>
inline constexpr bool is_registered_type_v = is_registered_type<i>::value;


template<DType dt>
struct single_dtype_to_type{
	using type = uint_bool_t;
};

template<DType... Rest>
struct dtype_to_type{
	using type = uint_bool_t;
};

template<DType dt, DType... Rest>
struct dtype_to_type<dt, Rest...>{
	using type = typename single_dtype_to_type<dt>::type;
};

template<DType... Rest>
using dtype_to_type_t = typename dtype_to_type<Rest...>::type;


#define DEFINE_DTYPE_TO_TYPE(dt, result) \
template<> \
struct single_dtype_to_type<dt> { \
	using type = result; \
};


template<typename T>
struct type_to_dtype_s{
	static constexpr DType dt = DType::Bool;
};

#define DEFINE_TYPE_TO_DTYPE(dtype_enum, type)\
template<>\
struct type_to_dtype_s<type>{\
	static constexpr DType dt = dtype_enum;\
};

template<typename T>
inline static constexpr DType type_to_dtype = type_to_dtype_s<T>::dt;

template<typename T>
inline static constexpr bool type_is_dtype = std::is_same_v<T, bool> || std::is_same_v<T, uint_bool_t> || type_to_dtype<T> != DType::Bool;

template <DType dt>
inline static constexpr bool dtype_is_num = is_dtype_num_v<dt>;


template<DType dt>
inline constexpr std::size_t size_of_dtype_c = sizeof(dtype_to_type_t<dt>);

template<std::size_t dt_size>
struct is_convertible_to_complex_s : std::false_type{};

template<DType dt>
inline constexpr bool is_convertible_to_complex = is_convertible_to_complex_s<size_of_dtype_c<dt>>::value;

template<std::size_t dt_size>
struct convert_to_complex_s{
	static constexpr DType dt = DType::Bool;
};

template<DType dt>
inline constexpr DType convert_to_complex = convert_to_complex_s<size_of_dtype_c<dt>>::dt;


template<std::size_t dt_size>
struct is_convertible_to_floating_s : std::false_type{};

template<DType dt>
inline constexpr bool is_convertible_to_floating = is_convertible_to_floating_s<size_of_dtype_c<dt>>::value;

template<std::size_t dt_size>
struct convert_to_floating_s{
	static constexpr DType dt = DType::Bool;
};

template<DType dt>
inline constexpr DType convert_to_floating = convert_to_floating_s<size_of_dtype_c<dt>>::dt;



template<std::size_t dt_size>
struct is_convertible_to_integer_s : std::false_type{};

template<DType dt>
inline constexpr bool is_convertible_to_integer = is_convertible_to_integer_s<size_of_dtype_c<dt>>::value;

template<std::size_t dt_size>
struct convert_to_integer_s{
	static constexpr DType dt = DType::Bool;
};

template<DType dt>
inline constexpr DType convert_to_integer = convert_to_integer_s<size_of_dtype_c<dt>>::dt;



template<std::size_t dt_size>
struct is_convertible_to_unsigned_s : std::false_type{};

template<DType dt>
constexpr bool is_convertible_to_unsigned = is_convertible_to_unsigned_s<size_of_dtype_c<dt>>::value;

template<std::size_t dt_size>
struct convert_to_unsigned_s{
	static constexpr DType dt = DType::Bool;
};

template<DType dt>
inline constexpr DType convert_to_unsigned = convert_to_unsigned_s<size_of_dtype_c<dt>>::dt;



#define _NT_REGISTER_FLOATING_TYPE(type, dtype_enum)\
namespace nt{ namespace DTypeFuncs{\
DEFINE_DTYPE_TO_TYPE(dtype_enum, type)\
DEFINE_TYPE_TO_DTYPE(dtype_enum, type)\
template<>\
struct is_dtype_floating<dtype_enum> : std::true_type {};\
template<>\
struct is_dtype_real_num<dtype_enum> : std::true_type {};\
template<>\
struct is_dtype_num<dtype_enum> : std::true_type {};\
template<>\
struct is_convertible_to_floating_s<size_of_dtype_c<dtype_enum>> : std::true_type{};\
template<>\
struct convert_to_floating_s<size_of_dtype_c<dtype_enum>>{\
	static constexpr DType dt = dtype_enum;\
};\
template<>\
struct is_registered_type<int(dtype_enum)> : std::true_type {};\
}} //nt::DTypeFuncs:: 


#define _NT_REGISTER_COMPLEX_TYPE(type, dtype_enum)\
namespace nt{ namespace DTypeFuncs{\
DEFINE_DTYPE_TO_TYPE(dtype_enum, type)\
DEFINE_TYPE_TO_DTYPE(dtype_enum, type)\
template<>\
struct is_dtype_complex<dtype_enum> : std::true_type {};\
template<>\
struct is_dtype_num<dtype_enum> : std::true_type {};\
template<>\
struct is_convertible_to_complex_s<size_of_dtype_c<dtype_enum>> : std::true_type{};\
template<>\
struct convert_to_complex_s<size_of_dtype_c<dtype_enum>>{\
	static constexpr DType dt = dtype_enum;\
};\
template<>\
struct is_registered_type<int(dtype_enum)> : std::true_type {};\
}} //nt::DTypeFuncs:: 


#define _NT_REGISTER_INTEGER_TYPE(type, dtype_enum)\
namespace nt{ namespace DTypeFuncs{\
DEFINE_DTYPE_TO_TYPE(dtype_enum, type)\
DEFINE_TYPE_TO_DTYPE(dtype_enum, type)\
template<>\
struct is_dtype_integer<dtype_enum> : std::true_type {};\
template<>\
struct is_dtype_signed<dtype_enum> : std::true_type {};\
template<>\
struct is_dtype_real_num<dtype_enum> : std::true_type {};\
template<>\
struct is_dtype_num<dtype_enum> : std::true_type {};\
template<>\
struct is_convertible_to_integer_s<size_of_dtype_c<dtype_enum>> : std::true_type{};\
template<>\
struct convert_to_integer_s<size_of_dtype_c<dtype_enum>>{\
	static constexpr DType dt = dtype_enum;\
};\
template<>\
struct is_registered_type<int(dtype_enum)> : std::true_type {};\
}} //nt::DTypeFuncs:: 


#define _NT_REGISTER_UNSIGNED_TYPE(type, dtype_enum)\
namespace nt{ namespace DTypeFuncs{\
DEFINE_DTYPE_TO_TYPE(dtype_enum, type)\
DEFINE_TYPE_TO_DTYPE(dtype_enum, type)\
template<>\
struct is_dtype_integer<dtype_enum> : std::true_type {};\
template<>\
struct is_dtype_unsigned<dtype_enum> : std::true_type {};\
template<>\
struct is_dtype_real_num<dtype_enum> : std::true_type {};\
template<>\
struct is_dtype_num<dtype_enum> : std::true_type {};\
template<>\
struct is_convertible_to_unsigned_s<size_of_dtype_c<dtype_enum>> : std::true_type{};\
template<>\
struct convert_to_unsigned_s<size_of_dtype_c<dtype_enum>>{\
	static constexpr DType dt = dtype_enum;\
};\
template<>\
struct is_registered_type<int(dtype_enum)> : std::true_type {};\
}} //nt::DTypeFuncs:: 


#define _NT_REGISTER_OTHER_TYPE(type, dtype_enum)\
namespace nt{ namespace DTypeFuncs{\
DEFINE_DTYPE_TO_TYPE(dtype_enum, type)\
DEFINE_TYPE_TO_DTYPE(dtype_enum, type)\
template<>\
struct is_dtype_other<dtype_enum> : std::true_type {};\
template<>\
struct is_registered_type<int(dtype_enum)> : std::true_type {};\
}} //nt::DTypeFuncs::



//now going to have all the iterators registered
//(1) first going to make a constexpr dtype list to hold all registered dtypes

template<int... Is>\
struct AllDTypeListK {};\
template<int I, int... Is>\
struct AllDTypeListK<I, Is...>{\
	using type = AllDTypeListK<I, Is...>;\
	using minus_type = AllDTypeListK<Is...>;\
	static constexpr size_t size = sizeof...(Is) + 1;\
	static constexpr int index = size-1;\
	static constexpr int element = I;\
};
template<>
struct AllDTypeListK<>{
	using type = AllDTypeListK<>;
	using minus_type = AllDTypeListK<>;
	static constexpr size_t size = 0;
	static constexpr int index = -1;
	static constexpr int element = -1;
};

template<int I, typename List>
struct Append_AllDTypeListK;

template<int I, int... Is>
struct Append_AllDTypeListK<I, AllDTypeListK<Is...>>{
	using type = AllDTypeListK<I, Is...>;
};

template<int Start, typename CurrentList = AllDTypeListK<>>\
struct MakeDTypeListK{
	using DTypeList_result = std::conditional_t<is_registered_type_v<Start>,
				typename Append_AllDTypeListK<Start, CurrentList>::type,
				CurrentList>;
	using DTypeList = typename MakeDTypeListK<Start + 1, DTypeList_result>::DTypeList;
};
template<typename CurrentList>
struct MakeDTypeListK<-1, CurrentList>{
	using DTypeList_result = CurrentList;
	using DTypeList = MakeDTypeListK<-1, CurrentList>;
};
template<typename CurrentList>
struct MakeDTypeListK<200, CurrentList>{
	using DTypeList_result = std::conditional_t<is_registered_type_v<200>,
				typename Append_AllDTypeListK<200, CurrentList>::type,
				CurrentList>;
	using DTypeList = MakeDTypeListK<-1, CurrentList>;
};

template <DType NewType, typename ExistingEnum>
struct append_to_dtype_enum;

// Base case: Appending to an empty DTypeEnum
template <DType NewType>
struct append_to_dtype_enum<NewType, double> {
    using type = DTypeEnum<NewType>;
};

// Recursive case: Append to a non-empty DTypeEnum
template <DType NewType, DType First, DType... Rest>
struct append_to_dtype_enum<NewType, DTypeEnum<First, Rest...>> {
    using type = DTypeEnum<NewType, First, Rest...>;
};



#define _NT_COLLECT_OF_TYPE_(name)\
template<typename previous, typename List>\
struct collect_##name##_types{\
	using enum_result = typename std::conditional_t< \
		is_dtype_##name##_v<DType(List::element)>,\
		collect_##name##_types< \
			typename append_to_dtype_enum<DType(List::element), previous>::type,\
			typename List::minus_type>,\
		collect_##name##_types<previous, typename List::minus_type> >::enum_result;\
};\
template<typename previous>\
struct collect_##name##_types<previous, AllDTypeListK<> >{\
	using enum_result = previous;\
};

_NT_COLLECT_OF_TYPE_(floating)
_NT_COLLECT_OF_TYPE_(complex)
_NT_COLLECT_OF_TYPE_(integer)
_NT_COLLECT_OF_TYPE_(signed)
_NT_COLLECT_OF_TYPE_(unsigned)
_NT_COLLECT_OF_TYPE_(real_num)
_NT_COLLECT_OF_TYPE_(num)

template<typename previous, typename List>
struct collect_all_types{
	using enum_result = typename collect_all_types<
			typename append_to_dtype_enum<DType(List::element), previous>::type,
			typename List::minus_type>::enum_result;
};

template<typename previous>
struct collect_all_types<previous, AllDTypeListK<> >{
	using enum_result = previous;
};

template<typename previous, typename List>
struct collect_all_nbool_types{
	using enum_result = typename std::conditional_t<
		DType(List::element) != DType::Bool,
		collect_all_nbool_types<
			typename append_to_dtype_enum<DType(List::element), previous>::type,
			typename List::minus_type>,
		collect_all_nbool_types<previous, typename List::minus_type> >::enum_result;
};

template<typename previous>
struct collect_all_nbool_types<previous, AllDTypeListK<> >{
	using enum_result = previous;
};



#define _NT_REGISTER_ALL_ITERATORS_()\
namespace nt{ namespace DTypeFuncs{\
using kDTypeListIt = MakeDTypeListK<0>::DTypeList::DTypeList_result;\
template<int i, typename List = kDTypeListIt>\
struct get_dtype_list_element_index{\
	static constexpr int index = (i == List::element) ? (List::index) :\
		(List::index == 0) ? 0 : get_dtype_list_element_index<i, typename List::minus_type>::index;\
};\
template<int i>\
struct get_dtype_list_element_index<i, AllDTypeListK<> >{\
	static constexpr int index = 0;\
};\
template<int i, typename List = kDTypeListIt>\
struct get_dtype_element_at_index_from_list{\
	static constexpr DType element = List::index == i ? DType(List::element)\
		: get_dtype_element_at_index_from_list<i, typename List::minus_type>::element;\
};\
template<int i>\
struct get_dtype_element_at_index_from_list<i, AllDTypeListK<> >{\
	static constexpr DType element = DType::Bool;\
};\
template<DType dtype>\
constexpr DType next_dtype_it = \
	get_dtype_list_element_index<int(dtype), kDTypeListIt>::index == 0\
	? DType(kDTypeListIt::element) \
	: get_dtype_element_at_index_from_list<get_dtype_list_element_index<int(dtype), kDTypeListIt>::index-1>::element;\
} \
using ComplexTypesL = DTypeFuncs::collect_complex_types<double, DTypeFuncs::kDTypeListIt>::enum_result;\
using FloatingTypesL = DTypeFuncs::collect_floating_types<double, DTypeFuncs::kDTypeListIt>::enum_result;\
using IntegerTypesL = DTypeFuncs::collect_integer_types<double, DTypeFuncs::kDTypeListIt>::enum_result;\
using SignedTypesL = DTypeFuncs::collect_signed_types<double, DTypeFuncs::kDTypeListIt>::enum_result;\
using UnsignedTypesL = DTypeFuncs::collect_unsigned_types<double, DTypeFuncs::kDTypeListIt>::enum_result;\
using AllTypesL = DTypeFuncs::collect_all_types<double, DTypeFuncs::kDTypeListIt>::enum_result;\
using AllTypesNBoolL = DTypeFuncs::collect_all_nbool_types<double, DTypeFuncs::kDTypeListIt>::enum_result;\
using NumberTypesL = DTypeFuncs::collect_num_types<double, DTypeFuncs::kDTypeListIt>::enum_result;\
using RealNumberTypesL = DTypeFuncs::collect_real_num_types<double, DTypeFuncs::kDTypeListIt>::enum_result;\
} //nt::






//checks in place to make static asserts to make sure everything was set up properly:

#ifdef _HALF_FLOAT_SUPPORT_
#define _NT_CHECK_COMPLEX_CONVERT_(dtype_enum) \
static_assert( \
	(size_of_dtype_c<dtype_enum> == size_of_dtype_c<DType::Complex32> || size_of_dtype_c<dtype_enum> == size_of_dtype_c<DType::Complex64> || size_of_dtype_c<dtype_enum> == size_of_dtype_c<DType::Complex128>)\
	? is_convertible_to_complex<dtype_enum> : !is_convertible_to_complex<dtype_enum>, "Expected to be able to convert "#dtype_enum" to a complex number");\
static_assert( \
	is_convertible_to_complex<dtype_enum> \
	? convert_to_complex<dtype_enum> != DType::Bool && size_of_dtype_c<convert_to_complex<dtype_enum>> == size_of_dtype_c<dtype_enum> \
	: convert_to_complex<dtype_enum> == DType::Bool, "Expected to get proper complex number from convert on "#dtype_enum" but did not");
#else
#define _NT_CHECK_COMPLEX_CONVERT_(dtype_enum) \
static_assert( \
	(size_of_dtype_c<dtype_enum> == size_of_dtype_c<DType::Complex64> || size_of_dtype_c<dtype_enum> == size_of_dtype_c<DType::Complex128>)\
	? is_convertible_to_complex<dtype_enum> : !is_convertible_to_complex<dtype_enum>, "Expected to be able to convert "#dtype_enum" to a complex number");\
static_assert( \
	is_convertible_to_complex<dtype_enum> \
	? convert_to_complex<dtype_enum> != DType::Bool && size_of_dtype_c<convert_to_complex<dtype_enum>> == size_of_dtype_c<dtype_enum> \
	: convert_to_complex<dtype_enum> == DType::Bool, "Expected to get proper complex number from convert on "#dtype_enum" but did not");


#endif


#if defined(_128_FLOAT_SUPPORT_) && defined(_HALF_FLOAT_SUPPORT_)
#define _NT_CHECK_FLOATING_CONVERT_(dtype_enum) \
static_assert( \
	(size_of_dtype_c<dtype_enum> == size_of_dtype_c<DType::Float16> ||\
	 size_of_dtype_c<dtype_enum> == size_of_dtype_c<DType::Float32> ||\
	 size_of_dtype_c<dtype_enum> == size_of_dtype_c<DType::Float64> ||\
	 size_of_dtype_c<dtype_enum> == size_of_dtype_c<DType::Float128>)\
	? is_convertible_to_floating<dtype_enum> : !is_convertible_to_floating<dtype_enum>, "Expected to be able to convert "#dtype_enum" to a floating number");\
static_assert( \
	is_convertible_to_floating<dtype_enum> \
	? convert_to_floating<dtype_enum> != DType::Bool && size_of_dtype_c<convert_to_floating<dtype_enum>> == size_of_dtype_c<dtype_enum> \
	: convert_to_floating<dtype_enum> == DType::Bool, "Expected to get proper floating number from convert on "#dtype_enum" but did not");

#elif defined(_128_FLOAT_SUPPORT_) && !defined(_HALF_FLOAT_SUPPORT)
#define _NT_CHECK_FLOATING_CONVERT_(dtype_enum) \
static_assert( \
	(size_of_dtype_c<dtype_enum> == size_of_dtype_c<DType::Float32> ||\
	 size_of_dtype_c<dtype_enum> == size_of_dtype_c<DType::Float64> ||\
	 size_of_dtype_c<dtype_enum> == size_of_dtype_c<DType::Float128>)\
	? is_convertible_to_floating<dtype_enum> : !is_convertible_to_floating<dtype_enum>, "Expected to be able to convert "#dtype_enum" to a floating number");\
static_assert( \
	is_convertible_to_floating<dtype_enum> \
	? convert_to_floating<dtype_enum> != DType::Bool && size_of_dtype_c<convert_to_floating<dtype_enum>> == size_of_dtype_c<dtype_enum> \
	: convert_to_floating<dtype_enum> == DType::Bool, "Expected to get proper floating number from convert on "#dtype_enum" but did not");

#elif !defined(_128_FLOAT_SUPPORT_) && defined(_HALF_FLOAT_SUPPORT_)


#define _NT_THIS_STRING_OR_THAT_STRING_CONDITIONAL(condition, string_a, string_b) condition ? string_a : string_b


#define _NT_CHECK_FLOATING_CONVERT_(dtype_enum) \
static_assert( \
	(size_of_dtype_c<dtype_enum> == size_of_dtype_c<DType::Float16> ||\
	 size_of_dtype_c<dtype_enum> == size_of_dtype_c<DType::Float32> ||\
	 size_of_dtype_c<dtype_enum> == size_of_dtype_c<DType::Float64>)\
	? is_convertible_to_floating<dtype_enum> : !is_convertible_to_floating<dtype_enum>,\
	"Expected to not be able to convert "#dtype_enum" to a floating number");\
static_assert( \
	is_convertible_to_floating<dtype_enum> \
	? convert_to_floating<dtype_enum> != DType::Bool && size_of_dtype_c<convert_to_floating<dtype_enum>> == size_of_dtype_c<dtype_enum> \
	: convert_to_floating<dtype_enum> == DType::Bool, "Expected to get proper floating number from convert on "#dtype_enum" but did not");


#elif !defined(_128_FLOAT_SUPPORT_) && !defined(_HALF_FLOAT_SUPPORT_)

#define _NT_CHECK_FLOATING_CONVERT_(dtype_enum) \
static_assert( \
	(size_of_dtype_c<dtype_enum> == size_of_dtype_c<DType::Float32> ||\
	 size_of_dtype_c<dtype_enum> == size_of_dtype_c<DType::Float64>)\
	? is_convertible_to_floating<dtype_enum> : !is_convertible_to_floating<dtype_enum>, "Expected to be able to convert "#dtype_enum" to a floating number");\
static_assert( \
	is_convertible_to_floating<dtype_enum> \
	? convert_to_floating<dtype_enum> != DType::Bool && size_of_dtype_c<convert_to_floating<dtype_enum>> == size_of_dtype_c<dtype_enum> \
	: convert_to_floating<dtype_enum> == DType::Bool, "Expected to get proper floating number from convert on "#dtype_enum" but did not");


#endif


#ifdef __SIZEOF_INT128__
#define _NT_CHECK_INTEGER_CONVERT_(dtype_enum) \
static_assert( \
	(size_of_dtype_c<dtype_enum> == size_of_dtype_c<DType::int8> ||\
	 size_of_dtype_c<dtype_enum> == size_of_dtype_c<DType::int16> ||\
	 size_of_dtype_c<dtype_enum> == size_of_dtype_c<DType::int32> ||\
	 size_of_dtype_c<dtype_enum> == size_of_dtype_c<DType::int64> ||\
	 size_of_dtype_c<dtype_enum> == size_of_dtype_c<DType::int128>)\
	? is_convertible_to_integer<dtype_enum> : !is_convertible_to_integer<dtype_enum>, "Expected to be able to convert "#dtype_enum" to an integer");\
static_assert( \
	is_convertible_to_integer<dtype_enum> \
	? convert_to_integer<dtype_enum> != DType::Bool && size_of_dtype_c<convert_to_integer<dtype_enum>> == size_of_dtype_c<dtype_enum> \
	: convert_to_integer<dtype_enum> == DType::Bool, "Expected to get proper integer number from convert on "#dtype_enum" but did not");

#define _NT_CHECK_UNSIGNED_CONVERT_(dtype_enum) \
static_assert( \
	(size_of_dtype_c<dtype_enum> == size_of_dtype_c<DType::uint8> ||\
	 size_of_dtype_c<dtype_enum> == size_of_dtype_c<DType::uint16> ||\
	 size_of_dtype_c<dtype_enum> == size_of_dtype_c<DType::uint32> ||\
	 size_of_dtype_c<dtype_enum> == size_of_dtype_c<DType::uint128>)\
	? is_convertible_to_unsigned<dtype_enum> : !is_convertible_to_unsigned<dtype_enum>, "Expected to be able to convert "#dtype_enum" to an unsigned integer ");\
static_assert( \
	is_convertible_to_unsigned<dtype_enum> \
	? convert_to_unsigned<dtype_enum> != DType::Bool && size_of_dtype_c<convert_to_unsigned<dtype_enum>> == size_of_dtype_c<dtype_enum> \
	: convert_to_unsigned<dtype_enum> == DType::Bool, "Expected to get proper unsigned number from convert on "#dtype_enum" but did not");


#else
#define _NT_CHECK_INTEGER_CONVERT_(dtype_enum) \
static_assert( \
	(size_of_dtype_c<dtype_enum> == size_of_dtype_c<DType::int8> ||\
	 size_of_dtype_c<dtype_enum> == size_of_dtype_c<DType::int16> ||\
	 size_of_dtype_c<dtype_enum> == size_of_dtype_c<DType::int32> ||\
	 size_of_dtype_c<dtype_enum> == size_of_dtype_c<DType::int64> )\
	? is_convertible_to_integer<dtype_enum> : !is_convertible_to_integer<dtype_enum>, "Expected to be able to convert "#dtype_enum" to an integer");\
static_assert( \
	is_convertible_to_integer<dtype_enum> \
	? convert_to_integer<dtype_enum> != DType::Bool && size_of_dtype_c<convert_to_integer<dtype_enum>> == size_of_dtype_c<dtype_enum> \
	: convert_to_integer<dtype_enum> == DType::Bool, "Expected to get proper integer number from convert on "#dtype_enum" but did not");

#define _NT_CHECK_UNSIGNED_CONVERT_(dtype_enum) \
static_assert( \
	(size_of_dtype_c<dtype_enum> == size_of_dtype_c<DType::uint8> ||\
	 size_of_dtype_c<dtype_enum> == size_of_dtype_c<DType::uint16> ||\
	 size_of_dtype_c<dtype_enum> == size_of_dtype_c<DType::uint32>)\
	? is_convertible_to_unsigned<dtype_enum> : !is_convertible_to_unsigned<dtype_enum>, "Expected to be able to convert "#dtype_enum" to an unsigned integer");\
static_assert( \
	is_convertible_to_unsigned<dtype_enum> \
	? convert_to_unsigned<dtype_enum> != DType::Bool && size_of_dtype_c<convert_to_unsigned<dtype_enum>> == size_of_dtype_c<dtype_enum> \
	: convert_to_unsigned<dtype_enum> == DType::Bool, "Expected to get proper unsigned number from convert on "#dtype_enum" but did not");


#endif


#define _NT_CHECK_SIGNED_(dtype_enum, type) \
static_assert(is_dtype_integer_v<dtype_enum>, "Expected "#dtype_enum" to be an integer");\
static_assert(is_dtype_real_num_v<dtype_enum>, "Expected "#dtype_enum" to be an real number");\
static_assert(is_dtype_num_v<dtype_enum>, "Expected "#dtype_enum" to be a number");\
static_assert(!is_dtype_floating_v<dtype_enum>, "Expected "#dtype_enum" to not be floating");\
static_assert(!is_dtype_complex_v<dtype_enum>, "Expected "#dtype_enum" to not be complex");\
static_assert(!is_dtype_other_v<dtype_enum>, "Expected "#dtype_enum" to not be an other type");\
static_assert(std::is_same_v<dtype_to_type_t<dtype_enum, DType::Float32, DType::Float64, DType::uint16>, type>, #dtype_enum" does not correlate to expected type");\
static_assert(type_to_dtype<type> == dtype_enum, "Expected to get proper dtype "#dtype_enum" for type "#type);\
static_assert(sizeof(type) == size_of_dtype_c<dtype_enum>, "Did not get proper size when converting from "#dtype_enum" back to regular type");\
static_assert(!is_in_dtype_enum<ComplexTypesL>(dtype_enum), "Expected "#dtype_enum" to not be in iterable complex constexpr");\
static_assert(!is_in_dtype_enum<FloatingTypesL>(dtype_enum), "Expected "#dtype_enum" to not be in iterable floating constexpr");\
static_assert(is_in_dtype_enum<IntegerTypesL>(dtype_enum), "Expected "#dtype_enum" to be in iterable integer constexpr");\
static_assert(is_in_dtype_enum<SignedTypesL>(dtype_enum), "Expected "#dtype_enum" to be in iterable signed integer constexpr");\
static_assert(!is_in_dtype_enum<UnsignedTypesL>(dtype_enum), "Expected "#dtype_enum" to not be in iterable unsigned integer constexpr");\
static_assert(is_in_dtype_enum<AllTypesL>(dtype_enum), "Expected "#dtype_enum" to be in iterable all types constexpr");\
static_assert(is_in_dtype_enum<AllTypesNBoolL>(dtype_enum), "Expected "#dtype_enum" to be in iterable all types nbool constexpr");\
static_assert(is_in_dtype_enum<NumberTypesL>(dtype_enum), "Expected "#dtype_enum" to be in iterable number types constexpr");\
static_assert(is_in_dtype_enum<RealNumberTypesL>(dtype_enum), "Expected "#dtype_enum" to be in iterable number types constexpr");\
_NT_CHECK_COMPLEX_CONVERT_(dtype_enum)\
_NT_CHECK_FLOATING_CONVERT_(dtype_enum)\
_NT_CHECK_UNSIGNED_CONVERT_(dtype_enum)\
_NT_CHECK_INTEGER_CONVERT_(dtype_enum)


#define _NT_CHECK_UNSIGNED_(dtype_enum, type) \
static_assert(is_dtype_integer_v<dtype_enum>, "Expected "#dtype_enum" to be an integer");\
static_assert(is_dtype_real_num_v<dtype_enum>, "Expected "#dtype_enum" to be an real number");\
static_assert(is_dtype_num_v<dtype_enum>, "Expected "#dtype_enum" to be a number");\
static_assert(!is_dtype_floating_v<dtype_enum>, "Expected "#dtype_enum" to not be floating");\
static_assert(!is_dtype_complex_v<dtype_enum>, "Expected "#dtype_enum" to not be complex");\
static_assert(!is_dtype_other_v<dtype_enum>, "Expected "#dtype_enum" to not be an other type");\
static_assert(std::is_same_v<dtype_to_type_t<dtype_enum, DType::Float32, DType::Float64, DType::uint16>, type>, #dtype_enum" does not correlate to expected type");\
static_assert(type_to_dtype<type> == dtype_enum, "Expected to get proper dtype "#dtype_enum" for type "#type);\
static_assert(sizeof(type) == size_of_dtype_c<dtype_enum>, "Did not get proper size when converting from "#dtype_enum" back to regular type");\
static_assert(!is_in_dtype_enum<ComplexTypesL>(dtype_enum), "Expected "#dtype_enum" to not be in iterable complex constexpr");\
static_assert(!is_in_dtype_enum<FloatingTypesL>(dtype_enum), "Expected "#dtype_enum" to not be in iterable floating constexpr");\
static_assert(is_in_dtype_enum<IntegerTypesL>(dtype_enum), "Expected "#dtype_enum" to be in iterable integer constexpr");\
static_assert(!is_in_dtype_enum<SignedTypesL>(dtype_enum), "Expected "#dtype_enum" to not be in iterable signed integer constexpr");\
static_assert(is_in_dtype_enum<UnsignedTypesL>(dtype_enum), "Expected "#dtype_enum" to be in iterable unsigned integer constexpr");\
static_assert(is_in_dtype_enum<AllTypesL>(dtype_enum), "Expected "#dtype_enum" to be in iterable all types constexpr");\
static_assert(is_in_dtype_enum<AllTypesNBoolL>(dtype_enum), "Expected "#dtype_enum" to be in iterable all types nbool constexpr");\
static_assert(is_in_dtype_enum<NumberTypesL>(dtype_enum), "Expected "#dtype_enum" to be in iterable number types constexpr");\
static_assert(is_in_dtype_enum<RealNumberTypesL>(dtype_enum), "Expected "#dtype_enum" to be in iterable number types constexpr");\
_NT_CHECK_COMPLEX_CONVERT_(dtype_enum)\
_NT_CHECK_FLOATING_CONVERT_(dtype_enum)\
_NT_CHECK_UNSIGNED_CONVERT_(dtype_enum)\
_NT_CHECK_INTEGER_CONVERT_(dtype_enum)


#define _NT_CHECK_COMPLEX_(dtype_enum, type) \
static_assert(!is_dtype_integer_v<dtype_enum>, "Expected "#dtype_enum" to not be an integer");\
static_assert(!is_dtype_real_num_v<dtype_enum>, "Expected "#dtype_enum" to not be a real number");\
static_assert(is_dtype_num_v<dtype_enum>, "Expected "#dtype_enum" to be a number");\
static_assert(!is_dtype_floating_v<dtype_enum>, "Expected "#dtype_enum" to not be floating");\
static_assert(is_dtype_complex_v<dtype_enum>, "Expected "#dtype_enum" to be complex");\
static_assert(!is_dtype_other_v<dtype_enum>, "Expected "#dtype_enum" to not be an other type");\
static_assert(std::is_same_v<dtype_to_type_t<dtype_enum, DType::Float32, DType::Float64, DType::uint16>, type>, #dtype_enum" does not correlate to expected type");\
static_assert(type_to_dtype<type> == dtype_enum, "Expected to get proper dtype "#dtype_enum" for type "#type);\
static_assert(sizeof(type) == size_of_dtype_c<dtype_enum>, "Did not get proper size when converting from "#dtype_enum" back to regular type");\
static_assert(is_in_dtype_enum<ComplexTypesL>(dtype_enum), "Expected "#dtype_enum" to be in iterable complex constexpr");\
static_assert(!is_in_dtype_enum<FloatingTypesL>(dtype_enum), "Expected "#dtype_enum" to not be in iterable floating constexpr");\
static_assert(!is_in_dtype_enum<IntegerTypesL>(dtype_enum), "Expected "#dtype_enum" to not be in iterable integer constexpr");\
static_assert(!is_in_dtype_enum<SignedTypesL>(dtype_enum), "Expected "#dtype_enum" to not be in iterable signed integer constexpr");\
static_assert(!is_in_dtype_enum<UnsignedTypesL>(dtype_enum), "Expected "#dtype_enum" to not be in iterable unsigned integer constexpr");\
static_assert(is_in_dtype_enum<AllTypesL>(dtype_enum), "Expected "#dtype_enum" to be in iterable all types constexpr");\
static_assert(is_in_dtype_enum<AllTypesNBoolL>(dtype_enum), "Expected "#dtype_enum" to be in iterable all types nbool constexpr");\
static_assert(is_in_dtype_enum<NumberTypesL>(dtype_enum), "Expected "#dtype_enum" to be in iterable number types constexpr");\
static_assert(!is_in_dtype_enum<RealNumberTypesL>(dtype_enum), "Expected "#dtype_enum" to not be in iterable number types constexpr");\
_NT_CHECK_COMPLEX_CONVERT_(dtype_enum)\
_NT_CHECK_FLOATING_CONVERT_(dtype_enum)\
_NT_CHECK_UNSIGNED_CONVERT_(dtype_enum)\
_NT_CHECK_INTEGER_CONVERT_(dtype_enum)



#define _NT_CHECK_FLOATING_(dtype_enum, type) \
static_assert(!is_dtype_integer_v<dtype_enum>, "Expected "#dtype_enum" to not be an integer");\
static_assert(is_dtype_real_num_v<dtype_enum>, "Expected "#dtype_enum" to be a real number");\
static_assert(is_dtype_num_v<dtype_enum>, "Expected "#dtype_enum" to be a number");\
static_assert(is_dtype_floating_v<dtype_enum>, "Expected "#dtype_enum" to be floating");\
static_assert(!is_dtype_complex_v<dtype_enum>, "Expected "#dtype_enum" to not be complex");\
static_assert(!is_dtype_other_v<dtype_enum>, "Expected "#dtype_enum" to not be an other type");\
static_assert(std::is_same_v<dtype_to_type_t<dtype_enum, DType::Float32, DType::Float64, DType::uint16>, type>, #dtype_enum" does not correlate to expected type");\
static_assert(type_to_dtype<type> == dtype_enum, "Expected to get proper dtype "#dtype_enum" for type "#type);\
static_assert(sizeof(type) == size_of_dtype_c<dtype_enum>, "Did not get proper size when converting from "#dtype_enum" back to regular type");\
static_assert(!is_in_dtype_enum<ComplexTypesL>(dtype_enum), "Expected "#dtype_enum" to not be in iterable complex constexpr");\
static_assert(is_in_dtype_enum<FloatingTypesL>(dtype_enum), "Expected "#dtype_enum" to be in iterable floating constexpr");\
static_assert(!is_in_dtype_enum<IntegerTypesL>(dtype_enum), "Expected "#dtype_enum" to not be in iterable integer constexpr");\
static_assert(!is_in_dtype_enum<SignedTypesL>(dtype_enum), "Expected "#dtype_enum" to not be in iterable signed integer constexpr");\
static_assert(!is_in_dtype_enum<UnsignedTypesL>(dtype_enum), "Expected "#dtype_enum" to not be in iterable unsigned integer constexpr");\
static_assert(is_in_dtype_enum<AllTypesL>(dtype_enum), "Expected "#dtype_enum" to be in iterable all types constexpr");\
static_assert(is_in_dtype_enum<AllTypesNBoolL>(dtype_enum), "Expected "#dtype_enum" to be in iterable all types nbool constexpr");\
static_assert(is_in_dtype_enum<NumberTypesL>(dtype_enum), "Expected "#dtype_enum" to be in iterable number types constexpr");\
static_assert(is_in_dtype_enum<RealNumberTypesL>(dtype_enum), "Expected "#dtype_enum" to be in iterable number types constexpr");\
_NT_CHECK_COMPLEX_CONVERT_(dtype_enum)\
_NT_CHECK_FLOATING_CONVERT_(dtype_enum)\
_NT_CHECK_UNSIGNED_CONVERT_(dtype_enum)\
_NT_CHECK_INTEGER_CONVERT_(dtype_enum)


#define _NT_CHECK_OTHER_(dtype_enum, type) \
static_assert(!is_dtype_integer_v<dtype_enum>, "Expected "#dtype_enum" to not be an integer");\
static_assert(!is_dtype_real_num_v<dtype_enum>, "Expected "#dtype_enum" to not be a real number");\
static_assert(!is_dtype_num_v<dtype_enum>, "Expected "#dtype_enum" to not be a number");\
static_assert(!is_dtype_floating_v<dtype_enum>, "Expected "#dtype_enum" to not be a floating number");\
static_assert(!is_dtype_complex_v<dtype_enum>, "Expected "#dtype_enum" to not be complex");\
static_assert(is_dtype_other_v<dtype_enum>, "Expected "#dtype_enum" to be an other type");\
static_assert(type_to_dtype<type> == dtype_enum, "Expected to get proper dtype "#dtype_enum" for type "#type);\
static_assert(!is_in_dtype_enum<IntegerTypesL>(dtype_enum), "Expected "#dtype_enum" to not be in iterable integer constexpr");\
static_assert(!is_in_dtype_enum<ComplexTypesL>(dtype_enum), "Expected "#dtype_enum" to not be in iterable complex constexpr");\
static_assert(!is_in_dtype_enum<FloatingTypesL>(dtype_enum), "Expected "#dtype_enum" to not be in iterable floating constexpr");\
static_assert(!is_in_dtype_enum<IntegerTypesL>(dtype_enum), "Expected "#dtype_enum" to not be in iterable integer constexpr");\
static_assert(!is_in_dtype_enum<SignedTypesL>(dtype_enum), "Expected "#dtype_enum" to not be in iterable signed integer constexpr");\
static_assert(!is_in_dtype_enum<UnsignedTypesL>(dtype_enum), "Expected "#dtype_enum" to not be in iterable unsigned integer constexpr");\
static_assert(is_in_dtype_enum<AllTypesL>(dtype_enum), "Expected "#dtype_enum" to be in iterable all types constexpr");\
static_assert(dtype_enum == DType::Bool ? \
		!is_in_dtype_enum<AllTypesNBoolL>(dtype_enum)\
		: is_in_dtype_enum<AllTypesNBoolL>(dtype_enum), "Expected "#dtype_enum" to be handled properly in iterable all types nbool constexpr");\
static_assert(!is_in_dtype_enum<NumberTypesL>(dtype_enum), "Expected "#dtype_enum" to not be in iterable number types constexpr");\
static_assert(!is_in_dtype_enum<RealNumberTypesL>(dtype_enum), "Expected "#dtype_enum" to not be in iterable number types constexpr");\





}} //nt::DTypeFuncs::


#endif //__DTYPE_COMPATIBLE_MACRO_H__
