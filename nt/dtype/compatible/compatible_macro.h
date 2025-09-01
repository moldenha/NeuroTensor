#ifndef NT_DTYPE_COMPATIBLE_MACROS_H__
#define NT_DTYPE_COMPATIBLE_MACROS_H__

#include "fameta/include/fameta/counter.hpp"
#include "../../types/Types.h"
#include "../DType.h"
#include "../DType_enum.h"
#include "../../utils/utils.h"


namespace nt::DTypeFuncs{

//There are a few structure definitions that need to be laid out for every incoming type
//This is with a new implementation that is meant to incorperate every single type
template<DType... dts>
inline constexpr bool integer_is_in_dtype_v = (is_in_v<DType::Integer, dts...>
					|| is_in_v<DType::Byte, dts...>
					|| is_in_v<DType::Short, dts...>
					|| is_in_v<DType::UnsignedShort, dts...>
					|| is_in_v<DType::LongLong, dts...>
					|| is_in_v<DType::Long, dts...>
					|| is_in_v<DType::Char, dts...>
					|| is_in_v<DType::int128, dts...>
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
//this is used to keep track of if a type has been registered
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

template<typename T>
struct type_to_dtype_s{
    static constexpr DType dt = DType::Bool;
};


template<typename T>
inline static constexpr DType type_to_dtype = type_to_dtype_s<T>::dt;

template<typename T>
inline static constexpr bool type_is_dtype = std::is_same_v<T, bool> || std::is_same_v<T, uint_bool_t> || type_to_dtype<T> != DType::Bool;

#define NT_DEFINE_DTYPE_TYPE_CONVERSION(dtype_enum, T)\
template<>\
struct single_dtype_to_type<dtype_enum>{\
    using type = T;\
};\
template<>\
struct dtype_to_type<dtype_enum>{\
    using type = T;\
};\
template<>\
struct type_to_dtype_s<T>{\
    static constexpr DType dt = dtype_enum;\
};


template <DType dt>
inline static constexpr bool dtype_is_num = is_dtype_num_v<dt>;


template<DType dt>
inline constexpr std::size_t size_of_dtype_c = sizeof(dtype_to_type_t<dt>);


namespace details{
template<DType... dt>
struct DTypeList {};

template <typename TypeListT, DType dt>
struct AppendDType;

template <DType... dts, DType dt>
struct AppendDType<DTypeEnum<dts...>, dt> {
    using type = DTypeEnum<dts..., dt>;
};

template <DType... dts, DType dt>
struct AppendDType<DTypeList<dts...>, dt> {
    using type = DTypeList<dts..., dt>;
};


template<typename T>
struct DTypeList_to_DTypeEnum;

template<DType... dts>
struct DTypeList_to_DTypeEnum<DTypeList<dts...>>{
    using type = DTypeEnum<dts...>;
};


template<typename T>
using DTypeList_to_DTypeEnum_t = typename DTypeList_to_DTypeEnum<T>::type;

template<typename T>
struct DTypeEnum_to_DTypeList;
template<DType... dts>
struct DTypeEnum_to_DTypeList<DTypeEnum<dts...>>{
    using type = DTypeList<dts...>;
};

template <typename TypeListT, DType dt>
using append_type_t = typename AppendDType<TypeListT, dt>::type;

template<std::size_t i>
struct registered_order_to_dtype{
    static constexpr DType value = DType::Bool;
};

template<DType dt>
struct registered_order_to_dtype_num{
    static constexpr std::size_t value = 0;
};



template <int I>
struct U;

constexpr fameta::counter<U<1>, 0> C;
template<std::size_t... I>
inline constexpr DTypeList<registered_order_to_dtype<I>::value...> getRegisteredDTypeListImpl(std::index_sequence<I...>){
    return {};
}

}

#define NT_CONCATENATE_IMPL(x, y) x##y
#define NT_CONCATENATE(x, y) NT_CONCATENATE_IMPL(x, y)
#define NT_CONCAT3(x, y, z) NT_CONCATENATE(NT_CONCATENATE(x, y), z)

#define NT_DTYPE_REGISTER_COUNTER_MACRO_(dtype_enum, input_type)\
namespace details{\
static constexpr std::size_t registered_##dtype_enum##_counter = C.next<__COUNTER__>();\
template<>\
struct registered_order_to_dtype< registered_##dtype_enum##_counter >{\
    static constexpr DType value = dtype_enum;\
};\
template<>\
struct registered_order_to_dtype_num<dtype_enum>{\
    static constexpr std::size_t value = registered_##dtype_enum##_counter;\
};\
}\

#define NT_MAKE_DTYPE_ENUM_OTHER_FALSE(dtype_enum)\
template<>\
struct is_dtype_other<dtype_enum> : std::false_type {};\

#define NT_REGISTER_FLOATING_TYPE(type, dtype_enum)\
namespace nt::DTypeFuncs{\
NT_DEFINE_DTYPE_TYPE_CONVERSION(dtype_enum, type)\
template<>\
struct is_dtype_floating<dtype_enum> : std::true_type {};\
template<>\
struct is_dtype_real_num<dtype_enum> : std::true_type {};\
template<>\
struct is_dtype_num<dtype_enum> : std::true_type {};\
template<>\
struct is_registered_type<int(dtype_enum)> : std::true_type {};\
NT_MAKE_DTYPE_ENUM_OTHER_FALSE(dtype_enum)\
NT_DTYPE_REGISTER_COUNTER_MACRO_(dtype_enum, type)\
}\


//

#define NT_REGISTER_COMPLEX_TYPE(type, dtype_enum)\
namespace nt::DTypeFuncs{\
NT_DEFINE_DTYPE_TYPE_CONVERSION(dtype_enum, type)\
template<>\
struct is_dtype_complex<dtype_enum> : std::true_type {};\
template<>\
struct is_dtype_num<dtype_enum> : std::true_type {};\
template<>\
struct is_registered_type<int(dtype_enum)> : std::true_type {};\
NT_MAKE_DTYPE_ENUM_OTHER_FALSE(dtype_enum)\
NT_DTYPE_REGISTER_COUNTER_MACRO_(dtype_enum, type)\
} //nt::DTypeFuncs:: 


#define NT_REGISTER_INTEGER_TYPE(type, dtype_enum)\
namespace nt::DTypeFuncs{\
NT_DEFINE_DTYPE_TYPE_CONVERSION(dtype_enum, type)\
template<>\
struct is_dtype_integer<dtype_enum> : std::true_type {};\
template<>\
struct is_dtype_signed<dtype_enum> : std::true_type {};\
template<>\
struct is_dtype_real_num<dtype_enum> : std::true_type {};\
template<>\
struct is_dtype_num<dtype_enum> : std::true_type {};\
template<>\
struct is_registered_type<int(dtype_enum)> : std::true_type {};\
NT_MAKE_DTYPE_ENUM_OTHER_FALSE(dtype_enum)\
NT_DTYPE_REGISTER_COUNTER_MACRO_(dtype_enum, type)\
} //nt::DTypeFuncs:: 


#define NT_REGISTER_UNSIGNED_TYPE(type, dtype_enum)\
namespace nt::DTypeFuncs{\
NT_DEFINE_DTYPE_TYPE_CONVERSION(dtype_enum, type)\
template<>\
struct is_dtype_integer<dtype_enum> : std::true_type {};\
template<>\
struct is_dtype_unsigned<dtype_enum> : std::true_type {};\
template<>\
struct is_dtype_real_num<dtype_enum> : std::true_type {};\
template<>\
struct is_dtype_num<dtype_enum> : std::true_type {};\
template<>\
struct is_registered_type<int(dtype_enum)> : std::true_type {};\
NT_MAKE_DTYPE_ENUM_OTHER_FALSE(dtype_enum)\
NT_DTYPE_REGISTER_COUNTER_MACRO_(dtype_enum, type)\
} //nt::DTypeFuncs:: 


#define NT_REGISTER_OTHER_TYPE(type, dtype_enum)\
namespace nt::DTypeFuncs{\
NT_DEFINE_DTYPE_TYPE_CONVERSION(dtype_enum, type)\
template<>\
struct is_dtype_other<dtype_enum> : std::true_type {};\
template<>\
struct is_registered_type<int(dtype_enum)> : std::true_type {};\
NT_DTYPE_REGISTER_COUNTER_MACRO_(dtype_enum, type)\
} //nt::DTypeFuncs::
//
//



// Recursive filter
template <typename TypeListT, template<DType> class Predicate, typename Accum = details::DTypeList<>>
struct collect_all_types;

// Recursive case
template <DType Head, DType... Tail, template<DType> class Predicate, typename Accum>
struct collect_all_types<details::DTypeList<Head, Tail...>, Predicate, Accum> {
    using next_accum = std::conditional_t<
        Predicate<Head>::value,
        details::append_type_t<Accum, Head>,
        Accum
    >;
    using type = typename collect_all_types<details::DTypeList<Tail...>, Predicate, next_accum>::type;
};

// Base case
template <template<DType> class Predicate, typename Accum>
struct collect_all_types<details::DTypeList<>, Predicate, Accum> {
    using type = Accum;
};

template<DType dt>
struct is_not_dtype_bool : std::true_type{};

template<>
struct is_not_dtype_bool<DType::Bool> : std::false_type{};

template<DType dt>
static constexpr bool is_not_dtype_bool_v = is_not_dtype_bool<dt>::value;



template<DType dt, typename List>
struct sub_is_convertible_to;


template<DType dt>
struct sub_is_convertible_to<dt, details::DTypeList<> >{
    static constexpr bool value = false;
};

template<DType dt, DType head, DType... dts>
struct sub_is_convertible_to<dt, details::DTypeList<head, dts...> >{
    static constexpr bool value = size_of_dtype_c<dt> == size_of_dtype_c<head> || sub_is_convertible_to<dt, details::DTypeList<dts...> >::value;
};

template<typename List, DType dt>
struct is_convertible_to{
    static constexpr bool value = sub_is_convertible_to<dt, typename details::DTypeEnum_to_DTypeList<List>::type>::value; 
};

template<DType dt, typename List>
struct sub_convert_dtype_to;


template<DType dt>
struct sub_convert_dtype_to<dt, details::DTypeList<> >{
    static constexpr DType value = DType::Bool;
};

template<DType dt, DType head, DType... dts>
struct sub_convert_dtype_to<dt, details::DTypeList<head, dts...> >{
    static constexpr DType value = (size_of_dtype_c<dt> == size_of_dtype_c<head>) ?  head : sub_convert_dtype_to<dt, details::DTypeList<dts...> >::value;
};

template<typename List, DType dt>
struct convert_dtype_to{
    static constexpr DType value = sub_convert_dtype_to<dt, typename details::DTypeEnum_to_DTypeList<List>::type>::value; 
};



#define NT_MAKE_CONVERTIBLE_(Name, name)\
template<DType dt>\
inline constexpr bool is_convertible_to_##name = is_convertible_to<Name##TypesL, dt>::value;\
template<DType dt>\
inline constexpr DType convert_to_##name = convert_dtype_to<Name##TypesL, dt>::value;

// details::DTypeList_to_DTypeEnum
// DTypeList <-> DTypeEnum

#define NT_MAKE_TYPE_ITERATOR_(Name, name)\
using Name##TypesL = DTypeFuncs::details::DTypeList_to_DTypeEnum_t<DTypeFuncs::collect_all_types<DTypeFuncs::details::DTypeEnum_to_DTypeList<AllTypesL>::type, DTypeFuncs::is_dtype_##name, DTypeFuncs::details::DTypeList<> >::type>;

#define NT_FINISH_DTYPE_REGISTER()\
namespace nt{\
namespace DTypeFuncs::details{\
static constexpr std::size_t total_registered_elements = C.next<__COUNTER__>();\
inline constexpr decltype(auto) getRegisteredDTypeList(){\
    return getRegisteredDTypeListImpl(std::make_index_sequence<total_registered_elements>{});\
}\
}\
using AllTypesL = typename DTypeFuncs::details::DTypeList_to_DTypeEnum<decltype(DTypeFuncs::details::getRegisteredDTypeList())>::type;\
NT_MAKE_TYPE_ITERATOR_(Complex, complex)\
NT_MAKE_TYPE_ITERATOR_(Floating, floating)\
NT_MAKE_TYPE_ITERATOR_(Integer, integer)\
NT_MAKE_TYPE_ITERATOR_(Signed, signed)\
NT_MAKE_TYPE_ITERATOR_(Unsigned, unsigned)\
using AllTypesNBoolL = DTypeFuncs::details::DTypeList_to_DTypeEnum_t<DTypeFuncs::collect_all_types<DTypeFuncs::details::DTypeEnum_to_DTypeList<AllTypesL>::type, DTypeFuncs::is_not_dtype_bool, DTypeFuncs::details::DTypeList<> >::type>;\
NT_MAKE_TYPE_ITERATOR_(Number, num)\
NT_MAKE_TYPE_ITERATOR_(RealNumber, real_num)\
namespace DTypeFuncs{\
namespace details{\
template<DType dt, std::size_t max>\
struct next_registered_dtype{\
    static_assert(registered_order_to_dtype<registered_order_to_dtype_num<dt>::value>::value == dt, "Error, got unregistered dtype");\
    static constexpr std::size_t current_num = registered_order_to_dtype_num<dt>::value;\
    static constexpr DType value = (current_num == (max-1)) ? registered_order_to_dtype<0>::value : registered_order_to_dtype<current_num+1>::value;\
};\
}\
template<DType dtype>\
constexpr DType next_dtype_it = details::next_registered_dtype<dtype, details::total_registered_elements>::value;\
NT_MAKE_CONVERTIBLE_(Complex, complex)\
NT_MAKE_CONVERTIBLE_(Floating, floating)\
NT_MAKE_CONVERTIBLE_(Integer, integer)\
NT_MAKE_CONVERTIBLE_(Signed, signed)\
NT_MAKE_CONVERTIBLE_(Unsigned, unsigned)\
}}



#define NT_CHECK_COMPLEX_CONVERT_(dtype_enum) \
static_assert( \
	(size_of_dtype_c<dtype_enum> == size_of_dtype_c<DType::Complex32> || size_of_dtype_c<dtype_enum> == size_of_dtype_c<DType::Complex64> || size_of_dtype_c<dtype_enum> == size_of_dtype_c<DType::Complex128>)\
	? is_convertible_to_complex<dtype_enum> : !is_convertible_to_complex<dtype_enum>, "Expected to be able to convert "#dtype_enum" to a complex number");\
static_assert( \
	is_convertible_to_complex<dtype_enum> \
	? convert_to_complex<dtype_enum> != DType::Bool && size_of_dtype_c<convert_to_complex<dtype_enum>> == size_of_dtype_c<dtype_enum> \
	: convert_to_complex<dtype_enum> == DType::Bool, "Expected to get proper complex number from convert on "#dtype_enum" but did not");

#define NT_CHECK_FLOATING_CONVERT_(dtype_enum) \
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



#define NT_CHECK_INTEGER_CONVERT_(dtype_enum) \
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

#define NT_CHECK_UNSIGNED_CONVERT_(dtype_enum) \
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



#define NT_CHECK_SIGNED_(dtype_enum, type) \
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
NT_CHECK_COMPLEX_CONVERT_(dtype_enum)\
NT_CHECK_FLOATING_CONVERT_(dtype_enum)\
NT_CHECK_UNSIGNED_CONVERT_(dtype_enum)\
NT_CHECK_INTEGER_CONVERT_(dtype_enum)



#define NT_CHECK_UNSIGNED_(dtype_enum, type) \
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
NT_CHECK_COMPLEX_CONVERT_(dtype_enum)\
NT_CHECK_FLOATING_CONVERT_(dtype_enum)\
NT_CHECK_UNSIGNED_CONVERT_(dtype_enum)\
NT_CHECK_INTEGER_CONVERT_(dtype_enum)

#define NT_CHECK_COMPLEX_(dtype_enum, type) \
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
NT_CHECK_COMPLEX_CONVERT_(dtype_enum)\
NT_CHECK_FLOATING_CONVERT_(dtype_enum)\
NT_CHECK_UNSIGNED_CONVERT_(dtype_enum)\
NT_CHECK_INTEGER_CONVERT_(dtype_enum)


#define NT_CHECK_FLOATING_(dtype_enum, type) \
static_assert(!is_dtype_other_v<dtype_enum>, "Expected "#dtype_enum" to not be an other type");\
static_assert(!is_dtype_integer_v<dtype_enum>, "Expected "#dtype_enum" to not be an integer");\
static_assert(is_dtype_real_num_v<dtype_enum>, "Expected "#dtype_enum" to be a real number");\
static_assert(is_dtype_num_v<dtype_enum>, "Expected "#dtype_enum" to be a number");\
static_assert(is_dtype_floating_v<dtype_enum>, "Expected "#dtype_enum" to be floating");\
static_assert(!is_dtype_complex_v<dtype_enum>, "Expected "#dtype_enum" to not be complex");\
static_assert(std::is_same_v<dtype_to_type_t<dtype_enum, DType::Float32, DType::Float64, DType::Float16>, type>, #dtype_enum" does not correlate to expected type");\
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
NT_CHECK_COMPLEX_CONVERT_(dtype_enum)\
NT_CHECK_FLOATING_CONVERT_(dtype_enum)\
NT_CHECK_UNSIGNED_CONVERT_(dtype_enum)\
NT_CHECK_INTEGER_CONVERT_(dtype_enum)

#define NT_CHECK_OTHER_(dtype_enum, type) \
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





}



#endif
