#include "DType.h"
#include "DType_enum.h"
#include "../convert/Convert.h"
#include "compatible/DType_compatible.h"
#include <_types/_uint32_t.h>
#include <_types/_uint8_t.h>
#include <complex>
#include <functional>
#include <sys/_types/_int64_t.h>
#include <typeinfo>
#include <iostream>
#include <type_traits>
#include <utility>

namespace nt{

/* uint_bool_t::uint_bool_t():value(0){} */

/* uint_bool_t::uint_bool_t(const bool &val) */
/* 	:value(val ? 1 : 0) */
/* {} */

/* uint_bool_t::uint_bool_t(const uint_bool_t &inp) */
/* 	:value(inp.value) */
/* {} */

/* uint_bool_t::uint_bool_t(uint_bool_t&& inp) */
/* 	:value(inp.value){} */

/* uint_bool_t& uint_bool_t::operator=(const bool &val){value = val ? 1 : 0; return *this;} */
/* uint_bool_t& uint_bool_t::operator=(const uint8_t &val){value = val > 0 ? 1 : 0; return *this;} */
/* uint_bool_t& uint_bool_t::operator=(const uint_bool_t &val){value = val.value; return *this;} */
/* uint_bool_t& uint_bool_t::operator=(uint_bool_t&& val){value = val.value; return *this;} */
/* bool uint_bool_t::operator==(const uint_bool_t& val) const {return value == val.value;} */

std::ostream& operator<<(std::ostream &out, const uint_bool_t &data){
	out << bool(data.value);
	return out;
}


std::ostream& operator<< (std::ostream &out, DType const& data) {
	switch(data){
		case DType::Float:
		      out << "DTypeFloat32";
		      break;
		case DType::Double:
		      out << "DTypeFloat64";
		      break;
		case DType::Complex64:
		      out << "DTypeComplex64";
		      break;
		case DType::Complex128:
		      out << "DTypeComplex128";
		      break;
		case DType::Byte:
		      out << "DTypeByteSigned8";
		      break;
		case DType::Short:
			out << "DTypeShortSigned16";
			break;
		case DType::UnsignedShort:
			out << "DTypeUnsignedShort16";
			break;
		case DType::Long:
		      out << "DTypeUnsignedLong32";
		      break;
		case DType::Integer:
		      out << "DTypeInteger32";
		      break;
		case DType::LongLong:
		      out << "DTypeInteger64";
		      break;
		case DType::Bool:
		      out << "DTypeBool";
		      break;
		case DType::TensorObj:
		      out << "DTypeTensor";
		      break;
#ifdef __SIZEOF_INT128__
		case DType::int128:
		      out << "DTypeInt128";
		      break;
		case DType::uint128:
		      out << "DTypeUnsignedInt128";
		      break;
#endif
#ifdef _HALF_FLOAT_SUPPORT_
		case DType::Float16:
		      out << "DTypeFloat16";
		      break;
		case DType::Complex32:
		      out << "DTypeComplex32";
		      break;
#endif
#ifdef _128_FLOAT_SUPPORT_
		case DType::Float128:
		      out << "DTypeFloat128";
		      break;
#endif
		default:
		      out << "UnknownType";
		      break;
	}
	return out;
}



namespace DTypeFuncs{

template<DType dt>
std::size_t size_of_dtype_p(const DType& d){
	if(d == dt){
		using value_t = dtype_to_type_t<dt>;
		value_t* ptr = nullptr;
		return sizeof(ptr);
	}
	return size_of_dtype_p<next_dtype_it<dt> >(d);
}

template std::size_t size_of_dtype_p<DType::Float>(const DType&);
template std::size_t size_of_dtype_p<DType::Double>(const DType&);
template std::size_t size_of_dtype_p<DType::Complex64>(const DType&);
template std::size_t size_of_dtype_p<DType::Complex128>(const DType&);
template std::size_t size_of_dtype_p<DType::uint8>(const DType&);
template std::size_t size_of_dtype_p<DType::int8>(const DType&);
template std::size_t size_of_dtype_p<DType::int16>(const DType&);
template std::size_t size_of_dtype_p<DType::uint16>(const DType&);
template std::size_t size_of_dtype_p<DType::Integer>(const DType&);
template std::size_t size_of_dtype_p<DType::Long>(const DType&);
template std::size_t size_of_dtype_p<DType::int64>(const DType&);
template std::size_t size_of_dtype_p<DType::Bool>(const DType&);
template std::size_t size_of_dtype_p<DType::TensorObj>(const DType&);
#ifdef _HALF_FLOAT_SUPPORT_
template std::size_t size_of_dtype_p<DType::Float16>(const DType&);
template std::size_t size_of_dtype_p<DType::Complex32>(const DType&);
#endif
#ifdef _128_FLOAT_SUPPORT_
template std::size_t size_of_dtype_p<DType::Float128>(const DType&);
#endif
#ifdef __SIZEOF_INT128__
template std::size_t size_of_dtype_p<DType::int128>(const DType&);
template std::size_t size_of_dtype_p<DType::uint128>(const DType&);
#endif

template<DType dt>
bool is_in(const DType inp){
	return inp == dt;
}
/*
enum_values = ['Float', 'Double', 'Complex64', 'Complex128', 'uint8', 'int8', 'int16', 'uint16', 'Integer', 'Long', 'int64', 'Bool', 'TensorObj']
half_enum_values = ['Float16', 'Complex32']
f_128_enum_values = ['Float128']
i_128_enum_values = ['int128', 'uint128']

def print_enum_special(evs, halfs, f_128s, i_128s, start, end):
	part_a = ''.join(start + 'DType::'+x+end + '\n' for x in evs)
	part_b = part_a + '#ifdef _HALF_FLOAT_SUPPORT_\n'
	part_c = part_b + ''.join(start + 'DType::'+x+end + '\n' for x in halfs)
	part_d = part_c + '#endif\n#ifdef _128_FLOAT_SUPPORT_\n'
	part_e = part_d + ''.join(start + 'DType::'+x+end + '\n' for x in f_128s)
	part_f = part_e + '#endif\n#ifdef __SIZEOF_INT128__\n'
	part_g = part_f + ''.join(start + 'DType::'+x+end + '\n' for x in i_128s) 
	print(part_g + '#endif')

print_enum_special(enum_values,  half_enum_values,  f_128_enum_values, i_128_enum_values

print_enum_special(enum_values,  half_enum_values,  f_128_enum_values, i_128_enum_values, 'template bool is_in<', '>(const DType);')

 */

template bool is_in<DType::Float>(const DType);
template bool is_in<DType::Double>(const DType);
template bool is_in<DType::Complex64>(const DType);
template bool is_in<DType::Complex128>(const DType);
template bool is_in<DType::uint8>(const DType);
template bool is_in<DType::int8>(const DType);
template bool is_in<DType::int16>(const DType);
template bool is_in<DType::uint16>(const DType);
template bool is_in<DType::Integer>(const DType);
template bool is_in<DType::Long>(const DType);
template bool is_in<DType::int64>(const DType);
template bool is_in<DType::Bool>(const DType);
template bool is_in<DType::TensorObj>(const DType);
#ifdef _HALF_FLOAT_SUPPORT_
template bool is_in<DType::Float16>(const DType);
template bool is_in<DType::Complex32>(const DType);
#endif
#ifdef _128_FLOAT_SUPPORT_
template bool is_in<DType::Float128>(const DType);
#endif
#ifdef __SIZEOF_INT128__
template bool is_in<DType::int128>(const DType);
template bool is_in<DType::uint128>(const DType);
#endif




template<class T>
void is_same(DType a, bool& outp, T b){
	if constexpr(std::is_same_v<T, DType>){
		if(a == b) outp = true;
	}
}
template void is_same<DType>(DType a, bool& outp, DType b);

/*
enum_values = ['Float', 'Double', 'Complex64', 'Complex128', 'uint8', 'int8', 'int16', 'uint16', 'Integer', 'Long', 'int64', 'Bool', 'TensorObj']
half_enum_values = ['Float16', 'Complex32']
f_128_enum_values = ['Float128']
i_128_enum_values = ['int128', 'uint128']

#this time it needs to do <DType A, DType B>
def print_enum_special_2(evs, halfs, f_128s, i_128s, start, end):
	part_a = ''
	for ev in evs:
		part_a = part_a + ''.join(start + 'DType::'+ev+',DType::'+x+end + '\n' for x in evs)
	part_a = part_a + '#ifdef _HALF_FLOAT_SUPPORT_\n'
	for half in halfs:
		part_a = part_a + ''.join(start + 'DType::'+half+',DType::'+x+end + '\n' for x in halfs)
		part_a = part_a + ''.join(start + 'DType::'+half+',DType::'+x+end + '\n' for x in evs)
	part_a = part_a + '#ifdef _128_FLOAT_SUPPORT_\n'
	for half in halfs:
		part_a = part_a + ''.join(start + 'DType::'+half+',DType::'+x+end + '\n' for x in f_128s)
	part_a = part_a + '#endif\n#ifdef __SIZEOF_INT128__\n'
	for half in halfs:
		part_a = part_a + ''.join(start + 'DType::'+half+',DType::'+x+end + '\n' for x in i_128s)
	part_a = part_a + '#endif\n#endif\n#ifdef _128_FLOAT_SUPPORT_'
	for f128 in f_128s:
		part_a = part_a + ''.join(start + 'DType::'+f128+',DType::'+x+end + '\n' for x in f_128s)
		part_a = part_a + ''.join(start + 'DType::'+f128+',DType::'+x+end + '\n' for x in evs)
	part_a = part_a + '#ifdef _HALF_FLOAT_SUPPORT_\n'
	for f128 in f_128s:
		part_a = part_a + ''.join(start + 'DType::'+f128+',DType::'+x+end + '\n' for x in halfs)
	part_a = part_a + '#endif\n#ifdef __SIZEOF_INT128__\n'
	for f128 in f_128s:
		part_a = part_a + ''.join(start + 'DType::'+f128+',DType::'+x+end + '\n' for x in i_128s)
	part_a = part_a + '#endif\n#endif\n#ifdef __SIZEOF_INT128__'
	for i128 in i_128s:
		part_a = part_a + ''.join(start + 'DType::'+i128+',DType::'+x+end + '\n' for x in i_128s)
		part_a = part_a + ''.join(start + 'DType::'+i128+',DType::'+x+end + '\n' for x in evs)
	part_a = part_a + '#ifdef _128_FLOAT_SUPPORT_\n'
	for i128 in i_128s:
		part_a = part_a + ''.join(start + 'DType::'+i128+',DType::'+x+end + '\n' for x in f_128s)
	part_a = part_a + '#endif\n#ifdef _HALF_FLOAT_SUPPORT_\n'
	for i128 in i_128s:
		part_a = part_a + ''.join(start + 'DType::'+i128+',DType::'+x+end + '\n' for x in halfs)
	part_a = part_a + '#endif\n#endif'
	print(part_a)

print_enum_special(enum_values,  half_enum_values,  f_128_enum_values, i_128_enum_values

print_enum_special(enum_values,  half_enum_values,  f_128_enum_values, i_128_enum_values, 'template bool is_in<', '>(const DType);')
print_enum_special_2(enum_values,  half_enum_values,  f_128_enum_values, i_128_enum_values, 'template bool convert_this_typed_array<', '>(void*, const DType&, const DType&, const std::size_t&);')
 */


template<DType F, DType T>
bool convert_this_typed_array(void* arr, const DType& from, const DType& to, const std::size_t& total){
	if(F != from){return convert_this_typed_array<next_dtype_it<F>, T>(arr, from, to, total);}
	if(T != to){return convert_this_typed_array<F, next_dtype_it<T>>(arr, from, to, total);}
	dtype_to_type_t<F>* it = reinterpret_cast<dtype_to_type_t<F>*>(arr);
	dtype_to_type_t<T>* start = reinterpret_cast<dtype_to_type_t<T>*>(arr);
	for(uint32_t i = 0; i < total; ++i, ++it, ++start){
		*start = convert::convert<T>(*it); 
	}
	return true;
}

template bool convert_this_typed_array<DType::Float,DType::Float>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Float,DType::Double>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Float,DType::Complex64>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Float,DType::Complex128>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Float,DType::uint8>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Float,DType::int8>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Float,DType::int16>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Float,DType::uint16>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Float,DType::Integer>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Float,DType::Long>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Float,DType::int64>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Float,DType::Bool>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Float,DType::TensorObj>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Double,DType::Float>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Double,DType::Double>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Double,DType::Complex64>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Double,DType::Complex128>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Double,DType::uint8>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Double,DType::int8>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Double,DType::int16>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Double,DType::uint16>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Double,DType::Integer>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Double,DType::Long>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Double,DType::int64>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Double,DType::Bool>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Double,DType::TensorObj>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Complex64,DType::Float>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Complex64,DType::Double>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Complex64,DType::Complex64>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Complex64,DType::Complex128>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Complex64,DType::uint8>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Complex64,DType::int8>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Complex64,DType::int16>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Complex64,DType::uint16>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Complex64,DType::Integer>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Complex64,DType::Long>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Complex64,DType::int64>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Complex64,DType::Bool>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Complex64,DType::TensorObj>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Complex128,DType::Float>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Complex128,DType::Double>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Complex128,DType::Complex64>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Complex128,DType::Complex128>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Complex128,DType::uint8>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Complex128,DType::int8>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Complex128,DType::int16>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Complex128,DType::uint16>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Complex128,DType::Integer>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Complex128,DType::Long>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Complex128,DType::int64>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Complex128,DType::Bool>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Complex128,DType::TensorObj>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::uint8,DType::Float>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::uint8,DType::Double>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::uint8,DType::Complex64>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::uint8,DType::Complex128>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::uint8,DType::uint8>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::uint8,DType::int8>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::uint8,DType::int16>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::uint8,DType::uint16>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::uint8,DType::Integer>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::uint8,DType::Long>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::uint8,DType::int64>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::uint8,DType::Bool>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::uint8,DType::TensorObj>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::int8,DType::Float>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::int8,DType::Double>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::int8,DType::Complex64>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::int8,DType::Complex128>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::int8,DType::uint8>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::int8,DType::int8>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::int8,DType::int16>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::int8,DType::uint16>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::int8,DType::Integer>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::int8,DType::Long>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::int8,DType::int64>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::int8,DType::Bool>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::int8,DType::TensorObj>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::int16,DType::Float>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::int16,DType::Double>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::int16,DType::Complex64>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::int16,DType::Complex128>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::int16,DType::uint8>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::int16,DType::int8>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::int16,DType::int16>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::int16,DType::uint16>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::int16,DType::Integer>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::int16,DType::Long>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::int16,DType::int64>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::int16,DType::Bool>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::int16,DType::TensorObj>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::uint16,DType::Float>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::uint16,DType::Double>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::uint16,DType::Complex64>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::uint16,DType::Complex128>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::uint16,DType::uint8>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::uint16,DType::int8>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::uint16,DType::int16>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::uint16,DType::uint16>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::uint16,DType::Integer>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::uint16,DType::Long>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::uint16,DType::int64>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::uint16,DType::Bool>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::uint16,DType::TensorObj>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Integer,DType::Float>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Integer,DType::Double>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Integer,DType::Complex64>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Integer,DType::Complex128>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Integer,DType::uint8>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Integer,DType::int8>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Integer,DType::int16>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Integer,DType::uint16>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Integer,DType::Integer>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Integer,DType::Long>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Integer,DType::int64>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Integer,DType::Bool>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Integer,DType::TensorObj>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Long,DType::Float>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Long,DType::Double>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Long,DType::Complex64>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Long,DType::Complex128>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Long,DType::uint8>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Long,DType::int8>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Long,DType::int16>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Long,DType::uint16>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Long,DType::Integer>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Long,DType::Long>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Long,DType::int64>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Long,DType::Bool>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Long,DType::TensorObj>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::int64,DType::Float>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::int64,DType::Double>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::int64,DType::Complex64>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::int64,DType::Complex128>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::int64,DType::uint8>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::int64,DType::int8>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::int64,DType::int16>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::int64,DType::uint16>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::int64,DType::Integer>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::int64,DType::Long>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::int64,DType::int64>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::int64,DType::Bool>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::int64,DType::TensorObj>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Bool,DType::Float>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Bool,DType::Double>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Bool,DType::Complex64>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Bool,DType::Complex128>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Bool,DType::uint8>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Bool,DType::int8>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Bool,DType::int16>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Bool,DType::uint16>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Bool,DType::Integer>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Bool,DType::Long>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Bool,DType::int64>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Bool,DType::Bool>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Bool,DType::TensorObj>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::TensorObj,DType::Float>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::TensorObj,DType::Double>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::TensorObj,DType::Complex64>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::TensorObj,DType::Complex128>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::TensorObj,DType::uint8>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::TensorObj,DType::int8>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::TensorObj,DType::int16>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::TensorObj,DType::uint16>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::TensorObj,DType::Integer>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::TensorObj,DType::Long>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::TensorObj,DType::int64>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::TensorObj,DType::Bool>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::TensorObj,DType::TensorObj>(void*, const DType&, const DType&, const std::size_t&);
#ifdef _HALF_FLOAT_SUPPORT_
template bool convert_this_typed_array<DType::Float16,DType::Float16>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Float16,DType::Complex32>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Float16,DType::Float>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Float16,DType::Double>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Float16,DType::Complex64>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Float16,DType::Complex128>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Float16,DType::uint8>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Float16,DType::int8>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Float16,DType::int16>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Float16,DType::uint16>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Float16,DType::Integer>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Float16,DType::Long>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Float16,DType::int64>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Float16,DType::Bool>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Float16,DType::TensorObj>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Complex32,DType::Float16>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Complex32,DType::Complex32>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Complex32,DType::Float>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Complex32,DType::Double>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Complex32,DType::Complex64>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Complex32,DType::Complex128>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Complex32,DType::uint8>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Complex32,DType::int8>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Complex32,DType::int16>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Complex32,DType::uint16>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Complex32,DType::Integer>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Complex32,DType::Long>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Complex32,DType::int64>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Complex32,DType::Bool>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Complex32,DType::TensorObj>(void*, const DType&, const DType&, const std::size_t&);
#ifdef _128_FLOAT_SUPPORT_
template bool convert_this_typed_array<DType::Float16,DType::Float128>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Complex32,DType::Float128>(void*, const DType&, const DType&, const std::size_t&);
#endif
#ifdef __SIZEOF_INT128__
template bool convert_this_typed_array<DType::Float16,DType::int128>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Float16,DType::uint128>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Complex32,DType::int128>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Complex32,DType::uint128>(void*, const DType&, const DType&, const std::size_t&);
#endif
#endif
#ifdef _128_FLOAT_SUPPORT_
template bool convert_this_typed_array<DType::Float128,DType::Float128>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Float128,DType::Float>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Float128,DType::Double>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Float128,DType::Complex64>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Float128,DType::Complex128>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Float128,DType::uint8>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Float128,DType::int8>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Float128,DType::int16>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Float128,DType::uint16>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Float128,DType::Integer>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Float128,DType::Long>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Float128,DType::int64>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Float128,DType::Bool>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Float128,DType::TensorObj>(void*, const DType&, const DType&, const std::size_t&);
#ifdef _HALF_FLOAT_SUPPORT_
template bool convert_this_typed_array<DType::Float128,DType::Float16>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Float128,DType::Complex32>(void*, const DType&, const DType&, const std::size_t&);
#endif
#ifdef __SIZEOF_INT128__
template bool convert_this_typed_array<DType::Float128,DType::int128>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::Float128,DType::uint128>(void*, const DType&, const DType&, const std::size_t&);
#endif
#endif
#ifdef __SIZEOF_INT128__
template bool convert_this_typed_array<DType::int128,DType::int128>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::int128,DType::uint128>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::int128,DType::Float>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::int128,DType::Double>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::int128,DType::Complex64>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::int128,DType::Complex128>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::int128,DType::uint8>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::int128,DType::int8>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::int128,DType::int16>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::int128,DType::uint16>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::int128,DType::Integer>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::int128,DType::Long>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::int128,DType::int64>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::int128,DType::Bool>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::int128,DType::TensorObj>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::uint128,DType::int128>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::uint128,DType::uint128>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::uint128,DType::Float>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::uint128,DType::Double>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::uint128,DType::Complex64>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::uint128,DType::Complex128>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::uint128,DType::uint8>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::uint128,DType::int8>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::uint128,DType::int16>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::uint128,DType::uint16>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::uint128,DType::Integer>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::uint128,DType::Long>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::uint128,DType::int64>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::uint128,DType::Bool>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::uint128,DType::TensorObj>(void*, const DType&, const DType&, const std::size_t&);
#ifdef _128_FLOAT_SUPPORT_
template bool convert_this_typed_array<DType::int128,DType::Float128>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::uint128,DType::Float128>(void*, const DType&, const DType&, const std::size_t&);
#endif
#ifdef _HALF_FLOAT_SUPPORT_
template bool convert_this_typed_array<DType::int128,DType::Float16>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::int128,DType::Complex32>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::uint128,DType::Float16>(void*, const DType&, const DType&, const std::size_t&);
template bool convert_this_typed_array<DType::uint128,DType::Complex32>(void*, const DType&, const DType&, const std::size_t&);
#endif
#endif


void convert_this_dtype_array(void* arr, const DType& from, const DType& to, const std::size_t& total){
	utils::throw_exception(can_convert(from, to), "The DType $ cannot be converted to $ without copying or making new memory, each type is not the same size", from, to);
	convert_this_typed_array<DType::Integer, DType::Integer>(arr, from, to, total);
}

std::shared_ptr<void> make_shared_array(size_t size, const DType& dt){
	switch(dt){
		case DType::Integer:
			return std::make_unique<int32_t[]>(size);
#ifdef __SIZEOF_INT128__
		case DType::int128:
			return std::make_unique<int128_t[]>(size);
		case DType::uint128:
			return std::make_unique<uint128_t[]>(size);
#endif
#ifdef _HALF_FLOAT_SUPPORT_
		case DType::Float16:
			return std::make_unique<float16_t[]>(size);
		case DType::Complex32:
			return std::make_unique<complex_32[]>(size);
#endif
#ifdef _128_FLOAT_SUPPORT_
		case DType::Float128:
			return std::make_unique<float128_t[]>(size);
#endif
		case DType::Double:
			return std::make_unique<double[]>(size);
		case DType::Float:
			return std::make_unique<float[]>(size);
		case DType::Long:
			return std::make_unique<uint32_t[]>(size);
		case DType::cfloat:
			return std::make_unique<complex_64[]>(size);
		case DType::cdouble:
			return std::make_unique<complex_128[]>(size);
		case DType::uint8:
			return std::make_unique<uint8_t[]>(size);
		case DType::int8:
			return std::make_unique<int8_t[]>(size);
		case DType::int16:
			return std::make_unique<int16_t[]>(size);
		case DType::uint16:
			return std::make_unique<uint16_t[]>(size);
		case DType::int64:
			return std::make_unique<int64_t[]>(size);
		case DType::TensorObj:
			return std::make_unique<Tensor[]>(size);
		case DType::Bool:
			return std::make_unique<uint_bool_t[]>(size);
	}
	/* if(dt == DType::TensorObj){ */
	/* 	return std::make_unique<Tensor[]>(size); */
	/* } */
	/* return std::shared_ptr<void>(calloc(size, size_of_dtype(dt)), free); */


}

template<DType dt>
std::shared_ptr<void> share_part_ptr(const uint32_t& index, const DType& m_dt, const std::shared_ptr<void>& ptr){
	if(dt != m_dt){return share_part_ptr<DTypeFuncs::next_dtype_it<dt>>(index, m_dt, ptr);}
	using value_t = dtype_to_type_t<dt>;
	const std::shared_ptr<value_t[]> *p = reinterpret_cast<const std::shared_ptr<value_t[]>*>(&ptr);
	return std::shared_ptr<value_t>((*p), &(*p)[index]);
}

template std::shared_ptr<void> share_part_ptr<DType::Float>(const uint32_t& index, const DType& m_dt, const std::shared_ptr<void>& ptr);
template std::shared_ptr<void> share_part_ptr<DType::Double>(const uint32_t& index, const DType& m_dt, const std::shared_ptr<void>& ptr);
template std::shared_ptr<void> share_part_ptr<DType::Complex64>(const uint32_t& index, const DType& m_dt, const std::shared_ptr<void>& ptr);
template std::shared_ptr<void> share_part_ptr<DType::Complex128>(const uint32_t& index, const DType& m_dt, const std::shared_ptr<void>& ptr);
template std::shared_ptr<void> share_part_ptr<DType::uint8>(const uint32_t& index, const DType& m_dt, const std::shared_ptr<void>& ptr);
template std::shared_ptr<void> share_part_ptr<DType::int8>(const uint32_t& index, const DType& m_dt, const std::shared_ptr<void>& ptr);
template std::shared_ptr<void> share_part_ptr<DType::int16>(const uint32_t& index, const DType& m_dt, const std::shared_ptr<void>& ptr);
template std::shared_ptr<void> share_part_ptr<DType::uint16>(const uint32_t& index, const DType& m_dt, const std::shared_ptr<void>& ptr);
template std::shared_ptr<void> share_part_ptr<DType::Integer>(const uint32_t& index, const DType& m_dt, const std::shared_ptr<void>& ptr);
template std::shared_ptr<void> share_part_ptr<DType::Long>(const uint32_t& index, const DType& m_dt, const std::shared_ptr<void>& ptr);
template std::shared_ptr<void> share_part_ptr<DType::int64>(const uint32_t& index, const DType& m_dt, const std::shared_ptr<void>& ptr);
template std::shared_ptr<void> share_part_ptr<DType::Bool>(const uint32_t& index, const DType& m_dt, const std::shared_ptr<void>& ptr);
template std::shared_ptr<void> share_part_ptr<DType::TensorObj>(const uint32_t& index, const DType& m_dt, const std::shared_ptr<void>& ptr);
#ifdef _HALF_FLOAT_SUPPORT_
template std::shared_ptr<void> share_part_ptr<DType::Float16>(const uint32_t& index, const DType& m_dt, const std::shared_ptr<void>& ptr);
template std::shared_ptr<void> share_part_ptr<DType::Complex32>(const uint32_t& index, const DType& m_dt, const std::shared_ptr<void>& ptr);
#endif
#ifdef _128_FLOAT_SUPPORT_
template std::shared_ptr<void> share_part_ptr<DType::Float128>(const uint32_t& index, const DType& m_dt, const std::shared_ptr<void>& ptr);
#endif
#ifdef __SIZEOF_INT128__
template std::shared_ptr<void> share_part_ptr<DType::int128>(const uint32_t& index, const DType& m_dt, const std::shared_ptr<void>& ptr);
template std::shared_ptr<void> share_part_ptr<DType::uint128>(const uint32_t& index, const DType& m_dt, const std::shared_ptr<void>& ptr);
#endif


std::size_t size_of_dtype(const DType& dt){
	switch(dt){
		case DType::Integer:
			return sizeof(int32_t);
#ifdef __SIZEOF_INT128__
		case DType::int128:
			return sizeof(int128_t);
		case DType::uint128:
			return sizeof(uint128_t);
#endif
#ifdef _HALF_FLOAT_SUPPORT_
		case DType::Float16:
			return sizeof(float16_t);
		case DType::Complex32:
			return sizeof(complex_32);
#endif
#ifdef _128_FLOAT_SUPPORT_
		case DType::Float128:
			return sizeof(float128_t);
#endif
		case DType::Double:
			return sizeof(double);
		case DType::Float:
			return sizeof(float);
		case DType::Long:
			return sizeof(uint32_t);
		case DType::cfloat:
			return sizeof(complex_64);
		case DType::cdouble:
			return sizeof(complex_128);
		case DType::uint8:
			return sizeof(uint8_t);
		case DType::int8:
			return sizeof(int8_t);
		case DType::int16:
			return sizeof(int16_t);
		case DType::uint16:
			return sizeof(uint16_t);
		case DType::int64:
			return sizeof(int64_t);
		case DType::TensorObj:
			return sizeof(Tensor);
		case DType::Bool:
			return sizeof(uint_bool_t);
	}
}

bool can_convert(const DType& from, const DType& to){
	return size_of_dtype(from) == size_of_dtype(to);
}

//print_enum_special(enum_values,  half_enum_values,  f_128_enum_values, i_128_enum_values, 'template void initialize_strides<', '>(void**, void*, const std::size_t&, const DType&);')

template<DType dt>
void initialize_strides(void** ptrs, void* cast, const std::size_t& s, const DType& ds){
	if(dt != ds){
		initialize_strides<next_dtype_it<dt> >(ptrs, cast, s, ds);
		return;
	}
	using value_t = dtype_to_type_t<dt>;
	value_t* ptr = static_cast<value_t*>(cast);
	for(uint32_t i = 0; i < s; ++i)
		ptrs[i] = &ptr[i];
}

template void initialize_strides<DType::Float>(void**, void*, const std::size_t&, const DType&);
template void initialize_strides<DType::Double>(void**, void*, const std::size_t&, const DType&);
template void initialize_strides<DType::Complex64>(void**, void*, const std::size_t&, const DType&);
template void initialize_strides<DType::Complex128>(void**, void*, const std::size_t&, const DType&);
template void initialize_strides<DType::uint8>(void**, void*, const std::size_t&, const DType&);
template void initialize_strides<DType::int8>(void**, void*, const std::size_t&, const DType&);
template void initialize_strides<DType::int16>(void**, void*, const std::size_t&, const DType&);
template void initialize_strides<DType::uint16>(void**, void*, const std::size_t&, const DType&);
template void initialize_strides<DType::Integer>(void**, void*, const std::size_t&, const DType&);
template void initialize_strides<DType::Long>(void**, void*, const std::size_t&, const DType&);
template void initialize_strides<DType::int64>(void**, void*, const std::size_t&, const DType&);
template void initialize_strides<DType::Bool>(void**, void*, const std::size_t&, const DType&);
template void initialize_strides<DType::TensorObj>(void**, void*, const std::size_t&, const DType&);
#ifdef _HALF_FLOAT_SUPPORT_
template void initialize_strides<DType::Float16>(void**, void*, const std::size_t&, const DType&);
template void initialize_strides<DType::Complex32>(void**, void*, const std::size_t&, const DType&);
#endif
#ifdef _128_FLOAT_SUPPORT_
template void initialize_strides<DType::Float128>(void**, void*, const std::size_t&, const DType&);
#endif
#ifdef __SIZEOF_INT128__
template void initialize_strides<DType::int128>(void**, void*, const std::size_t&, const DType&);
template void initialize_strides<DType::uint128>(void**, void*, const std::size_t&, const DType&);
#endif


bool is_unsigned(const DType& dt){
	return ((dt == DType::Long)
			|| (dt == DType::uint16)
			|| (dt == DType::uint8)
#ifdef __SIZEOF_INT128__
			|| (dt == DType::uint128)
#endif
			);}
bool is_integer(const DType& dt){
	return ((dt == DType::int64)
			|| (dt == DType::uint32)
			|| (dt == DType::int32)
			|| (dt == DType::uint16)
			|| (dt == DType::int16)
			|| (dt == DType::uint8)
			|| (dt == DType::int8)

#ifdef __SIZEOF_INT128__
			|| (dt == DType::uint128)
			|| (dt == DType::int128)
#endif
			);}
bool is_floating(const DType& dt){
	return ((dt == DType::Float) 
			|| (dt == DType::Double)
#ifdef _HALF_FLOAT_SUPPORT_
			|| (dt == DType::Float16) 
#endif
#ifdef _128_FLOAT_SUPPORT_
			|| (dt == DType::Float128)
#endif
			);}
bool is_complex(const DType& dt){
	return ((dt == DType::Complex64)
				|| (dt == DType::Complex128) 
#ifdef _HALF_FLOAT_SUPPORT_
				|| (dt == DType::Complex32)
#endif
			);}


DType complex_size(const std::size_t& s){
	switch(s){
		case sizeof(complex_128):
			return DType::Complex128;
		case sizeof(complex_64):
			return DType::Complex64;
#ifdef _HALF_FLOAT_SUPPORT_
		case sizeof(complex_32):
			return DType::Complex32;
#endif
		default:
			return DType::Bool;
	}
}


DType floating_size(const std::size_t& s){
	switch(s){
#ifdef _128_FLOAT_SUPPORT_
		case sizeof(float128_t):
			return DType::Float128;
#endif
		case sizeof(double):
			return DType::Float64;
		case sizeof(float):
			return DType::Float32;
#ifdef _HALF_FLOAT_SUPPORT_
		case sizeof(float16_t):
			return DType::Float16;
#endif
		default:
			return DType::Bool;
	}
}

DType integer_size(const std::size_t& s){
	switch(s){
#ifdef __SIZEOF_INT128__
		case sizeof(int128_t):
			return DType::int128;
#endif
		case sizeof(int64_t):
			return DType::int64;
		case sizeof(int32_t):
			return DType::int32;
		case sizeof(int16_t):
			return DType::int16;
		case sizeof(int8_t):
			return DType::int8;
		default:
			return DType::Bool;
	}
}


DType unsigned_size(const std::size_t& s){
		switch(s){
#ifdef __SIZEOF_INT128__
		case sizeof(uint128_t):
			return DType::uint128;
#endif
		case sizeof(uint32_t):
			return DType::uint32;
		case sizeof(uint16_t):
			return DType::uint16;
		case sizeof(uint8_t):
			return DType::uint8;
		default:
			return DType::Bool;
	}
}


}

}
