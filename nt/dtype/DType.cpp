#include "DType.h"
#include "DType_enum.h"
#include "../convert/Convert.h"
#include "compatible/DType_compatible.h"
#include "compatible/DTypeDeclareMacros.h"


#include <complex>
#include <cstddef>
#include <functional>

#include <typeinfo>
#include <iostream>
#include <type_traits>
#include <utility>

#include <cstdint>

namespace nt{


std::ostream& operator<<(std::ostream &out, const uint_bool_t &data){
	out << bool(data.value);
	return out;
}


std::ostream& operator<< (std::ostream &out, DType const& data) {
    return out << toString(data);
}



namespace DTypeFuncs{


std::size_t size_of_dtype(const DType& dt){
	switch(dt){
#define X(type, dtype_enum_a, dtype_enum_b)\
        case DType::dtype_enum_a: return sizeof(type);

NT_GET_X_FLOATING_DTYPES_ 
NT_GET_X_COMPLEX_DTYPES_
NT_GET_X_SIGNED_INTEGER_DTYPES_
NT_GET_X_UNSIGNED_INTEGER_DTYPES_
NT_GET_X_OTHER_DTYPES_

#undef X
        default:
            utils::THROW_EXCEPTION(false, "unknown dtype $", dt);
            return 0;
	}
}


uint8_t dtype_int_code(const DType& dt){
	switch(dt){
#define X(type, dtype_enum_a, dtype_enum_b)\
        case DType::dtype_enum_a : return details::registered_order_to_dtype_num<DType::dtype_enum_a>::value;

NT_GET_X_FLOATING_DTYPES_ 
NT_GET_X_COMPLEX_DTYPES_
NT_GET_X_SIGNED_INTEGER_DTYPES_
NT_GET_X_UNSIGNED_INTEGER_DTYPES_
NT_GET_X_OTHER_DTYPES_

#undef X
        default:
            utils::THROW_EXCEPTION(false, "unknown dtype $", dt);
            return 18;
	}
}
DType code_int_dtype(const uint8_t& i){
	switch(i){
#define X(type, dtype_enum_a, dtype_enum_b)\
        case details::registered_order_to_dtype_num<DType::dtype_enum_a>::value : return DType::dtype_enum_a;

NT_GET_X_FLOATING_DTYPES_ 
NT_GET_X_COMPLEX_DTYPES_
NT_GET_X_SIGNED_INTEGER_DTYPES_
NT_GET_X_UNSIGNED_INTEGER_DTYPES_
NT_GET_X_OTHER_DTYPES_

#undef X
		default:
			utils::throw_exception(i == 0, "Got unexpected number code $ for code int to dtype", i);
			return DType::Integer;
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

#define X(type, dtype_enum_a, dtype_enum_b)\
template void initialize_strides<DType::dtype_enum_a>(void**, void*, const std::size_t&, const DType&);

NT_GET_X_FLOATING_DTYPES_ 
NT_GET_X_COMPLEX_DTYPES_
NT_GET_X_SIGNED_INTEGER_DTYPES_
NT_GET_X_UNSIGNED_INTEGER_DTYPES_
NT_GET_X_OTHER_DTYPES_

#undef X


bool is_unsigned(const DType& dt){
    switch(dt){
#define X(type, dtype_enum_a, dtype_enum_b)\
        case DType::dtype_enum_a: return true;
NT_GET_X_UNSIGNED_INTEGER_DTYPES_
#undef X
        default: return false;
    }
}

bool is_integer(const DType& dt){
    switch(dt){
#define X(type, dtype_enum_a, dtype_enum_b)\
        case DType::dtype_enum_a: return true;
NT_GET_X_SIGNED_INTEGER_DTYPES_
NT_GET_X_UNSIGNED_INTEGER_DTYPES_
#undef X
        default: return false;
    }
}

bool is_floating(const DType& dt){
    switch(dt){
#define X(type, dtype_enum_a, dtype_enum_b)\
        case DType::dtype_enum_a: return true;
NT_GET_X_FLOATING_DTYPES_
#undef X
        default: return false;
    }
}
bool is_complex(const DType& dt){
    switch(dt){
#define X(type, dtype_enum_a, dtype_enum_b)\
        case DType::dtype_enum_a: return true;
NT_GET_X_COMPLEX_DTYPES_
#undef X
        default: return false;
    }

}


DType complex_size(const std::size_t& s){
	switch(s){
#define X(type, dtype_enum_a, dtype_enum_b)\
        case sizeof(type) : return DType::dtype_enum_a;
NT_GET_X_COMPLEX_DTYPES_
#undef X
		default:
			return DType::Bool;
	}
}


DType floating_size(const std::size_t& s){
	switch(s){
#define X(type, dtype_enum_a, dtype_enum_b)\
        case sizeof(type) : return DType::dtype_enum_a;
NT_GET_X_FLOATING_DTYPES_
#undef X
		default:
			return DType::Bool;
	}
}

DType integer_size(const std::size_t& s){
	switch(s){
#define X(type, dtype_enum_a, dtype_enum_b)\
        case sizeof(type) : return DType::dtype_enum_a;
NT_GET_X_SIGNED_INTEGER_DTYPES_
#undef X
		default:
			return DType::Bool;
	}
}


DType unsigned_size(const std::size_t& s){
	switch(s){
#define X(type, dtype_enum_a, dtype_enum_b)\
        case sizeof(type) : return DType::dtype_enum_a;
NT_GET_X_UNSIGNED_INTEGER_DTYPES_
#undef X
		default:
			return DType::Bool;
	}
}


}

}
