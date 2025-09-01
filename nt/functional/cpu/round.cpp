#include "../../mp/simde_traits.h"
#include "../../mp/simde_traits/simde_traits_iterators.h"
#include "round.h"
#include "../../dtype/ArrayVoid.h"
#include "../../dtype/Scalar.h"
#include "../../dtype/DType.h"
#include "../../refs/SizeRef.h"
#include "../../dtype/ArrayVoid_NTensor.hpp"
#include "../../convert/Convert.h"
#include <cmath>

#include "../../types/float128.h"
#include "../../utils/always_inline_macro.h"

//if this is defined
//this means that for float128_t boost's 128 bit floating point is used
#ifdef BOOST_MP_STANDALONE
namespace std{
NT_ALWAYS_INLINE ::nt::float128_t round(const ::nt::float128_t& x){
    ::nt::float128_t int_part = trunc(x);  // integer part (toward zero)
    ::nt::float128_t frac = x - int_part;  // fractional part

    if (x >= 0) {
        return frac < 0.5 ? int_part : int_part + 1;
    } else {
        return frac > -0.5 ? int_part : int_part - 1;
    }
}


NT_ALWAYS_INLINE ::nt::float128_t trunc(const ::nt::float128_t& x) {
    // Remove the fractional part by casting to integer type
    // then back to float128
    if (x >= 0)
        return ::nt::float128_t(static_cast<int128_t>(x));
    else
        return ::nt::float128_t(static_cast<int128_t>(x));
}

NT_ALWAYS_INLINE ::nt::float128_t floor(const ::nt::float128_t& x) {
    ::nt::float128_t int_part = trunc(x);
    if (x < int_part) {
        return int_part - 1;
    }
    return int_part;
}

NT_ALWAYS_INLINE ::nt::float128_t ceil(const ::nt::float128_t& x) {
    ::nt::float128_t int_part = trunc(x);
    if (x > int_part) {
        return int_part + 1;
    }
    return int_part;
}

}

#endif //BOOST_MP_STANDALONE

#ifdef SIMDE_FLOAT16_IS_SCALAR
namespace std{

NT_ALWAYS_INLINE ::nt::float16_t floor(const ::nt::float16_t& x) {
    return _NT_FLOAT16_TO_FLOAT32_(std::floor(_NT_FLOAT16_TO_FLOAT32_(x)));
}

NT_ALWAYS_INLINE ::nt::float16_t ceil(const ::nt::float16_t& x) {
    return _NT_FLOAT16_TO_FLOAT32_(std::ceil(_NT_FLOAT16_TO_FLOAT32_(x)));
}

NT_ALWAYS_INLINE ::nt::float16_t trunc(const ::nt::float16_t& x) {
    return _NT_FLOAT16_TO_FLOAT32_(std::trunc(_NT_FLOAT16_TO_FLOAT32_(x)));
}

NT_ALWAYS_INLINE ::nt::float16_t round(const ::nt::float16_t& x) {
    return _NT_FLOAT16_TO_FLOAT32_(std::round(_NT_FLOAT16_TO_FLOAT32_(x)));
}

}
#endif

namespace std{

NT_ALWAYS_INLINE ::nt::complex_32 floor(const ::nt::complex_32& x){
    return ::nt::complex_32(floor(x.real()), floor(x.imag()));
}

NT_ALWAYS_INLINE ::nt::complex_64 floor(const ::nt::complex_64& x){
    return ::nt::complex_64(floor(x.real()), floor(x.imag()));
}

NT_ALWAYS_INLINE ::nt::complex_128 floor(const ::nt::complex_128& x){
    return ::nt::complex_128(floor(x.real()), floor(x.imag()));
}

NT_ALWAYS_INLINE ::nt::complex_32 round(const ::nt::complex_32& x){
    return ::nt::complex_32(round(x.real()), round(x.imag()));
}

NT_ALWAYS_INLINE ::nt::complex_64 round(const ::nt::complex_64& x){
    return ::nt::complex_64(round(x.real()), round(x.imag()));
}

NT_ALWAYS_INLINE ::nt::complex_128 round(const ::nt::complex_128& x){
    return ::nt::complex_128(round(x.real()), round(x.imag()));
}


NT_ALWAYS_INLINE ::nt::complex_32 ceil(const ::nt::complex_32& x){
    return ::nt::complex_32(ceil(x.real()), ceil(x.imag()));
}

NT_ALWAYS_INLINE ::nt::complex_64 ceil(const ::nt::complex_64& x){
    return ::nt::complex_64(ceil(x.real()), ceil(x.imag()));
}

NT_ALWAYS_INLINE ::nt::complex_128 ceil(const ::nt::complex_128& x){
    return ::nt::complex_128(ceil(x.real()), ceil(x.imag()));
}


NT_ALWAYS_INLINE ::nt::complex_32 trunc(const ::nt::complex_32& x){
    return ::nt::complex_32(trunc(x.real()), trunc(x.imag()));
}

NT_ALWAYS_INLINE ::nt::complex_64 trunc(const ::nt::complex_64& x){
    return ::nt::complex_64(trunc(x.real()), trunc(x.imag()));
}

NT_ALWAYS_INLINE ::nt::complex_128 trunc(const ::nt::complex_128& x){
    return ::nt::complex_128(trunc(x.real()), trunc(x.imag()));
}

}

namespace nt::mp{

template<typename T, typename U>
inline void round(T begin, T end, U out){
	static_assert(std::is_same_v<utils::IteratorBaseType_t<T>, utils::IteratorBaseType_t<U> >, "Expected to get base types the same for simde optimized routes");
	using base_type = utils::IteratorBaseType_t<T>;
	if constexpr (simde_svml_supported_v<base_type>){
		static constexpr size_t pack_size = pack_size_v<base_type>;
		for(;begin + pack_size <= end; begin += pack_size, out += pack_size){
			simde_type<base_type> current = it_loadu(begin);
                                  current = SimdTraits<base_type>::round(current, SIMDE_MM_FROUND_TO_NEAREST_INT);
			it_storeu(out, current);
		}
		for(;begin < end; ++begin, ++out){
            *out = std::round(*begin);
        }
	}else{
        for(;begin != end; ++begin, ++out){
            *out = std::round(*begin);
        }
	}
}


template<typename T, typename U>
inline void round_decimal(T begin, T end, U out, int64_t decimals){
	static_assert(std::is_same_v<utils::IteratorBaseType_t<T>, utils::IteratorBaseType_t<U> >, "Expected to get base types the same for simde optimized routes");
	using base_type = utils::IteratorBaseType_t<T>;
    base_type ten = base_type(10);
    base_type mult = std::pow(ten, decimals);
	if constexpr (simde_svml_supported_v<base_type>){
		static constexpr size_t pack_size = pack_size_v<base_type>;
        simde_type<base_type> val = SimdTraits<base_type>::set1(mult);

		for(;begin + pack_size <= end; begin += pack_size, out += pack_size){
			simde_type<base_type> current = it_loadu(begin);
                                  current = SimdTraits<base_type>::multiply(current, val);
                                  current = SimdTraits<base_type>::round(current, SIMDE_MM_FROUND_TO_NEAREST_INT);
                                  current = SimdTraits<base_type>::divide(current, val);
			it_storeu(out, current);
		}
		for(;begin < end; ++begin, ++out){
            *out = std::round(*begin * mult) / mult;
        }
	}else{
        for(;begin != end; ++begin, ++out){
            *out = std::round(*begin * mult) / mult;
        }
	}
}

template<typename T, typename U>
inline void truncate(T begin, T end, U out){
	static_assert(std::is_same_v<utils::IteratorBaseType_t<T>, utils::IteratorBaseType_t<U> >, "Expected to get base types the same for simde optimized routes");
	using base_type = utils::IteratorBaseType_t<T>;
	if constexpr (simde_svml_supported_v<base_type>){
		static constexpr size_t pack_size = pack_size_v<base_type>;
		for(;begin + pack_size <= end; begin += pack_size, out += pack_size){
			simde_type<base_type> current = it_loadu(begin);
                                  current = SimdTraits<base_type>::round(current, SIMDE_MM_FROUND_TO_ZERO);
			it_storeu(out, current);
		}
		for(;begin < end; ++begin, ++out){
            *out = std::trunc(*begin);
        }
	}else{
        for(;begin != end; ++begin, ++out){
            *out = std::trunc(*begin);
        }
	}
}


template<typename T, typename U>
inline void floor(T begin, T end, U out){
	static_assert(std::is_same_v<utils::IteratorBaseType_t<T>, utils::IteratorBaseType_t<U> >, "Expected to get base types the same for simde optimized routes");
	using base_type = utils::IteratorBaseType_t<T>;
	if constexpr (simde_svml_supported_v<base_type>){
		static constexpr size_t pack_size = pack_size_v<base_type>;
		for(;begin + pack_size <= end; begin += pack_size, out += pack_size){
			simde_type<base_type> current = it_loadu(begin);
                                  current = SimdTraits<base_type>::floor(current);
			it_storeu(out, current);
		}
		for(;begin < end; ++begin, ++out){
            *out = std::floor(*begin);
        }
	}else{
        for(;begin != end; ++begin, ++out){
            *out = std::floor(*begin);
        }
	}
}

template<typename T, typename U>
inline void ceil(T begin, T end, U out){
	static_assert(std::is_same_v<utils::IteratorBaseType_t<T>, utils::IteratorBaseType_t<U> >, "Expected to get base types the same for simde optimized routes");
	using base_type = utils::IteratorBaseType_t<T>;
	if constexpr (simde_svml_supported_v<base_type>){
		static constexpr size_t pack_size = pack_size_v<base_type>;
		for(;begin + pack_size <= end; begin += pack_size, out += pack_size){
			simde_type<base_type> current = it_loadu(begin);
                                  current = SimdTraits<base_type>::ceil(current);
			it_storeu(out, current);
		}
		for(;begin < end; ++begin, ++out){
            *out = std::ceil(*begin);
        }
	}else{
        for(;begin != end; ++begin, ++out){
            *out = std::ceil(*begin);
        }
	}
}


}

namespace nt{
namespace functional{
namespace cpu{


void _round(const ArrayVoid& a, ArrayVoid& out){
    if(!(DTypeFuncs::is_floating(a.dtype()) || DTypeFuncs::is_complex(a.dtype()))){
        throw std::logic_error("Cannot run round on non-floating data type");
    }
    out = a.clone();
    out.execute_function_chunk<WRAP_DTYPES<FloatingTypesL, ComplexTypesL> >([](auto begin, auto end){
        mp::round(begin, end, begin);
    });
}

void _round_decimal(const ArrayVoid& a, ArrayVoid& out, int64_t decimal){
    if(!(DTypeFuncs::is_floating(a.dtype()) || DTypeFuncs::is_complex(a.dtype()))){
        throw std::logic_error("Cannot run round on non-floating data type");
    }
    out = a.clone();
    out.execute_function_chunk<WRAP_DTYPES<FloatingTypesL, ComplexTypesL> >([&decimal](auto begin, auto end){
        mp::round_decimal(begin, end, begin, decimal);
    });
}

void _trunc(const ArrayVoid& a, ArrayVoid& out){
    if(!(DTypeFuncs::is_floating(a.dtype()) || DTypeFuncs::is_complex(a.dtype()))){
        throw std::logic_error("Cannot run trunc on non-floating data type");
    }
    out = a.clone();
    out.execute_function_chunk<WRAP_DTYPES<FloatingTypesL, ComplexTypesL> >([](auto begin, auto end){
        mp::truncate(begin, end, begin);
    });
}

void _floor(const ArrayVoid& a, ArrayVoid& out){
    if(!(DTypeFuncs::is_floating(a.dtype()) || DTypeFuncs::is_complex(a.dtype()))){
        throw std::logic_error("Cannot run floor on non-floating data type");
    }
    out = a.clone();
    out.execute_function_chunk<WRAP_DTYPES<FloatingTypesL, ComplexTypesL> >([](auto begin, auto end){
        mp::floor(begin, end, begin);
    });
}

void _ceil(const ArrayVoid& a, ArrayVoid& out){
    if(!(DTypeFuncs::is_floating(a.dtype()) || DTypeFuncs::is_complex(a.dtype()))){
        throw std::logic_error("Cannot run ceil on non-floating data type");
    }
    out = a.clone();
    out.execute_function_chunk<WRAP_DTYPES<FloatingTypesL, ComplexTypesL> >([](auto begin, auto end){
        mp::ceil(begin, end, begin);
    });
}


}
}
}
