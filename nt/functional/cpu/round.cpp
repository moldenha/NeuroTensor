#include "../../mp/simde_traits.h"
#include "../../mp/simde_traits/simde_traits_iterators.h"
#include "round.h"
#include "../../dtype/ArrayVoid.h"
#include "../../dtype/Scalar.h"
#include "../../dtype/DType.h"
#include "../../refs/SizeRef.h"
#include "../../dtype/ArrayVoid_NTensor.hpp"
#include "../../convert/Convert.h"
#include "../../math/math.h"
#include <cmath>

#include "../../types/float128.h"
#include "../../utils/always_inline_macro.h"



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
            *out = ::nt::math::round(*begin);
        }
	}else{
        for(;begin != end; ++begin, ++out){
            *out = ::nt::math::round(*begin);
        }
	}
}


template<typename T, typename U>
inline void round_decimal(T begin, T end, U out, int64_t decimals){
	static_assert(std::is_same_v<utils::IteratorBaseType_t<T>, utils::IteratorBaseType_t<U> >, "Expected to get base types the same for simde optimized routes");
	using base_type = utils::IteratorBaseType_t<T>;
    base_type ten = base_type(10);
    base_type mult = ::nt::math::pow(ten, decimals);
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
            *out = ::nt::math::round(*begin * mult) / mult;
        }
	}else{
        for(;begin != end; ++begin, ++out){
            *out = ::nt::math::round(*begin * mult) / mult;
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
            *out = ::nt::math::trunc(*begin);
        }
	}else{
        for(;begin != end; ++begin, ++out){
            *out = ::nt::math::trunc(*begin);
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
            *out = ::nt::math::floor(*begin);
        }
	}else{
        for(;begin != end; ++begin, ++out){
            *out = ::nt::math::floor(*begin);
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
            *out = ::nt::math::ceil(*begin);
        }
	}else{
        for(;begin != end; ++begin, ++out){
            *out = ::nt::math::ceil(*begin);
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
