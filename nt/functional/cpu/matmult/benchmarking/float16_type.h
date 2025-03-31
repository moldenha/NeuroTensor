//replace neurotensors types header
#ifndef _NT_MY_TYPES_H_
#define _NT_MY_TYPES_H_

#include <complex>
#include <simde/simde-f16.h> //using this to convert between float32 and float16
#ifdef SIMDE_FLOAT16_IS_SCALAR
	typedef simde_float16 float16_t;
	#define _NT_FLOAT16_TO_FLOAT32_(f16) simde_float16_to_float32(f16)
	#define _NT_FLOAT32_TO_FLOAT16_(f)    simde_float16_from_float32(f)
	inline std::ostream& operator<<(std::ostream& os, const float16_t& val){
		return os << _NT_FLOAT16_TO_FLOAT32_(val);
	}
#else
	#include <half/half.hpp>
	using float16_t = half_float::half;
	#define _NT_FLOAT16_TO_FLOAT32_(f16) float(f16)
	#define _NT_FLOAT32_TO_FLOAT16_(f) float16_t(f)
	//by default has a way to print out the half_float::half
#endif

inline int16_t _NT_FLOAT16_TO_INT16_(float16_t a){
	union {
		float16_t f;
		int16_t i;
	} pun;
	pun.f = a;
	return pun.i;
}


template<typename T>
using my_complex = std::complex<T>;
using complex_32 = my_complex<float16_t>;
using complex_64 = my_complex<float>;
using complex_128 = my_complex<double>;


namespace std {
	template<size_t Index, typename T>
	inline constexpr T get(const my_complex<T>& c) noexcept {
		if constexpr (Index == 0){
			return c.real();
		}else if constexpr (Index == 1){
			return c.imag();
		}
		static_assert(Index < 2, "Cannot get complex number with index over 2");
	}
	template<size_t Index, typename T>
	inline constexpr T& get(my_complex<T>& c) noexcept {
		if constexpr (Index == 0){
			return c.real();
		}else if constexpr (Index == 1){
			return c.imag();
		}
		static_assert(Index < 2, "Cannot get complex number with index over 2");
	}
	template<size_t Index, typename T>
	inline constexpr T&& get(my_complex<T>&& c) noexcept {
		if constexpr (Index == 0){
			return std::move(c.real());
		}else if constexpr (Index == 1){
			return std::move(c.imag());
		}
		static_assert(Index < 2, "Cannot get complex number with index over 2");
	}


}  // namespace std

// Specialize `std::tuple_size` and `std::tuple_element` to make std::complex tuple-like
namespace std {
    template <typename T>
    struct tuple_size<my_complex<T>> : std::integral_constant<size_t, 2> {};

    template <size_t Index, typename T>
    struct tuple_element<Index, my_complex<T>> {
        using type = T;
    };
}

#endif //_NT_FLOAT16_H_
