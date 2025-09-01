#ifndef NT_SCALAR_H__
#define NT_SCALAR_H__
namespace nt{
class Scalar;
class ScalarRef;
class ConstScalarRef;
}

#include "DType_enum.h"
#include "DType.h"
#include "compatible/DType_compatible.h"
#include "../types/float16.h"
#include "../utils/api_macro.h"

#include <functional>

#include <type_traits>
#include <complex>
#include <string>
#include "compatible/DTypeDeclareMacros.h"


namespace nt{
namespace dtype_valid_checker_scalar{

/* template<typename T> */
/* static constexpr bool valid = (std::is_same_v<T, int32_t> ? true */ 
/* 			: std::is_same_v<T, int128_t> ? true */
/* 			: std::is_same_v<T, uint128_t> ? true */
/* 			: std::is_same_v<T, float16_t> ? true */
/* 			: std::is_same_v<T, float128_t> ? true */
/* 			: std::is_same_v<T, complex_32 > ? true */
/* 			: std::is_same_v<T, double> ? true */
/* 			: std::is_same_v<T, float> ? true */
/* 			: std::is_same_v<T, uint32_t> ? true */
/* 			: std::is_same_v<T, complex_64> ? true */
/* 			: std::is_same_v<T, complex_128> ? true */
/* 			: std::is_same_v<T, uint8_t> ? true */
/* 			: std::is_same_v<T, int8_t> ? true */
/* 			: std::is_same_v<T, int16_t> ? true */
/* 			: std::is_same_v<T, int64_t> ? true */
/* 			/1* : std::is_same_v<T, Tensor> ? DType::TensorObj *1/ */
/* 			: std::is_same_v<T, uint16_t> ? true: */ 
/* 			: std::is_same_v<T, bool> ? bool : false); */				


#define X(type, dtype_enum_a, dtype_enum_b)\
    std::is_same_v<T, type> ||

template<typename T>
static constexpr bool valid = (
            NT_GET_X_FLOATING_DTYPES_ 
            NT_GET_X_COMPLEX_DTYPES_
            NT_GET_X_SIGNED_INTEGER_DTYPES_
            NT_GET_X_UNSIGNED_INTEGER_DTYPES_
			std::is_same_v<T, uint_bool_t> ||
			std::is_same_v<T, bool>);

#undef X

}

class NEUROTENSOR_API Scalar{
	union v_t{
//boosts float128 does not have a trivial copy constructor
//which automatically deletes the copy constructor from this union
#ifdef BOOST_MP_STANDALONE
        double d{};
#else
        float128_t d{};
#endif


		int128_t i;
		complex_128 c;
		v_t() {}
	} v;
	DType dtype;

    template<typename T>
    void init_from(const T& vv);
	public:
		Scalar();
		Scalar(const Scalar&);
        
        //have to change the specialization because of MSVC
        template<typename T, std::enable_if_t<dtype_valid_checker_scalar::valid<T>, bool > = true>
        Scalar(T vv){init_from<T>(vv);}

            
#ifndef SIMDE_FLOAT16_IS_SCALAR
        Scalar(half_float::detail::expr val)
        :dtype(DType::Float16)
        {v.d = static_cast<decltype(v.d)>(double(val));} 
#endif 
        Scalar(std::string);

		bool isComplex() const;
		bool isFloatingPoint() const;
		bool isIntegral() const;
		bool isBoolean() const;
		bool isZero() const;
		bool isNegative() const;
        bool isInfinity() const;
        bool isNan() const;
        bool isEqual(Scalar) const;
        Scalar toComplex() const;
        Scalar toIntegral() const;
        Scalar toFloatingPoint() const;
        Scalar toBoolean() const;
        
		Scalar operator+(const Scalar&) const;
		Scalar operator-(const Scalar&) const;
		Scalar operator/(const Scalar&) const;
		Scalar operator*(const Scalar&) const;
		
		Scalar& operator+=(const Scalar&);
		Scalar& operator-=(const Scalar&);
		Scalar& operator/=(const Scalar&);
		Scalar& operator*=(const Scalar&);
		
		Scalar& operator=(const Scalar&);
		Scalar operator-() const;

		DType type() const;

		template<typename T, std::enable_if_t<DTypeFuncs::is_dtype_integer_v<DTypeFuncs::type_to_dtype<T>>, bool> = true>
		T to() const;

		template<typename T, std::enable_if_t<DTypeFuncs::is_dtype_complex_v<DTypeFuncs::type_to_dtype<T>>, bool> = true>
		T to() const;

		template<typename T, std::enable_if_t<DTypeFuncs::is_dtype_floating_v<DTypeFuncs::type_to_dtype<T>>, bool> = true>
		T to() const;
        

		template<typename T, std::enable_if_t<DTypeFuncs::type_to_dtype<T> == DType::Bool, bool> = true>
		uint_bool_t to() const;
		Scalar inverse() const;
		friend std::ostream& operator<<(std::ostream&, const Scalar&);
};

static Scalar inf = Scalar("inf");
static Scalar nan = Scalar("nan");

namespace utils{

//a scalar is either the Scalar class, or a type that can be made into a scalar
template<typename T>
inline constexpr bool is_scalar_value_v = std::is_constructible<Scalar, T>::value;
}




}

#endif
