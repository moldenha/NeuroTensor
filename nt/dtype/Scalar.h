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


#include <functional>

#include <type_traits>
#include <complex>
#include <string>

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


template<typename T>
static constexpr bool valid = (std::is_same_v<T, int32_t> ||
#ifdef __SIZEOF_INT128__
			std::is_same_v<T, int128_t> ||
			std::is_same_v<T, uint128_t> ||
#endif
#ifdef _HALF_FLOAT_SUPPORT_
			std::is_same_v<T, complex_32 > ||
			std::is_same_v<T, float16_t> ||
#endif
#ifdef _128_FLOAT_SUPPORT_
			std::is_same_v<T, float128_t> ||
#endif
			std::is_same_v<T, double> ||
			std::is_same_v<T, float> ||
			std::is_same_v<T, uint32_t> ||
			std::is_same_v<T, complex_64> ||
			std::is_same_v<T, complex_128> ||
			std::is_same_v<T, uint8_t> ||
			std::is_same_v<T, int8_t> ||
			std::is_same_v<T, int16_t> ||
			std::is_same_v<T, int64_t> ||
			std::is_same_v<T, uint16_t> ||
			std::is_same_v<T, Tensor> ||
			std::is_same_v<T, bool>);	
}

class Scalar{
	union v_t{
//boosts float128 does not have a trivial copy constructor
//which automatically deletes the copy constructor from this union
#ifdef BOOST_MP_STANDALONE
        double d{};
#else
        float128_t d{};
#endif


#ifdef __SIZEOF_INT128__
		int128_t i;
#else
		int64_t i;
#endif
		complex_128 c;
		v_t() {}
	} v;
	DType dtype;
	public:
		Scalar();
		Scalar(const Scalar&);

		template<typename T, std::enable_if_t<DTypeFuncs::is_dtype_integer_v<DTypeFuncs::type_to_dtype<T>>, bool> = true>
		Scalar(T vv);

		template<typename T, std::enable_if_t<DTypeFuncs::is_dtype_complex_v<DTypeFuncs::type_to_dtype<T>>, bool> = true>
		Scalar(T vv);

		template<typename T, std::enable_if_t<DTypeFuncs::is_dtype_floating_v<DTypeFuncs::type_to_dtype<T>>, bool> = true>
		Scalar(T vv);

		template<typename T, std::enable_if_t<std::is_same_v<T, bool>, bool> = true>
		Scalar(T vv);

		template<typename T, std::enable_if_t<std::is_same_v<T, uint_bool_t>, bool> = true>
		Scalar(T vv);
            
#ifndef SIMDE_FLOAT16_IS_SCALAR
        Scalar(half_float::detail::expr val)
        :dtype(DType::Float16)
        {v.d = static_cast<decltype(v.d)>(double(val));} 
        
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


class ScalarRef{
	union data_t{
		std::reference_wrapper<double> f64;
		std::reference_wrapper<float> f32;
		std::reference_wrapper<int32_t> i32;
		std::reference_wrapper<uint32_t> i32_u;
		std::reference_wrapper<int64_t> i64;
		std::reference_wrapper<int16_t> i16;
		std::reference_wrapper<uint16_t> i16_u;
		std::reference_wrapper<int8_t> i8;
		std::reference_wrapper<uint8_t> i8_u;
		std::reference_wrapper<complex_64> c64;
		std::reference_wrapper<complex_128> c128;
		std::reference_wrapper<uint_bool_t> b;
		std::reference_wrapper<Tensor> t;
#ifdef __SIZEOF_INT128__
		std::reference_wrapper<int128_t> i128;
		std::reference_wrapper<uint128_t> i128_u;
#endif
#ifdef _HALF_FLOAT_SUPPORT_
		std::reference_wrapper<float16_t> f16;
		std::reference_wrapper<complex_32> c32;
#endif
#ifdef _128_FLOAT_SUPPORT_
		std::reference_wrapper<float128_t> f128;
#endif

		data_t(double& val)
			:f64(std::reference_wrapper<double>(val)){}
		data_t(float& val)
			:f32(std::reference_wrapper<float>(val)){}
		data_t(int32_t& val)
			:i32(std::reference_wrapper<int32_t>(val)){}
		data_t(uint32_t& val)
			:i32_u(std::reference_wrapper<uint32_t>(val)){}
		data_t(int64_t& val)
			:i64(std::reference_wrapper<int64_t>(val)){}
		data_t(int16_t& val)
			:i16(std::reference_wrapper<int16_t>(val)){}
		data_t(uint16_t& val)
			:i16_u(std::reference_wrapper<uint16_t>(val)){}
		data_t(int8_t& val)
			:i8(std::reference_wrapper<int8_t>(val)){}
		data_t(uint8_t& val)
			:i8_u(std::ref(val)){}
		data_t(complex_64& val)
			:c64(std::ref(val)){}
		data_t(complex_128& val)
			:c128(std::ref(val)){}
		data_t(uint_bool_t& val)
			:b(std::ref(val)){}
		data_t(Tensor& val)
			:t(std::ref(val)){}
#ifdef __SIZEOF_INT128__
		data_t(int128_t& val)
			:i128(std::ref(val)){}
		data_t(uint128_t& val)
			:i128_u(std::ref(val)){}
#endif
#ifdef _HALF_FLOAT_SUPPORT_
		data_t(float16_t& val)
			:f16(std::ref(val)){}
		data_t(complex_32& val)
			:c32(std::ref(val)){}
#endif
#ifdef _128_FLOAT_SUPPORT_
		data_t(float128_t& val)
			:f128(std::ref(val)){}
#endif

	} data;
	DType dtype;
	public:
		template<typename T, std::enable_if_t<dtype_valid_checker_scalar::valid<T>, bool> = true>
		ScalarRef(T& val)
		:dtype(DTypeFuncs::type_to_dtype<T>),
		data(val)
		{}

		ScalarRef(uint_bool_t& val)
			:dtype(DType::Bool),
			data(val)
		{}

		template<DType dt>
		DTypeFuncs::dtype_to_type_t<dt>& get(){
			if(dt != dtype)
				throw std::runtime_error("DTypes must match");
			if constexpr (dt == DType::Integer)
				return data.i32.get();
			else if constexpr (dt == DType::Long)
				return data.i32_u.get();
			else if constexpr (dt == DType::Float)
				return data.f32.get();
			else if constexpr (dt == DType::Double)
				return data.f64.get();
			else if constexpr (dt == DType::cfloat)
				return data.c64.get();
			else if constexpr (dt == DType::cdouble)
				return data.c128.get();
			else if constexpr (dt == DType::uint8)
				return data.i8_u.get();
			else if constexpr (dt == DType::int8)
				return data.i8.get();
			else if constexpr (dt == DType::uint16)
				return data.i16_u.get();
			else if constexpr (dt == DType::int16)
				return data.i16.get();
			else if constexpr (dt == DType::int16)
				return data.i64.get();
			else if constexpr (dt == DType::Bool)
				return data.b.get();
#ifdef __SIZEOF_INT128__
			else if constexpr (dt == DType::uint128)
				return data.i128_u.get();
			else if constexpr (dt == DType::int128)
				return data.i128.get();
#endif
#ifdef _HALF_FLOAT_SUPPORT_
			else if constexpr (dt == DType::Float16)
				return data.f16.get();
			else if constexpr (dt == DType::Complex32)
				return data.c32.get();
#endif
#ifdef _128_FLOAT_SUPPORT_
			else if constexpr(dt == DType::Float128)
				return data.f128.get();
#endif
			else if constexpr(dt == DType::TensorObj)
				return data.t.get();
			
		}

		template<DType dt = DType::Integer>
		inline ScalarRef& operator=(const Scalar val) noexcept{
			if(dt != dtype)
				return (*this).operator=<DTypeFuncs::next_dtype_it<dt>>(val);
			get<dt>() = val.to<dt>();
			return *this;
		}

		ScalarRef& operator=(const Tensor& val);
		friend std::ostream& operator<<(std::ostream&, const ScalarRef&);
};


class ConstScalarRef{
	union data_t{
		std::reference_wrapper<const double> f64;
		std::reference_wrapper<const float> f32;
		std::reference_wrapper<const int32_t> i32;
		std::reference_wrapper<const uint32_t> i32_u;
		std::reference_wrapper<const int64_t> i64;
		std::reference_wrapper<const int16_t> i16;
		std::reference_wrapper<const uint16_t> i16_u;
		std::reference_wrapper<const int8_t> i8;
		std::reference_wrapper<const uint8_t> i8_u;
		std::reference_wrapper<const complex_64> c64;
		std::reference_wrapper<const complex_128> c128;
		std::reference_wrapper<const uint_bool_t> b;
		std::reference_wrapper<const Tensor> t;
#ifdef __SIZEOF_INT128__
		std::reference_wrapper<const int128_t> i128;
		std::reference_wrapper<const uint128_t> i128_u;
#endif
#ifdef _HALF_FLOAT_SUPPORT_
		std::reference_wrapper<const float16_t> f16;
		std::reference_wrapper<const complex_32> c32;
#endif
#ifdef _128_FLOAT_SUPPORT_
		std::reference_wrapper<const float128_t> f128;
#endif

		data_t(const double& val)
			:f64(std::reference_wrapper<const double>(val)){}
		data_t(const float& val)
			:f32(std::reference_wrapper<const float>(val)){}
		data_t(const int32_t& val)
			:i32(std::reference_wrapper<const int32_t>(val)){}
		data_t(const uint32_t& val)
			:i32_u(std::reference_wrapper<const uint32_t>(val)){}
		data_t(const int64_t& val)
			:i64(std::reference_wrapper<const int64_t>(val)){}
		data_t(const int16_t& val)
			:i16(std::reference_wrapper<const int16_t>(val)){}
		data_t(const uint16_t& val)
			:i16_u(std::reference_wrapper<const uint16_t>(val)){}
		data_t(const int8_t& val)
			:i8(std::reference_wrapper<const int8_t>(val)){}
		data_t(const uint8_t& val)
			:i8_u(std::reference_wrapper<const uint8_t>(val)){}
		data_t(const complex_64& val)
			:c64(std::ref(val)){}
		data_t(const complex_128& val)
			:c128(std::ref(val)){}
		data_t(const uint_bool_t& val)
			:b(std::ref(val)){}
		data_t(const Tensor& val)
			:t(std::ref(val)){}
#ifdef __SIZEOF_INT128__
		data_t(const int128_t& val)
			:i128(std::ref(val)){}
		data_t(const uint128_t& val)
			:i128_u(std::ref(val)){}
#endif
#ifdef _HALF_FLOAT_SUPPORT_
		data_t(const float16_t& val)
			:f16(std::ref(val)){}
		data_t(const complex_32& val)
			:c32(std::ref(val)){}
#endif
#ifdef _128_FLOAT_SUPPORT_
		data_t(const float128_t& val)
			:f128(std::ref(val)){}
#endif

	} data;
	DType dtype;
	public:
		template<typename T, std::enable_if_t<dtype_valid_checker_scalar::valid<T>, bool> = true>
		ConstScalarRef(const T& val)
		:dtype(DTypeFuncs::type_to_dtype<T>),
		data(val)
		{}

		ConstScalarRef(const uint_bool_t& val)
			:dtype(DType::Bool),
			data(val)
		{}

		template<DType dt>
		const DTypeFuncs::dtype_to_type_t<dt>& get(){
			if(dt != dtype)
				throw std::runtime_error("DTypes must match");
			if constexpr (dt == DType::Integer)
				return data.i32.get();
			else if constexpr (dt == DType::Long)
				return data.i32_u.get();
			else if constexpr (dt == DType::Float)
				return data.f32.get();
			else if constexpr (dt == DType::Double)
				return data.f64.get();
			else if constexpr (dt == DType::cfloat)
				return data.c64.get();
			else if constexpr (dt == DType::cdouble)
				return data.c128.get();
			else if constexpr (dt == DType::uint8)
				return data.i8_u.get();
			else if constexpr (dt == DType::int8)
				return data.i8.get();
			else if constexpr (dt == DType::uint16)
				return data.i16_u.get();
			else if constexpr (dt == DType::int16)
				return data.i16.get();
			else if constexpr (dt == DType::int16)
				return data.i64.get();
			else if constexpr (dt == DType::Bool)
				return data.b.get();
#ifdef __SIZEOF_INT128__
			else if constexpr (dt == DType::uint128)
				return data.i128_u.get();
			else if constexpr (dt == DType::int128)
				return data.i128.get();
#endif
#ifdef _HALF_FLOAT_SUPPORT_
			else if constexpr (dt == DType::Float16)
				return data.f16.get();
			else if constexpr (dt == DType::Complex32)
				return data.c32.get();
#endif
#ifdef _128_FLOAT_SUPPORT_
			else if constexpr(dt == DType::Float128)
				return data.f128.get();
#endif
			else if constexpr(dt == DType::TensorObj)
				return data.t.get();
			
		}
		friend std::ostream& operator<<(std::ostream&, const ConstScalarRef&);
};


}

#endif
