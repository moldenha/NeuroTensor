#ifndef DTYPE_H
#define DTYPE_H
#include "DType_enum.h"
#include "../utils/utils.h"
namespace nt{
struct d_type;
class d_type_reference;
struct uint_bool_t{
	unsigned value : 1;
	uint_bool_t();
	uint_bool_t(const bool&);
	uint_bool_t(const uint_bool_t&);
	uint_bool_t(uint_bool_t&&);
	uint_bool_t& operator=(const bool&);
	uint_bool_t& operator=(const uint8_t&);
	uint_bool_t& operator=(uint_bool_t&&);
	uint_bool_t& operator=(const uint_bool_t&);
	bool operator==(const uint_bool_t&) const;
};


namespace detect_not_tensor {
	template<class T, class...Ts>
	constexpr bool is_charlike(tag_t<T>, Ts&&...){ return false; }
	constexpr bool is_charlike( tag_t<double> ){ return true; }
	constexpr bool is_charlike( tag_t<float> ){ return true; }
	constexpr bool is_charlike( tag_t<std::complex<double> >){ return true; }
	constexpr bool is_charlike( tag_t<std::complex<float> >){ return true; }
	constexpr bool is_charlike( tag_t<int32_t> ){ return true; }
	constexpr bool is_charlike( tag_t<uint32_t> ){ return true;}
	constexpr bool is_charlike( tag_t<int64_t> ){return true;}
	constexpr bool is_charlike( tag_t<int16_t> ){ return true; }
	constexpr bool is_charlike( tag_t<uint16_t> ){ return true; }
	constexpr bool is_charlike( tag_t<int8_t> ){ return true; }
	constexpr bool is_charlike( tag_t<uint8_t> ){ return true; }
	constexpr bool is_charlike( tag_t<uint_bool_t> ){ return true; }
	template<class T>
	constexpr bool detect=is_charlike(tag<T>);
};

}

#include <_types/_uint16_t.h>
#include <_types/_uint32_t.h>
#include <functional>
#include <iostream>
#include <typeinfo>
#include <variant>
#include "../Tensor.h"
#include <complex>
#include <type_traits>


namespace nt{
std::ostream& operator<< (std::ostream &out, DType const& data);


namespace detect_dtype{
	template<class T, class...Ts>
	constexpr bool is_charlike(tag_t<T>, Ts&&...){ return false; }
	constexpr bool is_charlike( tag_t<double> ){ return true; }
	constexpr bool is_charlike( tag_t<float> ){ return true; }
	constexpr bool is_charlike( tag_t<std::complex<double> >){ return true; }
	constexpr bool is_charlike( tag_t<std::complex<float> >){ return true; }
	constexpr bool is_charlike( tag_t<int32_t> ){ return true; }
	constexpr bool is_charlike( tag_t<uint32_t> ){ return true;}
	constexpr bool is_charlike( tag_t<int64_t> ){return true;}
	constexpr bool is_charlike( tag_t<int16_t> ){ return true; }
	constexpr bool is_charlike( tag_t<uint16_t> ){ return true; }
	constexpr bool is_charlike( tag_t<int8_t> ){ return true; }
	constexpr bool is_charlike( tag_t<uint8_t> ){ return true; }
	constexpr bool is_charlike( tag_t<uint_bool_t> ){ return true; }
	constexpr bool is_charlike( tag_t<Tensor> ){ return true; }
	template<class T>
	constexpr bool detect=is_charlike(tag<T>);

};



std::ostream& operator<<(std::ostream &out, const uint_bool_t &data);

struct d_type{
	DType type;
	using var_t = std::variant<float,
					double,
					std::complex<float>,
					std::complex<double>,
					uint8_t,
					int8_t,
					int16_t,
					uint16_t,
					int32_t,
					uint32_t,
					int64_t,
					uint_bool_t,
					std::reference_wrapper<const Tensor> >;
	var_t data;
	d_type(const std::complex<float> _cf);
	d_type(const std::complex<double> _cd);
	d_type(const int32_t _i);
	d_type(const float _f);
	d_type(const double _d);
	d_type(const uint32_t _l);
	d_type(const Tensor& _t);
	d_type(const int16_t);
	d_type(const uint16_t);
	d_type(const int8_t);
	d_type(const uint8_t);
	d_type(const int64_t);
	d_type(const uint_bool_t);

	d_type(const int32_t _i, DType _t);
	d_type(const float _i, DType _t);
	d_type(const double _i, DType _t);
	d_type(const uint32_t _i, DType _t);
	d_type(const d_type& _t);
	d_type(d_type&& _t);
	d_type& operator=(const d_type& _t);
	d_type& operator=(d_type&& _t);
	template<typename T>
	T cast_num() const;
	const std::type_info& m_type() const noexcept;
	private:
		void set(const int32_t _i, DType _t);
		void set(const float _i, DType _t);
		void set(const double _i, DType _t);
		void set(const uint32_t _i, DType _t);


};


class d_type_reference{
	
	void set_type();
	/* DType type; */
	DTypeConst type;
	public:
	using var_t = std::variant<std::reference_wrapper<float>,
					std::reference_wrapper<double>,
					std::reference_wrapper<std::complex<float> >,
					std::reference_wrapper<std::complex<double> >,
					std::reference_wrapper<uint8_t>,
					std::reference_wrapper<int8_t>,
					std::reference_wrapper<int16_t>,
					std::reference_wrapper<uint16_t>,
					std::reference_wrapper<int32_t>,
					std::reference_wrapper<uint32_t>,
					std::reference_wrapper<int64_t>,
					std::reference_wrapper<uint_bool_t>,
					std::reference_wrapper<Tensor>,
					std::reference_wrapper<const float>,
					std::reference_wrapper<const double>,
					std::reference_wrapper<const std::complex<float> >,
					std::reference_wrapper<const std::complex<double> >,
					std::reference_wrapper<const uint8_t>,
					std::reference_wrapper<const int8_t>,
					std::reference_wrapper<const int16_t>,
					std::reference_wrapper<const uint16_t>,
					std::reference_wrapper<const int32_t>,
					std::reference_wrapper<const uint32_t>,
					std::reference_wrapper<const int64_t>,
					std::reference_wrapper<const uint_bool_t>,
					std::reference_wrapper<const Tensor> >;		


		d_type_reference(var_t);
		const bool is_const() const;
		template<typename T>
		const T& item() const;
		template<typename T>
		T& get();
		d_type_reference& operator=(const d_type_reference& inp);
		d_type_reference& operator=(const int32_t&);
		d_type_reference& operator=(const float&);
		d_type_reference& operator=(const double&);
		d_type_reference& operator=(const uint32_t&);
		d_type_reference& operator=(const std::complex<double>&);
		d_type_reference& operator=(const std::complex<float>&);
		d_type_reference& operator=(const Tensor&);
		d_type_reference& operator=(Tensor&&);
		template<typename T>
		T operator+(const T&) const;
		template<typename T>
		d_type_reference& operator+=(const T&);
		friend std::ostream& operator<< (std::ostream& out, const d_type_reference&);

	private:
		var_t data;
	
};

namespace DTypeFuncs{

template<DType dt>
bool is_in(const DType inp){
	return inp == dt;
}

template<DType dt, DType M, DType... Rest>
bool is_in(const DType inp){
	return (inp == dt) ? true : is_in<M, Rest...>(inp);
}

template<DType dt, DType M, DType... Rest>
struct is_in_t{
	static constexpr bool value =  (dt == M) ? true : is_in_t<dt, Rest...>::value;
};

template<DType dt, DType M>
struct is_in_t<dt, M>{
	static constexpr bool value = (dt == M);
}

template<DType dt, DType M, DType... Rest>
inline constexpr bool is_in_v = is_in_t<dt, M, Rest...>::value;

template<DType dt>
inline constexpr bool is_dtype_integer_v = is_in_v<dt, DType::Integer, DType::Byte, DType::Short, DType::UnsignedShort, DType::LongLong, DType::Long>;
template<DType dt>
inline constexpr bool is_dtype_floating_v = is_in_v<dt, DType::Float, DType::Double>;
template<DType dt>
inline constexpr bool is_dtype_complex_v = is_in_v<dt, DType::Complex128, DType::Complex64>;
template<DType dt>
inline constexpr bool is_dtype_other_v = is_in_v<dt, DType::Bool, DType::TensorObj>;

template<DType dt>
inline constexpr bool is_dtype_real_num_v = (is_dtype_integer_v<dt> || is_dtype_floating_v<dt>);

template<DType dt>
inline constexpr bool is_dtype_num_v = (is_dtype_integer_v<dt> || is_dtype_floating_v<dt> || is_dtype_complex_v<dt>);


template<DType... dts>
inline constexpr bool integer_is_in_dtype_v = (is_in_v<DType::Integer, dts...>
					|| is_in_v<DType::Byte, dts...>
					|| is_in_v<DType::Short, dts...>
					|| is_in_v<DType::UnsignedShort, dts...>
					|| is_in_v<DType::LongLong, dts...>
					|| is_in_v<DType::Long, dts...>
					|| is_in_v<DType::Char, dts...>);
template<DType... dts>
inline constexpr bool floating_is_in_dtype_v = (is_in_v<DType::Double, dts...> || is_in_v<DType::Float, dts...>);

template<DType... dts>
inline constexpr bool complex_is_in_dtype_v = (is_in_v<DType::Complex64, dts...> || is_in_v<DType::Complex128, dts...>);

template<DType... dts>
inline constexpr bool other_is_in_dtype_v = (is_in_v<DType::Bool, dts...> || is_in_v<DType::TensorObj, dts...>);

template<DType dt>
std::ostream& print_dtypes(std::ostream& os){
	os << dt << "}";
	return os;
}

template<DType dt, DType M, DType... Rest>
std::ostream& print_dtypes(std::ostream& os){
	os <<dt << ", ";
	return print_dtypes<M, Rest...>(os);
}

template<DType... Rest>
struct dtype_to_type{
	using type = std::conditional_t<is_in_v<DType::Integer, Rest...>, int32_t, 
			std::conditional_t<is_in_v<DType::Double, Rest...>, double,
			std::conditional_t<is_in_v<DType::Float, Rest...>, float,
			std::conditional_t<is_in_v<DType::Long, Rest...>, uint32_t,
			std::conditional_t<is_in_v<DType::TensorObj, Rest...>, Tensor,
			std::conditional_t<is_in_v<DType::cfloat, Rest...>, std::complex<float>,
			std::conditional_t<is_in_v<DType::cdouble, Rest...>, std::complex<double>,
			std::conditional_t<is_in_v<DType::uint8, Rest...>, uint8_t,
			std::conditional_t<is_in_v<DType::int8, Rest...>, int8_t,
			std::conditional_t<is_in_v<DType::int16, Rest...>, int16_t,
			std::conditional_t<is_in_v<DType::uint16, Rest...>, uint16_t,
			std::conditional_t<is_in_v<DType::int64, Rest...>, int64_t, uint_bool_t> > > > > > > > > > > >;
};

template<DType dt>
struct dtype_to_type<dt>{
	using type = std::conditional_t<dt == DType::Integer, int32_t, 
			std::conditional_t<dt == DType::Double, double,
			std::conditional_t<dt == DType::Float, float,
			std::conditional_t<dt == DType::Long, uint32_t,
			std::conditional_t<dt == DType::TensorObj, Tensor,
			std::conditional_t<dt == DType::cfloat, std::complex<float>,
			std::conditional_t<dt == DType::cdouble, std::complex<double>,
			std::conditional_t<dt == DType::uint8, uint8_t,
			std::conditional_t<dt == DType::int8, int8_t,
			std::conditional_t<dt == DType::int16, int16_t,
			std::conditonal_t<dt == DType::uint16, uint16_t,
			std::conditional_t<dt == DType::int64, int64_t, uint_bool_t> > > > > > > > > > > >;
};


template<DType... Rest>
using dtype_to_type_t = typename dtype_to_type<Rest...>::type;

template<DType dt>
using dtype_to_type_t = typename dtype_to_type<dt>::type;

template<typename T>
inline static constexpr DType type_to_dtype = (std::is_same_v<T, int32_t> ? DType::Integer
					: std::is_same_v<T, double> ? DType::Double
					: std::is_same_v<T, float> ? DType::Float
					: std::is_same_v<T, uint32_t> ? DType::Long
					: std::is_same_v<T, std::complex<float>> ? DType::cfloat
					: std::is_same_v<T, std::complex<double>> ? DType::cdouble
					: std::is_same_v<T, uint8_t> ? DType::uint8
					: std::is_same_v<T, int8_t> ? DType::int8
					: std::is_same_v<T, int16_t> ? DType::int16
					: std::is_same_v<T, int64_t> ? DType::int64
					: std::is_same_v<T, Tensor> ? DType::TensorObj
					: std::is_same_v<T, uint16_t> ? DType::uint16 : DType::Bool);

template <DType dt>
inline static constexpr bool dtype_is_num = is_in_v<dt, DType::Integer, DType::Long, DType::uint8, DType::int8, DType::int16, DType::int64, DType::uint64, DType::Float, DType::Double, DType::Complex128, DType::Complex64>;

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

bool is_unsiged(const DType& dt){return ((dt == DType::Long) || (dt == DType::uint16) || (dt == DType::uint8));}
bool is_integer(const DType& dt){return is_in<DType::Integer, DType::Long, DType::uint8, DType::int8, DType::int16, DType::int64, DType::uint64>(dt);}
bool is_floating(const DType& dt){return ((dt == DType::Float) || (dt == DType::Double));}

template <class... DTs>
bool is_in(DType dt, DTs... dts){
	if constexpr(!all_dtype_v<DTs...>){
		throw std::runtime_error("expected only DType types");
	}
	bool outp = false;
	(is_same(dt, outp, dts), ...);
	return outp;
}


#include "Convert.h"

template<DType dt, std::enable_if_t<dtype_is_num<dt>, bool> = true> 
class MultiplyThis{
	using type = dtype_to_type_t<dt>;
	template<typename A, std::enable_if_t<std::is_same_v<A, type>,bool>=true>
	type& operator()(type& a, const A& b){return a *= b;}
	template<typename A>
	type& operator()(type& a, const A& b){return a *= convert::convert<dt>(b);}
};

template<DType dt, std::enable_if_t<dtype_is_num<dt>, bool> = true> 
class DivideThis{
	using type = dtype_to_type_t<dt>;
	template<typename A, std::enable_if_t<std::is_same_v<A, type>,bool>=true>
	type& operator()(type& a, const A& b){return a /= b;}
	template<typename A>
	type& operator()(type& a, const A& b){return a /= convert::convert<dt>(b);}
};

template<DType dt, std::enable_if_t<dtype_is_num<dt>, bool> = true> 
class AddThis{
	using type = dtype_to_type_t<dt>;
	template<typename A, std::enable_if_t<std::is_same_v<A, type>,bool>=true>
	type& operator()(type& a, const A& b){return a += b;}
	template<typename A>
	type& operator()(type& a, const A& b){return a += convert::convert<dt>(b);}
};

template<DType dt, std::enable_if_t<dtype_is_num<dt>, bool> = true> 
class SubtractThis{
	using type = dtype_to_type_t<dt>;
	template<typename A, std::enable_if_t<std::is_same_v<A, type>,bool>=true>
	type& operator()(type& a, const A& b){return a -= b;}
	template<typename A>
	type& operator()(type& a, const A& b){return a -= convert::convert<dt>(b);}
};

template<DType dt, std::enable_if_t<dtype_is_num<dt>, bool> = true> 
class Multiply{
	using type = dtype_to_type_t<dt>;
	template<typename A, std::enable_if_t<std::is_same_v<A, type>,bool>=true>
	type operator()(const type& a, const type& b){return a * b;}
	template<typename A>
	type& operator()(type& a, const A& b){return a * convert::convert<dt>(b);}
};

template<DType dt, std::enable_if_t<dtype_is_num<dt>, bool> = true> 
class Divide{
	using type = dtype_to_type_t<dt>;
	template<typename A, std::enable_if_t<std::is_same_v<A, type>,bool>=true>
	type operator()(const type& a, const type& b){return a / b;}
	template<typename A>
	type& operator()(type& a, const A& b){return a / convert::convert<dt>(b);}
};

template<DType dt, std::enable_if_t<dtype_is_num<dt>, bool> = true> 
class Add{
	using type = dtype_to_type_t<dt>;
	template<typename A, std::enable_if_t<std::is_same_v<A, type>,bool>=true>
	type operator()(const type& a, const type& b){return a + b;}
	template<typename A>
	type& operator()(type& a, const A& b){return a + convert::convert<dt>(b);}
};
template<DType dt, std::enable_if_t<dtype_is_num<dt>, bool> = true> 
class Subtract{
	using type = dtype_to_type_t<dt>;
	template<typename A, std::enable_if_t<std::is_same_v<A, type>,bool>=true>
	type operator()(const type& a, const type& b){return a - b;}
	template<typename A>
	type& operator()(type& a, const A& b){return a - convert::convert<dt>(b);}
};

}
#include "Scalar.h"

namespace nt{
namespace DTypeFuncs{
template<DType dt, std::enable_if_t<dt == DType::TensorObj, bool> = true> 
class MultiplyThis{
	Tensor& operator()(Tensor& a, const Scalar& b){return a *= b;}
	Tensor& operator()(Tensor& a, const Tensor& b){return a += b;}
};

template<DType dt, std::enable_if_t<dt == DType::TensorObj, bool> = true> 
class DivideThis{
	Tensor& operator()(Tensor& a, const Scalar& b){return a /= b;}
	Tensor& operator()(Tensor& a, const Tensor& b){return a /= b;}
};

template<DType dt, std::enable_if_t<dt == DType::TensorObj, bool> = true> 
class SubtractThis{
	Tensor& operator()(Tensor& a, const Scalar& b){return a -= b;}
	Tensor& operator()(Tensor& a, const Tensor& b){return a -= b;}
};

template<DType dt, std::enable_if_t<dt == DType::TensorObj, bool> = true> 
class AddThis{
	Tensor& operator()(Tensor& a, const Scalar& b){return a += b;}
	Tensor& operator()(Tensor& a, const Tensor& b){return a += b;}
};

template<DType dt, std::enable_if_t<dt == DType::TensorObj, bool> = true> 
class Multiply{
	Tensor operator()(const Tensor& a, const Scalar& b){return a * b;}
	Tensor operator()(const Tensor& a, const Tensor& b){return a * b;}
};

template<DType dt, std::enable_if_t<dt == DType::TensorObj, bool> = true> 
class Divide{
	Tensor operator()(const Tensor& a, const Scalar& b){return a / b;}
	Tensor operator()(const Tensor& a, const Tensor& b){return a / b;}
};

template<DType dt, std::enable_if_t<dt == DType::TensorObj, bool> = true> 
class Subtract{
	Tensor operator()(const Tensor& a, const Scalar& b){return a - b;}
	Tensor operator()(const Tensor& a, const Tensor& b){return a - b;}
};

template<DType dt, std::enable_if_t<dt == DType::TensorObj, bool> = true> 
class Add{
	Tensor operator()(const Tensor& a, const Scalar& b){return a + b;}
	Tensor operator()(const Tensor& a, const Tensor& b){return a + b;}
};

}
}

}
#endif
