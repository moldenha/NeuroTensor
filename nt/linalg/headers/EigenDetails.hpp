#ifndef NT_EIGEN_DETAILS_HPP__
#define NT_EIGEN_DETAILS_HPP__

#include <Eigen/Dense>
#include <complex>
#include "../../dtype/DType_enum.h"

namespace nt{
namespace linalg{
namespace detail{
template<typename T>
struct NT_Type_To_Eigen_Type{
    using type = T;
};

#define _NT_MAKE_TYPE_TO_EIGEN_TYPE_(from, to)\
template<>\
struct NT_Type_To_Eigen_Type<from>{\
    using type = to;\
};

_NT_MAKE_TYPE_TO_EIGEN_TYPE_(complex_32, std::complex<float>)
_NT_MAKE_TYPE_TO_EIGEN_TYPE_(complex_64, std::complex<float>)
_NT_MAKE_TYPE_TO_EIGEN_TYPE_(complex_128, std::complex<double>)
_NT_MAKE_TYPE_TO_EIGEN_TYPE_(float16_t, float)
_NT_MAKE_TYPE_TO_EIGEN_TYPE_(float128_t, double)
_NT_MAKE_TYPE_TO_EIGEN_TYPE_(int128_t, int64_t)
_NT_MAKE_TYPE_TO_EIGEN_TYPE_(uint128_t, int64_t)

#undef _NT_MAKE_TYPE_TO_EIGEN_TYPE_

template<typename T>
using NT_Type_To_Eigen_Type_t = typename NT_Type_To_Eigen_Type<T>::type;



//will the type need to be converted
template<typename T>
inline static constexpr bool NT_Transform_Type_To_Eigen_v = !std::is_same_v<T, NT_Type_To_Eigen_Type_t<T>>; 


template<typename T>
struct EigenType_to_DType{
    static constexpr DType dt = DType::Bool;
};

#define _NT_MAKE_EIGENTYPE_TO_DTYPE_(type, dtype)\
template <>\
struct EigenType_to_DType<type> {\
    static constexpr DType dt = dtype;\
};


_NT_MAKE_EIGENTYPE_TO_DTYPE_(float, DType::Float32)
_NT_MAKE_EIGENTYPE_TO_DTYPE_(double, DType::Float64)
_NT_MAKE_EIGENTYPE_TO_DTYPE_(std::complex<float>, DType::Complex64)
_NT_MAKE_EIGENTYPE_TO_DTYPE_(std::complex<double>, DType::Complex128)
_NT_MAKE_EIGENTYPE_TO_DTYPE_(int64_t, DType::int64)
_NT_MAKE_EIGENTYPE_TO_DTYPE_(uint32_t, DType::uint32)
_NT_MAKE_EIGENTYPE_TO_DTYPE_(int32_t, DType::int32)
_NT_MAKE_EIGENTYPE_TO_DTYPE_(uint16_t, DType::uint16)
_NT_MAKE_EIGENTYPE_TO_DTYPE_(int16_t, DType::int16)
_NT_MAKE_EIGENTYPE_TO_DTYPE_(uint8_t, DType::uint8)
_NT_MAKE_EIGENTYPE_TO_DTYPE_(int8_t, DType::int8)

#undef _NT_MAKE_EIGENTYPE_TO_DTYPE_

//supported complex types for eigen
template<typename T>
inline static constexpr bool is_eigen_complex = std::is_same_v<std::complex<float>, T> || std::is_same_v<std::complex<double>, T>;

template<typename T>
struct from_complex_eigen{
    using type = T;
};

template<>
struct from_complex_eigen<std::complex<float>> { using type = float; };
template<>
struct from_complex_eigen<std::complex<double>> { using type = double; };

template<typename T>
using from_complex_eigen_t = typename from_complex_eigen<T>::type;

} //namespace detail

#define _NT_DECLARE_EIGEN_TYPES_(func)\
func(float)\
func(double)\
func(std::complex<float>)\
func(std::complex<double>)\
func(int64_t)\
func(uint32_t)\
func(int32_t)\
func(uint16_t)\
func(int16_t)\
func(uint8_t)\
func(int8_t)


}}


#endif
