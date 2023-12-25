#include "Convert.h"
#include "../dtype/DType_enum.h"
#include "../types/Types.h"
#include "../types/Cast16.h"
#include "../Tensor.h"
#include <__config>
#include <_types/_uint8_t.h>
#include "../dtype/compatible/DType_compatible.h"

namespace nt{
namespace convert{

#ifdef _HALF_FLOAT_SUPPORT_
template<> float16_t convert<DType::Float16>(const float& v){return static_cast<float16_t>(v);}
template<> float16_t convert<DType::Float16>(const double& v){return static_cast<float16_t>(v);}
#ifdef _HALF_FLOAT_SUPPORT_
template<> float16_t convert<DType::Float16>(const float16_t& v){return v;}
template<> float16_t convert<DType::Float16>(const complex_32& v){return v.real();}
#endif
#ifdef _128_FLOAT_SUPPORT_
template<> float16_t convert<DType::Float16>(const float128_t& v){return static_cast<float16_t>(v);}
#endif
#ifdef __SIZEOF_INT128__
template<> float16_t convert<DType::Float16>(const int128_t& v){return static_cast<float16_t>(float(v));}
template<> float16_t convert<DType::Float16>(const uint128_t& v){return static_cast<float16_t>(float(v));}
#endif
template<> float16_t convert<DType::Float16>(const complex_64& v){return static_cast<float16_t>(v.real());}
template<> float16_t convert<DType::Float16>(const complex_128& v){return static_cast<float16_t>(v.real());}
template<> float16_t convert<DType::Float16>(const int64_t& v){return static_cast<float16_t>(v);}
template<> float16_t convert<DType::Float16>(const int32_t& v){return static_cast<float16_t>(v);}
template<> float16_t convert<DType::Float16>(const uint32_t& v){return static_cast<float16_t>(v);}
template<> float16_t convert<DType::Float16>(const int16_t& v){return static_cast<float16_t>(v);}
template<> float16_t convert<DType::Float16>(const uint16_t& v){return static_cast<float16_t>(v);}
template<> float16_t convert<DType::Float16>(const int8_t& v){return static_cast<float16_t>(v);}
template<> float16_t convert<DType::Float16>(const uint8_t& v){return static_cast<float16_t>(v);}
template<> float16_t convert<DType::Float16>(const Tensor& v){return v.toScalar().to<float16_t>();}
template<> float16_t convert<DType::Float16>(const uint_bool_t& v){return static_cast<float16_t>(v.value);}
template<> float16_t convert<DType::Float16>(const bool &v){return float16_t(0 ? v : 1);}
#endif


template<> float convert<DType::Float32>(const float& v){return v;}
template<> float convert<DType::Float32>(const double& v){return static_cast<float>(v);}
#ifdef _HALF_FLOAT_SUPPORT_
template<> float convert<DType::Float32>(const float16_t& v){return static_cast<float>(v);}
template<> float convert<DType::Float32>(const complex_32& v){return static_cast<float>(v.real());}
#endif
#ifdef _128_FLOAT_SUPPORT_
template<> float convert<DType::Float32>(const float128_t& v){return static_cast<float>(v);}
#endif
#ifdef __SIZEOF_INT128__
template<> float convert<DType::Float32>(const int128_t& v){return static_cast<float>(v);}
template<> float convert<DType::Float32>(const uint128_t& v){return static_cast<float>(v);}
#endif
template<> float convert<DType::Float32>(const complex_64& v){return v.real();}
template<> float convert<DType::Float32>(const complex_128& v){return static_cast<float>(v.real());}
template<> float convert<DType::Float32>(const int64_t& v){return static_cast<float>(v);}
template<> float convert<DType::Float32>(const int32_t& v){return static_cast<float>(v);}
template<> float convert<DType::Float32>(const uint32_t& v){return static_cast<float>(v);}
template<> float convert<DType::Float32>(const int16_t& v){return static_cast<float>(v);}
template<> float convert<DType::Float32>(const uint16_t& v){return static_cast<float>(v);}
template<> float convert<DType::Float32>(const int8_t& v){return static_cast<float>(v);}
template<> float convert<DType::Float32>(const uint8_t& v){return static_cast<float>(v);}
template<> float convert<DType::Float32>(const Tensor& v){return v.toScalar().to<float>();}
template<> float convert<DType::Float32>(const uint_bool_t& v){return static_cast<float>(v.value);}
template<> float convert<DType::Float32>(const bool &v){return float(0 ? v : 1);}


template<> double convert<DType::Float64>(const float& v){return static_cast<double>(v);}
template<> double convert<DType::Float64>(const double& v){return v;}
#ifdef _HALF_FLOAT_SUPPORT_
template<> double convert<DType::Float64>(const float16_t& v){return static_cast<double>(v);}
template<> double convert<DType::Float64>(const complex_32& v){return static_cast<double>(v.real());}
#endif
#ifdef _128_FLOAT_SUPPORT_
template<> double convert<DType::Float64>(const float128_t& v){return static_cast<double>(v);}
#endif
#ifdef __SIZEOF_INT128__
template<> double convert<DType::Float64>(const int128_t& v){return static_cast<double>(v);}
template<> double convert<DType::Float64>(const uint128_t& v){return static_cast<double>(v);}
#endif
template<> double convert<DType::Float64>(const complex_64& v){return static_cast<double>(v.real());}
template<> double convert<DType::Float64>(const complex_128& v){return v.real();}
template<> double convert<DType::Float64>(const int64_t& v){return static_cast<double>(v);}
template<> double convert<DType::Float64>(const int32_t& v){return static_cast<double>(v);}
template<> double convert<DType::Float64>(const uint32_t& v){return static_cast<double>(v);}
template<> double convert<DType::Float64>(const int16_t& v){return static_cast<double>(v);}
template<> double convert<DType::Float64>(const uint16_t& v){return static_cast<double>(v);}
template<> double convert<DType::Float64>(const int8_t& v){return static_cast<double>(v);}
template<> double convert<DType::Float64>(const uint8_t& v){return static_cast<double>(v);}
template<> double convert<DType::Float64>(const Tensor& v){return v.toScalar().to<double>();}
template<> double convert<DType::Float64>(const uint_bool_t& v){return static_cast<double>(v.value);}
template<> double convert<DType::Float64>(const bool &v){return double(0 ? v : 1);}

#ifdef _128_FLOAT_SUPPORT_
template<> float128_t convert<DType::Float128>(const float& v){return static_cast<float128_t>(v);}
template<> float128_t convert<DType::Float128>(const double& v){return static_cast<float128_t>(v);}
#ifdef _HALF_FLOAT_SUPPORT_
template<> float128_t convert<DType::Float128>(const float16_t& v){return static_cast<float128_t>(v);}
template<> float128_t convert<DType::Float128>(const complex_32& v){return static_cast<float128_t>(v.real());}
#endif
#ifdef _128_FLOAT_SUPPORT_
template<> float128_t convert<DType::Float128>(const float128_t& v){return v;}
#endif
#ifdef __SIZEOF_INT128__
template<> float128_t convert<DType::Float128>(const int128_t& v){return static_cast<float128_t>(v);}
template<> float128_t convert<DType::Float128>(const uint128_t& v){return static_cast<float128_t>(v);}
#endif
template<> float128_t convert<DType::Float128>(const complex_64& v){return static_cast<float128_t>(v.real());}
template<> float128_t convert<DType::Float128>(const complex_128& v){return static_cast<float128_t>(v.real());}
template<> float128_t convert<DType::Float128>(const int64_t& v){return static_cast<float128_t>(v);}
template<> float128_t convert<DType::Float128>(const int32_t& v){return static_cast<float128_t>(v);}
template<> float128_t convert<DType::Float128>(const uint32_t& v){return static_cast<float128_t>(v);}
template<> float128_t convert<DType::Float128>(const int16_t& v){return static_cast<float128_t>(v);}
template<> float128_t convert<DType::Float128>(const uint16_t& v){return static_cast<float128_t>(v);}
template<> float128_t convert<DType::Float128>(const int8_t& v){return static_cast<float128_t>(v);}
template<> float128_t convert<DType::Float128>(const uint8_t& v){return static_cast<float128_t>(v);}
template<> float128_t convert<DType::Float128>(const Tensor& v){return v.toScalar().to<float128_t>();}
template<> float128_t convert<DType::Float128>(const uint_bool_t& v){return static_cast<float128_t>(v.value);}
template<> float128_t convert<DType::Float128>(const bool &v){return float128_t(0 ? v : 1);}
#endif

#ifdef __SIZEOF_INT128__
template<> int128_t convert<DType::int128>(const float& v){return static_cast<int128_t>(v);}
template<> int128_t convert<DType::int128>(const double& v){return static_cast<int128_t>(v);}
#ifdef _HALF_FLOAT_SUPPORT_
template<> int128_t convert<DType::int128>(const float16_t& v){return static_cast<int128_t>(float(v));}
template<> int128_t convert<DType::int128>(const complex_32& v){return static_cast<int128_t>(float(v.real()));}
#endif
#ifdef _128_FLOAT_SUPPORT_
template<> int128_t convert<DType::int128>(const float128_t& v){return static_cast<int128_t>(v);}
#endif
#ifdef __SIZEOF_INT128__
template<> int128_t convert<DType::int128>(const int128_t& v){return v;}
template<> int128_t convert<DType::int128>(const uint128_t& v){return static_cast<int128_t>(v);}
#endif
template<> int128_t convert<DType::int128>(const complex_64& v){return static_cast<int128_t>(v.real());}
template<> int128_t convert<DType::int128>(const complex_128& v){return static_cast<int128_t>(v.real());}
template<> int128_t convert<DType::int128>(const int64_t& v){return static_cast<int128_t>(v);}
template<> int128_t convert<DType::int128>(const int32_t& v){return static_cast<int128_t>(v);}
template<> int128_t convert<DType::int128>(const uint32_t& v){return static_cast<int128_t>(v);}
template<> int128_t convert<DType::int128>(const int16_t& v){return static_cast<int128_t>(v);}
template<> int128_t convert<DType::int128>(const uint16_t& v){return static_cast<int128_t>(v);}
template<> int128_t convert<DType::int128>(const int8_t& v){return static_cast<int128_t>(v);}
template<> int128_t convert<DType::int128>(const uint8_t& v){return static_cast<int128_t>(v);}
template<> int128_t convert<DType::int128>(const Tensor& v){return v.toScalar().to<int128_t>();}
template<> int128_t convert<DType::int128>(const uint_bool_t& v){return static_cast<int128_t>(v.value);}
template<> int128_t convert<DType::int128>(const bool &v){return int128_t(0 ? v : 1);}
#endif

#ifdef __SIZEOF_INT128__
template<> uint128_t convert<DType::uint128>(const float& v){return static_cast<uint128_t>(v);}
template<> uint128_t convert<DType::uint128>(const double& v){return static_cast<uint128_t>(v);}
#ifdef _HALF_FLOAT_SUPPORT_
template<> uint128_t convert<DType::uint128>(const float16_t& v){return static_cast<uint128_t>(float(v));}
template<> uint128_t convert<DType::uint128>(const complex_32& v){return static_cast<uint128_t>(float(v.real()));}
#endif
#ifdef _128_FLOAT_SUPPORT_
template<> uint128_t convert<DType::uint128>(const float128_t& v){return static_cast<uint128_t>(v);}
#endif
#ifdef __SIZEOF_INT128__
template<> uint128_t convert<DType::uint128>(const int128_t& v){return static_cast<uint128_t>(v);}
template<> uint128_t convert<DType::uint128>(const uint128_t& v){return v;}
#endif
template<> uint128_t convert<DType::uint128>(const complex_64& v){return static_cast<uint128_t>(v.real());}
template<> uint128_t convert<DType::uint128>(const complex_128& v){return static_cast<uint128_t>(v.real());}
template<> uint128_t convert<DType::uint128>(const int64_t& v){return static_cast<uint128_t>(v);}
template<> uint128_t convert<DType::uint128>(const int32_t& v){return static_cast<uint128_t>(v);}
template<> uint128_t convert<DType::uint128>(const uint32_t& v){return static_cast<uint128_t>(v);}
template<> uint128_t convert<DType::uint128>(const int16_t& v){return static_cast<uint128_t>(v);}
template<> uint128_t convert<DType::uint128>(const uint16_t& v){return static_cast<uint128_t>(v);}
template<> uint128_t convert<DType::uint128>(const int8_t& v){return static_cast<uint128_t>(v);}
template<> uint128_t convert<DType::uint128>(const uint8_t& v){return static_cast<uint128_t>(v);}
template<> uint128_t convert<DType::uint128>(const Tensor& v){return v.toScalar().to<uint128_t>();}
template<> uint128_t convert<DType::uint128>(const uint_bool_t& v){return static_cast<uint128_t>(v.value);}
template<> uint128_t convert<DType::uint128>(const bool &v){return uint128_t(0 ? v : 1);}
#endif

#ifdef _HALF_FLOAT_SUPPORT_
template<> complex_32 convert<DType::Complex32>(const float& v){return complex_32(static_cast<float16_t>(v), 0);}
template<> complex_32 convert<DType::Complex32>(const double& v){return complex_32(static_cast<float16_t>(v), 0);}
#ifdef _HALF_FLOAT_SUPPORT_
template<> complex_32 convert<DType::Complex32>(const float16_t& v){return complex_32(v, 0);}
template<> complex_32 convert<DType::Complex32>(const complex_32& v){return v;}
#endif
#ifdef _128_FLOAT_SUPPORT_
template<> complex_32 convert<DType::Complex32>(const float128_t& v){return complex_32(static_cast<float16_t>(v), 0);}
#endif
#ifdef __SIZEOF_INT128__
template<> complex_32 convert<DType::Complex32>(const int128_t& v){return complex_32(static_cast<float16_t>(float(v)), 0);}
template<> complex_32 convert<DType::Complex32>(const uint128_t& v){return complex_32(static_cast<float16_t>(float(v)), 0);}
#endif
template<> complex_32 convert<DType::Complex32>(const complex_64& v){return complex_32(v);}
template<> complex_32 convert<DType::Complex32>(const complex_128& v){return complex_32(v);}
template<> complex_32 convert<DType::Complex32>(const int64_t& v){return complex_32(static_cast<float16_t>(v), 0);}
template<> complex_32 convert<DType::Complex32>(const int32_t& v){return complex_32(static_cast<float16_t>(v), 0);}
template<> complex_32 convert<DType::Complex32>(const uint32_t& v){return complex_32(static_cast<float16_t>(v), 0);}
template<> complex_32 convert<DType::Complex32>(const int16_t& v){return complex_32(static_cast<float16_t>(v), 0);}
template<> complex_32 convert<DType::Complex32>(const uint16_t& v){return complex_32(static_cast<float16_t>(v), 0);}
template<> complex_32 convert<DType::Complex32>(const int8_t& v){return complex_32(static_cast<float16_t>(v), 0);}
template<> complex_32 convert<DType::Complex32>(const uint8_t& v){return complex_32(static_cast<float16_t>(v), 0);}
template<> complex_32 convert<DType::Complex32>(const Tensor& v){return v.toScalar().to<complex_32>();}
template<> complex_32 convert<DType::Complex32>(const uint_bool_t& v){return complex_32(static_cast<float16_t>(v.value), 0);}
template<> complex_32 convert<DType::Complex32>(const bool &v){return complex_32(float16_t(0 ? v : 1), 0);}
#endif

template<> complex_64 convert<DType::Complex64>(const float& v){return complex_64(v, 0);}
template<> complex_64 convert<DType::Complex64>(const double& v){return complex_64(static_cast<float>(v), 0);}
#ifdef _HALF_FLOAT_SUPPORT_
template<> complex_64 convert<DType::Complex64>(const float16_t& v){return complex_64(static_cast<float>(v), 0);}
template<> complex_64 convert<DType::Complex64>(const complex_32& v){return complex_64(v);}
#endif
#ifdef _128_FLOAT_SUPPORT_
template<> complex_64 convert<DType::Complex64>(const float128_t& v){return complex_64(static_cast<float>(v), 0);}
#endif
#ifdef __SIZEOF_INT128__
template<> complex_64 convert<DType::Complex64>(const int128_t& v){return complex_64(static_cast<float>(v), 0);}
template<> complex_64 convert<DType::Complex64>(const uint128_t& v){return complex_64(static_cast<float>(v), 0);}
#endif
template<> complex_64 convert<DType::Complex64>(const complex_64& v){return v;}
template<> complex_64 convert<DType::Complex64>(const complex_128& v){return complex_64(v);}
template<> complex_64 convert<DType::Complex64>(const int64_t& v){return complex_64(static_cast<float>(v), 0);}
template<> complex_64 convert<DType::Complex64>(const int32_t& v){return complex_64(static_cast<float>(v), 0);}
template<> complex_64 convert<DType::Complex64>(const uint32_t& v){return complex_64(static_cast<float>(v), 0);}
template<> complex_64 convert<DType::Complex64>(const int16_t& v){return complex_64(static_cast<float>(v), 0);}
template<> complex_64 convert<DType::Complex64>(const uint16_t& v){return complex_64(static_cast<float>(v), 0);}
template<> complex_64 convert<DType::Complex64>(const int8_t& v){return complex_64(static_cast<float>(v), 0);}
template<> complex_64 convert<DType::Complex64>(const uint8_t& v){return complex_64(static_cast<float>(v), 0);}
template<> complex_64 convert<DType::Complex64>(const Tensor& v){return v.toScalar().to<complex_64>();}
template<> complex_64 convert<DType::Complex64>(const uint_bool_t& v){return complex_64(static_cast<float>(v.value), 0);}
template<> complex_64 convert<DType::Complex64>(const bool &v){return complex_64(float(0 ? v : 1), 0);}

template<> complex_128 convert<DType::Complex128>(const float& v){return complex_128(static_cast<double>(v), 0);}
template<> complex_128 convert<DType::Complex128>(const double& v){return complex_128(v, 0);}
#ifdef _HALF_FLOAT_SUPPORT_
template<> complex_128 convert<DType::Complex128>(const float16_t& v){return complex_128(static_cast<double>(v), 0);}
template<> complex_128 convert<DType::Complex128>(const complex_32& v){return complex_128(v);}
#endif
#ifdef _128_FLOAT_SUPPORT_
template<> complex_128 convert<DType::Complex128>(const float128_t& v){return complex_128(static_cast<double>(v), 0);}
#endif
#ifdef __SIZEOF_INT128__
template<> complex_128 convert<DType::Complex128>(const int128_t& v){return complex_128(static_cast<double>(v), 0);}
template<> complex_128 convert<DType::Complex128>(const uint128_t& v){return complex_128(static_cast<double>(v), 0);}
#endif
template<> complex_128 convert<DType::Complex128>(const complex_64& v){return complex_128(v);}
template<> complex_128 convert<DType::Complex128>(const complex_128& v){return v;}
template<> complex_128 convert<DType::Complex128>(const int64_t& v){return complex_128(static_cast<double>(v), 0);}
template<> complex_128 convert<DType::Complex128>(const int32_t& v){return complex_128(static_cast<double>(v), 0);}
template<> complex_128 convert<DType::Complex128>(const uint32_t& v){return complex_128(static_cast<double>(v), 0);}
template<> complex_128 convert<DType::Complex128>(const int16_t& v){return complex_128(static_cast<double>(v), 0);}
template<> complex_128 convert<DType::Complex128>(const uint16_t& v){return complex_128(static_cast<double>(v), 0);}
template<> complex_128 convert<DType::Complex128>(const int8_t& v){return complex_128(static_cast<double>(v), 0);}
template<> complex_128 convert<DType::Complex128>(const uint8_t& v){return complex_128(static_cast<double>(v), 0);}
template<> complex_128 convert<DType::Complex128>(const Tensor& v){return v.toScalar().to<complex_128>();}
template<> complex_128 convert<DType::Complex128>(const uint_bool_t& v){return complex_128(static_cast<double>(v.value), 0);}
template<> complex_128 convert<DType::Complex128>(const bool &v){return complex_128(double(0 ? v : 1), 0);}


template<> uint8_t convert<DType::uint8>(const float& v){return static_cast<uint8_t>(v);}
template<> uint8_t convert<DType::uint8>(const double& v){return static_cast<uint8_t>(v);}
#ifdef _HALF_FLOAT_SUPPORT_
template<> uint8_t convert<DType::uint8>(const float16_t& v){return static_cast<uint8_t>(v);}
template<> uint8_t convert<DType::uint8>(const complex_32& v){return static_cast<uint8_t>(v.real());}
#endif
#ifdef _128_FLOAT_SUPPORT_
template<> uint8_t convert<DType::uint8>(const float128_t& v){return static_cast<uint8_t>(v);}
#endif
#ifdef __SIZEOF_INT128__
template<> uint8_t convert<DType::uint8>(const int128_t& v){return static_cast<uint8_t>(v);}
template<> uint8_t convert<DType::uint8>(const uint128_t& v){return static_cast<uint8_t>(v);}
#endif
template<> uint8_t convert<DType::uint8>(const complex_64& v){return static_cast<uint8_t>(v.real());}
template<> uint8_t convert<DType::uint8>(const complex_128& v){return static_cast<uint8_t>(v.real());}
template<> uint8_t convert<DType::uint8>(const int64_t& v){return static_cast<uint8_t>(v);}
template<> uint8_t convert<DType::uint8>(const int32_t& v){return static_cast<uint8_t>(v);}
template<> uint8_t convert<DType::uint8>(const uint32_t& v){return static_cast<uint8_t>(v);}
template<> uint8_t convert<DType::uint8>(const int16_t& v){return static_cast<uint8_t>(v);}
template<> uint8_t convert<DType::uint8>(const uint16_t& v){return static_cast<uint8_t>(v);}
template<> uint8_t convert<DType::uint8>(const int8_t& v){return static_cast<uint8_t>(v);}
template<> uint8_t convert<DType::uint8>(const uint8_t& v){return v;}
template<> uint8_t convert<DType::uint8>(const Tensor& v){return v.toScalar().to<uint8_t>();}
template<> uint8_t convert<DType::uint8>(const uint_bool_t& v){return static_cast<uint8_t>(v.value);}
template<> uint8_t convert<DType::uint8>(const bool &v){return uint8_t(0 ? v : 1);}

template<> int8_t convert<DType::int8>(const float& v){return static_cast<int8_t>(v);}
template<> int8_t convert<DType::int8>(const double& v){return static_cast<int8_t>(v);}
#ifdef _HALF_FLOAT_SUPPORT_
template<> int8_t convert<DType::int8>(const float16_t& v){return static_cast<int8_t>(v);}
template<> int8_t convert<DType::int8>(const complex_32& v){return static_cast<int8_t>(v.real());}
#endif
#ifdef _128_FLOAT_SUPPORT_
template<> int8_t convert<DType::int8>(const float128_t& v){return static_cast<int8_t>(v);}
#endif
#ifdef __SIZEOF_INT128__
template<> int8_t convert<DType::int8>(const int128_t& v){return static_cast<int8_t>(v);}
template<> int8_t convert<DType::int8>(const uint128_t& v){return static_cast<int8_t>(v);}
#endif
template<> int8_t convert<DType::int8>(const complex_64& v){return static_cast<int8_t>(v.real());}
template<> int8_t convert<DType::int8>(const complex_128& v){return static_cast<int8_t>(v.real());}
template<> int8_t convert<DType::int8>(const int64_t& v){return static_cast<int8_t>(v);}
template<> int8_t convert<DType::int8>(const int32_t& v){return static_cast<int8_t>(v);}
template<> int8_t convert<DType::int8>(const uint32_t& v){return static_cast<int8_t>(v);}
template<> int8_t convert<DType::int8>(const int16_t& v){return static_cast<int8_t>(v);}
template<> int8_t convert<DType::int8>(const uint16_t& v){return static_cast<int8_t>(v);}
template<> int8_t convert<DType::int8>(const int8_t& v){return v;}
template<> int8_t convert<DType::int8>(const uint8_t& v){return static_cast<int8_t>(v);}
template<> int8_t convert<DType::int8>(const Tensor& v){return v.toScalar().to<int8_t>();}
template<> int8_t convert<DType::int8>(const uint_bool_t& v){return static_cast<int8_t>(v.value);}
template<> int8_t convert<DType::int8>(const bool &v){return int8_t(0 ? v : 1);}


template<> int16_t convert<DType::int16>(const float& v){return static_cast<int16_t>(v);}
template<> int16_t convert<DType::int16>(const double& v){return static_cast<int16_t>(v);}
#ifdef _HALF_FLOAT_SUPPORT_
template<> int16_t convert<DType::int16>(const float16_t& v){return static_cast<int16_t>(v);}
template<> int16_t convert<DType::int16>(const complex_32& v){return static_cast<int16_t>(v.real());}
#endif
#ifdef _128_FLOAT_SUPPORT_
template<> int16_t convert<DType::int16>(const float128_t& v){return static_cast<int16_t>(v);}
#endif
#ifdef __SIZEOF_INT128__
template<> int16_t convert<DType::int16>(const int128_t& v){return static_cast<int16_t>(v);}
template<> int16_t convert<DType::int16>(const uint128_t& v){return static_cast<int16_t>(v);}
#endif
template<> int16_t convert<DType::int16>(const complex_64& v){return static_cast<int16_t>(v.real());}
template<> int16_t convert<DType::int16>(const complex_128& v){return static_cast<int16_t>(v.real());}
template<> int16_t convert<DType::int16>(const int64_t& v){return static_cast<int16_t>(v);}
template<> int16_t convert<DType::int16>(const int32_t& v){return static_cast<int16_t>(v);}
template<> int16_t convert<DType::int16>(const uint32_t& v){return static_cast<int16_t>(v);}
template<> int16_t convert<DType::int16>(const int16_t& v){return v;}
template<> int16_t convert<DType::int16>(const uint16_t& v){return static_cast<int16_t>(v);}
template<> int16_t convert<DType::int16>(const int8_t& v){return static_cast<int16_t>(v);}
template<> int16_t convert<DType::int16>(const uint8_t& v){return static_cast<int16_t>(v);}
template<> int16_t convert<DType::int16>(const Tensor& v){return v.toScalar().to<int16_t>();}
template<> int16_t convert<DType::int16>(const uint_bool_t& v){return static_cast<int16_t>(v.value);}
template<> int16_t convert<DType::int16>(const bool &v){return int16_t(0 ? v : 1);}

template<> uint16_t convert<DType::uint16>(const float& v){return static_cast<uint16_t>(v);}
template<> uint16_t convert<DType::uint16>(const double& v){return static_cast<uint16_t>(v);}
#ifdef _HALF_FLOAT_SUPPORT_
template<> uint16_t convert<DType::uint16>(const float16_t& v){return static_cast<uint16_t>(v);}
template<> uint16_t convert<DType::uint16>(const complex_32& v){return static_cast<uint16_t>(v.real());}
#endif
#ifdef _128_FLOAT_SUPPORT_
template<> uint16_t convert<DType::uint16>(const float128_t& v){return static_cast<uint16_t>(v);}
#endif
#ifdef __SIZEOF_INT128__
template<> uint16_t convert<DType::uint16>(const int128_t& v){return static_cast<uint16_t>(v);}
template<> uint16_t convert<DType::uint16>(const uint128_t& v){return static_cast<uint16_t>(v);}
#endif
template<> uint16_t convert<DType::uint16>(const complex_64& v){return static_cast<uint16_t>(v.real());}
template<> uint16_t convert<DType::uint16>(const complex_128& v){return static_cast<uint16_t>(v.real());}
template<> uint16_t convert<DType::uint16>(const int64_t& v){return static_cast<uint16_t>(v);}
template<> uint16_t convert<DType::uint16>(const int32_t& v){return static_cast<uint16_t>(v);}
template<> uint16_t convert<DType::uint16>(const uint32_t& v){return static_cast<uint16_t>(v);}
template<> uint16_t convert<DType::uint16>(const int16_t& v){return static_cast<uint16_t>(v);}
template<> uint16_t convert<DType::uint16>(const uint16_t& v){return v;}
template<> uint16_t convert<DType::uint16>(const int8_t& v){return static_cast<uint16_t>(v);}
template<> uint16_t convert<DType::uint16>(const uint8_t& v){return static_cast<uint16_t>(v);}
template<> uint16_t convert<DType::uint16>(const Tensor& v){return v.toScalar().to<uint16_t>();}
template<> uint16_t convert<DType::uint16>(const uint_bool_t& v){return static_cast<uint16_t>(v.value);}
template<> uint16_t convert<DType::uint16>(const bool &v){return uint16_t(0 ? v : 1);}

template<> int32_t convert<DType::int32>(const float& v){return static_cast<int32_t>(v);}
template<> int32_t convert<DType::int32>(const double& v){return static_cast<int32_t>(v);}
#ifdef _HALF_FLOAT_SUPPORT_
template<> int32_t convert<DType::int32>(const float16_t& v){return static_cast<int32_t>(v);}
template<> int32_t convert<DType::int32>(const complex_32& v){return static_cast<int32_t>(v.real());}
#endif
#ifdef _128_FLOAT_SUPPORT_
template<> int32_t convert<DType::int32>(const float128_t& v){return static_cast<int32_t>(v);}
#endif
#ifdef __SIZEOF_INT128__
template<> int32_t convert<DType::int32>(const int128_t& v){return static_cast<int32_t>(v);}
template<> int32_t convert<DType::int32>(const uint128_t& v){return static_cast<int32_t>(v);}
#endif
template<> int32_t convert<DType::int32>(const complex_64& v){return static_cast<int32_t>(v.real());}
template<> int32_t convert<DType::int32>(const complex_128& v){return static_cast<int32_t>(v.real());}
template<> int32_t convert<DType::int32>(const int64_t& v){return static_cast<int32_t>(v);}
template<> int32_t convert<DType::int32>(const int32_t& v){return v;}
template<> int32_t convert<DType::int32>(const uint32_t& v){return static_cast<int32_t>(v);}
template<> int32_t convert<DType::int32>(const int16_t& v){return static_cast<int32_t>(v);}
template<> int32_t convert<DType::int32>(const uint16_t& v){return static_cast<int32_t>(v);}
template<> int32_t convert<DType::int32>(const int8_t& v){return static_cast<int32_t>(v);}
template<> int32_t convert<DType::int32>(const uint8_t& v){return static_cast<int32_t>(v);}
template<> int32_t convert<DType::int32>(const Tensor& v){return v.toScalar().to<int32_t>();}
template<> int32_t convert<DType::int32>(const uint_bool_t& v){return static_cast<int32_t>(v.value);}
template<> int32_t convert<DType::int32>(const bool &v){return int32_t(0 ? v : 1);}

template<> uint32_t convert<DType::uint32>(const float& v){return static_cast<uint32_t>(v);}
template<> uint32_t convert<DType::uint32>(const double& v){return static_cast<uint32_t>(v);}
#ifdef _HALF_FLOAT_SUPPORT_
template<> uint32_t convert<DType::uint32>(const float16_t& v){return static_cast<uint32_t>(v);}
template<> uint32_t convert<DType::uint32>(const complex_32& v){return static_cast<uint32_t>(v.real());}
#endif
#ifdef _128_FLOAT_SUPPORT_
template<> uint32_t convert<DType::uint32>(const float128_t& v){return static_cast<uint32_t>(v);}
#endif
#ifdef __SIZEOF_INT128__
template<> uint32_t convert<DType::uint32>(const int128_t& v){return static_cast<uint32_t>(v);}
template<> uint32_t convert<DType::uint32>(const uint128_t& v){return static_cast<uint32_t>(v);}
#endif
template<> uint32_t convert<DType::uint32>(const complex_64& v){return static_cast<uint32_t>(v.real());}
template<> uint32_t convert<DType::uint32>(const complex_128& v){return static_cast<uint32_t>(v.real());}
template<> uint32_t convert<DType::uint32>(const int64_t& v){return static_cast<uint32_t>(v);}
template<> uint32_t convert<DType::uint32>(const int32_t& v){return static_cast<uint32_t>(v);}
template<> uint32_t convert<DType::uint32>(const uint32_t& v){return v;}
template<> uint32_t convert<DType::uint32>(const int16_t& v){return static_cast<uint32_t>(v);}
template<> uint32_t convert<DType::uint32>(const uint16_t& v){return static_cast<uint32_t>(v);}
template<> uint32_t convert<DType::uint32>(const int8_t& v){return static_cast<uint32_t>(v);}
template<> uint32_t convert<DType::uint32>(const uint8_t& v){return static_cast<uint32_t>(v);}
template<> uint32_t convert<DType::uint32>(const Tensor& v){return v.toScalar().to<uint32_t>();}
template<> uint32_t convert<DType::uint32>(const uint_bool_t& v){return static_cast<uint32_t>(v.value);}
template<> uint32_t convert<DType::uint32>(const bool &v){return uint32_t(0 ? v : 1);}


template<> int64_t convert<DType::int64>(const float& v){return static_cast<int64_t>(v);}
template<> int64_t convert<DType::int64>(const double& v){return static_cast<int64_t>(v);}
#ifdef _HALF_FLOAT_SUPPORT_
template<> int64_t convert<DType::int64>(const float16_t& v){return static_cast<int64_t>(v);}
template<> int64_t convert<DType::int64>(const complex_32& v){return static_cast<int64_t>(v.real());}
#endif
#ifdef _128_FLOAT_SUPPORT_
template<> int64_t convert<DType::int64>(const float128_t& v){return static_cast<int64_t>(v);}
#endif
#ifdef __SIZEOF_INT128__
template<> int64_t convert<DType::int64>(const int128_t& v){return static_cast<int64_t>(v);}
template<> int64_t convert<DType::int64>(const uint128_t& v){return static_cast<int64_t>(v);}
#endif
template<> int64_t convert<DType::int64>(const complex_64& v){return static_cast<int64_t>(v.real());}
template<> int64_t convert<DType::int64>(const complex_128& v){return static_cast<int64_t>(v.real());}
template<> int64_t convert<DType::int64>(const int32_t& v){return static_cast<int64_t>(v);}
template<> int64_t convert<DType::int64>(const uint32_t& v){return static_cast<int64_t>(v);}
template<> int64_t convert<DType::int64>(const int64_t& v){return v;}
template<> int64_t convert<DType::int64>(const int16_t& v){return static_cast<int64_t>(v);}
template<> int64_t convert<DType::int64>(const uint16_t& v){return static_cast<int64_t>(v);}
template<> int64_t convert<DType::int64>(const int8_t& v){return static_cast<int64_t>(v);}
template<> int64_t convert<DType::int64>(const uint8_t& v){return static_cast<int64_t>(v);}
template<> int64_t convert<DType::int64>(const Tensor& v){return v.toScalar().to<int64_t>();}
template<> int64_t convert<DType::int64>(const uint_bool_t& v){return static_cast<int64_t>(v.value);}
template<> int64_t convert<DType::int64>(const bool &v){return int64_t(0 ? v : 1);}

template<> uint_bool_t convert<DType::Bool>(const float& v){return uint_bool_t(v == 1);}
template<> uint_bool_t convert<DType::Bool>(const double& v){return uint_bool_t(v == 1);}
#ifdef _HALF_FLOAT_SUPPORT_
template<> uint_bool_t convert<DType::Bool>(const float16_t& v){return uint_bool_t(v == 1);}
template<> uint_bool_t convert<DType::Bool>(const complex_32& v){return uint_bool_t(v.real() == 1);}
#endif
#ifdef _128_FLOAT_SUPPORT_
template<> uint_bool_t convert<DType::Bool>(const float128_t& v){return uint_bool_t(v == 1);}
#endif
#ifdef __SIZEOF_INT128__
template<> uint_bool_t convert<DType::Bool>(const int128_t& v){return uint_bool_t(v == 1);}
template<> uint_bool_t convert<DType::Bool>(const uint128_t& v){return uint_bool_t(v == 1);}
#endif
template<> uint_bool_t convert<DType::Bool>(const complex_64& v){return uint_bool_t(v.real() == 1);}
template<> uint_bool_t convert<DType::Bool>(const complex_128& v){return uint_bool_t(v.real() == 1);}
template<> uint_bool_t convert<DType::Bool>(const int32_t& v){return uint_bool_t(v == 1);}
template<> uint_bool_t convert<DType::Bool>(const uint32_t& v){return uint_bool_t(v == 1);}
template<> uint_bool_t convert<DType::Bool>(const int64_t& v){return uint_bool_t(v == 1);}
template<> uint_bool_t convert<DType::Bool>(const int16_t& v){return uint_bool_t(v == 1);}
template<> uint_bool_t convert<DType::Bool>(const uint16_t& v){return uint_bool_t(v == 1);}
template<> uint_bool_t convert<DType::Bool>(const int8_t& v){return uint_bool_t(v == 1);}
template<> uint_bool_t convert<DType::Bool>(const uint8_t& v){return uint_bool_t(v == 1);}
template<> uint_bool_t convert<DType::Bool>(const Tensor& v){return v.toScalar().to<uint_bool_t>();}
template<> uint_bool_t convert<DType::Bool>(const uint_bool_t& v){return v;}
template<> uint_bool_t convert<DType::Bool>(const bool &v){return uint_bool_t(v);}


template<> Tensor convert<DType::TensorObj>(const float& v){
	Tensor outp({1}, DType::Float32);
	outp = v;
	return std::move(outp);
}
template<> Tensor convert<DType::TensorObj>(const double& v){
	Tensor outp({1}, DType::Float64);
	outp = v;
	return std::move(outp);
}
#ifdef _HALF_FLOAT_SUPPORT_
template<> Tensor convert<DType::TensorObj>(const float16_t& v){
	Tensor outp({1}, DType::Float16);
	outp = v;
	return std::move(outp);
}
template<> Tensor convert<DType::TensorObj>(const complex_32& v){
	Tensor outp({1}, DType::Complex32);
	outp = v;
	return std::move(outp);
}
#endif
#ifdef _128_FLOAT_SUPPORT_
template<> Tensor convert<DType::TensorObj>(const float128_t& v){
	Tensor outp({1}, DType::Float128);
	outp = v;
	return std::move(outp);
}
#endif
#ifdef __SIZEOF_INT128__
template<> Tensor convert<DType::TensorObj>(const int128_t& v){
	Tensor outp({1}, DType::int128);
	outp = v;
	return std::move(outp);
}
template<> Tensor convert<DType::TensorObj>(const uint128_t& v){
	Tensor outp({1}, DType::uint128);
	outp = v;
	return std::move(outp);
}
#endif
template<> Tensor convert<DType::TensorObj>(const complex_64& v){
	Tensor outp({1}, DType::Complex64);
	outp = v;
	return std::move(outp);
}
template<> Tensor convert<DType::TensorObj>(const complex_128& v){
	Tensor outp({1}, DType::Complex128);
	outp = v;
	return std::move(outp);

}
template<> Tensor convert<DType::TensorObj>(const int32_t& v){
	Tensor outp({1}, DType::int32);
	outp = v;
	return std::move(outp);
}
template<> Tensor convert<DType::TensorObj>(const uint32_t& v){
	Tensor outp({1}, DType::uint32);
	outp = v;
	return std::move(outp);
}
template<> Tensor convert<DType::TensorObj>(const int64_t& v){
	Tensor outp({1}, DType::int64);
	outp = v;
	return std::move(outp);
}
template<> Tensor convert<DType::TensorObj>(const int16_t& v){
	Tensor outp({1}, DType::int16);
	outp = v;
	return std::move(outp);
}
template<> Tensor convert<DType::TensorObj>(const uint16_t& v){
	Tensor outp({1}, DType::uint16);
	outp = v;
	return std::move(outp);
}
template<> Tensor convert<DType::TensorObj>(const int8_t& v){
	Tensor outp({1}, DType::int8);
	outp = v;
	return std::move(outp);
}
template<> Tensor convert<DType::TensorObj>(const uint8_t& v){
	Tensor outp({1}, DType::uint8);
	outp = v;
	return std::move(outp);
}
template<> Tensor convert<DType::TensorObj>(const Tensor& v){return v;}
template<> Tensor convert<DType::TensorObj>(const uint_bool_t& v){
	Tensor outp({1}, DType::Bool);
	outp = v;
	return std::move(outp);
}
template<> Tensor convert<DType::TensorObj>(const bool &v){
	Tensor outp({1}, DType::Bool);
	outp = v;
	return std::move(outp);
}

template<typename T, typename A>
T convert(const A& val){return convert<::nt::DTypeFuncs::type_to_dtype<T>>(val);}

template float convert<float,float>(const float&);
template float convert<float,double>(const double&);
template float convert<float,complex_64>(const complex_64&);
template float convert<float,complex_128>(const complex_128&);
template float convert<float,uint8_t>(const uint8_t&);
template float convert<float,int8_t>(const int8_t&);
template float convert<float,int16_t>(const int16_t&);
template float convert<float,uint16_t>(const uint16_t&);
template float convert<float,int32_t>(const int32_t&);
template float convert<float,uint32_t>(const uint32_t&);
template float convert<float,int64_t>(const int64_t&);
template float convert<float,uint_bool_t>(const uint_bool_t&);
template float convert<float,Tensor>(const Tensor&);
#ifdef _HALF_FLOAT_SUPPORT_
template float convert<float, complex_32>(const complex_32&);
template float convert<float, float16_t>(const float16_t&);
#endif
#ifdef _128_FLOAT_SUPPORT_
template float convert<float, float128_t>(const float128_t&);
#endif
#ifdef __SIZEOF_INT128__
template float convert<float, int128_t>(const int128_t&);
template float convert<float, uint128_t>(const uint128_t&);
#endif
template double convert<double,float>(const float&);
template double convert<double,double>(const double&);
template double convert<double,complex_64>(const complex_64&);
template double convert<double,complex_128>(const complex_128&);
template double convert<double,uint8_t>(const uint8_t&);
template double convert<double,int8_t>(const int8_t&);
template double convert<double,int16_t>(const int16_t&);
template double convert<double,uint16_t>(const uint16_t&);
template double convert<double,int32_t>(const int32_t&);
template double convert<double,uint32_t>(const uint32_t&);
template double convert<double,int64_t>(const int64_t&);
template double convert<double,uint_bool_t>(const uint_bool_t&);
template double convert<double,Tensor>(const Tensor&);
#ifdef _HALF_FLOAT_SUPPORT_
template double convert<double, complex_32>(const complex_32&);
template double convert<double, float16_t>(const float16_t&);
#endif
#ifdef _128_FLOAT_SUPPORT_
template double convert<double, float128_t>(const float128_t&);
#endif
#ifdef __SIZEOF_INT128__
template double convert<double, int128_t>(const int128_t&);
template double convert<double, uint128_t>(const uint128_t&);
#endif
template complex_64 convert<complex_64,float>(const float&);
template complex_64 convert<complex_64,double>(const double&);
template complex_64 convert<complex_64,complex_64>(const complex_64&);
template complex_64 convert<complex_64,complex_128>(const complex_128&);
template complex_64 convert<complex_64,uint8_t>(const uint8_t&);
template complex_64 convert<complex_64,int8_t>(const int8_t&);
template complex_64 convert<complex_64,int16_t>(const int16_t&);
template complex_64 convert<complex_64,uint16_t>(const uint16_t&);
template complex_64 convert<complex_64,int32_t>(const int32_t&);
template complex_64 convert<complex_64,uint32_t>(const uint32_t&);
template complex_64 convert<complex_64,int64_t>(const int64_t&);
template complex_64 convert<complex_64,uint_bool_t>(const uint_bool_t&);
template complex_64 convert<complex_64,Tensor>(const Tensor&);
#ifdef _HALF_FLOAT_SUPPORT_
template complex_64 convert<complex_64, complex_32>(const complex_32&);
template complex_64 convert<complex_64, float16_t>(const float16_t&);
#endif
#ifdef _128_FLOAT_SUPPORT_
template complex_64 convert<complex_64, float128_t>(const float128_t&);
#endif
#ifdef __SIZEOF_INT128__
template complex_64 convert<complex_64, int128_t>(const int128_t&);
template complex_64 convert<complex_64, uint128_t>(const uint128_t&);
#endif
template complex_128 convert<complex_128,float>(const float&);
template complex_128 convert<complex_128,double>(const double&);
template complex_128 convert<complex_128,complex_64>(const complex_64&);
template complex_128 convert<complex_128,complex_128>(const complex_128&);
template complex_128 convert<complex_128,uint8_t>(const uint8_t&);
template complex_128 convert<complex_128,int8_t>(const int8_t&);
template complex_128 convert<complex_128,int16_t>(const int16_t&);
template complex_128 convert<complex_128,uint16_t>(const uint16_t&);
template complex_128 convert<complex_128,int32_t>(const int32_t&);
template complex_128 convert<complex_128,uint32_t>(const uint32_t&);
template complex_128 convert<complex_128,int64_t>(const int64_t&);
template complex_128 convert<complex_128,uint_bool_t>(const uint_bool_t&);
template complex_128 convert<complex_128,Tensor>(const Tensor&);
#ifdef _HALF_FLOAT_SUPPORT_
template complex_128 convert<complex_128, complex_32>(const complex_32&);
template complex_128 convert<complex_128, float16_t>(const float16_t&);
#endif
#ifdef _128_FLOAT_SUPPORT_
template complex_128 convert<complex_128, float128_t>(const float128_t&);
#endif
#ifdef __SIZEOF_INT128__
template complex_128 convert<complex_128, int128_t>(const int128_t&);
template complex_128 convert<complex_128, uint128_t>(const uint128_t&);
#endif
template uint8_t convert<uint8_t,float>(const float&);
template uint8_t convert<uint8_t,double>(const double&);
template uint8_t convert<uint8_t,complex_64>(const complex_64&);
template uint8_t convert<uint8_t,complex_128>(const complex_128&);
template uint8_t convert<uint8_t,uint8_t>(const uint8_t&);
template uint8_t convert<uint8_t,int8_t>(const int8_t&);
template uint8_t convert<uint8_t,int16_t>(const int16_t&);
template uint8_t convert<uint8_t,uint16_t>(const uint16_t&);
template uint8_t convert<uint8_t,int32_t>(const int32_t&);
template uint8_t convert<uint8_t,uint32_t>(const uint32_t&);
template uint8_t convert<uint8_t,int64_t>(const int64_t&);
template uint8_t convert<uint8_t,uint_bool_t>(const uint_bool_t&);
template uint8_t convert<uint8_t,Tensor>(const Tensor&);
#ifdef _HALF_FLOAT_SUPPORT_
template uint8_t convert<uint8_t, complex_32>(const complex_32&);
template uint8_t convert<uint8_t, float16_t>(const float16_t&);
#endif
#ifdef _128_FLOAT_SUPPORT_
template uint8_t convert<uint8_t, float128_t>(const float128_t&);
#endif
#ifdef __SIZEOF_INT128__
template uint8_t convert<uint8_t, int128_t>(const int128_t&);
template uint8_t convert<uint8_t, uint128_t>(const uint128_t&);
#endif
template int8_t convert<int8_t,float>(const float&);
template int8_t convert<int8_t,double>(const double&);
template int8_t convert<int8_t,complex_64>(const complex_64&);
template int8_t convert<int8_t,complex_128>(const complex_128&);
template int8_t convert<int8_t,uint8_t>(const uint8_t&);
template int8_t convert<int8_t,int8_t>(const int8_t&);
template int8_t convert<int8_t,int16_t>(const int16_t&);
template int8_t convert<int8_t,uint16_t>(const uint16_t&);
template int8_t convert<int8_t,int32_t>(const int32_t&);
template int8_t convert<int8_t,uint32_t>(const uint32_t&);
template int8_t convert<int8_t,int64_t>(const int64_t&);
template int8_t convert<int8_t,uint_bool_t>(const uint_bool_t&);
template int8_t convert<int8_t,Tensor>(const Tensor&);
#ifdef _HALF_FLOAT_SUPPORT_
template int8_t convert<int8_t, complex_32>(const complex_32&);
template int8_t convert<int8_t, float16_t>(const float16_t&);
#endif
#ifdef _128_FLOAT_SUPPORT_
template int8_t convert<int8_t, float128_t>(const float128_t&);
#endif
#ifdef __SIZEOF_INT128__
template int8_t convert<int8_t, int128_t>(const int128_t&);
template int8_t convert<int8_t, uint128_t>(const uint128_t&);
#endif
template int16_t convert<int16_t,float>(const float&);
template int16_t convert<int16_t,double>(const double&);
template int16_t convert<int16_t,complex_64>(const complex_64&);
template int16_t convert<int16_t,complex_128>(const complex_128&);
template int16_t convert<int16_t,uint8_t>(const uint8_t&);
template int16_t convert<int16_t,int8_t>(const int8_t&);
template int16_t convert<int16_t,int16_t>(const int16_t&);
template int16_t convert<int16_t,uint16_t>(const uint16_t&);
template int16_t convert<int16_t,int32_t>(const int32_t&);
template int16_t convert<int16_t,uint32_t>(const uint32_t&);
template int16_t convert<int16_t,int64_t>(const int64_t&);
template int16_t convert<int16_t,uint_bool_t>(const uint_bool_t&);
template int16_t convert<int16_t,Tensor>(const Tensor&);
#ifdef _HALF_FLOAT_SUPPORT_
template int16_t convert<int16_t, complex_32>(const complex_32&);
template int16_t convert<int16_t, float16_t>(const float16_t&);
#endif
#ifdef _128_FLOAT_SUPPORT_
template int16_t convert<int16_t, float128_t>(const float128_t&);
#endif
#ifdef __SIZEOF_INT128__
template int16_t convert<int16_t, int128_t>(const int128_t&);
template int16_t convert<int16_t, uint128_t>(const uint128_t&);
#endif
template uint16_t convert<uint16_t,float>(const float&);
template uint16_t convert<uint16_t,double>(const double&);
template uint16_t convert<uint16_t,complex_64>(const complex_64&);
template uint16_t convert<uint16_t,complex_128>(const complex_128&);
template uint16_t convert<uint16_t,uint8_t>(const uint8_t&);
template uint16_t convert<uint16_t,int8_t>(const int8_t&);
template uint16_t convert<uint16_t,int16_t>(const int16_t&);
template uint16_t convert<uint16_t,uint16_t>(const uint16_t&);
template uint16_t convert<uint16_t,int32_t>(const int32_t&);
template uint16_t convert<uint16_t,uint32_t>(const uint32_t&);
template uint16_t convert<uint16_t,int64_t>(const int64_t&);
template uint16_t convert<uint16_t,uint_bool_t>(const uint_bool_t&);
template uint16_t convert<uint16_t,Tensor>(const Tensor&);
#ifdef _HALF_FLOAT_SUPPORT_
template uint16_t convert<uint16_t, complex_32>(const complex_32&);
template uint16_t convert<uint16_t, float16_t>(const float16_t&);
#endif
#ifdef _128_FLOAT_SUPPORT_
template uint16_t convert<uint16_t, float128_t>(const float128_t&);
#endif
#ifdef __SIZEOF_INT128__
template uint16_t convert<uint16_t, int128_t>(const int128_t&);
template uint16_t convert<uint16_t, uint128_t>(const uint128_t&);
#endif
template int32_t convert<int32_t,float>(const float&);
template int32_t convert<int32_t,double>(const double&);
template int32_t convert<int32_t,complex_64>(const complex_64&);
template int32_t convert<int32_t,complex_128>(const complex_128&);
template int32_t convert<int32_t,uint8_t>(const uint8_t&);
template int32_t convert<int32_t,int8_t>(const int8_t&);
template int32_t convert<int32_t,int16_t>(const int16_t&);
template int32_t convert<int32_t,uint16_t>(const uint16_t&);
template int32_t convert<int32_t,int32_t>(const int32_t&);
template int32_t convert<int32_t,uint32_t>(const uint32_t&);
template int32_t convert<int32_t,int64_t>(const int64_t&);
template int32_t convert<int32_t,uint_bool_t>(const uint_bool_t&);
template int32_t convert<int32_t,Tensor>(const Tensor&);
#ifdef _HALF_FLOAT_SUPPORT_
template int32_t convert<int32_t, complex_32>(const complex_32&);
template int32_t convert<int32_t, float16_t>(const float16_t&);
#endif
#ifdef _128_FLOAT_SUPPORT_
template int32_t convert<int32_t, float128_t>(const float128_t&);
#endif
#ifdef __SIZEOF_INT128__
template int32_t convert<int32_t, int128_t>(const int128_t&);
template int32_t convert<int32_t, uint128_t>(const uint128_t&);
#endif
template uint32_t convert<uint32_t,float>(const float&);
template uint32_t convert<uint32_t,double>(const double&);
template uint32_t convert<uint32_t,complex_64>(const complex_64&);
template uint32_t convert<uint32_t,complex_128>(const complex_128&);
template uint32_t convert<uint32_t,uint8_t>(const uint8_t&);
template uint32_t convert<uint32_t,int8_t>(const int8_t&);
template uint32_t convert<uint32_t,int16_t>(const int16_t&);
template uint32_t convert<uint32_t,uint16_t>(const uint16_t&);
template uint32_t convert<uint32_t,int32_t>(const int32_t&);
template uint32_t convert<uint32_t,uint32_t>(const uint32_t&);
template uint32_t convert<uint32_t,int64_t>(const int64_t&);
template uint32_t convert<uint32_t,uint_bool_t>(const uint_bool_t&);
template uint32_t convert<uint32_t,Tensor>(const Tensor&);
#ifdef _HALF_FLOAT_SUPPORT_
template uint32_t convert<uint32_t, complex_32>(const complex_32&);
template uint32_t convert<uint32_t, float16_t>(const float16_t&);
#endif
#ifdef _128_FLOAT_SUPPORT_
template uint32_t convert<uint32_t, float128_t>(const float128_t&);
#endif
#ifdef __SIZEOF_INT128__
template uint32_t convert<uint32_t, int128_t>(const int128_t&);
template uint32_t convert<uint32_t, uint128_t>(const uint128_t&);
#endif
template int64_t convert<int64_t,float>(const float&);
template int64_t convert<int64_t,double>(const double&);
template int64_t convert<int64_t,complex_64>(const complex_64&);
template int64_t convert<int64_t,complex_128>(const complex_128&);
template int64_t convert<int64_t,uint8_t>(const uint8_t&);
template int64_t convert<int64_t,int8_t>(const int8_t&);
template int64_t convert<int64_t,int16_t>(const int16_t&);
template int64_t convert<int64_t,uint16_t>(const uint16_t&);
template int64_t convert<int64_t,int32_t>(const int32_t&);
template int64_t convert<int64_t,uint32_t>(const uint32_t&);
template int64_t convert<int64_t,int64_t>(const int64_t&);
template int64_t convert<int64_t,uint_bool_t>(const uint_bool_t&);
template int64_t convert<int64_t,Tensor>(const Tensor&);
#ifdef _HALF_FLOAT_SUPPORT_
template int64_t convert<int64_t, complex_32>(const complex_32&);
template int64_t convert<int64_t, float16_t>(const float16_t&);
#endif
#ifdef _128_FLOAT_SUPPORT_
template int64_t convert<int64_t, float128_t>(const float128_t&);
#endif
#ifdef __SIZEOF_INT128__
template int64_t convert<int64_t, int128_t>(const int128_t&);
template int64_t convert<int64_t, uint128_t>(const uint128_t&);
#endif
template uint_bool_t convert<uint_bool_t,float>(const float&);
template uint_bool_t convert<uint_bool_t,double>(const double&);
template uint_bool_t convert<uint_bool_t,complex_64>(const complex_64&);
template uint_bool_t convert<uint_bool_t,complex_128>(const complex_128&);
template uint_bool_t convert<uint_bool_t,uint8_t>(const uint8_t&);
template uint_bool_t convert<uint_bool_t,int8_t>(const int8_t&);
template uint_bool_t convert<uint_bool_t,int16_t>(const int16_t&);
template uint_bool_t convert<uint_bool_t,uint16_t>(const uint16_t&);
template uint_bool_t convert<uint_bool_t,int32_t>(const int32_t&);
template uint_bool_t convert<uint_bool_t,uint32_t>(const uint32_t&);
template uint_bool_t convert<uint_bool_t,int64_t>(const int64_t&);
template uint_bool_t convert<uint_bool_t,uint_bool_t>(const uint_bool_t&);
template uint_bool_t convert<uint_bool_t,Tensor>(const Tensor&);
#ifdef _HALF_FLOAT_SUPPORT_
template uint_bool_t convert<uint_bool_t, complex_32>(const complex_32&);
template uint_bool_t convert<uint_bool_t, float16_t>(const float16_t&);
#endif
#ifdef _128_FLOAT_SUPPORT_
template uint_bool_t convert<uint_bool_t, float128_t>(const float128_t&);
#endif
#ifdef __SIZEOF_INT128__
template uint_bool_t convert<uint_bool_t, int128_t>(const int128_t&);
template uint_bool_t convert<uint_bool_t, uint128_t>(const uint128_t&);
#endif
template Tensor convert<Tensor,float>(const float&);
template Tensor convert<Tensor,double>(const double&);
template Tensor convert<Tensor,complex_64>(const complex_64&);
template Tensor convert<Tensor,complex_128>(const complex_128&);
template Tensor convert<Tensor,uint8_t>(const uint8_t&);
template Tensor convert<Tensor,int8_t>(const int8_t&);
template Tensor convert<Tensor,int16_t>(const int16_t&);
template Tensor convert<Tensor,uint16_t>(const uint16_t&);
template Tensor convert<Tensor,int32_t>(const int32_t&);
template Tensor convert<Tensor,uint32_t>(const uint32_t&);
template Tensor convert<Tensor,int64_t>(const int64_t&);
template Tensor convert<Tensor,uint_bool_t>(const uint_bool_t&);
template Tensor convert<Tensor,Tensor>(const Tensor&);
#ifdef _HALF_FLOAT_SUPPORT_
template Tensor convert<Tensor, complex_32>(const complex_32&);
template Tensor convert<Tensor, float16_t>(const float16_t&);
#endif
#ifdef _128_FLOAT_SUPPORT_
template Tensor convert<Tensor, float128_t>(const float128_t&);
#endif
#ifdef __SIZEOF_INT128__
template Tensor convert<Tensor, int128_t>(const int128_t&);
template Tensor convert<Tensor, uint128_t>(const uint128_t&);
#endif
#ifdef _HALF_FLOAT_SUPPORT_
template float16_t convert<float16_t,float16_t>(const float16_t&);
template float16_t convert<float16_t,complex_32>(const complex_32&);
template float16_t convert<float16_t,float>(const float&);
template float16_t convert<float16_t,double>(const double&);
template float16_t convert<float16_t,complex_64>(const complex_64&);
template float16_t convert<float16_t,complex_128>(const complex_128&);
template float16_t convert<float16_t,uint8_t>(const uint8_t&);
template float16_t convert<float16_t,int8_t>(const int8_t&);
template float16_t convert<float16_t,int16_t>(const int16_t&);
template float16_t convert<float16_t,uint16_t>(const uint16_t&);
template float16_t convert<float16_t,int32_t>(const int32_t&);
template float16_t convert<float16_t,uint32_t>(const uint32_t&);
template float16_t convert<float16_t,int64_t>(const int64_t&);
template float16_t convert<float16_t,uint_bool_t>(const uint_bool_t&);
template float16_t convert<float16_t,Tensor>(const Tensor&);
template complex_32 convert<complex_32,float16_t>(const float16_t&);
template complex_32 convert<complex_32,complex_32>(const complex_32&);
template complex_32 convert<complex_32,float>(const float&);
template complex_32 convert<complex_32,double>(const double&);
template complex_32 convert<complex_32,complex_64>(const complex_64&);
template complex_32 convert<complex_32,complex_128>(const complex_128&);
template complex_32 convert<complex_32,uint8_t>(const uint8_t&);
template complex_32 convert<complex_32,int8_t>(const int8_t&);
template complex_32 convert<complex_32,int16_t>(const int16_t&);
template complex_32 convert<complex_32,uint16_t>(const uint16_t&);
template complex_32 convert<complex_32,int32_t>(const int32_t&);
template complex_32 convert<complex_32,uint32_t>(const uint32_t&);
template complex_32 convert<complex_32,int64_t>(const int64_t&);
template complex_32 convert<complex_32,uint_bool_t>(const uint_bool_t&);
template complex_32 convert<complex_32,Tensor>(const Tensor&);
#ifdef _128_FLOAT_SUPPORT_
template float16_t convert<float16_t,float128_t>(const float128_t&);
template complex_32 convert<complex_32,float128_t>(const float128_t&);
#endif
#ifdef __SIZEOF_INT128__
template float16_t convert<float16_t,int128_t>(const int128_t&);
template float16_t convert<float16_t,uint128_t>(const uint128_t&);
template complex_32 convert<complex_32,int128_t>(const int128_t&);
template complex_32 convert<complex_32,uint128_t>(const uint128_t&);
#endif
#endif
#ifdef _128_FLOAT_SUPPORT_
template float128_t convert<float128_t,float128_t>(const float128_t&);
template float128_t convert<float128_t,float>(const float&);
template float128_t convert<float128_t,double>(const double&);
template float128_t convert<float128_t,complex_64>(const complex_64&);
template float128_t convert<float128_t,complex_128>(const complex_128&);
template float128_t convert<float128_t,uint8_t>(const uint8_t&);
template float128_t convert<float128_t,int8_t>(const int8_t&);
template float128_t convert<float128_t,int16_t>(const int16_t&);
template float128_t convert<float128_t,uint16_t>(const uint16_t&);
template float128_t convert<float128_t,int32_t>(const int32_t&);
template float128_t convert<float128_t,uint32_t>(const uint32_t&);
template float128_t convert<float128_t,int64_t>(const int64_t&);
template float128_t convert<float128_t,uint_bool_t>(const uint_bool_t&);
template float128_t convert<float128_t,Tensor>(const Tensor&);
#ifdef _HALF_FLOAT_SUPPORT_
template float128_t convert<float128_t,float16_t>(const float16_t&);
template float128_t convert<float128_t,complex_32>(const complex_32&);
#endif
#ifdef __SIZEOF_INT128__
template float128_t convert<float128_t,int128_t>(const int128_t&);
template float128_t convert<float128_t,uint128_t>(const uint128_t&);
#endif
#endif
#ifdef __SIZEOF_INT128__
template int128_t convert<int128_t,int128_t>(const int128_t&);
template int128_t convert<int128_t,uint128_t>(const uint128_t&);
template int128_t convert<int128_t,float>(const float&);
template int128_t convert<int128_t,double>(const double&);
template int128_t convert<int128_t,complex_64>(const complex_64&);
template int128_t convert<int128_t,complex_128>(const complex_128&);
template int128_t convert<int128_t,uint8_t>(const uint8_t&);
template int128_t convert<int128_t,int8_t>(const int8_t&);
template int128_t convert<int128_t,int16_t>(const int16_t&);
template int128_t convert<int128_t,uint16_t>(const uint16_t&);
template int128_t convert<int128_t,int32_t>(const int32_t&);
template int128_t convert<int128_t,uint32_t>(const uint32_t&);
template int128_t convert<int128_t,int64_t>(const int64_t&);
template int128_t convert<int128_t,uint_bool_t>(const uint_bool_t&);
template int128_t convert<int128_t,Tensor>(const Tensor&);
template uint128_t convert<uint128_t,int128_t>(const int128_t&);
template uint128_t convert<uint128_t,uint128_t>(const uint128_t&);
template uint128_t convert<uint128_t,float>(const float&);
template uint128_t convert<uint128_t,double>(const double&);
template uint128_t convert<uint128_t,complex_64>(const complex_64&);
template uint128_t convert<uint128_t,complex_128>(const complex_128&);
template uint128_t convert<uint128_t,uint8_t>(const uint8_t&);
template uint128_t convert<uint128_t,int8_t>(const int8_t&);
template uint128_t convert<uint128_t,int16_t>(const int16_t&);
template uint128_t convert<uint128_t,uint16_t>(const uint16_t&);
template uint128_t convert<uint128_t,int32_t>(const int32_t&);
template uint128_t convert<uint128_t,uint32_t>(const uint32_t&);
template uint128_t convert<uint128_t,int64_t>(const int64_t&);
template uint128_t convert<uint128_t,uint_bool_t>(const uint_bool_t&);
template uint128_t convert<uint128_t,Tensor>(const Tensor&);
#ifdef _128_FLOAT_SUPPORT_
template int128_t convert<int128_t,float128_t>(const float128_t&);
template uint128_t convert<uint128_t,float128_t>(const float128_t&);
#endif
#ifdef _HALF_FLOAT_SUPPORT_
template int128_t convert<int128_t,float16_t>(const float16_t&);
template int128_t convert<int128_t,complex_32>(const complex_32&);
template uint128_t convert<uint128_t,float16_t>(const float16_t&);
template uint128_t convert<uint128_t,complex_32>(const complex_32&);
#endif
#endif

}
}
