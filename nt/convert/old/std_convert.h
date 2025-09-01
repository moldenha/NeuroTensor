#ifndef NT_STD_CONVERT_DTYPE_H__
#define NT_STD_CONVERT_DTYPE_H__

#include "../dtype/DType_enum.h"
#include "../utils/api_macro.h"
#include "../types/Types.h"
#include <type_traits>
#include "../dtype/compatible/DTypeDeclareMacros.h"



namespace nt{
namespace convert{
namespace details{
template <DType dt>
using my_convert_dtype_to_type_t = std::conditional_t<dt == DType::Float, float,
					std::conditional_t<dt == DType::Double, double,
					std::conditional_t<dt == DType::int128, int128_t,
					std::conditional_t<dt == DType::uint128, uint128_t,
					std::conditional_t<dt == DType::Float16, float16_t,
					std::conditional_t<dt == DType::Complex32, complex_32,
					std::conditional_t<dt == DType::Float128, float128_t,
					std::conditional_t<dt == DType::int64, int64_t,
					std::conditional_t<dt == DType::uint32, uint32_t,
					std::conditional_t<dt == DType::int32, int32_t,
					std::conditional_t<dt == DType::uint16, uint16_t,
					std::conditional_t<dt == DType::int16, int16_t,
					std::conditional_t<dt == DType::uint8, uint8_t,
					std::conditional_t<dt == DType::int8, int8_t,
					std::conditional_t<dt == DType::Complex64, complex_64,
					std::conditional_t<dt == DType::Complex128, complex_128,
					std::conditional_t<dt == DType::Bool, uint_bool_t, float> > > > > > > > > > > > >
						> >
						> >;

}

template<DType T, typename A, std::enable_if_t<T != DType::TensorObj, bool> = true>
NEUROTENSOR_API details::my_convert_dtype_to_type_t<T> convert(const A&);

template<typename T, typename A>
NEUROTENSOR_API T convert(const A& val);



}
}

#endif //NT_STD_CONVERT_DTYPE_H__
