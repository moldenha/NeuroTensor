//This is a header file that ensures that nt::float16_t is included
#ifndef _NT_TYPES_FLOAT16_ENSURE_H_
#define _NT_TYPES_FLOAT16_ENSURE_H_


#include <simde/simde-f16.h> //using this to convert between float32 and float16
#define _HALF_FLOAT_SUPPORT_ //by default is going to have half float support
#ifdef SIMDE_FLOAT16_IS_SCALAR
namespace nt{
    using float16_t = simde_float16;
	#define _NT_FLOAT16_TO_FLOAT32_(f16) simde_float16_to_float32(f16)
	#define _NT_FLOAT32_TO_FLOAT16_(f)    simde_float16_from_float32(f)
	inline std::ostream& operator<<(std::ostream& os, const float16_t& val){
		return os << _NT_FLOAT16_TO_FLOAT32_(val);
	}
}
#else
#include <half/half.hpp>
namespace nt{
	using float16_t = half_float::half;
	#define _NT_FLOAT16_TO_FLOAT32_(f16) float(f16)
	#define _NT_FLOAT32_TO_FLOAT16_(f) float16_t(f)
	//by default has a way to print out the half_float::half
}
#endif

namespace nt{
inline int16_t _NT_FLOAT16_TO_INT16_(float16_t fp) noexcept {
	int16_t out = *reinterpret_cast<int16_t*>(&fp);
	return out;
}
}

#endif
