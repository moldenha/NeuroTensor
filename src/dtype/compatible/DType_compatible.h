#ifndef __DTYPE_COMPATIBLE_T__
#define __DTYPE_COMPATIBLE_T__

#include "../../types/Types.h"
#if defined(_HALF_FLOAT_SUPPORT_) && defined(_128_FLOAT_SUPPORT_) && defined(__SIZEOF_INT128__)
#include "DType_compatible_all.h"
#elif !defined(_HALF_FLOAT_SUPPORT_) && defined(_128_FLOAT_SUPPORT_) && !defined(__SIZEOF_INT128__) 
#include "DType_compatible_float128.h"
#elif !defined(_HALF_FLOAT_SUPPORT_) && defined(_128_FLOAT_SUPPORT_) && defined(__SIZEOF_INT128__)
#include "DType_compatible_float128_int128.h"
#elif defined(_HALF_FLOAT_SUPPORT_) && !defined(_128_FLOAT_SUPPORT_) && !defined(__SIZEOF_INT128__)
#include "DType_compatible_float16.h"
#elif defined(_HALF_FLOAT_SUPPORT_) && defined(_128_FLOAT_SUPPORT_) && !defined(__SIZEOF_INT128__)
#include "DType_compatible_float16_float128.h"
#elif defined(_HALF_FLOAT_SUPPORT_) && !defined(_128_FLOAT_SUPPORT_) && defined(__SIZEOF_INT128__)
#include "DType_compatible_float16_int128.h"
#elif !defined(_HALF_FLOAT_SUPPORT_) && !defined(_128_FLOAT_SUPPORT_) && defined(__SIZEOF_INT128__)
#include "DType_compatible_int128.h"
#else
#include "DType_compatible_standard.h"
#endif

#endif
