#ifndef NT_TYPES_TYPES_H__
#define NT_TYPES_TYPES_H__

#include "uint_bool.h"
#include "complex.h"
#include "float128.h"
#include "float16.h"
#include "bit_128_integer.h"
#include "../utils/type_traits.h"

namespace nt::type_traits{

template<>
struct make_unsigned<::nt::float128_t>{
    using type = ::nt::uint128_t;
};

template<>
struct make_signed<::nt::float128_t>{
    using type = ::nt::int128_t;
};


}


#endif
