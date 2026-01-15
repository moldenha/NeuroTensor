/*
 * This header file is for casting between different types at a bit level
 *  - Use:
 *      - Class1 ex(value);
 *      - Class2 Bits = nt::bit_cast<Class2>(ex);
 *  - Limitations:
 *      - sizeof(Class1) == sizeof(Class2)
 *      - Not Constexpr (look at kbit_cast.h <has its own limitations> )
 *      - Class1 and Class2 are trivially copyable
*/

#ifndef NT_BIT_BIT_CAST_H__
#define NT_BIT_BIT_CAST_H__

#include "../utils/always_inline_macro.h"
#include "../utils/type_traits.h"

#if defined(__cpp_lib_bit_cast)
    #define NT_CPP20_AVAILABLE_
    #define NT_CPP20_CONSTEXPR constexpr
#else
    #define NT_CPP20_CONSTEXPR
#endif


namespace nt{

#ifdef NT_CPP20_AVAILABLE_
template<typename To, typename From>
inline constexpr To bit_cast(const From& src) noexcept {
    return std::bit_cast<To>(src);
}
#else
template<class To, class From_>
inline std::enable_if_t<
    sizeof(To) == sizeof(From_) &&
    std::is_trivially_copyable_v<From_> &&
    std::is_trivially_copyable_v<To>,
    To>
// constexpr support needs compiler magic
bit_cast(const From_& src) noexcept
{
    using From = std::decay_t<From_>;
    // the union is trivially constructible
    // Therefore From_ and To don't need to be
    union u__{u__(){}; char bits[sizeof(From)]; std::decay_t<To> dst;} u;
    std::memcpy(&u.dst, &src, sizeof(From));
    return u.dst;
}

#endif

}

#endif
