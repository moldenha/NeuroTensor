#ifndef NT_REFLECT_DEVICE_DEVICE_FUNC_H__
#define NT_REFLECT_DEVICE_DEVICE_FUNC_H__

#include "../../memory/DeviceEnum.h"
#include "../../utils/type_traits.h"
#include "devices_func_macro.h"
#include "../../dtype/ArrayVoid.h"
#include <utility>
#include <string_view>

namespace nt::functional::reflect_device{

template<char... Str>
struct NT_ReflectDeviceStringLiteral_{
    template<char... Str2>
    inline constexpr bool operator==(NT_ReflectDeviceStringLiteral_<Str2...>) const noexcept {
        if constexpr (sizeof...(Str) != sizeof...(Str2))
            return false;
        else
            return ((Str == Str2) && ...);
    }
};

namespace detail{
template<std::size_t N>
inline constexpr std::size_t string_literal_size(const char (&)[N]){
    return N-1;
}

// Concatenate a character `C` to an existing NT_ReflectDeviceStringLiteral_
template<typename T, char C>
struct concat_nt_string; // primary template

// Specialization for NT_ReflectDeviceStringLiteral_<...>
template<char... Cs, char C>
struct concat_nt_string<NT_ReflectDeviceStringLiteral_<Cs...>, C> {
    using type = NT_ReflectDeviceStringLiteral_<Cs..., C>;
};

// Primary template — not defined
template<typename T, std::size_t N>
struct take_first_nt_chars;

// Base case: empty string
template<std::size_t N>
struct take_first_nt_chars<NT_ReflectDeviceStringLiteral_<>, N> {
    using type = NT_ReflectDeviceStringLiteral_<>;
};

// Base case: N == 0
template<char... Cs>
struct take_first_nt_chars<NT_ReflectDeviceStringLiteral_<Cs...>, 0> {
    using type = NT_ReflectDeviceStringLiteral_<>;
};

template <char... N_i,size_t... M_i>
inline constexpr auto take_first_nt_chars_helper(std::index_sequence<M_i...>)
{
    constexpr char values[] = {N_i...};
    return NT_ReflectDeviceStringLiteral_<values[M_i]...>{};
}

template<char C, char... Cs, std::size_t N>
struct take_first_nt_chars<NT_ReflectDeviceStringLiteral_<C, Cs...>, N> {
    static_assert(N > 0, "N must be > 0 in recursive case");
    using type = decltype(take_first_nt_chars_helper<C, Cs...>(std::make_index_sequence<N>{}));
};



} // nt::functional::reflect_device::detail 
// inline constexpr auto first_letter(std::string_view s) {
//     return s.empty() ? '\0' : s[0];
// }


template<std::size_t I, std::size_t N>
inline constexpr auto get_n_letter(const char (& str)[N]){
    if constexpr (I < (N-1)){
        return str[I];
    }else{
        return '\0';
    }
    // return I < (N-1) ? str[I] : '\0';
}

// inline constexpr auto last_letters(std::string_view s){
//     s.remove_prefix(1);
//     return s;
//     // return s.empty() ? s : s.remove_prefix(1);
// }

} // nt::functional::reflect_device::


#define NT_FUNCTIONAL_MAKE_DEVICE_REFLECT_REGISTRY(device, other)\
namespace device{\
template<typename SV>\
struct check_bare_metal_registry{\
    static constexpr auto ptr = nullptr;\
};\
}

namespace nt::functional{
NT_GET_DEVICES_FUNCTIONAL_FUNC(NT_FUNCTIONAL_MAKE_DEVICE_REFLECT_REGISTRY, 0)
} //nt::fuctional::

#undef NT_FUNCTIONAL_MAKE_DEVICE_REFLECT_REGISTRY 

#define NT_REGISTER_OP_EXTEND_NAME_CHARACTERS__(name)\
        reflect_device::get_n_letter<0>(name), \
        reflect_device::get_n_letter<1>(name), \
        reflect_device::get_n_letter<2>(name), \
        reflect_device::get_n_letter<3>(name), \
        reflect_device::get_n_letter<4>(name), \
        reflect_device::get_n_letter<5>(name), \
        reflect_device::get_n_letter<6>(name), \
        reflect_device::get_n_letter<7>(name), \
        reflect_device::get_n_letter<8>(name), \
        reflect_device::get_n_letter<9>(name), \
        reflect_device::get_n_letter<10>(name), \
        reflect_device::get_n_letter<11>(name), \
        reflect_device::get_n_letter<12>(name), \
        reflect_device::get_n_letter<13>(name), \
        reflect_device::get_n_letter<14>(name), \
        reflect_device::get_n_letter<15>(name), \
        reflect_device::get_n_letter<16>(name), \
        reflect_device::get_n_letter<17>(name), \
        reflect_device::get_n_letter<18>(name), \
        reflect_device::get_n_letter<19>(name), \
        reflect_device::get_n_letter<20>(name), \
        reflect_device::get_n_letter<21>(name), \
        reflect_device::get_n_letter<22>(name), \
        reflect_device::get_n_letter<23>(name), \
        reflect_device::get_n_letter<24>(name), \
        reflect_device::get_n_letter<25>(name), \
        reflect_device::get_n_letter<26>(name), \
        reflect_device::get_n_letter<27>(name), \
        reflect_device::get_n_letter<28>(name), \
        reflect_device::get_n_letter<29>(name), \
        reflect_device::get_n_letter<30>(name)



// this is meant to be used inside the namespace
#define NT_DEVICE_REFLECT_REGISTER_OP(name)\
template<>\
struct check_bare_metal_registry< \
        reflect_device::detail::take_first_nt_chars<reflect_device::NT_ReflectDeviceStringLiteral_< \
            NT_REGISTER_OP_EXTEND_NAME_CHARACTERS__(#name) \
    >, reflect_device::detail::string_literal_size(#name)>::type>{ \
    static constexpr auto ptr = &name; \
};

#define NT_DEVICE_REFLECT_CHECK_REGISTRY(device, name) \
    device::check_bare_metal_registry< \
        reflect_device::detail::take_first_nt_chars<reflect_device::NT_ReflectDeviceStringLiteral_< \
            NT_REGISTER_OP_EXTEND_NAME_CHARACTERS__(#name) \
    >, reflect_device::detail::string_literal_size(#name)>::type>::ptr != nullptr

#define NT_DEVICE_REFLECT_GET_REGISTRY(device, name) \
    device::check_bare_metal_registry< \
        reflect_device::detail::take_first_nt_chars<reflect_device::NT_ReflectDeviceStringLiteral_< \
            NT_REGISTER_OP_EXTEND_NAME_CHARACTERS__(#name) \
    >, reflect_device::detail::string_literal_size(#name)>::type>::ptr



// use nt::type_traits::remove_cvref_t for the following:

namespace nt::functional::reflect_device{
namespace details{

template<typename T, typename... Ts>
struct nt__has__arr__void__{
    static constexpr bool value = ::nt::type_traits::is_same_v<T, 
};

} // nt::functional::reflect_device::details::


} // nt::functional::reflect_device


#endif //NT_REFLECT_DEVICE_DEVICE_FUNC_H__  
