#ifndef NT_DEVICE_BARE_METAL_REFLECTION_MACROS_H__
#define NT_DEVICE_BARE_METAL_REFLECTION_MACROS_H__
#include <initializer_list>
#include <type_traits>
#include <string_view>
#include <utility>
#include <exception>
#include <tuple>


#include "../../memory/DeviceEnum.h"
#include "../../utils/type_traits.h"
#include "../../Tensor.h"


namespace nt::reflect_device{
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

namespace detail {
    template<std::size_t... I>
    constexpr auto to_reflect_device_string_literal_impl(const char* str, std::index_sequence<I...>) {
        return NT_ReflectDeviceStringLiteral_<str[I]...>{};
    }
    
    template<std::size_t N>
    using size_t_ = type_traits::integral_constant<std::size_t, N>;
    
    template<std::size_t N>
    inline constexpr std::size_t string_literal_size(const char (&)[N]) {
        return N - 1;  // exclude null terminator
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

}

inline constexpr auto first_letter(std::string_view s) {
    return s.empty() ? '\0' : s[0];
}


template<std::size_t I, std::size_t N>
inline constexpr auto get_n_letter(const char (& str)[N]){
    if constexpr (I < (N-1)){
        return str[I];
    }else{
        return '\0';
    }
    // return I < (N-1) ? str[I] : '\0';
}


inline constexpr auto last_letters(std::string_view s){
    s.remove_prefix(1);
    return s;
    // return s.empty() ? s : s.remove_prefix(1);
}

// template<std::size_t N, char... Cs>
// inline constexpr auto get_nt_string(std::string_view str, NT_ReflectDeviceStringLiteral_<Cs...>){
//     return detail::concat_nt_string<NT_ReflectDeviceStringLiteral_<Cs...>, first_letter(str)>::type{};
// }

namespace expand_details{
// Helper: Expands an integer_sequence into a lambda
template<typename F, typename T, T... Is>
inline constexpr auto expand_sequence_to_lambda(F&& f, std::integer_sequence<T, Is...>) {
    return f(std::integral_constant<T, Is>{}...);
}


}
}


#define NT_MAKE_DEVICE_REGISTRY(device, other)\
namespace device{\
template<typename SV>\
struct check_bare_metal_registry{\
    static constexpr auto ptr = nullptr;\
};\
}


NT_GET_DEVICES_FUNC(NT_MAKE_DEVICE_REGISTRY, 0)


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
    

// #define NT_REGISTER_OP_STRING_SUB(reflection, rest)\
//     reflect_device


// this is meant to be used inside the namespace
#define NT_REGISTER_OP(name)\
template<>\
struct check_bare_metal_registry< \
        reflect_device::detail::take_first_nt_chars<reflect_device::NT_ReflectDeviceStringLiteral_< \
            NT_REGISTER_OP_EXTEND_NAME_CHARACTERS__(#name) \
    >, reflect_device::detail::string_literal_size(#name)>::type>{ \
    static constexpr auto ptr = &name; \
};


#define NT_CHECK_REGISTRY(device, name) \
    device::check_bare_metal_registry< \
        reflect_device::detail::take_first_nt_chars<reflect_device::NT_ReflectDeviceStringLiteral_< \
            NT_REGISTER_OP_EXTEND_NAME_CHARACTERS__(#name) \
    >, reflect_device::detail::string_literal_size(#name)>::type>::ptr != nullptr

#define NT_GET_REGISTRY(device, name) \
    device::check_bare_metal_registry< \
        reflect_device::detail::take_first_nt_chars<reflect_device::NT_ReflectDeviceStringLiteral_< \
            NT_REGISTER_OP_EXTEND_NAME_CHARACTERS__(#name) \
    >, reflect_device::detail::string_literal_size(#name)>::type>::ptr



template<typename T, typename... Ts>
struct nt__has__tensor__{
    static constexpr bool value = type_traits::is_same_v<type_traits::remove_cvref_t<T>, Tensor> || nt__has__devicet__<Ts...>::value;
};

template<typename T>
struct nt__has__tensor__<T>{
    static constexpr bool value = type_traits::is_same_v<type_traits::remove_cvref_t<T>, Tensor>;
};

template<>
struct nt__has__tensor__<int> : type_traits::false_type{};

template<typename T>
Tensor get_tensor(const T& arg){
    if constexpr (type_traits::is_same_v<type_traits::remove_cvref_t<T>, Tensor>){
        return arg;
    }
    return Tensor::Null();
}

template<typename T, typename... Ts>
Tensor get_tensor(const T& arg, const Ts&... args){
    if constexpr (type_traits::is_same_v<type_traits::remove_cvref_t<T>, Tensor>){
        return arg;
    }
    return get_tensor(args...);
}

template<typename T>
void ensure_same_device_types(const DeviceType& device, const T& arg){
    if constexpr (type_traits::is_same_v<type_traits::remove_cvref_t<T>, Tensor>){
        if(!(device == arg.device())){
            throw std::logic_error("Error, given 2 different device types");
        }
    }
}

template<typename T, typename... Ts>
void ensure_same_device_types(const DeviceType& device, const T& arg, const Ts&... args){
    if constexpr (type_traits::is_same_v<type_traits::remove_cvref_t<T>, Tensor>){
        if(!(device == arg.device())){
            throw std::logic_error("Error, given 2 different device types");
        }
    }
    ensure_same_device_types(device, args...);
}

template <std::size_t Index, typename NewType, typename Tuple, std::size_t... Is>
auto transform_tuple_element_impl(Tuple&& t, NewType&& new_value, std::index_sequence<Is...>) {
    return std::make_tuple(
        (Is == Index
             ? std::forward<NewType>(new_value)
             : std::get<Is>(std::forward<Tuple>(t)))...
    );
}

template <std::size_t Index, typename NewType, typename Tuple>
auto transform_tuple_element(Tuple&& t, NewType&& new_value) {
    constexpr std::size_t N = std::tuple_size<type_traits::decay_t<Tuple>>::value;
    static_assert(Index < N, "Index out of range in transform_tuple_element");
    return transform_tuple_element_impl<Index>(
        std::forward<Tuple>(t),
        std::forward<NewType>(new_value),
        std::make_index_sequence<N>{}
    );
}

// this changes all of them to the same device type of cpu (for when the function doesnt exist)
template<typename tuple_type, std::size_t index, typename... T>
inline auto nt__run__device__func__choose__edit__one__cpu__(std::tuple<T...>& tup){
    if constexpr (index == sizeof...(T)){return;}
    else{
        if constexpr (type_traits::is_same_v<std::tuple_element_t<index, tuple_type>, Tensor&>){
            std::get<index>(tup).to_(DeviceType::cpu);
        }else if constexpr (type_traits::is_same_v<type_traits::remove_cvref_t<std::tuple_element_t<index, tuple_type>>, Tensor>){
            if constexpr (type_traits::is_same_v<std::tuple_element_t<index, std::tuple<T...>>, Tensor&>){
                tup = transform_tuple_element<index, Tensor>(tup, std::get<index>(tup).to(DeviceType::cpu));
            }else if constexpr (std::is_same_v<std::tuple_element_t<index, std::tuple<T...>>, const Tensor&>){
                tup = transform_tuple_element<index, Tensor>(tup, std::get<index>(tup).to(DeviceType::cpu));
            }else{
                std::get<index>(tup) = std::get<index>(tup).to(DeviceType::cpu);
            }
        }
        nt__run__device__func__choose__edit__one__cpu__<tuple_type, index+1, T...>(tup);
    }
}

template<typename tuple_type, std::size_t index, typename... T>
inline auto nt__run__device__func__choose__edit__one__original__(Device d, std::tuple<T...>& tup){
    if constexpr (index == sizeof...(T)){return;}
    else{
        if constexpr (std::is_same_v<std::tuple_element_t<index, tuple_type>, DeviceT&>){
            std::get<index>(tup).to_(d);
        }else if constexpr (std::is_same_v<remove_cvref_t<std::tuple_element_t<index, tuple_type>>, DeviceT>){
            // nothing needed because the original element in the tuple wasn't changed
            // if constexpr (std::is_same_v<std::tuple_element_t<index, std::tuple<T...>>, DeviceT&>){
            //     tup = transform_tuple_element<index, DeviceT>(tup, std::get<index>(tup).to(d));
            // }else if constexpr (std::is_same_v<std::tuple_element_t<index, std::tuple<T...>>, const DeviceT&>){
            //     tup = transform_tuple_element<index, DeviceT>(tup, std::get<index>(tup).to(d));
            // }else{
            //     std::get<index>(tup) = std::get<index>(tup).to(d);
            // }
        }
        nt__run__device__func__choose__edit__one__original__<tuple_type, index+1, T...>(d, tup);
    }
}

// needs to have non-void return
template<typename... Tf, typename... T>
inline void nt__run__device__func__choose__(Device original, 
                                            void(*cpu_func) (Tf...), 
                                            void(*other_func) (Tf...),
                                            T&&... args){
    if(other_func != nullptr){
        other_func(std::forward<T&&>(args)...);
        return;
    }
    auto tup = std::make_tuple(std::forward<T&&>(args)...);
    using func_tup_type = std::tuple<Tf...>;
    nt__run__device__func__choose__edit__one__cpu__<func_tup_type, 0>(tup);
    std::apply(cpu_func, tup);
    nt__run__device__func__choose__edit__one__original__<func_tup_type, 0>(original, tup);
}

template<typename out, typename... Tf, typename... T>
inline out nt__run__device__func__(out(*cpu_func) (Tf...),
                                    out(*metal_func) (Tf...),
                                   T&&... args){
    static_assert(nt__has__devicet__<T...>::value, "Error, not given a device");
    Tensor d = get_tensor(args...);
    DeviceType device = d.device();
    ensure_same_device_types(device, args...);
    if constexpr (std::is_void<out>::value){
        switch(device){
            case DeviceType::META:
                throw std::logic_error("Error, given device type of meta");
                return;
            case DeviceType::CPU:
                cpu_func(std::forward<T>(args)...);
                return;
            case DeviceType::CPUShared:
                cpu_func(std::forward<T>(args)...);
                return;
            case DeviceType::METAL:
                nt__run__device__func__choose__(device, cpu_func, metal_func, std::forward<T>(args)...);
                return;
        }
    }else{
        switch(device){
            case DeviceType::META:
                throw std::logic_error("Error, given device type of meta");
                return cpu_func(std::forward<T>(args)...);
            case DeviceType::CPU:
                return cpu_func(std::forward<T>(args)...);
            case DeviceType::CPUShared:
                return cpu_func(std::forward<T>(args)...);
            case DeviceType::METAL:
                return nt__run__device__func__choose__(device, cpu_func, metal_func, std::forward<T>(args)...);
        }
    }
}


#define NT_CALL_SPECIFIC_BARE_METAL_FUNCTION__(device, func_name)\
    static_cast<func_name##func_type>(NT_GET_REGISTRY(device, func_name)), 

#define NT_RUN_BARE_METAL_FUNC(func_name, ...)\
    static_assert(NT_CHECK_REGISTRY(cpu, func_name), "Error, function has not been implemented on cpu, cannot progress");\
    using func_name##func_type = decltype(NT_GET_REGISTRY(cpu, func_name));\
    nt__run__device__func__(NT_GET_DEVICES_FUNC(NT_CALL_SPECIFIC_BARE_METAL_FUNCTION__, func_name) __VA_ARGS__); 




}


#endif
