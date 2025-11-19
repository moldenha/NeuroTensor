// This is a header file designed to make the use of functions on different devices easier
// Such that, it will automatically decide if there needs to be any device switching, and so on

// So for example
// If there is a function that is called for example add,
// but lets say that you only have it implemented for the cpu, but the user wants to run it on cuda
// This will automatically detect if the function has been made in cuda
// After seeing that it hasn't it will switch all the devices to the cpu, and then run the cpu version,
// then it will switch the result to the cuda device so that there is no issue for the user and returns expected behavior
//
// lets say that it is a += add, and again it has only been implemented on the cpu
// Then it will switch both devices over to the cpu, then after the operation, back to their original device
// As long as the functions in this header file are used properly, there should be no issue with device switching
//
// The reason for this header file:
//  - Yes, all this could be implemented manually for each function with less typing per function 
//          by manually checking if the function exists for that device
//  - However, that manual checking invites bugs and human error, 
//          that is what this header file is designed to avoid
//  - Persistent bugs based on device switching would be an innevitable outcome if this header file isn't implemented
//  
//  - Also, once the functions are made for different devices (me as the programmer) will not have
//      to worry about implementing this part in the tensor files, so it saves me some time and headache

#ifndef NT_DEVICE_BARE_METAL_MACROS_H__
#define NT_DEVICE_BARE_METAL_MACROS_H__
#include <initializer_list>
#include <type_traits>
#include <string_view>
#include <utility>
#include <exception>
#include <tuple>

#define NT_GET_DEVICES_FUNC(func, ...)\
    func(cpu, __VA_ARGS__)\
    func(mkl, __VA_ARGS__)\
    func(cuda, __VA_ARGS__)

#define NT_HEADER_PATH(device, op) "../"#device"/"#op".h"

#define NT_STR_(n) #n

#define NT_CHECK_HEADER_PATH(device, op) \
    __has_include(NT_STR_(NT_HEADER_PATH(device, op)))


#if __has_include("../cpu/add.h")
    #include "../cpu/add.h"
#endif

#if NT_CHECK_HEADER_PATH(cpu, add)
    #include NT_HEADER_PATH(cpu, add)
#endif


// #define NT_CHECK_HEADER(device, op)\
//     #if NT_CHECK_HEADER_PATH(device, op)\
//         #include NT_HEADER_PATH(device, op)\
//     #endif

// #define NT_CHECK_HEADER(device, op)\
//     _Pragma("clang diagnostic push") \
//     _Pragma("clang diagnostic ignored \"-Wunknown-pragmas\"") \
//     #if NT_CHECK_HEADER_PATH(device, op) \
//         #include "../" #device "/" #op ".h"\
//     #endif\
//     _Pragma("clang diagnostic pop")


// this is designed to parse through all of the possible devices
// and if the header with this name exists, include it
// #define NT_INCLUDE_BARE_OP(op)\
//     NT_GET_DEVICES_FUNC(NT_CHECK_HEADER, op)


// this macro is designed to be used before a function definition
#ifndef NEUROTENSOR_API
#define NEUROTENSOR_API
#endif // NEUROTENSOR_API

#include "device.h"

namespace reflect_device{
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
    using size_t_ = std::integral_constant<std::size_t, N>;
    
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


template<class T>
struct remove_cvref
{
    using type = std::remove_cv_t<std::remove_reference_t<T>>;
};

template<class T>
using remove_cvref_t = typename remove_cvref<T>::type;

template<typename T, typename... Ts>
struct nt__has__devicet__{
    static constexpr bool value = std::is_same_v<remove_cvref_t<T>, DeviceT> || nt__has__devicet__<Ts...>::value;
};

template<typename T>
struct nt__has__devicet__<T>{
    static constexpr bool value = std::is_same_v<remove_cvref_t<T>, DeviceT>;
};

template<>
struct nt__has__devicet__<int> : std::false_type{};

template<typename T>
DeviceT get_device(const T& arg){
    if constexpr (std::is_same_v<remove_cvref_t<T>, DeviceT>){
        return arg;
    }
    return DeviceT{Device::meta};
}


template<typename T, typename... Ts>
DeviceT get_device(const T& arg, const Ts&... args){
    if constexpr (std::is_same_v<remove_cvref_t<T>, DeviceT>){
        return arg;
    }
    return get_device(args...);
}

template<typename T>
void ensure_same_device_types(const Device& device, const T& arg){
    if constexpr (std::is_same_v<remove_cvref_t<T>, DeviceT>){
        if(!(device == arg.d)){
            throw std::logic_error("Error, given 2 different device types");
        }
    }
}

template<typename T, typename... Ts>
void ensure_same_device_types(const Device& device, const T& arg, const Ts&... args){
    if constexpr (std::is_same_v<remove_cvref_t<T>, DeviceT>){
        if(!(device == arg.d)){
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
    constexpr std::size_t N = std::tuple_size<std::decay_t<Tuple>>::value;
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
        if constexpr (std::is_same_v<std::tuple_element_t<index, tuple_type>, DeviceT&>){
            std::get<index>(tup).to_(Device::cpu);
        }else if constexpr (std::is_same_v<remove_cvref_t<std::tuple_element_t<index, tuple_type>>, DeviceT>){
            if constexpr (std::is_same_v<std::tuple_element_t<index, std::tuple<T...>>, DeviceT&>){
                tup = transform_tuple_element<index, DeviceT>(tup, std::get<index>(tup).to(Device::cpu));
            }else if constexpr (std::is_same_v<std::tuple_element_t<index, std::tuple<T...>>, const DeviceT&>){
                tup = transform_tuple_element<index, DeviceT>(tup, std::get<index>(tup).to(Device::cpu));
            }else{
                std::get<index>(tup) = std::get<index>(tup).to(Device::cpu);
            }
        }
        nt__run__device__func__choose__edit__one__cpu__<tuple_type, index+1, T...>(tup);
    }
}

// this changes it back to the original device type
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
                                    out(*mkl_func) (Tf...),
                                    out(*cuda_func) (Tf...),
                                   T&&... args){
    static_assert(nt__has__devicet__<T...>::value, "Error, not given a device");
    DeviceT d = get_device(args...);
    Device device = d.d;
    ensure_same_device_types(device, args...);
    if constexpr (std::is_void<out>::value){
        switch(d.d){
            case Device::meta:
                throw std::logic_error("Error, given device type of meta");
                return;
            case Device::cpu:
                cpu_func(std::forward<T>(args)...);
                return;
            case Device::mkl:
                nt__run__device__func__choose__(device, cpu_func, mkl_func, std::forward<T>(args)...);
                return;
            case Device::cuda:
                if(cuda_func == nullptr){ cpu_func(std::forward<T>(args)...); break; }
                cuda_func(std::forward<T>(args)...);
                return;
        }
    }else{
        switch(d.d){
            case Device::meta:
                throw std::logic_error("Error, given device type of meta");
                return cpu_func(std::forward<T>(args)...);
            case Device::cpu:
                return cpu_func(std::forward<T>(args)...);
            case Device::mkl:
                if(mkl_func == nullptr){ return cpu_func(std::forward<T>(args)...); }
                return mkl_func(std::forward<T>(args)...);
            case Device::cuda:
                if(cuda_func == nullptr){ return cpu_func(d, std::forward<T>(args)...); }
                return cuda_func(std::forward<T>(args)...);
        }
    }
}


#define NT_CALL_SPECIFIC_BARE_METAL_FUNCTION__(device, func_name)\
    static_cast<func_name##func_type>(NT_GET_REGISTRY(device, func_name)), 

#define NT_RUN_BARE_METAL_FUNC(func_name, ...)\
    static_assert(NT_CHECK_REGISTRY(cpu, func_name), "Error, function has not been implemented on cpu, cannot progress");\
    using func_name##func_type = decltype(NT_GET_REGISTRY(cpu, func_name));\
    nt__run__device__func__(NT_GET_DEVICES_FUNC(NT_CALL_SPECIFIC_BARE_METAL_FUNCTION__, func_name) __VA_ARGS__); 



#endif //NT_DEVICE_BARE_METAL_MACROS_H__ 
