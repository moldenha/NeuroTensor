/*
 * This header file is for casting between different types at a bit level
 * Specifically in a constexpr way
 *  - Sometimes the ability to cast between different numerical types is needed in a constexpr way within NeuroTensor
 *  - This specific class only supports numerical types created by NeuroTensor 
 *      - (Everythig that is a DType numeric equivalent except for the Tensor class <-> DType::TensorObj
 *  - Use:
 *      - constexpr Class1 ex(value);
 *      - constexpr Class2 Bits = nt::kbit_cast<Class2>(ex);
 *  - Limitations:
 *      - sizeof(Class1) == sizeof(Class2)
 *      - Class1 and Class2 are some type of nt::(number type) thatis all it supports 
*/

#ifndef NT_BIT_KBIT_CAST_H__
#define NT_BIT_KBIT_CAST_H__

#include "../utils/always_inline_macro.h"
#include "../utils/type_traits.h"
#include "../types/Types.h"
#include "bitset.h"
#include "float_bits.h"
#include "complex_bits.h"

namespace nt{

namespace kbit_cast_details{

template<typename In, typename Out>
inline constexpr Out kbit_cast_integral_to_integral(const In& src) noexcept {
    static_assert(type_traits::is_integral_v<In>, 
                  "Error,  kbit_cast_signed_unsigned_integers expects integral values");
    static_assert(type_traits::is_integral_v<Out>, 
                  "Error,  kbit_cast_signed_unsigned_integers expects integral values");
    static_assert(sizeof(In) == sizeof(Out),
                  "Error,  kbit_cast_signed_unsigned_integers expects In and Out types to be the same size");
    
    constexpr std::size_t num_bits = sizeof(In) * CHAR_BIT;
    ::nt::bitset<num_bits, In> in_value(src);
    ::nt::bitset<num_bits, Out> out_value(0);
    
    for(std::size_t i = 0; i < num_bits; ++i){
        if(in_value[i]){
            out_value.set(i, true);
        }
    }

    return out_value.lo_type();
}

template<typename In, typename Out>
inline constexpr Out kbit_cast_floating_to_floating(const In& src) noexcept {
    static_assert(type_traits::is_floating_point_v<In>, 
                  "Error,  kbit_cast_floats_to_floats expects floating values");
    static_assert(type_traits::is_floating_point_v<Out>, 
                  "Error,  kbit_cast_floats_to_floats expects floating values");
    static_assert(sizeof(In) == sizeof(Out),
                  "Error,  kbit_cast_floats_to_floats expects In and Out types to be the same size");
    
    ::nt::float_bits<In> start(src);
    ::nt::float_bits<Out> end(0);
    constexpr std::size_t num_bits = decltype(end)::NUM_BITS;
    
    for(std::size_t i = 0; i < num_bits; ++i){
        if(start[i]){
            end.set(i, true);
        }
    }

    return end.get();
}


template<typename In, typename Out>
inline constexpr Out kbit_cast_floating_to_integral(const In& src) noexcept {
    static_assert(type_traits::is_floating_point_v<In>, 
                  "Error,  kbit_cast_floats_to_integers expects input floating values");
    static_assert(type_traits::is_floating_point_v<Out>, 
                  "Error,  kbit_cast_floats_to_integers expects integral out values");
    static_assert(sizeof(In) == sizeof(Out),
                  "Error,  kbit_cast_floats_to_integers expects In and Out types to be the same size");
 
    ::nt::float_bits<In> start(src);
    typename ::nt::float_bits<In>::integer_type cur_type = start.get_tracker().lo_type();
    if constexpr (type_traits::is_same_v<typename ::nt::float_bits<In>::integer_type, Out>){
        return cur_type;
    }else{
        return kbit_cast_integral_to_integral<Out>(cur_type); 
    } 
}

template<typename In, typename Out>
inline constexpr Out kbit_cast_integral_to_floating(const In& src) noexcept {
    static_assert(type_traits::is_floating_point_v<Out>, 
                  "Error,  kbit_cast_integers_to_floats expects output floating values");
    static_assert(type_traits::is_integral_v<In>, 
                  "Error,  kbit_cast_integers_to_floats expects integral input values");
    static_assert(sizeof(In) == sizeof(Out),
                  "Error,  kbit_cast_floats_to_integers expects In and Out types to be the same size");
    
    return ::nt::float_bits<Out>(src).get();
    if constexpr (type_traits::is_same_v<typename ::nt::float_bits<Out>::integer_type, In>){
        return ::nt::float_bits<Out, In>(src);
    }
    ::nt::float_bits<In> start(src);
    typename ::nt::float_bits<In>::integer_type cur_type = start.get_tracker().lo_type();
    if constexpr (type_traits::is_same_v<typename ::nt::float_bits<In>::integer_type, Out>){
        return cur_type;
    }else{
        return kbit_cast_integral_to_integral<Out>(cur_type); 
    } 
}

template<typename In, typename Out>
inline constexpr Out kbit_cast_complex_to_floating(const In& src) noexcept {
    static_assert(type_traits::is_complex_v<In>, 
                  "Error,  kbit_cast_complex_to_floats expects complex values in");
    static_assert(type_traits::is_floating_point_v<Out>, 
                  "Error,  kbit_cast_complex_to_floats expects floating values out");
    static_assert(sizeof(In) == sizeof(Out),
                  "Error,  kbit_cast_complex_to_floats expects In and Out types to be the same size");
    
    ::nt::complex_bits<In> start(src);
    ::nt::float_bits<Out> end(0);
    constexpr std::size_t num_bits = decltype(end)::NUM_BITS;
    
    for(std::size_t i = 0; i < num_bits; ++i){
        if(start[i]){
            end.set(i, true);
        }
    }
    return end.get();
}


template<typename In, typename Out>
inline constexpr Out kbit_cast_floating_to_complex(const In& src) noexcept {
    static_assert(type_traits::is_complex_v<Out>, 
                  "Error,  kbit_cast_floats_to_complex expects complex values out");
    static_assert(type_traits::is_floating_point_v<In>, 
                  "Error,  kbit_cast_floats_to_complex expects floating values in");
    static_assert(sizeof(In) == sizeof(Out),
                  "Error,  kbit_cast_floats_to_complex expects In and Out types to be the same size");
    
    ::nt::float_bits<In> start(src);
    ::nt::complex_bits<Out> end(0);
    constexpr std::size_t num_bits = decltype(end)::NUM_BITS;
    
    for(std::size_t i = 0; i < num_bits; ++i){
        if(start[i]){
            end.set(i, true);
        }
    }
    return end.get();
}


template<typename In, typename Out>
inline constexpr Out kbit_cast_integral_to_complex(const In& src) noexcept {
    return kbit_cast_floats_to_complex(kbit_cast_integers_to_floats(src));
}

template<typename In, typename Out>
inline constexpr Out kbit_cast_complex_to_integral(const In& src) noexcept {
    return kbit_cast_floats_to_integers(kbit_cast_complex_to_floats(src));
}

template<typename In, typename Out>
inline constexpr Out kbit_cast_complex_to_complex(const In& src) noexcept {
    static_assert(type_traits::is_complex_v<Out>, 
                  "Error,  kbit_cast_complex_to_complex expects complex values out");
    static_assert(type_traits::is_complex_v<In>, 
                  "Error,  kbit_cast_complex_to_complex expects complex values in");
    static_assert(sizeof(In) == sizeof(Out),
                  "Error,  kbit_cast_complex_to_complex expects In and Out types to be the same size");

    ::nt::complex_bits<In> start(src);
    ::nt::complex_bits<Out> end;
    constexpr std::size_t num_bits = decltype(end)::NUM_BITS;
    for(std::size_t i = 0; i < num_bits; ++i){
        if(start[i]){
            end.set(i, true);
        }
    }
    return end.get();
}

template<typename In, typename Out>
inline constexpr Out kbit_cast_meta(const In& src) noexcept {
    return Out();
}

enum class kbit_cast_val_type{
    integral = 0,
    floating = 1,
    complex = 2,
    meta = 3
};

template<typename T>
inline constexpr kbit_cast_val_type get_cast(){
    using Type = type_traits::remove_cvref_t<type_traits::decay_t<T>>;
    if constexpr (type_traits::is_integral_v<Type>){
        return kbit_cast_val_type::integral;
    }else if constexpr (type_traits::is_complex_v<Type>){
        return kbit_cast_val_type::complex;
    }else if constexpr (type_traits::is_floating_point_v<Type>){
        return kbit_cast_val_type::floating;
    }else{
        return kbit_cast_val_type::meta;
    }
}


// ORGANIZED TEMPLATE SPECIALIZATION FOR RUNNING BITCAST

// template<typename In, typename Out, kbit_cast_val_type a, kbit_cast_val_type b>
// inline constexpr Out run(const In& src){
//     return kbit_cast_meta<Out>(src);
// }

template<kbit_cast_val_type A, kbit_cast_val_type B>
struct kbit_cast_runner {
    template<typename In, typename Out>
    static constexpr Out run(const In& src) {
        return kbit_cast_meta<Out>(src);
    }
};

#define NT_MAKE_BITCAST_RUN_SPECIALIZTION(A, B)                         \
template<>                                                              \
struct kbit_cast_runner<kbit_cast_val_type::A,                          \
                        kbit_cast_val_type::B> {                        \
    template<typename In, typename Out>                                 \
    static constexpr Out run(const In& src) {                           \
        return kbit_cast_##A##_to_##B<Out>(src);                        \
    }                                                                   \
};

NT_MAKE_BITCAST_RUN_SPECIALIZTION(integral, integral);
NT_MAKE_BITCAST_RUN_SPECIALIZTION(integral, floating);
NT_MAKE_BITCAST_RUN_SPECIALIZTION(integral, complex);
NT_MAKE_BITCAST_RUN_SPECIALIZTION(floating, floating);
NT_MAKE_BITCAST_RUN_SPECIALIZTION(floating, integral);
NT_MAKE_BITCAST_RUN_SPECIALIZTION(floating, complex);
NT_MAKE_BITCAST_RUN_SPECIALIZTION(complex, complex);
NT_MAKE_BITCAST_RUN_SPECIALIZTION(complex, integral);
NT_MAKE_BITCAST_RUN_SPECIALIZTION(complex, floating);

#undef NT_MAKE_BITCAST_RUN_SPECIALIZTION

// END ORGANIZED TEMPLATE SPECIALIZATION FOR RUNNING BITCAST


template<typename T>
struct is_kbit_castable : type_traits::bool_constant<
                    type_traits::is_arithmetic_v<T> || type_traits::is_complex_v<T> || type_traits::is_same_v<::nt::uint_bool_t, T>
        > {};

template<typename T>
inline static constexpr bool is_kbit_castable_v = is_kbit_castable<type_traits::decay_t<T>>::value;

}

template<class To, class From_>
NT_ALWAYS_INLINE std::enable_if_t<
    sizeof(To) == sizeof(From_) &&
    kbit_cast_details::is_kbit_castable_v<To> &&
    kbit_cast_details::is_kbit_castable_v<From_> ,
    To>
// constexpr support needs compiler magic
kbit_cast(const From_& src) noexcept
{

    using From = type_traits::remove_cvref_t<type_traits::decay_t<From_>>;
    if constexpr(type_traits::is_same_v<::nt::uint_bool_t, From>){
        return kbit_cast<To>(src.value); // inputted uint8_t
    }else if constexpr (type_traits::is_same_v<::nt::uint_bool_t, To>){
        return ::nt::uint_bool_t(kbit_cast<uint8_t>(src));
    }else{
        return ::nt::kbit_cast_details::kbit_cast_runner<::nt::kbit_cast_details::get_cast<From>(), ::nt::kbit_cast_details::get_cast<To>()>::template run<From, To>(src);
    }
}


}

#endif
