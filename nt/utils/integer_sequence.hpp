#ifndef NT_UTILS_INTEGER_SQUENCE_HPP__
#define NT_UTILS_INTEGER_SQUENCE_HPP__

#include "type_traits.h"
#include <iostream>

namespace nt::utils{

template<typename T, T... Ints>
struct integer_sequence{
    static_assert(type_traits::is_integral_v<T>, "Error, expected integer sequence to be of integers");
    using value_type = T;
    static constexpr size_t size = sizeof...(Ints);
};

template< std::size_t... Ints >
using index_sequence = integer_sequence<std::size_t, Ints...>;

namespace detail {
template<class T, T I, T N, T... integers>
struct make_integer_sequence_helper
{
    using type = typename make_integer_sequence_helper<T, I + 1, N, integers..., I>::type;
};

template<class T, T N, T... integers>
struct make_integer_sequence_helper<T, N, N, integers...>
{
    using type = integer_sequence<T, integers...>;
};

}

template<class T, T N>
using make_integer_sequence = typename detail::make_integer_sequence_helper<T, 0, N>::type;

template<std::size_t N>
using make_index_seqence = make_integer_sequence<std::size_t, N>;

template<class T, T Check, T... Integers>
inline constexpr bool is_in_integer_sequence(integer_sequence<T, Integers...>){
    return ((Check == Integers) || ...);
}

template<std::size_t Check, std::size_t... Integers>
inline constexpr bool is_in_index_sequence(index_sequence<Integers...>){
    return ((Check == Integers) || ...);
}

namespace detail{
template<class T, T Check, T N, T I, T... integers>
inline constexpr T index_of_integer_sequence_helper(integer_sequence<T, I, integers...>){
    return (Check == I) ? N : index_of_integer_sequence_helper<T, Check, N+1, integers...>();
}

template<class T, T Check, T N, T I>
inline constexpr T index_of_integer_sequence_helper(integer_sequence<T, I>){
    return (Check == I) ? N : N + 1;
}

template<class T, T Check, T N>
inline constexpr T index_of_integer_sequence_helper(integer_sequence<T>){
    return N + 1;
}


}

template<class T, T Check, T... integers>
inline constexpr T index_of_integer_sequence(integer_sequence<T, integers...>){
    return detail::index_of_integer_sequence_helper<T, Check, 0, integers...>(integer_sequence<T, integers...>{});
}

template<std::size_t Check, std::size_t... integers>
inline constexpr std::size_t index_of_index_sequence(index_sequence<integers...>){
    return detail::index_of_integer_sequence_helper<std::size_t, Check, 0, integers...>(index_sequence<integers...>{});
}

namespace detail{

template<typename Seq1, typename Seq2>
struct concat_integer_sequence;

template<typename T, T... A, T... B>
struct concat_integer_sequence<
    integer_sequence<T, A...>,
    integer_sequence<T, B...>
> {
    using type = integer_sequence<T, A..., B...>;
};

}

template<typename Seq1, typename Seq2>
using integer_sequence_concat = detail::concat_integer_sequence<Seq1, Seq2>;

template<typename Seq1, typename Seq2>
using index_sequence_concat = detail::concat_integer_sequence<Seq1, Seq2>;

namespace detail{

template<typename T, typename Check, typename Tuple, size_t I, size_t N, T... integers>
struct is_same_integer_sequence_helper
{
    using type = type_traits::conditional_t<
                    type_traits::is_decay_same_v<Check, typename std::tuple_element<I, Tuple>::type>,
                        typename is_same_integer_sequence_helper<T, Check, Tuple, I + 1, N, integers..., T(I)>::type,
                        typename is_same_integer_sequence_helper<T, Check, Tuple, I + 1, N, integers...>::type
    >;
};

template<class T, typename Check, typename Tuple, T N, T... integers>
struct is_same_integer_sequence_helper<T, Check, Tuple, N, N, integers...>
{
    using type = integer_sequence<T, integers...>;
};



}


// these are used to make an integer sequence where given a variadic template parameter, it returns the indexes corresponding to
// that specific type
template<typename T, typename Check, typename... Args>
using is_same_integer_sequence = typename detail::is_same_integer_sequence_helper<T, Check, std::tuple<Args...>, 
                                                                                    0, sizeof...(Args)>::type;
template<typename Check, typename... Args>
using is_same_index_sequence = typename detail::is_same_integer_sequence_helper<std::size_t, Check, std::tuple<Args...>, 
                                                                                        0, sizeof...(Args)>::type;


// checks if a specific number is inside of an integer sequence
template<typename T, T... integers>
inline constexpr bool is_integer_sequence_in(T Check, integer_sequence<T, integers...>){
    return ((Check == integers) || ...);
}

// checks if a specific number is inside of an integer sequence
template<std::size_t... integers>
inline constexpr bool is_index_sequence_in(std::size_t Check, integer_sequence<std::size_t, integers...>){
    return ((Check == integers) || ...);
}

// makes a constexpr std::array from an integer sequence
template<typename T, T... integers>
inline constexpr std::array<T, sizeof...(integers)> make_integer_array(integer_sequence<T, integers...>){
    return std::array<T, sizeof...(integers)>{integers, ...}
}

template<std::size_t... integers>
inline constexpr std::array<std::size_t, sizeof...(integers)> make_index_array(integer_sequence<std::size_t, integers...>){
    return std::array<std::size_t, sizeof...(integers)>{integers, ...}
}


}


// prints an integer sequence
template<typename T, T... integers>
inline std::ostream& operator<<(std::ostream& os, nt::utils::integer_sequence<T, integers...>){
    os <<  nt::utils::integer_sequence<T, integers...>::size << ": ";
    ((os << integers << " "), ...);
    return os;
}

#endif
