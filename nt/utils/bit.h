// Own version of bit cast and things involving direct bit casting
#include <cstring>
#include "always_inline_macro.h"
#include "type_traits.h"
#include <utility>

namespace nt{


template<class To, class From_>
NT_ALWAYS_INLINE std::enable_if_t<
    sizeof(To) == sizeof(From_) &&
    std::is_trivially_copyable_v<From_> &&
    std::is_trivially_copyable_v<To>,
    To>
// constexpr support needs compiler magic
bit_cast(const From_& src) noexcept
{
    using From = type_traits::remove_cvref_t<std::decay_t<From_>>;
    // the union is trivially constructible
    // Therefore From_ and To don't need to be
    union u__{u__(){}; char bits[sizeof(From)]; type_traits::remove_cvref_t<To> dst;} u;
    std::memcpy(&u.dst, &src, sizeof(From));
    return u.dst;
}


// Currently there is no reason for this at the moment
// Other than an attempt to make a constexpr bit_cast for c++17
// which would only slightly speed up float32 <-> float16 conversions

// template<std::size_t N>
// struct n_size_byte_holder{
//     uint8_t bytes[N]{};


//     // constexpr n_size_byte_holder() = default;

//     // template<uint8_t... I>
//     // constexpr n_size_byte_holder()
//     // :bytes({I...})
//     // {}

//     constexpr std::size_t size_in_bits() const {
//         return N * 8;
//     }
    
//     constexpr bool get_bit(std::size_t bit_index) const {
//         return (bytes[bit_index / 8] >> (bit_index % 8)) & 1;
//     }
    
//     constexpr void set_bit(std::size_t bit_index, bool value) {
//         if (value)
//             bytes[bit_index / 8] |= (1 << (bit_index % 8));
//         else
//             bytes[bit_index / 8] &= ~(1 << (bit_index % 8));
//     }


//     // template<typename T, std::enable_if_t<sizeof(T) == N && std::is_trivially_copyable<T>::value, int> = 0>
//     // constexpr void copy_from(const T& val) {
//     //     const uint8_t* src = reinterpret_cast<const uint8_t*>(&val);
//     //     for (std::size_t i = 0; i < N; ++i) {
//     //         bytes[i] = src[i];
//     //     }
//     // }

//     // template<typename T, std::enable_if_t<sizeof(T) == N && std::is_trivially_copyable<T>::value, int> = 0>
//     // constexpr void copy_to(T& val) {
//     //     const uint8_t* src = reinterpret_cast<const uint8_t*>(&val);
//     //     for (std::size_t i = 0; i < N; ++i) {
//     //         src[i] = bytes[i];
//     //     }
//     // }
// };

// template<typename T, std::size_t... I>
// constexpr n_size_byte_holder<sizeof...(I)> kget_byte_holder(T val, std::index_sequence<I...>){
//     return n_size_byte_holder<sizeof...(I)>{ (reinterpret_cast<const uint8_t*>(&val)[I])... };
// }


// template<std::size_t bit_index, typename From, std::enable_if_t<std::is_trivially_copyable<From>::value, int> = 0>
// constexpr bool kget_bit(From val){
//     constexpr std::size_t bytes = sizeof(From);
//     static_assert(bit_index < (bytes * 8), 
//     "Error, expected the bit index to be less than the size of the input");
//     constexpr n_size_byte_holder<bytes> bit_getter = kget_byte_holder<From>(val, std::make_index_sequence<bytes>{});
//     return bit_getter.get_bit(bit_index);
// }


// template<std::size_t bit_index, bool set, typename From, std::enable_if_t<std::is_trivially_copyable<From>::value, int> = 0>
// constexpr From kset_bit(const From& val){
//     static_assert(bit_index < bytes, 
//     "Error, expected the bit index to be less than the size of the input");
//     constexpr n_size_byte_holder<bytes> bit_getter;
//     bit_getter.copy_from(val);
//     bit_getter.set_bit(bit_index, set);
//     uinion u__{u__(){}, char bits[sizeof(From)], type_traits::remove_cvref_t<From> dst} u;
    

    
// }

}
