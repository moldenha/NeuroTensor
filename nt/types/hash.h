// This is a file containing hashes for all supported types except tensors
// For when specific number types need a hash

#include "Types.h"
#include "float128.h"
#include "float16.h"
#include "bit_128_integer.h"
#include "../convert/Convert.h"
#include "../utils/always_inline_macro.h"
#include <memory> // hash
#include <type_traits>
#include <unordered_map>
#include <unordered_set>

// It is in a types namespace because these hash values are specific to just numbers defined within NeuroTensor
// These cannot be used on Tensors, or anything but scalars
namespace nt::types{

template<typename T>
struct hash{
    NT_ALWAYS_INLINE std::size_t operator()(const T& val) const noexcept{
        if constexpr (std::is_default_constructible<std::hash<T>>::value){
            return std::hash<T>{}(val);
        }else{
            return std::hash<float>{}(::nt::convert::convert<float>(val));
        }
    }
};


template<>
struct hash<nt::float16_t>{
    NT_ALWAYS_INLINE std::size_t operator()(const nt::float16_t& s) const noexcept{
        return std::hash<float>{}(nt::convert::convert<float>(s));
    }
};

template<>
struct hash<nt::complex_32>{
    NT_ALWAYS_INLINE std::size_t operator()(const nt::complex_32& s) const noexcept{
        std::size_t h1 = std::hash<float>{}(nt::convert::convert<float>(s.real()));
        std::size_t h2 = std::hash<float>{}(nt::convert::convert<float>(s.imag())); 
        return h1 ^ (h2 << 1);
    }
};


template<>
struct hash<nt::complex_64>{
    NT_ALWAYS_INLINE std::size_t operator()(const nt::complex_64& s) const noexcept{
        std::size_t h1 = std::hash<float>{}(s.real());
        std::size_t h2 = std::hash<float>{}(s.imag()); 
        return h1 ^ (h2 << 1);
    }
};


template<>
struct hash<nt::complex_128>{
    NT_ALWAYS_INLINE std::size_t operator()(const nt::complex_128& s) const noexcept{
        std::size_t h1 = std::hash<double>{}(s.real());
        std::size_t h2 = std::hash<double>{}(s.imag()); 
        return h1 ^ (h2 << 1);
    }
};

template<>
struct hash<nt::uint_bool_t>{
    NT_ALWAYS_INLINE std::size_t operator()(const nt::uint_bool_t& s) const noexcept{
        return std::hash<float>{}(s ? float(1) : float(0));
    }
};


template<>
struct hash<nt::float128_t>{
    NT_ALWAYS_INLINE std::size_t operator()(const nt::float128_t& s) const noexcept{
        if constexpr (std::is_default_constructible<std::hash<nt::float128_t>>::value){
            return std::hash<nt::float128_t>{}(s);
        }else{
            return std::hash<double>{}(convert::convert<double>(s));
        }
    }
};


    
template <>
struct hash<nt::int128_t> {
    std::size_t operator()(const nt::int128_t& value) const noexcept {
        if constexpr (std::is_default_constructible<std::hash<nt::int128_t>>::value){
            return std::hash<nt::int128_t>{}(value);
        }else{
            int64_t low  = convert::convert<int64_t>(value);
            int64_t high = convert::convert<int64_t>(value >> 64);
            // Combine high and low into a hash
            return std::hash<int64_t>{}(low) ^ (std::hash<int64_t>{}(high) << 1);
        }
    }
};

template <>
struct hash<nt::uint128_t> {
    std::size_t operator()(const nt::uint128_t& value) const noexcept {
        if constexpr (std::is_default_constructible<std::hash<nt::uint128_t>>::value){
            return std::hash<nt::uint128_t>{}(value);
        }else{
            int64_t low  = convert::convert<int64_t>(value);
            int64_t high = convert::convert<int64_t>(value >> 64);
            // Combine high and low into a hash
            return std::hash<int64_t>{}(low) ^ (std::hash<int64_t>{}(high) << 1);
        }
    }
};

// consider using robin_hood::unordered_map in the future
template<
    class Key,
    class T,
    class Hash = hash<Key>,
    class KeyEqual = std::equal_to<Key>,
    class Allocator = std::allocator<std::pair<const Key, T>> >
using unordered_map = std::unordered_map<Key, T, Hash, KeyEqual, Allocator>;

template<
    class Key,
    class Hash = hash<Key>,
    class KeyEqual = std::equal_to<Key>,
    class Allocator = std::allocator<Key> >
using unordered_set = std::unordered_set<Key, Hash, KeyEqual, Allocator>;


}



