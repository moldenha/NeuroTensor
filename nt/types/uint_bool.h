/*
 * This type is for booleans and to have more control over their behavior
 *
 */

#ifndef NT_TYPES_UINT_BOOL_H__
#define NT_TYPES_UINT_BOOL_H__
#include <cstdint>

namespace nt{



struct uint_bool_t{
    // Note: the value : 1 basically just means memory wise this would be the first bit
    // However, in this constext, because there are not enough types in here to make a bit
    // it does not make a difference for memory, so instead, it is more of a mental note
	uint8_t value : 1;
	constexpr uint_bool_t():value(0) {}
	constexpr uint_bool_t(const bool& val) : value(val ? 1 : 0) {;}
	constexpr uint_bool_t(const uint_bool_t& val) : value(val.value) {;}
	constexpr uint_bool_t(uint_bool_t&& val) : value(val.value) {val.value = 0;}
	inline constexpr uint_bool_t& operator=(const bool& val) noexcept {value = val ? 1 : 0; return *this;}
	inline constexpr uint_bool_t& operator=(const uint8_t &val) noexcept {value = val > 0 ? 1 : 0; return *this;}
	inline constexpr uint_bool_t& operator=(const uint_bool_t &val) noexcept {value = val.value; return *this;}
	inline constexpr uint_bool_t& operator=(uint_bool_t&& val) noexcept {value = val.value; return *this;}
    inline constexpr operator bool() const noexcept {return value == 1;}
};

inline constexpr bool operator==(const uint_bool_t& a, const uint_bool_t& b) noexcept {
    return a.value == b.value;
}

inline constexpr bool operator==(const bool& a, const uint_bool_t& b) noexcept {
    return a == bool(b);
}

inline constexpr bool operator==(const uint_bool_t& a, const bool& b) noexcept {
    return bool(a) == b;
}

}


#endif
