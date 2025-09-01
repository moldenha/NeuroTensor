#ifndef NT_RANGES_H__
#define NT_RANGES_H__

#include <iostream>
#include <functional>
#include <numeric>
#include <cstdint>
#include "../utils/api_macro.h"
#include <utility>

namespace nt{


#define NT_RANGE_CMP_OPERATOR_(op) inline constexpr bool operator op (const range_& r) const {return this->begin op r.begin && this->end op r.end;}

//Note: constexpr functions since C++17 are implicitly inline
//      The constexpr functions are marked as inline for readability
struct range_{
	int64_t begin, end; // begining and end of the range_
	constexpr range_() : begin(0), end(-1) {}
    constexpr range_(const int64_t& s) : begin(s), end(s+1) {}
    constexpr range_(const int64_t& s, const int64_t& e) : begin(s), end(e) {}
    constexpr range_(const range_& r) : begin(r.begin), end(r.end) {}
    range_(range_&& r) : begin(std::exchange(r.begin, 0)), end(std::exchange(r.end, 0)) {}
    inline constexpr range_& operator=(const range_& r){this->begin = r.begin; this->end = r.end; return *this;}
    inline range_& operator=(range_&& r){this->begin = std::exchange(r.begin, 0); this->end = std::exchange(r.end, 0); return *this;}
    
    inline constexpr range_ operator>(const int64_t& e) const {return range_(begin, e);}
    inline constexpr range_ operator()(const int64_t& s, const int64_t& e) const {return range_(s, e);}
    inline constexpr range_ operator()(const int64_t& s) const {return range_(s, -1);}
    NT_RANGE_CMP_OPERATOR_(==);
    NT_RANGE_CMP_OPERATOR_(<);
    NT_RANGE_CMP_OPERATOR_(>);
    NT_RANGE_CMP_OPERATOR_(<=);
    NT_RANGE_CMP_OPERATOR_(>=);

    inline constexpr range_ operator+(const int64_t& v) const {return range_(begin, end + v);}
    inline constexpr range_ operator-(const int64_t& v) const {return range_(begin, end - v);}
    inline constexpr range_& operator+=(const int64_t& v) {begin += v; end += v; return *this;}
    inline constexpr range_& operator-=(const int64_t& v) {begin -= v; end -= v; return *this;}
    range_& fix(int64_t s);
	inline constexpr int64_t length() const {return end - begin;}
};

#undef NT_RANGE_CMP_OPERATOR_ 

inline constexpr range_ operator<(const int64_t& s, const range_ r){return range_(s, r.end);}
inline static constexpr range_ range = range_();

NEUROTENSOR_API std::ostream& operator<<(std::ostream& out, const range_& v);
NEUROTENSOR_API std::ostream& operator<<(std::ostream& out, const std::vector<range_>& v);
NEUROTENSOR_API std::ostream& operator<<(std::ostream& out, const std::vector<std::vector<range_> >& v);


}


#endif
