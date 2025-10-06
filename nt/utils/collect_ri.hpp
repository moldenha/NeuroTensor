// This is a helper file to collect all variadic template parameters into a vector of integers or ranges or ranges + integers
// so that basically something like this could be done:
//
// auto a = t.dilate(1, 2, 3)
// instead of:
// auto a = t.dilate({1, 2, 3})
//
//and implementation would be:
//
//template<typename... Args>
//inline Tensor dilate(int64_t i, Args&&... args) const{
//  std::vector<int64_t> vec;
//  vec.reserve(sizeof...(Args) + 1);
//  utils::collect_integers_impl(vec, i, std::forward<Args>(args)...);
//  this->dilate(std::move(vec);
//}
//
//
//on the other hand it also allows for this function:
//
//auto a = t(1, 1 <range> 10, 1 <range, range> 10);
//For when there is a mix of ranges and solid integers, it is all compiled into a vector of ranges
//to then be processed
#ifndef NT_COLLECT_INTEGERS_AND_RANGES_TO_VEC_HPP__
#define NT_COLLECT_INTEGERS_AND_RANGES_TO_VEC_HPP__
#include "../dtype/ranges.h"
#include <type_traits>
#include <vector>

namespace nt::utils{

template<typename... Args>
struct is_all_integers : std::true_type {};
template<typename Arg, typename... Args>
struct is_all_integers<Arg, Args...>{
    static constexpr bool value = std::is_integral_v<std::decay_t<Arg>> && is_all_integers<Args...>::value;
};

template<typename... Args>
struct is_all_ranges : std::true_type {};
template<typename Arg, typename... Args>
struct is_all_ranges<Arg, Args...>{
    static constexpr bool value = (std::is_integral_v<std::decay_t<Arg>> || std::is_same_v<range_, std::decay_t<Arg>>) && is_all_ranges<Args...>::value;
};

inline void collect_ranges_impl(std::vector<range_>&){;}

template<typename Arg, typename... Args>
inline void collect_ranges_impl(std::vector<range_>& ranges, Arg&& arg, Args&&... args){
    if constexpr (std::is_integral_v<Arg>){
        ranges.emplace_back(arg, arg+1);
    }else{
        ranges.emplace_back(std::forward<Arg>(arg));
    }
    collect_ranges_impl(ranges, std::forward<Args>(args)...);
}


template<typename outInteger>
inline void collect_integers_impl(std::vector<outInteger>&){;}
template<typename outInteger, typename Arg, typename... Args>
inline void collect_integers_impl(std::vector<outInteger>& integers, Arg&& arg, Args&&... args){
    integers.emplace_back(std::forward<Arg>(arg));
    collect_integers_impl(integers, std::forward<Args>(args)...);
}

template<typename outInteger, typename Arg, typename... Args>
inline decltype(auto) collect_integers_or_ranges(Arg&& arg, Args&&... args){
    static_assert(is_all_integers<Arg, Args...>::value || is_all_ranges<Arg, Args...>::value, "Expected to get all ranges or integers or a mix for this function");
   if constexpr (is_all_integers<Arg, Args...>::value){
        std::vector<outInteger> ints;
        ints.reserve(sizeof...(Args) + 1);
        collect_integers_impl(ints, std::forward<Arg>(arg), std::forward<Args>(args)...);
        return ints;
    }else{
        std::vector<range_> ranges;
        ranges.reserve(sizeof...(Args) + 1);
        collect_ranges_impl(ranges, std::forward<Arg>(arg), std::forward<Args>(args)...);
        return ranges;
    }
}

}

#endif
