#ifndef _UTILS_H
#define _UTILS_H
#include <iostream>
#include <sstream>
#include <string_view>

#include <cstddef>
#include <string>
#include <stdexcept>
#include <complex>
#include "../types/Types.h"
#include <initializer_list>
#include <type_traits>
#include <vector>
#include <tuple>
#include <variant>

namespace nt{
template<class T>struct tag_t{};
template<class T>constexpr tag_t<T> tag{};

namespace utils{

/* template <typename T> */
/* struct IsInitializerList : std::false_type {}; */

/* template <typename T> */
/* struct IsInitializerList<std::initializer_list<T>> : std::true_type {}; */

/* template<typename T> */
/* constexpr bool IsInitializerList_t = IsInitializerList<T>::value; */


/* template <typename T> */
/* struct InitializerList_size{ */
/* 	using type = std::conditional_t<IsInitializerList_t<T>, T::value_type, T>; */
/* 	std::size_t value = !IsInitializerList_t<T> ? 0 : 1 + InitializerList_size<type>::value; */
/* }; */


template<typename T, std::size_t N>
struct NestedInitializerLists_type{
	using type = typename std::initializer_list<typename NestedInitializerLists_type<T, N-1>::type>;
};

template<typename T>
struct NestedInitializerLists_type<T, 1>{
	using type = typename std::initializer_list<T>;
};

template<typename T, std::size_t N, std::enable_if_t<N == 1, bool> = true>
inline void get_shape(const typename NestedInitializerLists_type<T, N>::type& v, std::vector<uint32_t>& s){
	s.back() = v.size();
}

template<typename T, std::size_t N, std::enable_if_t<N != 1, bool> = true>
inline void get_shape(const typename NestedInitializerLists_type<T, N>::type& v, std::vector<uint32_t>& s){
	s[s.size() - N] = v.size();
	get_shape<T, N-1>(*v.begin(), s);

}


template<typename T, std::size_t N, std::enable_if_t<N == 1, bool> = true>
inline bool validate_shape(const typename NestedInitializerLists_type<T, N>::type& v, std::vector<uint32_t>& s){
	return s.back() == v.size();
}

template<typename T, std::size_t N, std::enable_if_t<N != 1, bool> = true>
inline bool validate_shape(const typename NestedInitializerLists_type<T, N>::type& v, std::vector<uint32_t>& s){
	if(s[s.size() - N] != v.size())
		return false;
	for(auto begin = v.begin(); begin != v.end(); ++begin){
		if(!validate_shape<T, N-1>(*begin, s))
			return false;
	}
	return true;
}



template<typename T>
inline void print_format_inner(std::ostringstream& out, std::string_view& str, size_t& last_pos, const T& var){
	size_t pos = str.substr(last_pos, str.size()).find('$');
	if(pos == std::string_view::npos){
		if(str.size() == last_pos)
			return;
		out << str.substr(last_pos, str.size());
		last_pos = str.size();
		return;
	}
	out << str.substr(last_pos, pos);
	out << var;
	last_pos += pos + 1;
}

template<typename... Args>
inline void throw_exception(const bool err, std::string_view str, const Args &...arg){
	if(err)
		return;
	std::ostringstream out;
	size_t last_pos = 0;
	(print_format_inner(out, str, last_pos, arg), ...);
	throw std::invalid_argument(out.str());
}

template<typename T, std::size_t N>
inline std::vector<uint32_t> aquire_shape(const typename NestedInitializerLists_type<T, N>::type& v){
	std::vector<uint32_t> shape(N);
	get_shape<T, N>(v, shape);
	if(!validate_shape<T, N>(v, shape))
		throw std::invalid_argument("\nRuntime Error: Got inconsistent shape");
	return std::move(shape);
}

template<typename T, std::size_t N, typename UnaryFunction, std::enable_if_t<N == 1, bool> = true>
inline void flatten_func(const typename NestedInitializerLists_type<T, N>::type& v, UnaryFunction&& unary_op){
	for(auto begin = v.begin(); begin != v.end(); ++begin){
		unary_op(*begin);
	}
}

template<typename T, std::size_t N, typename UnaryFunction, std::enable_if_t<N != 1, bool> = true>
inline void flatten_func(const typename NestedInitializerLists_type<T, N>::type& v, UnaryFunction&& unary_op){
	for(auto begin = v.begin(); begin != v.end(); ++begin){
		flatten_func<T, N-1, UnaryFunction>(*begin, std::forward<UnaryFunction&&>(unary_op));
	}
}


class my_tuple{
	uint32_t first, second;
	public:
		inline my_tuple(const uint32_t a, const uint32_t b)
			:first(a), second(b)
		{}

		inline my_tuple(const uint32_t a)
			:first(a), second(a)
		{}

		inline my_tuple(std::tuple<uint32_t, uint32_t> t)
			:first(std::get<0>(t)), second(std::get<1>(t))
		{}

		inline my_tuple& operator=(uint32_t x){first = x; second = x; return *this;}
		inline my_tuple& operator=(std::tuple<uint32_t, uint32_t> x){first = std::get<0>(x); second = std::get<1>(x); return *this;}
		inline const uint32_t& operator[](const uint32_t x) const {if(x == 0){return first;}return second;}
		inline bool operator==(const uint32_t x) const {return (first == x && second == x);}
		inline bool operator!=(const uint32_t x) const {return (first != x && second != x);}
		inline bool operator > (const uint32_t x) const {return (first > x && second > x);}
		inline bool operator < (const uint32_t x) const {return (first < x && second < x);}
		inline bool operator >= (const uint32_t x) const {return (first >= x && second >= x);}
		inline bool operator <= (const uint32_t x) const {return (first <= x && second <= x);}
};

class tuple_or_int{
	std::variant<uint32_t, std::tuple<uint32_t, uint32_t>> val;
	public:
		inline tuple_or_int(uint32_t x)
			:val(x)
		{}

		inline tuple_or_int(std::tuple<uint32_t, uint32_t> x)
			:val(x)
		{}

		inline tuple_or_int& operator=(uint32_t x){val = x; return *this;}
		inline tuple_or_int& operator=(std::tuple<uint32_t, uint32_t> x){val = x; return *this;}
		inline std::tuple<uint32_t, uint32_t> to_tuple() const{
			if(std::holds_alternative<std::tuple<uint32_t, uint32_t>>(val)){
				return std::get<1>(val);
			}
			return std::tuple<uint32_t, uint32_t>(std::get<0>(val), std::get<0>(val));
		}
		inline operator std::tuple<uint32_t, uint32_t>() const {return to_tuple();}
		inline my_tuple to_my_tuple() const {
			if(std::holds_alternative<std::tuple<uint32_t, uint32_t>>(val)){
				return my_tuple(std::get<1>(val));
			}
			return my_tuple(std::get<0>(val), std::get<0>(val));
	
		}
};

}
}


namespace nt{
namespace detect_number {
	template<class T, class...Ts>
	constexpr bool is_charlike(tag_t<T>, Ts&&...){ return false; }
	constexpr bool is_charlike( tag_t<double> ){ return true; }
	constexpr bool is_charlike( tag_t<float> ){ return true; }
	constexpr bool is_charlike( tag_t<int32_t> ){ return true; }
	constexpr bool is_charlike( tag_t<uint32_t> ){ return true;}
	constexpr bool is_charlike( tag_t<int64_t> ){return true;}
	constexpr bool is_charlike( tag_t<int16_t> ){ return true; }
	constexpr bool is_charlike( tag_t<uint16_t> ){ return true; }
	constexpr bool is_charlike( tag_t<int8_t> ){ return true; }
	constexpr bool is_charlike( tag_t<uint8_t> ){ return true; }
	template<class T>
	constexpr bool detect=is_charlike(tag<T>);
};

namespace detect_number_pc {
	template<class T, class...Ts>
	constexpr bool is_charlike(tag_t<T>, Ts&&...){ return false; }
	constexpr bool is_charlike( tag_t<double> ){ return true; }
	constexpr bool is_charlike( tag_t<float> ){ return true; }
	constexpr bool is_charlike( tag_t<complex_128 >){ return true; }
	constexpr bool is_charlike( tag_t<complex_64 >){ return true; }
	constexpr bool is_charlike( tag_t<int32_t> ){ return true; }
	constexpr bool is_charlike( tag_t<uint32_t> ){ return true;}
	constexpr bool is_charlike( tag_t<int64_t> ){return true;}
	constexpr bool is_charlike( tag_t<int16_t> ){ return true; }
	constexpr bool is_charlike( tag_t<uint16_t> ){ return true; }
	constexpr bool is_charlike( tag_t<int8_t> ){ return true; }
	constexpr bool is_charlike( tag_t<uint8_t> ){ return true; }
	template<class T>
	constexpr bool detect=is_charlike(tag<T>);
};





}

#endif
