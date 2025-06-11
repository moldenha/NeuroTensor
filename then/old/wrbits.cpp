#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <ios>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <string>
#include <cstring>
#include <memory>
#include <optional>
#include "wrbits.h"
#include <complex>
#include "../Tensor.h"
#include "../dtype/DType.h"

namespace nt{

template <typename T>
writer<T>::writer(const char* fname)
	:of(std::ofstream(fname,  std::ios::out | std::ios::binary | std::ios::trunc))
{}

template<typename T>
void writer<T>::operator()(const T& inp){std::memcpy(_arr.data(), &inp, Size);of->write(_arr.data(), Size);}

template<typename T>
bool writer<T>::write_num(T inp, std::ostream& outfile) {
    std::memcpy(_arr.data(), &inp, Size);
    return static_cast<bool>(outfile.write(_arr.data(), Size));
}
template<typename T>
bool writer<T>::write_num(T inp){
    std::memcpy(_arr.data(), &inp, Size);
    return static_cast<bool>(of->write(_arr.data(), Size));
}


template<typename T>
appender<T>::appender(std::ofstream& _of)
		:of(_of) {}

template<typename T>
void appender<T>::operator()(const T& inp){std::memcpy(_arr.data(), &inp, Size);of.write(_arr.data(), Size);}

template<typename T>
bool appender<T>::write_num(T inp, std::ostream& outfile) {
    std::memcpy(_arr.data(), &inp, Size);
    return static_cast<bool>(outfile.write(_arr.data(), Size));
}
template<typename T>
bool appender<T>::write_num(T inp){
    std::memcpy(_arr.data(), &inp, Size);
    return static_cast<bool>(of.write(_arr.data(), Size));
}

template<typename T>
bool bracket_appender<T>::write_list(const T *begin, const T *end, std::ofstream &outfile){
	outfile<<"{";
	bool check = true;
	for(;begin != end; ++begin){
		std::memcpy(_arr.data(), begin, Size);
		check = static_cast<bool>(outfile.write(_arr.data(), Size));
		if(check == false)
			return false;
	}
	outfile << "}";
	return true;
}


template<typename T>
bool reader<T>::add_nums(std::ifstream &in){
	return static_cast<bool>(in.read(_arr.data(), Size));
}



template<typename T>
bool reader<T>::convert() {
    /* if(std::any_of(_arr.cbegin(), _arr.cend(), [](auto val){return (int)val == -1;})) return false; */
    std::memcpy(&outp, _arr.data(), Size);
    return true;
}


template struct reader<Tensor>;
template struct reader<float>;
template struct reader<double>;
template struct reader<complex_64>;
template struct reader<complex_128>;
template struct reader<int64_t>;
template struct reader<uint32_t>;
template struct reader<int32_t>;
template struct reader<uint16_t>;
template struct reader<int16_t>;
template struct reader<uint8_t>;
template struct reader<int8_t>;
template struct reader<uint_bool_t>;
#ifdef _HALF_FLOAT_SUPPORT_
template struct reader<float16_t>;
template struct reader<complex_32>;
#endif
#ifdef __SIZEOF_INT128__
template struct reader<int128_t>;
template struct reader<uint128_t>;
#endif
#ifdef _128_FLOAT_SUPPORT_
template struct reader<float128_t>;
#endif

template struct bracket_appender<Tensor>;
template struct bracket_appender<float>;
template struct bracket_appender<double>;
template struct bracket_appender<complex_64>;
template struct bracket_appender<complex_128>;
template struct bracket_appender<int64_t>;
template struct bracket_appender<uint32_t>;
template struct bracket_appender<int32_t>;
template struct bracket_appender<uint16_t>;
template struct bracket_appender<int16_t>;
template struct bracket_appender<uint8_t>;
template struct bracket_appender<int8_t>;
template struct bracket_appender<uint_bool_t>;
#ifdef _HALF_FLOAT_SUPPORT_
template struct bracket_appender<float16_t>;
template struct bracket_appender<complex_32>;
#endif
#ifdef __SIZEOF_INT128__
template struct bracket_appender<int128_t>;
template struct bracket_appender<uint128_t>;
#endif
#ifdef _128_FLOAT_SUPPORT_
template struct bracket_appender<float128_t>;
#endif


template struct appender<Tensor>;
template struct appender<float>;
template struct appender<double>;
template struct appender<complex_64>;
template struct appender<complex_128>;
template struct appender<int64_t>;
template struct appender<uint32_t>;
template struct appender<int32_t>;
template struct appender<uint16_t>;
template struct appender<int16_t>;
template struct appender<uint8_t>;
template struct appender<int8_t>;
template struct appender<uint_bool_t>;
#ifdef _HALF_FLOAT_SUPPORT_
template struct appender<float16_t>;
template struct appender<complex_32>;
#endif
#ifdef __SIZEOF_INT128__
template struct appender<int128_t>;
template struct appender<uint128_t>;
#endif
#ifdef _128_FLOAT_SUPPORT_
template struct appender<float128_t>;
#endif

template struct writer<Tensor>;
template struct writer<float>;
template struct writer<double>;
template struct writer<complex_64>;
template struct writer<complex_128>;
template struct writer<int64_t>;
template struct writer<uint32_t>;
template struct writer<int32_t>;
template struct writer<uint16_t>;
template struct writer<int16_t>;
template struct writer<uint8_t>;
template struct writer<int8_t>;
template struct writer<uint_bool_t>;
#ifdef _HALF_FLOAT_SUPPORT_
template struct writer<float16_t>;
template struct writer<complex_32>;
#endif
#ifdef __SIZEOF_INT128__
template struct writer<int128_t>;
template struct writer<uint128_t>;
#endif
#ifdef _128_FLOAT_SUPPORT_
template struct writer<float128_t>;
#endif

}

