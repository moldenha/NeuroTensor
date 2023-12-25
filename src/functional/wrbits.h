#ifndef _WRBITS_H_
#define _WRBITS_H_
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
#include <optional>
#include "../Tensor.h"
#include <complex>

namespace nt{

template<typename T>
struct writer{
	static constexpr size_t Size = sizeof(T);
	std::array<char, Size> _arr;
	std::optional<std::ofstream> of;
	writer(const char*);
	void operator()(const T& inp);
	bool write_num(T inp, std::ostream& outfile);
	bool write_num(T inp);
};

template<typename T>
struct appender{
	static constexpr size_t Size = sizeof(T);
	std::array<char, Size> _arr;
	std::ofstream& of;
	appender(std::ofstream&);
	void operator()(const T& inp);
	bool write_num(T inp, std::ostream& outfile);
	bool write_num(T inp);
};

template<typename T>
struct bracket_appender{
	static constexpr size_t Size = sizeof(T);
	std::array<char, Size> _arr;
	bool write_list(const T* begin, const T* end, std::ofstream& outfile);
};

template<typename T>
struct reader{
	static constexpr size_t Size = sizeof(T);
	T outp;
	std::array<char, Size> _arr;
	bool add_nums(std::ifstream& in);
	bool convert();

};



}
#endif
