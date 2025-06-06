#include "utils.h"

#include <cstddef>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <string>
#include <string_view>

#ifdef USE_PARALLEL
#include <tbb/mutex.h>
#endif

#include "utils_systems/utils_systems_win32.hpp"
#include  "utils_systems/utils_systems_posix.hpp"

namespace nt{
namespace utils{


/* template<typename T> */
/* void print_format_inner(std::ostringstream& out, std::string_view& str, size_t& last_pos, const T& var){ */
/* 	size_t pos = str.substr(last_pos, str.size()).find('$'); */
/* 	if(pos == std::string_view::npos){ */
/* 		if(str.size() == last_pos) */
/* 			return; */
/* 		out << str.substr(last_pos, str.size()); */
/* 		last_pos = str.size(); */
/* 		return; */
/* 	} */
/* 	out << str.substr(last_pos, pos); */
/* 	out << var; */
/* 	last_pos += pos + 1; */
/* } */

/* template<> void print_format_inner<int>(std::ostringstream&, std::string_view&, size_t&, const int&); */
/* template<> void print_format_inner<float>(std::ostringstream&, std::string_view&, size_t&, const float&); */
/* template<> void print_format_inner<double>(std::ostringstream&, std::string_view&, size_t&, const double&); */
/* template<> void print_format_inner<uint32_t>(std::ostringstream&, std::string_view&, size_t&, const uint32_t&); */
/* template<> void print_format_inner<SizeRef>(std::ostringstream&, std::string_view&, size_t&, const SizeRef&); */
/* template<> void print_format_inner<DType>(std::ostringstream&, std::string_view&, size_t&, const DType&); */


namespace memory_details{
int64_t cpu_memory_allocated = 0;
int64_t shared_cpu_memory_allocated = 0;
int64_t meta_memory_allocated = 0;
}

#ifdef USE_PARALLEL


void printThreadingProgressBar(uint32_t progress, uint32_t total, std::string add, uint32_t width) {
    float percentage = static_cast<float>(progress) / total;
    int numChars = static_cast<int>(percentage * width);

    static tbb::mutex printMutex;
    tbb::mutex::scoped_lock lock(printMutex);
    std::cout << "\r[";
    for (int i = 0; i < numChars; ++i) {
        std::cout << "=";
    }
    for (int i = numChars; i < width; ++i) {
        std::cout << " ";
    }
    std::cout << "] " << static_cast<int>(percentage * 100.0) << "% "<<progress<<'/'<<total << add;
    std::cout.flush();
    lock.release();
}



#endif


void printProgressBar(uint32_t progress, uint32_t total, std::string add, uint32_t width) {
    float percentage = static_cast<float>(progress) / total;
    int numChars = static_cast<int>(percentage * width);

    std::cout << "\r[";
    for (int i = 0; i < numChars; ++i) {
        std::cout << "=";
    }
    for (int i = numChars; i < width; ++i) {
        std::cout << " ";
    }
    std::cout << "] " << static_cast<int>(percentage * 100.0) << "% "<<progress<<'/'<<total << add;
    std::cout.flush();
}

void beginDualProgressBar(uint32_t total_a, uint32_t total_b, uint32_t width){
	std::cout << "[";
	for(int i = 0; i < width; ++i)
		std::cout << ' ';
	std::cout << "] 0% 0/"<<total_a << "\n";
	std::cout << "[";
	for(int i = 0; i < width; ++i)
		std::cout << ' ';
	std::cout << "] 0% 0/"<<total_b;
	std::cout.flush();
}
void printDualProgressBar(uint32_t progress_a, uint32_t total_a, uint32_t progress_b, uint32_t total_b, uint32_t width){
	std::cout << "\033[F";
	printProgressBar(progress_a, total_a, "", width);
	std::cout << "\n";
	printProgressBar(progress_b, total_b, "", width);

}
void endDualProgressBar(uint32_t total_a, uint32_t total_b, uint32_t width){
	std::cout << "\033[F";
	printProgressBar(total_a, total_a, "", width);
	std::cout << "\n";
	printProgressBar(total_b, total_b, "", width);
	
}

tuple_or_int::tuple_or_int(int64_t x)
    :val(x)
{}


tuple_or_int::tuple_or_int(std::tuple<int64_t, int64_t> x)
    :val(x)
{}

tuple_or_int& tuple_or_int::operator=(int64_t x){val = x; return *this;}
tuple_or_int& tuple_or_int::operator=(std::tuple<int64_t, int64_t> x){val = x; return *this;}

std::tuple<int64_t, int64_t> tuple_or_int::to_tuple() const{
    if(std::holds_alternative<std::tuple<int64_t, int64_t>>(val)){
        return std::get<1>(val);
    }
    return std::tuple<int64_t, int64_t>(std::get<0>(val), std::get<0>(val));
}

tuple_or_int::operator std::tuple<int64_t, int64_t>() const {return this->to_tuple();}

my_tuple tuple_or_int::to_my_tuple() const {
    if(std::holds_alternative<std::tuple<int64_t, int64_t>>(val)){
        return my_tuple(std::get<1>(val));
    }
    return my_tuple(std::get<0>(val), std::get<0>(val));

}

my_tuple::my_tuple(const int64_t a, const int64_t b)
    :first(a), second(b)
{}

my_tuple::my_tuple(const int64_t a)
    :first(a), second(a)
{}

my_tuple::my_tuple(std::tuple<int64_t, int64_t> t)
    :first(std::get<0>(t)), second(std::get<1>(t))
{}

my_tuple& my_tuple::operator=(int64_t x){first = x; second = x; return *this;}
my_tuple& my_tuple::operator=(std::tuple<int64_t, int64_t> x){first = std::get<0>(x); second = std::get<1>(x); return *this;}
const int64_t& my_tuple::operator[](const int64_t x) const {if(x == 0){return first;}return second;}
bool my_tuple::operator==(const int64_t x) const {return (first == x && second == x);}
bool my_tuple::operator!=(const int64_t x) const {return (first != x && second != x);}
bool my_tuple::operator > (const int64_t x) const {return (first > x && second > x);}
bool my_tuple::operator < (const int64_t x) const {return (first < x && second < x);}
bool my_tuple::operator >= (const int64_t x) const {return (first >= x && second >= x);}
bool my_tuple::operator <= (const int64_t x) const {return (first <= x && second <= x);}

std::ostream& operator<<(std::ostream& out, const my_tuple& t){
	return out << '(' << t[0] << ',' << t[1] << ')';
}

template<size_t N>
my_n_tuple<N>::my_n_tuple(const int64_t a)
{
    for(size_t i = 0; i < N; ++i)
        arr[i] = a;
}

template<size_t N>
my_n_tuple<N>& my_n_tuple<N>::operator=(int64_t x){
    for(size_t i = 0; i < N; ++i)
        arr[i] = x;
    return *this;
}


template<size_t N>
const int64_t& my_n_tuple<N>::operator[](const int64_t x) const {return arr[x];}
template<size_t N>
bool my_n_tuple<N>::operator==(const int64_t x) const {
    for(size_t i = 0; i < N; ++i){
        if(arr[i] != x){return false;}}
    return true;
}
template<size_t N>
bool my_n_tuple<N>::operator!=(const int64_t x) const {
    for(size_t i = 0; i < N; ++i){
        if(arr[i] == x){return false;}}
    return true;
}
template<size_t N>
bool my_n_tuple<N>::operator > (const int64_t x) const {
    for(size_t i = 0; i < N; ++i){
        if(arr[i] <= x){return false;}}
    return true;
}
template<size_t N>
bool my_n_tuple<N>::operator < (const int64_t x) const {
    for(size_t i = 0; i < N; ++i){if(arr[i] >= x){return false;}}
    return true;
}
template<size_t N>
bool my_n_tuple<N>::operator >= (const int64_t x) const {
    for(size_t i = 0; i < N; ++i){if(arr[i] < x){return false;}}
    return true;
}
template<size_t N>
bool my_n_tuple<N>::operator <= (const int64_t x) const {
    for(size_t i = 0; i < N; ++i){if(arr[i] > x){return false;}}
    return true;
}


template class my_n_tuple<2>;
template class my_n_tuple<3>;
template class my_n_tuple<4>;

template <>
std::ostream& operator<<(std::ostream& out, const my_n_tuple<2>& t){
	return out << '(' << t[0] << ',' << t[1]  << ')';
}

template <>
std::ostream& operator<<(std::ostream& out, const my_n_tuple<3>& t){
	return out << '(' << t[0] << ',' << t[1] << ',' << t[2] << ')';
}

template <>
std::ostream& operator<<(std::ostream& out, const my_n_tuple<4>& t){
	return out << '(' << t[0] << ',' << t[1] << ',' << t[2] << ',' << t[3] << ')';
}


}

}

