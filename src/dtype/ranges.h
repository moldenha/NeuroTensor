#ifndef _RANGES_H_
#define _RANGES_H_

#include <iostream>
#include <functional>
#include <numeric>

namespace nt{

struct my_range{
	int64_t begin, end; // begining and end of the range
	my_range();
	explicit my_range(const int64_t& v); // a start and by default end = -1;
	explicit my_range(const int64_t& v, const int64_t &v2); // a start and an end
	my_range& operator+(const int64_t& v); // set the end to v;
	my_range& operator-(const int64_t& v); // set end to -v;
	my_range& operator+=(const int64_t& v); // increment begin and end by v
	my_range& operator-=(const int64_t& v); // decrement begin and end by v
	void fix(size_t s); // if begin or end is less than 0, add s to them
	const int64_t length() const; // end - begin
};


std::ostream& operator<<(std::ostream& out, const my_range& v);
inline std::ostream& operator<<(std::ostream& out, const std::vector<my_range>& v){
	if(v.size() == 0)
		return out << "[]"<<std::endl;
	out << "[";
	for(uint32_t i = 0; i < v.size() - 1; ++i)
		out << v[i].begin << ':' << v[i].end << ',';
	out << v.back().begin << ':' << v.back().end << ']';
	return out;
}
inline std::ostream& operator<<(std::ostream& out, const std::vector<std::vector<my_range> >& v){
	if(v.size() == 0)
		return out << "{}"<<std::endl;
	out << '{';
	for(uint32_t i = 0; i < v.size()-1; ++i)
		out << v[i] << ',';
	out << v.back() << '}';
	return out;
}


namespace literals{
#define NEG_1_LITERAL INT64_MAX
my_range operator ""_r(unsigned long long i);
}

}


#endif
