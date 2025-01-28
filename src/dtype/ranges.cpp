#include "ranges.h"
#include "../utils/utils.h"

namespace nt{
namespace literals{
my_range operator ""_r(unsigned long long i){
	if(i == INT64_MAX)
		return my_range();
	return my_range(i);
}
}

std::ostream& operator<<(std::ostream& out, const my_range& v){
	out <<"{"<<v.begin<<","<<v.end<<"}";
	return out;
}

my_range::my_range()
	:begin(0),end(-1){}

my_range::my_range(const int64_t& v) //this is just a single index
	:begin(v),end(v+1){}

my_range::my_range(const int64_t& v, const int64_t &v2)
	:begin(v),end(v2){}
my_range& my_range::operator+=(const int64_t& v){begin += v; end += v; return *this;}

my_range& my_range::operator-=(const int64_t& v){begin -= v; end -= v; return *this;}

my_range& my_range::operator+(const int64_t& v){end = v; return *this;}

my_range& my_range::operator-(const int64_t& v){end = (-1*v); return *this;}

bool my_range::operator==(const my_range& a) const { return a.begin == begin && a.end == end;}
bool my_range::operator<(const my_range& a) const { return a.begin < begin && a.end < end; }
bool my_range::operator>(const my_range& a) const { return a.begin > begin && a.end > end; }
bool my_range::operator<=(const my_range& a) const { return a.begin <= begin && a.end <= end; }
bool my_range::operator>=(const my_range& a) const { return a.begin >= begin && a.end >= end; }

void my_range::fix(size_t s){
	end = end < 0 ? (end + s): end;
	begin = begin < 0 ? (begin + s): begin;
	if(end < begin)
		std::swap(begin, end);
	utils::throw_exception(begin <= s && end <= s, "Runtime Error: Expected end and begin range to be less than $ but instead got $:$", s, begin, end);
}
const int64_t my_range::length() const {return end - begin;}




}
