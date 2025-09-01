#include "ranges.h"
#include "../utils/utils.h"

namespace nt{

range_& range_::fix(int64_t s){
    end = (end < 0) ? (end + (s+1)) : end;
    begin = (begin < 0) ? (begin + (s+1)) : begin;
    if(end < begin)
        std::swap(begin, end);
    utils::throw_exception(begin <= s && end <= s, "Runtime Error: Expected end and begin range to be less than $ but instead got {$,$}", s, begin, end);
    return *this;
}

std::ostream& operator<<(std::ostream& out, const range_& v){
	out <<"{"<<v.begin<<","<<v.end<<"}";
	return out;
}

std::ostream& operator<<(std::ostream& out, const std::vector<range_>& v){
	if(v.size() == 0)
		return out << "[]"<<std::endl;
	out << "[";
	for(uint32_t i = 0; i < v.size() - 1; ++i)
		out << v[i].begin << ':' << v[i].end << ',';
	out << v.back().begin << ':' << v.back().end << ']';
	return out;
}

std::ostream& operator<<(std::ostream& out, const std::vector<std::vector<range_> >& v){
	if(v.size() == 0)
		return out << "{}"<<std::endl;
	out << '{';
	for(uint32_t i = 0; i < v.size()-1; ++i)
		out << v[i] << ',';
	out << v.back() << '}';
	return out;
}


}
