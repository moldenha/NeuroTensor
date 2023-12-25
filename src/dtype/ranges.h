#ifndef _RANGES_H_
#define _RANGES_H_

#include <iostream>
#include <functional>
#include <numeric>

namespace nt{

struct my_range{
	int32_t begin, end;
	my_range();
	my_range(const uint32_t& v);
	my_range(const uint32_t& v, const int32_t &v2);
	my_range& operator+(const int32_t& v);
	my_range& operator-(const int32_t& v);
	void fix(size_t s);
	const uint32_t length() const;
};


std::ostream& operator<<(std::ostream& out, const my_range& v);

namespace literals{
#define NEG_1_LITERAL INT64_MAX
my_range operator ""_r(unsigned long long i);
}

}


#endif
