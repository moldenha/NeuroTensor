#include "functional.h"


#include <cstdint>
#include <iterator>
#include <memory>
#include <stdbool.h>
#include <stdio.h>
#include <string_view>
#include <string.h>
#include <iostream>
#include <fstream>
#include <sys/_types/_int8_t.h>
#include <vector>
#include <deque>
#include "../dtype/DType_enum.h"
#include "wrbits.h"
#include "../dtype/ArrayVoid.hpp"

namespace nt{
namespace functional{


DType get_dtype(uint8_t dt_i){
	switch(dt_i){
		case 0:
			return DType::Integer;
		case 1:
			return DType::Float;
		case 2:
			return DType::Double;
		case 3:
			return DType::Long;
		case 4:
			return DType::Complex64;
		case 5:
			return DType::Complex128;
		case 6:
			return DType::uint8;
		case 7:
			return DType::int8;
		case 8:
			return DType::int16;
		case 9:
			return DType::uint16;
		case 10:
			return DType::LongLong;
		case 11:
			return DType::Bool;
		case 12:
			return DType::TensorObj;
#ifdef _128_FLOAT_SUPPORT_
		case 13:
			return DType::Float128;
#endif
#ifdef _HALF_FLOAT_SUPPORT_
		case 14:
			return DType::Float16;
		case 15:
			return DType::Complex32;
#endif
#ifdef __SIZEOF_INT128__
		case 16:
			return DType::int128;
		case 17:
			return DType::uint128;
#endif
		default:
			return DType::Integer;
	}
}


static inline constexpr auto read_nums_binary = [](auto a_begin, auto a_end, size_t& total, std::ifstream& in, const char* filename) -> bool{
	using value_t = typename decltype(a_begin)::value_type;
	reader<value_t> reader_2;
	uint32_t counter = 0;
	while(in.good()){
		char checking = in.get();
		if(checking == '}')
			break;
		in.putback(checking);
		if(!reader_2.add_nums(in)){break;}
		if(!reader_2.convert()){
			std::cerr << "error reading, got -1 from num reading "<<filename<<" at "<<counter;
			return false;
		}
		*a_begin = reader_2.outp;
		++a_begin;
		++counter;
	}
	if(a_begin != a_end){
		std::cerr << "error loading from "<<filename;
		return false;
	}
	return true;

};

Tensor load(const char* filename){
	std::ifstream in(filename);
	if(!in.is_open()){
		std::cerr<< "error reading " << filename;
		return Tensor();
	}
	in.get();
	uint8_t dt_i = in.get();
	in.get();
	DType dt = get_dtype(dt_i);
	in.get(); //{
	std::vector<SizeRef::ArrayRefInt::value_type> nshape;
	reader<SizeRef::ArrayRefInt::value_type> my_reader;
	while(in.good()){
		char checking = in.get();
		if(checking == '}')
			break;
		in.putback(checking);
		if(!my_reader.add_nums(in)){break;}
		if(!my_reader.convert()){
			std::cerr << "error reading, got -1 from num reading " << filename;
			return Tensor();
		}
		nshape.push_back(my_reader.outp);
	}
	Tensor outTensor(SizeRef(std::move(nshape)), dt);
	size_t total = outTensor.shape().multiply();
	in.get();
	bool verify = outTensor.arr_void().execute_function_nbool(read_nums_binary, total, in, filename);
	/* if(!verify){std::cout<<"error reading "<<filename<<std::endl;} */
	return std::move(outTensor);

}

}
}
