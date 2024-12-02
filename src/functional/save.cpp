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
#include <vector>
#include <deque>
#include "../dtype/DType_enum.h"
#include "wrbits.h"
#include "../dtype/ArrayVoid.hpp"

//first I am going to make a bf interpreter


namespace nt{
namespace functional{

void write_shape(const SizeRef& _st, std::ofstream& outfile){
	std::for_each(_st.cbegin(), _st.cend(), appender<SizeRef::ArrayRefInt::value_type>(outfile));
}

void write_dtype(const DType& dtype, std::ofstream& outfile){
	uint8_t dt_i = 0;
	switch(dtype){
		case DType::Integer:
			dt_i = 0;
			break;
		case DType::Float:
			dt_i = 1;
			break;
		case DType::Double:
			dt_i = 2;
			break;
		case DType::Long:
			dt_i = 3;
			break;
		case DType::Complex64:
			dt_i = 4;
			break;
		case DType::Complex128:
			dt_i = 5;
			break;
		case DType::uint8:
			dt_i = 6;
			break;
		case DType::int8:
			dt_i = 7;
			break;
		case DType::int16:
			dt_i = 8;
			break;
		case DType::uint16:
			dt_i = 9;
			break;
		case DType::LongLong:
			dt_i = 10;
			break;
		case DType::Bool:
			dt_i = 11;
			break;
		case DType::TensorObj:
			dt_i = 12;
			break;
#ifdef _128_FLOAT_SUPPORT_
		case DType::Float128:
			dt_i = 13;
			break;
#endif
#ifdef _HALF_FLOAT_SUPPORT_
		case DType::Float16:
			dt_i = 14;
			break;
		case DType::Complex32:
			dt_i = 15;
			break;
#endif
#ifdef __SIZEOF_INT128__
		case DType::int128:
			dt_i = 16;
			break;
		case DType::uint128:
			dt_i = 17;
			break;
#endif
	}
	outfile << "{";
	outfile<<(char)dt_i;
	outfile << "}";
}


void save_tensorObj(const Tensor& t, const char* filename);
void save_bool(const Tensor& t, const char* filename);

/* inline static constexpr auto save_tensor_parts = [](auto a_begin, auto a_end, std::ofstream& f1){ */
/* 	using value_t = typename std::remove_const<typename decltype(a_begin)::value_type>::type; */
/* 	std::for_each(a_begin, a_end, writer<value_t>(f1)); */
/* } */

void save(const Tensor &t, const char* filename){
	if(t.dtype == DType::TensorObj){
		save_tensorObj(t, filename);
		return;
	}
	if(t.dtype == DType::Bool){
		save_bool(t, filename);
		return;
	}
	std::ofstream f1(filename, std::ios::out | std::ios::binary | std::ios::trunc);
	write_dtype(t.dtype, f1);
	bracket_appender<typename SizeRef::ArrayRefInt::value_type> app_shp;
	bool check = app_shp.write_list(t.shape().data(), t.shape().data() + t.shape().size(), f1);
	if(!check){std::cout<<"problem writing shape list"<<std::endl;}
	f1<<"{";
	t.arr_void().cexecute_function([&f1](auto a_begin, auto a_end){
			using value_t = utils::IteratorBaseType_t<decltype(a_begin)>;
			std::for_each(a_begin, a_end, appender<value_t>(f1));
			});
	f1<<"}";
}

void save_tensorObj(const Tensor& t, const char* filename){
	std::cerr<<"TensorObj DType is currently unsupported for save function";
	return;
}
void save_bool(const Tensor& t, const char* filename){
	std::cerr<<"Bool DType is currently unsupported for save function";
	return;
}

}
}
