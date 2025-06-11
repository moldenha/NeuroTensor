#ifdef _HALF_FLOAT_SUPPORT_
#ifndef _NT_BFLOAT_16_H_
#define _NT_BFLOAT_16_H_

//a library for bfloat16
//suport to be added to NeuroTensor soon

#include "Types.h"
#include <cstring>
#include "../dtype/DType_enum.h"
#include "../convert/Convert.h"

namespace nt{

struct bfloat16_t{
	uint16_t data;
	bfloat16_t() noexcept
	: data(0) 
	{}
	bfloat16_t(float value) noexcept {
		uint32_t bits;
		std::memcpy(&bits, &value, sizeof(bits));
		data = bits >> 16;
	}
	bfloat16_t(float16_t value) noexcept
		:bfloat16_t(convert::convert<DType::Float32>(value))
	{}
	inline operator float() const{
		uint32_t bits = static_cast<uint32_t>(data) << 16;
		float value;
		std::memcpy(&value, &bits, sizeof(bits));
		return value;
	}
	inline operator double() const {return double(float(*this));}
	inline operator float16_t() const {
		return convert::convert<DType::Float16>(float(*this));
	}

};

namespace convert{
//inplace is used to save memory
inline bfloat16_t* inplace_to_bfloat16_from_float16(float16_t* vals, const int64_t& num){
	bfloat16_t* n_vals = reinterpret_cast<bfloat16_t*>(vals);
	for(int64_t i = 0; i < num; ++i){
		n_vals[i] = bfloat16_t(vals[i]);
	}
	return n_vals;
}

inline float16_t* inplace_to_float16_from_bfloat16(bfloat16_t* vals, const int64_t& num){
	float16_t* n_vals = reinterpret_cast<float16_t*>(vals);
	for(int64_t i = 0; i < num; ++i){
		n_vals[i] = float16_t(vals[i]);
	}
	return n_vals;
}


}


}

#endif //_NT_BFLOAT_16_H_
#endif //_HALF_FLOAT_SUPPORT
