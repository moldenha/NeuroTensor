#include "permute.h"


#include <cstdint>
#include <iostream>
#include <numeric>
#include <vector>
#include <memory.h>
#include "../utils/utils.h"

namespace nt{
namespace permute{


Permuter::Permuter(void** s, const std::vector<uint32_t>& st, const std::vector<uint32_t>& sh)
	:strides_old(s),
	strides(st),
	shape(sh),
	shape_accumulate(sh.size())
{
	for(uint32_t i = 0; i < shape_accumulate.size(); ++i){
		shape_accumulate[i] = std::accumulate(shape.cbegin()+i+1, shape.cend(), 1, std::multiplies<uint32_t>());
	}
}

uint32_t Permuter::get_index(uint32_t idx) const {
	uint32_t n_index = 0;
	for(uint32_t i = 0; i < shape.size(); ++i){
		if(idx >= shape_accumulate[i]){
			uint32_t idx_n = idx / shape_accumulate[i];
			idx = idx % shape_accumulate[i];
			n_index += idx_n * strides[i];
		}
	}
	return n_index;
}

void* Permuter::get_ptr(uint32_t idx){
	void** o_copy = strides_old;
	for(uint32_t i = 0; i < shape.size() && idx > 0; ++i){
		if(idx >= shape_accumulate[i]){
			uint32_t idx_n = idx / shape_accumulate[i];
			idx = idx % shape_accumulate[i];
			o_copy += idx_n * strides[i];
		}
	}
	return *o_copy;
}


void Permute(void** ar, void** n_str, uint32_t size, const std::vector<uint32_t>& new_shape, const std::vector<uint32_t>& new_strides){
	Permuter perm(ar, new_strides, new_shape);
	for(uint32_t i = 0; i < size; ++i, ++n_str)
		*(n_str) = perm.get_ptr(i);
}

}
}
