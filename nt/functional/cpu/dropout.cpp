#include "dropout.h"
#include "../../dtype/ArrayVoid_NTensor.hpp"
#include <algorithm>
#include "../../mp/Threading.h"
#include <stdexcept>


#include "../../mp/simde_traits.h"
#include "../../mp/simde_traits/simde_traits_iterators.h"
#include "../../types/Types.h"

namespace nt::mp{

//this is generally the only one specified like this
//for a pointer specifically, this is basically just a faster std::memset(begin, end, 0)
inline void fill_zero(void* begin, void* end) noexcept{
	simde_type<int8_t> zero = SimdTraits<int8_t>::zero();
	int8_t* i_begin = reinterpret_cast<int8_t*>(begin);
	int8_t* i_end = reinterpret_cast<int8_t*>(end);
	static constexpr size_t pack_size = pack_size_v<int8_t>;
	for(;i_begin + pack_size <= i_end; i_begin += pack_size){
		SimdTraits<int8_t>::storeu(reinterpret_cast<mask_type*>(i_begin), zero);
	}
	for(;i_begin < i_end; ++i_begin){
		*i_begin = 0;
	}
}

template<typename T>
inline void fill_zero(BucketIterator_blocked<T>& begin, BucketIterator_blocked<T>& end) noexcept {
	if constexpr (simde_supported_v<T>){
	simde_type<T> zero = SimdTraits<T>::zero();
	static constexpr size_t pack_size = pack_size_v<T>;
	for(;begin + pack_size <= end; begin += pack_size){
		it_storeu(begin, zero);
	}
	for(;begin < end; ++begin){
		*begin = 0;
	}
	}else{
		std::fill(begin, end, T(0));	
	}
}
template<typename T>
inline void fill_zero(BucketIterator_list<T>& begin, BucketIterator_list<T>& end) noexcept {
	if constexpr (simde_supported_v<T>){
	simde_type<T> zero = SimdTraits<T>::zero();
	static constexpr size_t pack_size = pack_size_v<T>;
	for(;begin + pack_size <= end; begin += pack_size){
		it_storeu(begin, zero);
	}
	for(;begin < end; ++begin){
		*begin = 0;
	}
	}else{
		std::fill(begin, end, T(0));
	}
}



}

void ::nt::functional::cpu::_dropout2d_(nt::ArrayVoid& tensor, nt::ArrayVoid const& bools, const int64_t& rows, const int64_t& cols){
    const uint64_t total_size = tensor.Size();
    if(!total_size % (rows * cols) != 0){
        throw std::logic_error("Error: dropout2d was given invalid rows and columns");
    }
    const uint64_t matrices = total_size / (rows * cols);
    if(bools.dtype() != DType::Bool){
        throw std::logic_error("Expected bools array void to be bools");
    }
    if(matrices != bools.Size()){
        throw std::length_error("Expected bools array length to be the same as the number of matrices in the tensor");
    }
    if(!bools.is_contiguous()){
        throw std::logic_error("Expected bools array void to be contiguous");
    }

    tensor.execute_function<WRAP_DTYPES<NumberTypesL, DTypeEnum<DType::Bool> > >(
    [&bools, &rows, &cols, &matrices](auto begin, auto end){
        const bool* drop = reinterpret_cast<const bool*>(bools.data_ptr());
        threading::preferential_parallel_for(
            threading::block_ranges<1>(0, matrices),
            [&](threading::blocked_range<1> block){
            auto begin_cpy = begin + (block.begin[0] * (rows * cols));
            auto end_cpy = begin_cpy + (rows * cols);
            for(int64_t b = block.begin[0]; b != block.end[0]; ++b, begin_cpy += (rows*cols), end_cpy += (rows*cols)){
                if(!drop[b]) continue;
                mp::fill_zero(begin_cpy, end_cpy);
            }
        });
    });

}

namespace nt::functional::cpu{


void _dropout3d_(ArrayVoid& tensor, const ArrayVoid& bools, const int64_t& channels, const int64_t& rows, const int64_t& cols){
    const uint64_t total_size = tensor.Size();
    if(!total_size % (channels * rows * cols) != 0){
        throw std::logic_error("Error: dropout3d was given invalid channels, rows, and columns");
    }
    const uint64_t batches = total_size / (channels * rows * cols);
    if(bools.dtype() != DType::Bool){
        throw std::logic_error("Expected bools array void to be bools");
    }
    if(batches != bools.Size()){
        throw std::length_error("Expected bools array length to be the same as the number of 3D Tensors in the tensor");
    }
    if(!bools.is_contiguous()){
        throw std::logic_error("Expected bools array void to be contiguous");
    }

    tensor.execute_function<WRAP_DTYPES<NumberTypesL, DTypeEnum<DType::Bool> > >(
    [&bools, &rows, &cols, &channels, &batches](auto begin, auto end){
        const bool* drop = reinterpret_cast<const bool*>(bools.data_ptr());
        threading::preferential_parallel_for(
            threading::block_ranges<1>(0, batches),
            [&](threading::blocked_range<1> block){
            auto begin_cpy = begin + (block.begin[0] * (channels * rows * cols));
            auto end_cpy = begin_cpy + (channels * rows * cols);
            for(int64_t b = block.begin[0]; b != block.end[0]; ++b, begin_cpy += (channels*rows*cols), end_cpy += (channels*rows*cols)){
                if(!drop[b]) continue;
                mp::fill_zero(begin_cpy, end_cpy);
            }
        });
    });

}


}
