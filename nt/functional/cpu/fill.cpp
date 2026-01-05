#include "fill.h"
#include "../../dtype/ArrayVoid_NTensor.hpp"
#include <algorithm>
#include "../../mp/Threading.h"
#include <stdexcept>


#include "../../mp/simde_traits.h"
#include "../../mp/simde_traits/simde_traits_iterators.h"
#include "../../types/Types.h"
#include "../../utils/type_traits.h"

namespace nt{
namespace mp{

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

template<typename T, typename U>
inline void fill(T begin, T end, const U& value){
	using base_type = utils::IteratorBaseType_t<T>;
	static_assert(std::is_same_v<base_type, U>, "Need to be same for fill");
	if constexpr (simde_supported_v<base_type>){
		simde_type<base_type> val = SimdTraits<base_type>::set1(value);
		static constexpr size_t pack_size = pack_size_v<base_type>;
		for(;begin + pack_size <= end; begin += pack_size){
			it_storeu(begin, val);
		}
		for(;begin < end; ++begin)
			*begin = value;
	}else{
		std::fill(begin, end, value);
	}
}

// template<typename base_type>
// void print_simd_register(simde_type<base_type> t){
//     base_type l[SimdTraits<base_type>::pack_size];
//     SimdTraits<base_type>::storeu(l, t);
//     for(int i = 0; i < SimdTraits<base_type>::pack_size; ++i){
//         std::cout << l[i] << ' ';
//     }
//     std::cout << std::endl;
// }

template<typename T>
inline void iota(T begin, T end, utils::IteratorBaseType_t<T> value = 0){
	using base_type = utils::IteratorBaseType_t<T>;
	if constexpr (simde_supported_v<base_type>){
		static constexpr size_t pack_size = pack_size_v<base_type>;
		base_type loading[pack_size];
        utils::IteratorBaseType_t<T> value_cpy = value;
		for(size_t i = 0; i < pack_size; ++i, ++value_cpy){
			loading[i] = base_type(value_cpy);
		}
        base_type to_set_1__ = (type_traits::is_complex_v<base_type>) ? base_type(pack_size) : static_cast<base_type>(pack_size);
		simde_type<base_type> val;
		simde_type<base_type> add = SimdTraits<base_type>::set1(to_set_1__);

		if constexpr (type_traits::is_integral_v<base_type> || type_traits::is_unsigned_v<base_type>){
			val = SimdTraits<base_type>::loadu(reinterpret_cast<const simde_type<base_type>*>(loading));
		}else{
			val = SimdTraits<base_type>::loadu(loading);
		}
        
        static base_type adding_pack_size(pack_size);

		for(;begin + pack_size <= end; begin += pack_size, value += adding_pack_size){
			it_storeu(begin, val);
			val = SimdTraits<base_type>::add(val, add);
		}

		if constexpr (type_traits::is_complex_v<base_type>){
            base_type my_cur_one__(1, 1);
            for(;begin < end; ++begin, value += my_cur_one__){
                *begin = value;
            }
        }
        else{
            for(;begin < end; ++begin, ++value)
                *begin = value;
        }
	}else{
		std::iota(begin, end, value);
	}	
}



}
}

namespace nt{
namespace functional{
namespace cpu{

void _fill_diagonal_(ArrayVoid& arr, Scalar s, const int64_t& batches, const int64_t& rows, const int64_t& cols){
    arr.execute_function<WRAP_DTYPES<NumberTypesL, DTypeEnum<DType::Bool> > >( 
    [&s, &batches, &rows, &cols](auto begin, auto end){
            using value_t = utils::IteratorBaseType_t<decltype(begin)>;
            auto v = s.to<value_t>();
            int64_t min_rows = std::min(rows, cols);
            threading::preferential_parallel_for(
                threading::block_ranges<1>(0, batches),
                [&](threading::blocked_range<1> block){
                auto begin_cpy = begin;
                for(int64_t b = block.begin[0]; b < block.end[0]; ++b, begin_cpy += (rows * cols)){
                    auto next_begin = begin_cpy + (rows * cols);
                    for(int64_t r = 0; r < min_rows; ++r){
                        if(next_begin < begin_cpy || begin_cpy == next_begin) break;
                        *begin_cpy = v;
                        begin_cpy += cols + 1;
                    }
                    begin_cpy = next_begin;
                }
                });
    });
}

void _fill_scalar_(ArrayVoid& arr, Scalar s){
    const uint64_t size = arr.Size();
    if(s.isZero()){
        arr.execute_function_chunk<WRAP_DTYPES<NumberTypesL, DTypeEnum<DType::Bool> > >(
        [](auto begin, auto end){
            mp::fill_zero(begin, end);
        });
    }else{
        arr.execute_function_chunk<WRAP_DTYPES<NumberTypesL, DTypeEnum<DType::Bool> > >(
        [&s](auto begin, auto end){
            using value_t = utils::IteratorBaseType_t<decltype(begin)>;
            auto v = s.to<value_t>();
            mp::fill(begin, end, v);
        });
    }
}

void _set_(ArrayVoid& arr, const ArrayVoid& arr2){
    const uint64_t size = arr.Size();
    const uint64_t size2 = arr2.Size();
    if((size != size2) || (arr.dtype() != arr2.dtype())){throw std::invalid_argument("cpu::_set_ function requires both arrays to have the same size and dtype");}
    arr.execute_function<WRAP_DTYPES<NumberTypesL, DTypeEnum<DType::Bool> > >(
    [&size](auto _begin, auto _end, auto _begin2){
        threading::preferential_parallel_for(threading::block_ranges<1>(0, size),
        [&](threading::blocked_range<1> block){
            auto begin = _begin + block.begin[0];
            auto end = _begin + block.end[0];
            auto begin2 = _begin2 + block.begin[0];
            for(;begin != end; ++begin, ++begin2) *begin = *begin2;
        });
    }, const_cast<ArrayVoid&>(arr2)); 
    // const cast so that the memory can be accessed by this function
    // but it will remain unmodified
}


void _iota_(ArrayVoid& arr, Scalar start){
    arr.execute_function<WRAP_DTYPES<NumberTypesL> >([&start](auto begin, auto end){
        using value_t = utils::IteratorBaseType_t<decltype(begin)>;
		auto v = start.to<value_t>();
        mp::iota(begin, end, v);
    });
}

}
}
}
