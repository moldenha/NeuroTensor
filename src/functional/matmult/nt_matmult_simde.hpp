#ifndef _NT_MATMULT_SIMDE_TRAITS_H_
#define _NT_MATMULT_SIMDE_TRAITS_H_
//types fully working:
//float
//double
//int32
//uint32
//int64
//uint64
//int8
//uint8
//int16
//uint16
//float16



//simde trats already defined before this header is called
#include <simde/x86/avx.h>
#include <simde/x86/avx2.h>
#include <simde/x86/fma.h>  // only for FMA if supported
#include <cstddef>
#include <cstddef>
#include <array>
#include <type_traits>
#include "nt_kmatmult_simde.hpp" //this is the constexpr version that takes the what is to be masked as constexpr size_t variables
#include "nt_matmult_macros.h"



namespace nt{
namespace functional{
namespace std_functional{

//differs from original one
//this fills the Indices * 2 + 1 with zeros
template<typename T, size_t... Indices>
inline constexpr void load_c_elements_masked_zero(
		T* C, const size_t& src_c_cols, mp::mask_type& mask, mp::simde_type<T>* arr, std::index_sequence<Indices...>
) noexcept {
	if constexpr (std::is_unsigned<T>::value){
		((arr[Indices * 2] = mp::SimdTraits<T>::load_masked(reinterpret_cast<std::make_signed_t<T>*>(&C[src_c_cols * Indices]), mask)), ...);
	}else{
		((arr[Indices * 2] = mp::SimdTraits<T>::load_masked(&C[src_c_cols * Indices], mask)), ...);
	}
	((arr[Indices * 2 + 1] = mp::SimdTraits<T>::zero()), ...);
}

template<typename T, size_t start, size_t... Indices>
inline constexpr void zero_c_elements_masked(
		mp::simde_type<T>* arr, std::index_sequence<Indices...>) noexcept{
	((arr[Indices+start] = mp::SimdTraits<T>::zero()), ...);
}

//this is for the case when b is less than pack size
//takes the max case, the current element, and the current element minus one
//max is tile_size
//the 0 case should not be possible
#define _NT_LOAD_C_MASKED_1_CASE_(max, current, current_minus){\
	if constexpr (max == current){\
		load_c_elements_masked_zero<T>(C, src_c_cols, mask, rowCs, std::make_index_sequence<tile_size_v<T> >{});\
	}else if constexpr (current == 0){\
		zero_c_elements_masked<T, 0>(rowCs, std::make_index_sequence<tile_size_v<T> * 2>{});\
	}else{\
		load_c_elements_masked_zero<T>(C, src_c_cols, mask, rowCs, std::make_index_sequence<current>{});\
		zero_c_elements_masked<T, current * 2>(rowCs, std::make_index_sequence< (max - current) * 2>{});\
	}\
	break;\
}


#define _NT_LOAD_C_MASKED_2_CASE_(max, current, current_minus){\
	if constexpr (max == current){\
		load_c_elements_masked_2<T>(C, src_c_cols, mask, rowCs, std::make_index_sequence<tile_size_v<T> >{});\
	}else if constexpr (current == 0){\
		zero_c_elements_masked<T, 0>(rowCs, std::make_index_sequence<tile_size_v<T> * 2>{});\
	}else{\
		load_c_elements_masked_2<T>(C, src_c_cols, mask, rowCs, std::make_index_sequence<current>{});\
		zero_c_elements_masked<T, current * 2>(rowCs, std::make_index_sequence< (max - current) * 2>{});\
	}\
	break;\
}

//this is to load it up to a certain number of rows
//per_row is automatically 2 because B_COLS per the tiled block is going to be tile_size (tile_size / pack_size = 2)
//this corresponds to the number of vectors per row
#define _NT_LOAD_C_CASE_(max, current, current_minus){\
	if constexpr (max == current){\
		load_c_elements_2<T, 2>(C, src_c_cols, rowCs, std::make_index_sequence<tile_size_v<T> * 2>{});\
	}else if constexpr (current == 0){\
		zero_c_elements_masked<T, 0>(rowCs, std::make_index_sequence<tile_size_v<T> * 2>{});\
	}else{\
		load_c_elements_2<T, 2>(C, src_c_cols, rowCs, std::make_index_sequence<current * 2 >{});\
		zero_c_elements_masked<T, current * 2>(rowCs, std::make_index_sequence< (max - current) * 2>{});\
	}\
	break;\
}



template<typename T, size_t... Indices>
inline constexpr void store_c_elements_masked_double(
		T* C, const size_t& src_c_cols, mp::mask_type& mask, mp::simde_type<T>* arr, std::index_sequence<Indices...>
) noexcept {

	/* if constexpr (std::is_integral<T>::value){ */
	/* (mp::SimdTraits<T>::store_masked(static_cast<mp::simde_type<T>*>(&C[src_c_cols * Indices]), mask, arr[Indices]), ...); */
	/* }else{ */
	if constexpr (std::is_unsigned<T>::value){
	(mp::SimdTraits<T>::store_masked(reinterpret_cast<std::make_signed_t<T>*>(&C[src_c_cols * Indices]), mask, arr[Indices*2]), ...);
	}else{
	(mp::SimdTraits<T>::store_masked(&C[src_c_cols * Indices], mask, arr[Indices*2]), ...);
	}
	/* } */
}


//if 0 do nothing!
#define _NT_STORE_C_MASKED_2_CASE_(max, current, current_minus){\
	if constexpr (max == current){\
		store_c_elements_masked_2<T>(C, src_c_cols, mask, rowCs, std::make_index_sequence<tile_size_v<T> >{});\
	}else if constexpr (current == 0){\
		;\
	}else{\
		store_c_elements_masked_2<T>(C, src_c_cols, mask, rowCs, std::make_index_sequence<current>{});\
	}\
	break;\
}

#define _NT_STORE_C_MASKED_1_CASE_(max, current, current_minus){\
	if constexpr (max == current){\
		store_c_elements_masked_double<T>(C, src_c_cols, mask, rowCs, std::make_index_sequence<tile_size_v<T> >{});\
	}else if constexpr (current == 0){\
		;\
	}else{\
		store_c_elements_masked_double<T>(C, src_c_cols, mask, rowCs, std::make_index_sequence<current>{});\
	}\
	break;\
}


#define _NT_STORE_C_CASE_(max, current, current_minus){\
	if constexpr (max == current){\
		store_c_elements<T, 2>(C, src_c_cols, rowCs, std::make_index_sequence<tile_size_v<T> * 2>{});\
	}else if constexpr (current == 0){\
		;\
	}else{\
		store_c_elements<T, 2>(C, src_c_cols, rowCs, std::make_index_sequence<current * 2 >{});\
	}\
	break;\
}




//addition is the number of collumns in B packed
//takes the amount of rows in a to process
//and the amount of collumns in b to process
//this is the version for when b_cols > mp::pack_size_v<T>;
template<typename T, size_t ADDITION>
void matmult_simdeT_directly_threaded(const T* A, const T* B, T* C, const size_t& src_c_cols, const size_t& b_cols, const size_t& a_rows){
	//going to load all the row elements from B into vectors and store them in an array
	constexpr size_t tile_size = tile_size_v<T>; //this is going to be the rows and collumns of both A and B
	constexpr size_t pack_size = mp::pack_size_v<T>;
	constexpr size_t rowB_size = tile_size / mp::pack_size_v<T>;
	constexpr size_t total_row_elements = tile_size * rowB_size;
	/* std::cout << "src c cols: "<<src_c_cols << " b_cols: "<<b_cols<< " a_rows: "<<a_rows<<std::endl; */

	//instead total_c_row_elements is going to be 2 * tile_size (2 is per row, and tile_size is a_rows, 
	//there is just going to be a switch statement to see exactly how many are loaded)
	mp::simde_type<T> rowCs[tile_size * 2];
	mp::mask_type mask;
	if(b_cols != tile_size){
		if(b_cols < pack_size){
			mask = mp::generate_mask<T>(b_cols);
			_NT_MATMULT_SWITCHES_(a_rows, _NT_LOAD_C_MASKED_1_CASE_);
		}else{
			mask = mp::generate_mask<T>(b_cols - pack_size);
			_NT_MATMULT_SWITCHES_(a_rows, _NT_LOAD_C_MASKED_2_CASE_);
		}
	}else{
		_NT_MATMULT_SWITCHES_(a_rows, _NT_LOAD_C_CASE_);
	}
	//c elements now loaded, now to load B into an array of packed vectors
	//same except that b cols are guarenteed to be tile_size (not pack_size)
	//therefore there is nothing to skip per row
	const std::array<mp::simde_type<T>, total_row_elements> rowBs = load_threaded_row_elements_skip<T, ADDITION, 0>(
								B, std::make_index_sequence<total_row_elements>{});
	//the loading of the rows stays generally the same
	mp::simde_type<T> aVector;
	//and running the loops directly is a little different
	//this is where the dot products happen
	for(size_t i = 0; i < a_rows; ++i){
		second_loop_direct_2<T, total_row_elements>(aVector, A + (tile_size_v<T> * i), rowCs[i * 2], rowCs[i * 2 + 1], rowBs, std::make_index_sequence<tile_size_v<T>>{});
	}
	if(b_cols != tile_size){
		if(b_cols < pack_size){
			_NT_MATMULT_SWITCHES_(a_rows, _NT_STORE_C_MASKED_1_CASE_);
		}else{
			_NT_MATMULT_SWITCHES_(a_rows, _NT_STORE_C_MASKED_2_CASE_);
		}
	}else{
		_NT_MATMULT_SWITCHES_(a_rows, _NT_STORE_C_CASE_);
	}	
}



}}} //nt::functional::std_functional::



#endif
