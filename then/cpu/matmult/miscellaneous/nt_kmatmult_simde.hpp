#ifndef _NT_KMATMULT_SIMDE_TRAITS_HPP_
#define _NT_KMATMULT_SIMDE_TRAITS_HPP_
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
#include "../../mp/simde_traits.h"





namespace nt{
namespace functional{
namespace std_functional{

template<typename T>
inline static constexpr size_t tile_size_v = mp::pack_size_v<T> * 2;




//addition is the number of cols in a packed block
template<typename T, size_t ADDITION, size_t... Indices>
inline constexpr std::array<mp::simde_type<T>, sizeof...(Indices)> load_threaded_row_elements (
    const T* A, std::index_sequence<Indices...>
) noexcept {
    constexpr size_t ratio_tile_pack = tile_size_v<T> / mp::pack_size_v<T>;
    if constexpr (std::is_integral<T>::value || std::is_unsigned<T>::value){
	return { mp::SimdTraits<T>::load((mp::simde_type<T>*)&A[(mp::pack_size_v<T> * (Indices % ratio_tile_pack)) + (ADDITION * (Indices / ratio_tile_pack))])... };
    }else{
	return { mp::SimdTraits<T>::load(&A[(mp::pack_size_v<T> * (Indices % ratio_tile_pack)) + (ADDITION * (Indices / ratio_tile_pack))])... };
    }
}


//in this case,
//skip is the number of "Packs" to skip
template<typename T, size_t ADDITION, size_t skip, size_t... Indices>
inline constexpr std::array<mp::simde_type<T>, sizeof...(Indices)> load_threaded_row_elements_skip (
    const T* A, std::index_sequence<Indices...>
) noexcept {
    
	constexpr size_t ratio_tile_pack = tile_size_v<T> / mp::pack_size_v<T>;
	constexpr size_t skip_ratio = ratio_tile_pack - skip;
	if constexpr (std::is_integral<T>::value || std::is_unsigned<T>::value){
		return { mp::SimdTraits<T>::load((mp::simde_type<T>*)&A[(mp::pack_size_v<T> * (Indices % skip_ratio)) + (ADDITION * (Indices / skip_ratio))])... };
	}else{
		return { mp::SimdTraits<T>::load(&A[(mp::pack_size_v<T> * (Indices % skip_ratio)) + (ADDITION * (Indices / skip_ratio))])... };
	}
}


template<typename T, size_t per_row, size_t... Indices>
inline constexpr void load_c_elements_2(
		T* C, size_t src_c_cols, mp::simde_type<T>* arr, std::index_sequence<Indices...>
) noexcept {
	if constexpr (std::is_integral<T>::value || std::is_unsigned<T>::value){
		((arr[Indices] =  mp::SimdTraits<T>::loadu((mp::simde_type<T>*)&C[(mp::pack_size_v<T> * (Indices % per_row)) + (src_c_cols * (Indices / per_row))])), ...);

	}else{
		((arr[Indices] =  mp::SimdTraits<T>::loadu(&C[(mp::pack_size_v<T> * (Indices % per_row)) + (src_c_cols * (Indices / per_row))])), ...);
	}
}




template<typename T, size_t per_row, size_t... Indices>
inline constexpr void store_c_elements (
		T* C, const size_t& src_c_cols, mp::simde_type<T>* rowCs, std::index_sequence<Indices...>
) noexcept {
	if constexpr (std::is_integral<T>::value || std::is_unsigned<T>::value){
	(mp::SimdTraits<T>::storeu((mp::simde_type<T>*)&C[(mp::pack_size_v<T> * (Indices % per_row)) + (src_c_cols * (Indices / per_row))], rowCs[Indices]), ...);
	}else{
	(mp::SimdTraits<T>::storeu(&C[(mp::pack_size_v<T> * (Indices % per_row)) + (src_c_cols * (Indices / per_row))], rowCs[Indices]), ...);
	}
}


template<typename T, size_t... Indices>
inline constexpr void load_c_elements_masked(
		T* C, const size_t& src_c_cols, mp::mask_type& mask, mp::simde_type<T>* arr, std::index_sequence<Indices...>
) noexcept {
	if constexpr (std::is_unsigned<T>::value){
	((arr[Indices] = mp::SimdTraits<T>::load_masked(reinterpret_cast<std::make_signed_t<T>*>(&C[src_c_cols * Indices]), mask)), ...);
	}else{
	((arr[Indices] = mp::SimdTraits<T>::load_masked(&C[src_c_cols * Indices], mask)), ...);
	}
}

template<typename T, size_t... Indices>
inline constexpr void load_c_elements_masked_2(
		T* C, const size_t& src_c_cols, mp::mask_type& mask, mp::simde_type<T>* arr, std::index_sequence<Indices...>
) noexcept {
	if constexpr (std::is_integral<T>::value || std::is_unsigned<T>::value){
		((arr[Indices*2] = mp::SimdTraits<T>::loadu((mp::simde_type<T>*)&C[src_c_cols * Indices])), ...);
	}else{
		((arr[Indices*2] = mp::SimdTraits<T>::loadu(&C[src_c_cols * Indices])), ...);
	}
	if constexpr (std::is_unsigned<T>::value){
	((arr[Indices*2+1] = mp::SimdTraits<T>::load_masked(reinterpret_cast<std::make_signed_t<T>*>(&C[mp::pack_size_v<T>  + (src_c_cols * Indices)]), mask)), ...);
	}else{
	((arr[Indices*2+1] = mp::SimdTraits<T>::load_masked(&C[mp::pack_size_v<T>  + (src_c_cols * Indices)], mask)), ...);
	}
}


template<typename T, size_t... Indices>
inline constexpr void store_c_elements_masked(
		T* C, const size_t& src_c_cols, mp::mask_type& mask, mp::simde_type<T>* arr, std::index_sequence<Indices...>
) noexcept {

	/* if constexpr (std::is_integral<T>::value){ */
	/* (mp::SimdTraits<T>::store_masked(static_cast<mp::simde_type<T>*>(&C[src_c_cols * Indices]), mask, arr[Indices]), ...); */
	/* }else{ */
	if constexpr (std::is_unsigned<T>::value){
	(mp::SimdTraits<T>::store_masked(reinterpret_cast<std::make_signed_t<T>*>(&C[src_c_cols * Indices]), mask, arr[Indices]), ...);
	}else{
	(mp::SimdTraits<T>::store_masked(&C[src_c_cols * Indices], mask, arr[Indices]), ...);
	}
	/* } */
}

template<typename T, size_t... Indices>
inline constexpr void store_c_elements_masked_2(
		T* C, const size_t& src_c_cols, mp::mask_type& mask, mp::simde_type<T>* arr, std::index_sequence<Indices...>
) noexcept {
	if constexpr(std::is_integral<T>::value || std::is_unsigned<T>::value){
	(mp::SimdTraits<T>::storeu((mp::simde_type<T>*)&C[src_c_cols * Indices], arr[Indices*2]), ...);
	}else{
	(mp::SimdTraits<T>::storeu(&C[src_c_cols * Indices], arr[Indices*2]), ...);
	}
	if constexpr (std::is_unsigned<T>::value){
	(mp::SimdTraits<T>::store_masked(reinterpret_cast<std::make_signed_t<T>*>(&C[mp::pack_size_v<T>  + (src_c_cols * Indices)]), mask, arr[Indices*2+1]), ...);
	}else{
	(mp::SimdTraits<T>::store_masked(&C[mp::pack_size_v<T>  + (src_c_cols * Indices)], mask, arr[Indices*2+1]), ...);
	}
}


//this is only to be run if A_COLS > mp::pack_size_v<T>
//tile size is never more than 2 times mp::pack_size_v<T>
template<typename T>
inline constexpr void fused_product_2(mp::simde_type<T>& aVec, const T* A, mp::simde_type<T>& C0, mp::simde_type<T>& C1, const mp::simde_type<T>& B0, const mp::simde_type<T>& B1) noexcept{
	aVec = mp::SimdTraits<T>::broadcast(A);
	mp::SimdTraits<T>::fmadd(aVec, B0, C0);
	mp::SimdTraits<T>::fmadd(aVec, B1, C1);
}

template<typename T>
inline constexpr void fused_product_1(mp::simde_type<T>& aVec, const T* A, mp::simde_type<T>& C0, const mp::simde_type<T>& B0) noexcept{
	aVec = mp::SimdTraits<T>::broadcast(A);
	mp::SimdTraits<T>::fmadd(aVec, B0, C0);

}

template<typename T, size_t total_row_elements, size_t... colIndices>
inline constexpr void second_loop_direct_2(mp::simde_type<T>& aVec, const T* A, mp::simde_type<T>& C0, mp::simde_type<T>& C1,
		const std::array<mp::simde_type<T>, total_row_elements>& rowBs,
		std::index_sequence<colIndices...>) noexcept{
	(fused_product_2(aVec, A + colIndices, C0, C1, rowBs[colIndices * 2], rowBs[colIndices * 2 + 1]), ...);
}

template<typename T, size_t total_row_elements, size_t... colIndices>
inline constexpr void second_loop_direct_1(mp::simde_type<T>& aVec, const T* A, mp::simde_type<T>& C0, 
		const std::array<mp::simde_type<T>, total_row_elements>& rowBs,
		std::index_sequence<colIndices...>) noexcept{
	(fused_product_1(aVec, A + colIndices, C0, rowBs[colIndices]), ...);
}

//rowIndices corresponds to the number of rows in A
template<typename T, size_t per_row, size_t total_row_elements, size_t A_COLS, size_t... rowIndices>
inline constexpr void krun_loops_directly(mp::simde_type<T>& aVec, const T* A,
				const std::array<mp::simde_type<T>, total_row_elements>& rowBs,
				mp::simde_type<T>* rowCs,
				std::index_sequence<rowIndices...>) noexcept {
	if constexpr (per_row == 2){
		//the amount of collumns in A packed is tile_size_v<T>
		(second_loop_direct_2(aVec, A + (tile_size_v<T> * rowIndices), rowCs[rowIndices * 2], rowCs[rowIndices * 2 + 1], rowBs, std::make_index_sequence<A_COLS>{}), ...);
	}else if(per_row == 1){
		(second_loop_direct_1(aVec, A + (tile_size_v<T> * rowIndices), rowCs[rowIndices], rowBs, std::make_index_sequence<A_COLS>{}), ...);
		
	}
	
	
}



//addition is the number of collumns in B packed
template<typename T, size_t A_ROWS, size_t A_COLS, size_t B_ROWS, size_t B_COLS, size_t ADDITION>
void kmatmult_simdeT_directly_threaded(const T* A, const T* B, T* C, const size_t& src_c_cols){
	static_assert(A_COLS == B_ROWS, "Expected A_COLS to be the same as B_ROWS");
	//going to load all the row elements from B into vectors and store them in an array
	constexpr size_t rowB_size = B_COLS / mp::pack_size_v<T>;
	constexpr size_t total_row_elements = B_ROWS * rowB_size;
	const std::array<mp::simde_type<T>, total_row_elements> rowBs = load_threaded_row_elements_skip<T, ADDITION,
									(B_COLS == mp::pack_size_v<T> ? 1 : 0)>(B, std::make_index_sequence<total_row_elements>{});

	//now I need to load all of the rows of C into vectors
	constexpr size_t per_row = B_COLS / mp::pack_size_v<T>; // the amount of vectors per row
	static_assert(per_row == 1 || per_row == 2, "Error with per row logic!");
	static_assert(B_COLS % mp::pack_size_v<T> == 0, "Error, directly simdeT does not handle masking, use masked version");
	constexpr size_t total_c_row_elements = per_row * A_ROWS;
	mp::simde_type<T> rowCs[total_c_row_elements];
	load_c_elements_2<T, per_row>(C, src_c_cols, rowCs, std::make_index_sequence<total_c_row_elements>{});
	
	//an element to broadcast A to
	mp::simde_type<T> aVector;

	//run the appropriate dot products
	krun_loops_directly<T, per_row, total_row_elements, A_COLS>(aVector, A, rowBs, rowCs, std::make_index_sequence<A_ROWS>{});


	//now to store rowCs back into C:
	store_c_elements<T, per_row>(C, src_c_cols, rowCs, std::make_index_sequence<total_c_row_elements>{});
}



template<typename T, size_t A_ROWS, size_t A_COLS, size_t B_ROWS, size_t B_COLS, size_t ADDITION>
void kmatmult_simdeT_masked_threaded(const T* A, const T* B, T* C, const size_t& src_c_cols){
	static_assert(A_COLS == B_ROWS, "Expected A_COLS to be the same as B_ROWS");
	static_assert(B_COLS % mp::pack_size_v<T> != 0, "B cols should not be divisible by pack size, inefficiency error"); 
	//going to load all the row elements from B into vectors and store them in an array
	constexpr size_t rowB_size =  B_COLS / mp::pack_size_v<T> + 1;
	constexpr size_t total_row_elements = B_ROWS * rowB_size;
	const std::array<mp::simde_type<T>, total_row_elements> rowBs = load_threaded_row_elements_skip<T, ADDITION, 
	      B_COLS < mp::pack_size_v<T> ? 1 : 0>(B, std::make_index_sequence<total_row_elements>{});
	
	//now I need to load all of the rows of C into vectors
	mp::mask_type mask = mp::Kgenerate_mask<T, B_COLS < mp::pack_size_v<T> ? B_COLS : B_COLS - mp::pack_size_v<T>>();
	constexpr size_t per_row = (B_COLS < mp::pack_size_v<T> ? 1 : 2);// the amount of vectors per row
 
	constexpr size_t total_c_row_elements = per_row * A_ROWS;// the amount of vectors per row  
	mp::simde_type<T> rowCs[total_c_row_elements];
	if constexpr (B_COLS < mp::pack_size_v<T>){
		load_c_elements_masked<T>(C, src_c_cols, mask, rowCs, std::make_index_sequence<total_c_row_elements>{});	
	}else{
		load_c_elements_masked_2<T>(C, src_c_cols, mask, rowCs, std::make_index_sequence<total_c_row_elements/2>{});	
	}
	

	//an element to broadcast A to
	mp::simde_type<T> aVector;

	//run the appropriate dot products
	krun_loops_directly<T, per_row, total_row_elements, A_COLS>(aVector, A, rowBs, rowCs, std::make_index_sequence<A_ROWS>{});





	//now to store rowCs back into C:
	if constexpr (B_COLS < mp::pack_size_v<T>){
		store_c_elements_masked<T>(C, src_c_cols, mask, rowCs, std::make_index_sequence<total_c_row_elements>{});	
	}else{
		store_c_elements_masked_2<T>(C, src_c_cols, mask, rowCs, std::make_index_sequence<total_c_row_elements/2>{});	
	}
}

template<typename T, size_t A_ROWS, size_t A_COLS, size_t B_ROWS, size_t B_COLS, size_t ADDITION>
void kmatmult_simdeT_threaded_fma(const T* A, const T* B, T* C, const size_t& src_c_cols){
	if constexpr (B_COLS % mp::pack_size_v<T> == 0){
		kmatmult_simdeT_directly_threaded<T, A_ROWS, A_COLS, B_ROWS, B_COLS, ADDITION>(A, B, C, src_c_cols);
	}else{
		kmatmult_simdeT_masked_threaded<T, A_ROWS, A_COLS, B_ROWS, B_COLS, ADDITION>(A, B, C, src_c_cols);
	}
}



}}} //nt::functional::std_functional::

#endif // _NT_KMATMULT_SIMDE_TRAITS_HPP_
