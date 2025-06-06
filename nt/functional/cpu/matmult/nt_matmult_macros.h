
//this is a file that has macros to auto generate switch statements
//this way the matmult function can remain to have a templated type
//and a variable tile size and pack size depending on the type

#ifndef _NT_MATMULT_MACROS_H_
#define _NT_MATMULT_MACROS_H_

//silence depreciation warnings for certain needed headers
#ifdef _MSC_VER
#ifndef _SILENCE_CXX17_C_HEADER_DEPRECATION_WARNING
#define _SILENCE_CXX17_C_HEADER_DEPRECATION_WARNING
#endif
#endif

#define _NT_MATMULT_MIN_(x, y) x < y ? x : y
#define _NT_MATMULT_MAX_(x, y) x > y ? x : y


#define _NT_MATMULT_ENSURE_ALIGNMENT_(type, align_byte, amt) (((amt * sizeof(type)) % align_byte != 0) ? (amt * sizeof(type)) + align_byte - ((amt * sizeof(type)) % align_byte) : amt * sizeof(type))


#define _NT_MATMULT_NTHREADS_ 21


//just easier to have in one place so I don't have to modify 16 files in case the matmult function is ever decided to be changed

#define _NT_DECLARE_MATMULT_TYPE_CPP_(type)\
	template void nt_matmult<type>(const type* A, const type* B, type* C, int64_t a_rows, int64_t a_cols, int64_t b_rows, int64_t b_cols, bool transpose_a, bool transpose_b);\
	template void nt_matmult_batch<type>(const type** A, const type** B, type** C, int64_t batches, int64_t a_rows, int64_t a_cols, int64_t b_rows, int64_t b_cols, bool transpose_a, bool transpose_b);

//routes for when simd support is there for the type

// #if defined(_MSC_VER) && defined(_WIN32)
//     #define NT_MATMULT_DECLARE_STATIC_BLOCK__(type)\
//         static alignas(64) type blockA_packed_##type[_NT_MATMULT_ENSURE_ALIGNMENT_(type, 64, tile_size_v<type> * _NT_MATMULT_NTHREADS_ * tile_size_v<type> * _NT_MATMULT_NTHREADS_)];\
//         static alignas(64) type blockB_packed_##type[_NT_MATMULT_ENSURE_ALIGNMENT_(type, 64, tile_size_v<type> * _NT_MATMULT_NTHREADS_ * tile_size_v<type> * _NT_MATMULT_NTHREADS_)];\
// #else

#define _NT_MATMULT_DECLARE_STATIC_BLOCK_(type)\
	alignas(64) static type blockA_packed_##type[_NT_MATMULT_ENSURE_ALIGNMENT_(type, 64, tile_size_v<type> * _NT_MATMULT_NTHREADS_ * tile_size_v<type> * _NT_MATMULT_NTHREADS_)];
	alignas(64) static type blockB_packed_##type[_NT_MATMULT_ENSURE_ALIGNMENT_(type, 64, tile_size_v<type> * _NT_MATMULT_NTHREADS_ * tile_size_v<type> * _NT_MATMULT_NTHREADS_)];\
	template<>\
	type* get_blockA_packed<type>(){\
		return blockA_packed_##type;\
	}\
	template<>\
	type* get_blockB_packed<type>(){\
		return blockB_packed_##type;\
	}\
	_NT_DECLARE_MATMULT_TYPE_CPP_(type)

#define _NT_MATMULT_DO_NOT_DECLARE_STATIC_BLOCK_(type)\
	template<>\
	type* get_blockA_packed<type>(){return nullptr;}\
	template<>\
	type* get_blockB_packed<type>(){return nullptr;}\
	_NT_DECLARE_MATMULT_TYPE_CPP_(type)





#define _NT_MATMULT_DETERMINE_SIZE_(length, addition) length % addition == 0 ? length / addition : int64_t(length / addition) + 1


#define _NT_MATMULT_CASE_0_(max, case_macro)\
	default:\
		case_macro(max, 0, 0)
 
#define _NT_MATMULT_CASE_1_(max, case_macro)\
	case 1:\
		case_macro(max, 1, 0)\
	_NT_MATMULT_CASE_0_(max, case_macro)
 
#define _NT_MATMULT_CASE_2_(max, case_macro)\
	case 2:\
		case_macro(max, 2, 1)\
	_NT_MATMULT_CASE_1_(max, case_macro)
 
#define _NT_MATMULT_CASE_3_(max, case_macro)\
	case 3:\
		case_macro(max, 3, 2)\
	_NT_MATMULT_CASE_2_(max, case_macro)
 
#define _NT_MATMULT_CASE_4_(max, case_macro)\
	case 4:\
		case_macro(max, 4, 3)\
	_NT_MATMULT_CASE_3_(max, case_macro)
 
#define _NT_MATMULT_CASE_5_(max, case_macro)\
	case 5:\
		case_macro(max, 5, 4)\
	_NT_MATMULT_CASE_4_(max, case_macro)
 
#define _NT_MATMULT_CASE_6_(max, case_macro)\
	case 6:\
		case_macro(max, 6, 5)\
	_NT_MATMULT_CASE_5_(max, case_macro)
 
#define _NT_MATMULT_CASE_7_(max, case_macro)\
	case 7:\
		case_macro(max, 7, 6)\
	_NT_MATMULT_CASE_6_(max, case_macro)
 
#define _NT_MATMULT_CASE_8_(max, case_macro)\
	case 8:\
		case_macro(max, 8, 7)\
	_NT_MATMULT_CASE_7_(max, case_macro)
 
#define _NT_MATMULT_CASE_9_(max, case_macro)\
	case 9:\
		case_macro(max, 9, 8)\
	_NT_MATMULT_CASE_8_(max, case_macro)
 
#define _NT_MATMULT_CASE_10_(max, case_macro)\
	case 10:\
		case_macro(max, 10, 9)\
	_NT_MATMULT_CASE_9_(max, case_macro)
 
#define _NT_MATMULT_CASE_11_(max, case_macro)\
	case 11:\
		case_macro(max, 11, 10)\
	_NT_MATMULT_CASE_10_(max, case_macro)
 
#define _NT_MATMULT_CASE_12_(max, case_macro)\
	case 12:\
		case_macro(max, 12, 11)\
	_NT_MATMULT_CASE_11_(max, case_macro)
 
#define _NT_MATMULT_CASE_13_(max, case_macro)\
	case 13:\
		case_macro(max, 13, 12)\
	_NT_MATMULT_CASE_12_(max, case_macro)
 
#define _NT_MATMULT_CASE_14_(max, case_macro)\
	case 14:\
		case_macro(max, 14, 13)\
	_NT_MATMULT_CASE_13_(max, case_macro)
 
#define _NT_MATMULT_CASE_15_(max, case_macro)\
	case 15:\
		case_macro(max, 15, 14)\
	_NT_MATMULT_CASE_14_(max, case_macro)
 
#define _NT_MATMULT_CASE_16_(max, case_macro)\
	case 16:\
		case_macro(max, 16, 15)\
	_NT_MATMULT_CASE_15_(max, case_macro)
 
#define _NT_MATMULT_CASE_17_(max, case_macro)\
	case 17:\
		case_macro(max, 17, 16)\
	_NT_MATMULT_CASE_16_(max, case_macro)
 
#define _NT_MATMULT_CASE_18_(max, case_macro)\
	case 18:\
		case_macro(max, 18, 17)\
	_NT_MATMULT_CASE_17_(max, case_macro)
 
#define _NT_MATMULT_CASE_19_(max, case_macro)\
	case 19:\
		case_macro(max, 19, 18)\
	_NT_MATMULT_CASE_18_(max, case_macro)
 
#define _NT_MATMULT_CASE_20_(max, case_macro)\
	case 20:\
		case_macro(max, 20, 19)\
	_NT_MATMULT_CASE_19_(max, case_macro)
 
#define _NT_MATMULT_CASE_21_(max, case_macro)\
	case 21:\
		case_macro(max, 21, 20)\
	_NT_MATMULT_CASE_20_(max, case_macro)
 
#define _NT_MATMULT_CASE_22_(max, case_macro)\
	case 22:\
		case_macro(max, 22, 21)\
	_NT_MATMULT_CASE_21_(max, case_macro)
 
#define _NT_MATMULT_CASE_23_(max, case_macro)\
	case 23:\
		case_macro(max, 23, 22)\
	_NT_MATMULT_CASE_22_(max, case_macro)
 
#define _NT_MATMULT_CASE_24_(max, case_macro)\
	case 24:\
		case_macro(max, 24, 23)\
	_NT_MATMULT_CASE_23_(max, case_macro)
 
#define _NT_MATMULT_CASE_25_(max, case_macro)\
	case 25:\
		case_macro(max, 25, 24)\
	_NT_MATMULT_CASE_24_(max, case_macro)
 
#define _NT_MATMULT_CASE_26_(max, case_macro)\
	case 26:\
		case_macro(max, 26, 25)\
	_NT_MATMULT_CASE_25_(max, case_macro)
 
#define _NT_MATMULT_CASE_27_(max, case_macro)\
	case 27:\
		case_macro(max, 27, 26)\
	_NT_MATMULT_CASE_26_(max, case_macro)
 
#define _NT_MATMULT_CASE_28_(max, case_macro)\
	case 28:\
		case_macro(max, 28, 27)\
	_NT_MATMULT_CASE_27_(max, case_macro)
 
#define _NT_MATMULT_CASE_29_(max, case_macro)\
	case 29:\
		case_macro(max, 29, 28)\
	_NT_MATMULT_CASE_28_(max, case_macro)
 
#define _NT_MATMULT_CASE_30_(max, case_macro)\
	case 30:\
		case_macro(max, 30, 29)\
	_NT_MATMULT_CASE_29_(max, case_macro)
 
#define _NT_MATMULT_CASE_31_(max, case_macro)\
	case 31:\
		case_macro(max, 31, 30)\
	_NT_MATMULT_CASE_30_(max, case_macro)
 
#define _NT_MATMULT_CASE_32_(max, case_macro)\
	case 32:\
		case_macro(max, 32, 31)\
	_NT_MATMULT_CASE_31_(max, case_macro)
 
#define _NT_MATMULT_CASE_33_(max, case_macro)\
	case 33:\
		case_macro(max, 33, 32)\
	_NT_MATMULT_CASE_32_(max, case_macro)
 
#define _NT_MATMULT_CASE_34_(max, case_macro)\
	case 34:\
		case_macro(max, 34, 33)\
	_NT_MATMULT_CASE_33_(max, case_macro)
 
#define _NT_MATMULT_CASE_35_(max, case_macro)\
	case 35:\
		case_macro(max, 35, 34)\
	_NT_MATMULT_CASE_34_(max, case_macro)
 
#define _NT_MATMULT_CASE_36_(max, case_macro)\
	case 36:\
		case_macro(max, 36, 35)\
	_NT_MATMULT_CASE_35_(max, case_macro)
 
#define _NT_MATMULT_CASE_37_(max, case_macro)\
	case 37:\
		case_macro(max, 37, 36)\
	_NT_MATMULT_CASE_36_(max, case_macro)
 
#define _NT_MATMULT_CASE_38_(max, case_macro)\
	case 38:\
		case_macro(max, 38, 37)\
	_NT_MATMULT_CASE_37_(max, case_macro)
 
#define _NT_MATMULT_CASE_39_(max, case_macro)\
	case 39:\
		case_macro(max, 39, 38)\
	_NT_MATMULT_CASE_38_(max, case_macro)
 
#define _NT_MATMULT_CASE_40_(max, case_macro)\
	case 40:\
		case_macro(max, 40, 39)\
	_NT_MATMULT_CASE_39_(max, case_macro)
 
#define _NT_MATMULT_CASE_41_(max, case_macro)\
	case 41:\
		case_macro(max, 41, 40)\
	_NT_MATMULT_CASE_40_(max, case_macro)
 
#define _NT_MATMULT_CASE_42_(max, case_macro)\
	case 42:\
		case_macro(max, 42, 41)\
	_NT_MATMULT_CASE_41_(max, case_macro)
 
#define _NT_MATMULT_CASE_43_(max, case_macro)\
	case 43:\
		case_macro(max, 43, 42)\
	_NT_MATMULT_CASE_42_(max, case_macro)
 
#define _NT_MATMULT_CASE_44_(max, case_macro)\
	case 44:\
		case_macro(max, 44, 43)\
	_NT_MATMULT_CASE_43_(max, case_macro)
 
#define _NT_MATMULT_CASE_45_(max, case_macro)\
	case 45:\
		case_macro(max, 45, 44)\
	_NT_MATMULT_CASE_44_(max, case_macro)
 
#define _NT_MATMULT_CASE_46_(max, case_macro)\
	case 46:\
		case_macro(max, 46, 45)\
	_NT_MATMULT_CASE_45_(max, case_macro)
 
#define _NT_MATMULT_CASE_47_(max, case_macro)\
	case 47:\
		case_macro(max, 47, 46)\
	_NT_MATMULT_CASE_46_(max, case_macro)
 
#define _NT_MATMULT_CASE_48_(max, case_macro)\
	case 48:\
		case_macro(max, 48, 47)\
	_NT_MATMULT_CASE_47_(max, case_macro)
 
#define _NT_MATMULT_CASE_49_(max, case_macro)\
	case 49:\
		case_macro(max, 49, 48)\
	_NT_MATMULT_CASE_48_(max, case_macro)
 
#define _NT_MATMULT_CASE_50_(max, case_macro)\
	case 50:\
		case_macro(max, 50, 49)\
	_NT_MATMULT_CASE_49_(max, case_macro)
 
#define _NT_MATMULT_CASE_51_(max, case_macro)\
	case 51:\
		case_macro(max, 51, 50)\
	_NT_MATMULT_CASE_50_(max, case_macro)
 
#define _NT_MATMULT_CASE_52_(max, case_macro)\
	case 52:\
		case_macro(max, 52, 51)\
	_NT_MATMULT_CASE_51_(max, case_macro)
 
#define _NT_MATMULT_CASE_53_(max, case_macro)\
	case 53:\
		case_macro(max, 53, 52)\
	_NT_MATMULT_CASE_52_(max, case_macro)
 
#define _NT_MATMULT_CASE_54_(max, case_macro)\
	case 54:\
		case_macro(max, 54, 53)\
	_NT_MATMULT_CASE_53_(max, case_macro)
 
#define _NT_MATMULT_CASE_55_(max, case_macro)\
	case 55:\
		case_macro(max, 55, 54)\
	_NT_MATMULT_CASE_54_(max, case_macro)
 
#define _NT_MATMULT_CASE_56_(max, case_macro)\
	case 56:\
		case_macro(max, 56, 55)\
	_NT_MATMULT_CASE_55_(max, case_macro)
 
#define _NT_MATMULT_CASE_57_(max, case_macro)\
	case 57:\
		case_macro(max, 57, 56)\
	_NT_MATMULT_CASE_56_(max, case_macro)
 
#define _NT_MATMULT_CASE_58_(max, case_macro)\
	case 58:\
		case_macro(max, 58, 57)\
	_NT_MATMULT_CASE_57_(max, case_macro)
 
#define _NT_MATMULT_CASE_59_(max, case_macro)\
	case 59:\
		case_macro(max, 59, 58)\
	_NT_MATMULT_CASE_58_(max, case_macro)
 
#define _NT_MATMULT_CASE_60_(max, case_macro)\
	case 60:\
		case_macro(max, 60, 59)\
	_NT_MATMULT_CASE_59_(max, case_macro)
 
#define _NT_MATMULT_CASE_61_(max, case_macro)\
	case 61:\
		case_macro(max, 61, 60)\
	_NT_MATMULT_CASE_60_(max, case_macro)
 
#define _NT_MATMULT_CASE_62_(max, case_macro)\
	case 62:\
		case_macro(max, 62, 61)\
	_NT_MATMULT_CASE_61_(max, case_macro)
 
#define _NT_MATMULT_CASE_63_(max, case_macro)\
	case 63:\
		case_macro(max, 63, 62)\
	_NT_MATMULT_CASE_62_(max, case_macro)
 
#define _NT_MATMULT_CASE_64_(max, case_macro)\
	case 64:\
		case_macro(max, 64, 63)\
	_NT_MATMULT_CASE_63_(max, case_macro)
 
#define _NT_MATMULT_SWITCH_(max, var_name, case_macro)\
	switch(var_name){\
		_NT_MATMULT_CASE_##max##_(max, case_macro)\
	}
#define _NT_MATMULT_TILE_KIF_64_(start, var_name, case_macro)\
	start if constexpr (tile_size == 64){\
		_NT_MATMULT_SWITCH_(64, var_name, case_macro);\
	}\
	_NT_MATMULT_TILE_KIF_32_(else, var_name, case_macro)
 
#define _NT_MATMULT_TILE_KIF_32_(start, var_name, case_macro)\
	start if constexpr (tile_size == 32){\
		_NT_MATMULT_SWITCH_(32, var_name, case_macro);\
	}\
	_NT_MATMULT_TILE_KIF_16_(else, var_name, case_macro)
 
#define _NT_MATMULT_TILE_KIF_16_(start, var_name, case_macro)\
	start if constexpr (tile_size == 16){\
		_NT_MATMULT_SWITCH_(16, var_name, case_macro);\
	}\
	_NT_MATMULT_TILE_KIF_8_(else, var_name, case_macro)
 
#define _NT_MATMULT_TILE_KIF_8_(start, var_name, case_macro)\
	start if constexpr (tile_size == 8){\
		_NT_MATMULT_SWITCH_(8, var_name, case_macro);\
	}
 
#define _NT_MATMULT_SWITCHES_(var_name, case_macro) _NT_MATMULT_TILE_KIF_64_(, var_name, case_macro)

#endif //_NT_MATMULT_MACROS_H_
