#ifndef _NT_MATMULT_MACROS_H_
#define _NT_MATMULT_MACROS_H_

#define _NT_MATMULT_MIN_(x, y) x < y ? x : y
#define _NT_MATMULT_MAX_(x, y) x > y ? x : y


#define _NT_MATMULT_ENSURE_ALIGNMENT_(type, align_byte, amt) (((amt * sizeof(type)) % align_byte != 0) ? (amt * sizeof(type)) + align_byte - ((amt * sizeof(type)) % align_byte) : amt * sizeof(type))


#define _NT_MATMULT_NTHREADS_ 21
#define _NT_MATMULT_DECLARE_STATIC_BLOCK_(type) static type blockA_packed_##type[_NT_MATMULT_ENSURE_ALIGNMENT_(type, 64, tile_size_v<type> * _NT_MATMULT_NTHREADS_ * tile_size_v<type> * _NT_MATMULT_NTHREADS_)] __attribute((aligned(64)));\
				   static type blockB_packed_##type[_NT_MATMULT_ENSURE_ALIGNMENT_(type, 64, tile_size_v<type> * _NT_MATMULT_NTHREADS_ * tile_size_v<type> * _NT_MATMULT_NTHREADS_)] __attribute((aligned(64)));

#define DETER_NT_MATMULT_MIN_E_SIZE(length, addition) length % addition == 0 ? length / addition : int64_t(length / addition) + 1

#define _NT_MATMULT_DETERMINE_SIZE_(length, addition) length % addition == 0 ? length / addition : int64_t(length / addition) + 1


#endif //_NT_MATMULT_MACROS_H_
