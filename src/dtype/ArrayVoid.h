#ifndef ARRAY_VOID_H
#define ARRAY_VOID_H
#include "ranges.h"
namespace nt {
class ArrayVoid;
template<typename First, typename... Rest>
struct IsFirstVectorArrayVoid {
    static constexpr bool value = false;
};

// Specialization for when the first argument is a std::vector
template<typename... Args>
struct IsFirstVectorArrayVoid<std::vector<ArrayVoid>, Args...> {
    static constexpr bool value = true;
};

template<typename... Args>
struct IsFirstVectorArrayVoid<std::vector<std::reference_wrapper<const ArrayVoid> >, Args...> {
    static constexpr bool value = true;
};

}


#include <memory>
#include <cstdio>
#include <type_traits>

#include "DType_enum.h"
#include <vector>
#include <cstdlib>
#include "Scalar.h"
#include "../Tensor.h"
#include "../intrusive_ptr/intrusive_ptr.hpp"
#include "../memory/bucket.h"
#include "../memory/iterator.h"
#include <type_traits>
#include <functional>

namespace nt{

class ArrayVoid{
	friend class Bucket;
	Bucket bucket;
	uint64_t size;
	
	/* void unique_strides(bool pre_order=true); */
	/* intrusive_ptr<void*> make_unique_strides(bool pre_order=true) const; */
	/* intrusive_ptr<void*> make_unique_strides(std::size_t start, std::size_t end, bool copy=true) const; */
	const size_t dtype_size(DType) const;
	/* ArrayVoid(intrusive_ptr<void*>&&, const std::size_t, const std::size_t, const std::size_t, DType); */
	/* ArrayVoid(const intrusive_ptr<void*>&, const std::size_t, const std::size_t, const std::size_t, DType); */
	ArrayVoid(Bucket&&, uint64_t, DType);
	ArrayVoid(const Bucket&, uint64_t, DType);
	ArrayVoid(const Bucket&);
	ArrayVoid(Bucket&&);
	inline static ArrayVoid catV(const std::vector<ArrayVoid>& v){
		std::vector<std::reference_wrapper<const Bucket> > buckets;
		buckets.reserve(v.size());
		for(const ArrayVoid& arr : v){
			buckets.push_back(std::cref(arr.bucket));
		}
		return Bucket::cat(buckets);
	}
	inline static ArrayVoid catV(const std::vector<std::reference_wrapper<const ArrayVoid> >& v){
		std::vector<Bucket> buckets;
		for(const std::reference_wrapper<const ArrayVoid>& arr : v){
			buckets.push_back(arr.get().bucket);
		}
		return Bucket::cat(buckets);
	}

	public:
		DType dtype;
		ArrayVoid(int64_t, DType);
		ArrayVoid(int64_t, DType, void*, DeleterFnPtr);
#ifdef USE_PARALLEL
		ArrayVoid(int64_t, DTypeShared);
#endif
		ArrayVoid& operator=(const ArrayVoid&);
		ArrayVoid& operator=(ArrayVoid&&);
		ArrayVoid(const ArrayVoid&);
		ArrayVoid(ArrayVoid&&);
		inline Bucket& get_bucket() {return bucket;}
		inline const DeviceType& device_type() const noexcept {return bucket.device_type();}
		inline const Bucket& get_bucket() const {return bucket;}
		inline void nullify() {size = 0; bucket.nullify();}
		inline const uint64_t& Size() const {return size;}
		inline const void* data_ptr() const {return bucket.data_ptr();}
		inline void* data_ptr() {return bucket.data_ptr();}
		const void* data_ptr_end() const;
		void* data_ptr_end();
		void swap(ArrayVoid&);
		/* void** strides_cbegin() const; */
		void** stride_begin() const {return bucket.stride_begin();}
		/* void** strides_cend() const; */
		void** stride_end() const {return bucket.stride_end();}
		inline const bool is_shared() const {return bucket.is_shared();}
		inline const bool is_empty() const {return size == 0;}
		inline const bool is_null() const {return bucket.is_null();}
		ArrayVoid& operator=(Scalar);
		ArrayVoid& fill_ptr_(Scalar);
		/* std::shared_ptr<void> share_part(uint32_t) const; */ //not used anymore
		ArrayVoid share_array(uint64_t) const;
		ArrayVoid share_array(uint64_t, uint64_t) const;
		inline ArrayVoid force_contiguity(int64_t n_size=-1) const {return ArrayVoid(bucket.force_contiguity(n_size == -1 ? size : n_size));}
		inline ArrayVoid bucket_all_indices() const {return ArrayVoid(bucket.bucket_all_indices(), size, dtype);}
		inline ArrayVoid force_contiguity_and_bucket() const {
			Bucket b = bucket.force_contiguity_and_bucket();
			int64_t n_size = b.size();
			return ArrayVoid(std::move(b), n_size, dtype);
		}
		inline ArrayVoid bound_force_contiguity_bucket() const {
			Bucket b = bucket.bound_force_contiguity_bucket();
			int64_t n_size = b.size();
			return ArrayVoid(std::move(b), n_size, dtype);
		}

		ArrayVoid change_stride(const std::vector<std::pair<uint64_t, uint64_t> >&) const;
		ArrayVoid change_stride(const std::vector<uint64_t>&) const;
		ArrayVoid range(std::vector<my_range>) const;
		inline bool is_contiguous() const {return bucket.is_contiguous();}
		inline ArrayVoid contiguous() const {return ArrayVoid(bucket.contiguous());}
		inline ArrayVoid clone() const {return ArrayVoid(bucket.clone());}
		/* template<DType dt = DType::Integer> */
		/* ArrayVoid copy_strides(bool copy=true) const; */ 
		inline ArrayVoid new_strides(uint64_t nsize) const{return ArrayVoid(bucket.new_stride_size(nsize), nsize, dtype);}
		ArrayVoid copy_strides(bool copy=false) const;
#ifdef USE_PARALLEL
		ArrayVoid shared_memory() const; // this is going to use shmem to create a shared memory version of ArrayVoid so that the memory can be shared across multiple processes, making functions like Queue possible, enacting a shared-memory version of the pointers.
		ArrayVoid from_shared_memory() const;
#endif
		/* static intrusive_ptr<void[]> MakeContiguousMemory(uint32_t _size, DType _type); */
		/* static intrusive_ptr<void[]> MakeContiguousMemory(uint32_t _size, DType _type, Scalar s); */
/* #ifdef USE_PARALLEL */
		/* static intrusive_ptr<void[]> MakeContiguousMemory(uint32_t _size, DTypeShared _type, Scalar s); */
		/* static intrusive_ptr<void[]> MakeContiguousMemory(uint32_t _size, DTypeShared _type); */
		/* static ArrayVoid FromShared(intrusive_ptr<void[]> ptr, uint64_t s, DType d) { return ArrayVoid(Bucket::FromShared(ptr, s, d));} */
		/* inline static ArrayVoid FromShared(intrusive_ptr<void[]> ptr, uint32_t _size, DTypeShared _type){return ArrayVoid(Bucket::FromShared(ptr, _size, _type));} */
/* #endif */


		inline static ArrayVoid makeEmptyArray(DType dtype = DType::Float32){
			return ArrayVoid(Bucket::makeNullBucket(dtype), 0, dtype);
		}		
		ArrayVoid& operator*=(Scalar);
		ArrayVoid& operator/=(Scalar);
		ArrayVoid& operator+=(Scalar);
		ArrayVoid& operator-=(Scalar);

		ArrayVoid operator*(Scalar) const;
		ArrayVoid operator/(Scalar) const;
		ArrayVoid operator+(Scalar) const;
		ArrayVoid operator-(Scalar) const;

		ArrayVoid& operator*=(const ArrayVoid&);
		ArrayVoid& operator/=(const ArrayVoid&);
		ArrayVoid& operator+=(const ArrayVoid&);
		ArrayVoid& operator-=(const ArrayVoid&);

		ArrayVoid operator*(const ArrayVoid&) const;
		ArrayVoid operator/(const ArrayVoid&) const;
		ArrayVoid operator+(const ArrayVoid&) const;
		ArrayVoid operator-(const ArrayVoid&) const;

		ArrayVoid& operator*=(const Tensor&);
		ArrayVoid& operator/=(const Tensor&);
		ArrayVoid& operator+=(const Tensor&);
		ArrayVoid& operator-=(const Tensor&);

		ArrayVoid operator*(const Tensor&) const;
		ArrayVoid operator/(const Tensor&) const;
		ArrayVoid operator+(const Tensor&) const;
		ArrayVoid operator-(const Tensor&) const;

		ArrayVoid operator!=(Scalar) const;
		ArrayVoid operator==(Scalar) const;
		ArrayVoid operator>=(Scalar) const;
		ArrayVoid operator<=(Scalar) const;
		ArrayVoid operator>(Scalar) const;
		ArrayVoid operator<(Scalar) const;
		ArrayVoid inverse() const;
		ArrayVoid& inverse_();
		ArrayVoid pow(int64_t) const;

		
		/* const std::size_t* stride_cbegin() const; */
		/* const std::size_t* stride_cend() const; */
		/* std::size_t* stride_begin(); */
		/* std::size_t* stride_end(); */
		/* std::vector<std::size_t>::const_iterator stride_it_cbegin() const; */
		/* std::vector<std::size_t>::const_iterator stride_it_cend() const; */
		/* std::vector<std::size_t>::iterator stride_it_begin(); */
		/* std::vector<std::size_t>::iterator stride_it_end(); */

		/* dtype_list begin(); */
		/* dtype_list end(); */
		/* const_dtype_list cbegin() const; */
		/* const_dtype_list cend() const; */
		inline const int64_t use_count() const {return bucket.use_count();}
		ArrayVoid& iota(Scalar);
		void copy(ArrayVoid&, unsigned long long i=0) const;
		ArrayVoid uint32() const;
		ArrayVoid int32() const;
		ArrayVoid Double() const;
		ArrayVoid Float() const;
		ArrayVoid cfloat() const;
		ArrayVoid cdouble() const;
		ArrayVoid tensorobj() const;
		ArrayVoid uint8() const;
		ArrayVoid int8() const;
		ArrayVoid uint16() const;
		ArrayVoid int16() const;
		ArrayVoid int64() const;
		ArrayVoid Bool() const;
		ArrayVoid to(DType) const;
		ArrayVoid to(DeviceType) const;
#ifdef _HALF_FLOAT_SUPPORT_
		ArrayVoid Float16() const;
		ArrayVoid Complex32() const;
#endif
#ifdef _128_FLOAT_SUPPORT_
		ArrayVoid Float128() const;
#endif
#ifdef __SIZEOF_INT128__
		ArrayVoid Int128() const;
		ArrayVoid UInt128() const;
#endif
		ArrayVoid exp() const;
		ArrayVoid& exp_();
		ArrayVoid& complex_();
		ArrayVoid& floating_();
		ArrayVoid& integer_();
		ArrayVoid& unsigned_();
		Tensor split(const uint64_t) const;
		Tensor split(const uint64_t, SizeRef) const;
		
		template<typename... arrVds>
		inline static ArrayVoid cat(const arrVds&... arrs){
			if constexpr(IsFirstVectorArrayVoid<arrVds...>::value && sizeof...(arrs) == 1){
				return ArrayVoid::catV(arrs...);	
			}
			else{
				static_assert(utils::SameType<ArrayVoid, arrVds...>::value, "Expected to only get ArrayVoids");
				return ArrayVoid(Bucket::cat(arrs.bucket...));

			}
		}
	
		template<DType dt, DType... dts, typename UnaryFunction, typename... Args>
		auto execute_function(UnaryFunction&& unary_op, Args&&... args);
		template<DType dt, DType... dts, typename UnaryFunction, typename... Args>
		auto execute_function(UnaryFunction&& unary_op, ArrayVoid& inp_arr, Args&&... args);
		template<DType dt, DType... dts>
		auto execute_function(bool throw_error=true, const char* func_name = __builtin_FUNCTION());
		template<DType dt, typename UnaryFunction, typename Output, typename... Args>
		void sub_handle_execute_function(UnaryFunction&& unary_op, Output& v, bool& called, ArrayVoid& inp_arr, Args&&... args);
		template<DType dt, typename UnaryFunction, typename Output, typename... Args>
		void sub_handle_execute_function(UnaryFunction&& unary_op, Output& v, bool& called, Args&&... args);
		template<DType dt, typename UnaryFunction, typename... Args>
		void sub_handle_execute_function_void(UnaryFunction&& unary_op, bool& called, ArrayVoid& inp_arr, Args&&... args);
		template<DType dt, typename UnaryFunction, typename... Args>
		void sub_handle_execute_function_void(UnaryFunction&& unary_op, bool& called, Args&&... args);
		
		template<typename WrappedTypes, typename UnaryFunction, typename... Args, std::enable_if_t<
			DTypeFuncs::is_wrapped_dtype<WrappedTypes>::value
			&& !DTypeFuncs::is_wrapped_dtype<UnaryFunction>::value, bool> = true>
		auto execute_function(UnaryFunction&& unary_op, Args&&... args);
		template<typename WrappedTypes, typename UnaryFunction, typename... Args>
		auto execute_function(UnaryFunction&& unary_op, ArrayVoid& inp_arr, Args&&... args); 
		template<typename WrappedTypes, std::enable_if_t<DTypeFuncs::is_wrapped_dtype<WrappedTypes>::value, bool> = true>
		auto execute_function(bool throw_error=true, const char* func_name = __builtin_FUNCTION()); 

		//this is where the execute function parallel is
/* #ifdef USE_PARALLEL */
		
/* 		template<DType dt, typename UnaryFunction, typename... Args> */
/* 		void sub_handle_execute_function_parallel_1(UnaryFunction&& unary_op, bool& called, Args&&... args); */
/* 		template<DType dt, typename UnaryFunction, typename... Args> */
/* 		void sub_handle_execute_function_parallel_2(UnaryFunction&& unary_op, bool& called, Args&&... args); */
/* 		template<DType dt, typename UnaryFunction, typename... Args> */
/* 		void sub_handle_execute_function_parallel_3(UnaryFunction&& unary_op, bool& called, Args&&... args); */

/* 		template<typename WrappedTypes, typename UnaryFunction, typename... Args, std::enable_if_t< */
/* 			DTypeFuncs::is_wrapped_dtype<WrappedTypes>::value */
/* 			&& !DTypeFuncs::is_wrapped_dtype<UnaryFunction>::value, bool> = true> */
/* 		void execute_function_parallel(UnaryFunction&& unary_op, Args&&... args); */
/* #endif */

		//this is to basically chunk up the data when the order in which the data is recieved, and the index of the data does not matter
		//you want to use the execute_function_chunk when can and appropriate
		//these have been the most optimized based on the way the memory is stored
		template<DType dt, typename UnaryFunction, typename... Args>
		void sub_handle_execute_function_chunk_1(UnaryFunction&& unary_op, bool& called, Args&&... args);
		template<DType dt, typename UnaryFunction, typename... Args>
		void sub_handle_execute_function_chunk_2(UnaryFunction&& unary_op, bool& called, Args&&... args);
		template<DType dt, typename UnaryFunction, typename... Args>
		void sub_handle_execute_function_chunk_3(UnaryFunction&& unary_op, bool& called, Args&&... args);	
		template<typename WrappedTypes, typename UnaryFunction, typename... Args, std::enable_if_t<
			DTypeFuncs::is_wrapped_dtype<WrappedTypes>::value
			&& !DTypeFuncs::is_wrapped_dtype<UnaryFunction>::value, bool> = true>
		void execute_function_chunk(UnaryFunction&& unary_op, Args&&... args);

		//execute 2 ArrayVoids at the same time
		//assumes ArrayVoids are same dtype (this will probably change in the future, no real reason for it other than easier to code)
		//if not bucketed, must be the same size
		//if is bucketed, must be the same size, or, same bucket_amt
		template<typename WrappedTypes, typename UnaryFunction, typename... Args, std::enable_if_t<
			DTypeFuncs::is_wrapped_dtype<WrappedTypes>::value
			&& !DTypeFuncs::is_wrapped_dtype<UnaryFunction>::value, bool> = true>
		void execute_function_chunk_execute(UnaryFunction&& unary_op, ArrayVoid& inp_arr, Args&&... args); 
		template<DType dt, typename UnaryFunction, typename... Args>
		void sub_handle_execute_function_chunk_1_1(UnaryFunction&& unary_op, ArrayVoid& inp_arr, bool& called, Args&&... args);
		template<DType dt, typename UnaryFunction, typename... Args>
		void sub_handle_execute_function_chunk_1_2(UnaryFunction&& unary_op, ArrayVoid& inp_arr, bool& called, Args&&... args);
		template<DType dt, typename UnaryFunction, typename... Args>
		void sub_handle_execute_function_chunk_1_3(UnaryFunction&& unary_op, ArrayVoid& inp_arr, bool& called, Args&&... args);
		template<DType dt, typename UnaryFunction, typename... Args>
		void sub_handle_execute_function_chunk_3_1(UnaryFunction&& unary_op, ArrayVoid& inp_arr, bool& called, Args&&... args);
		template<DType dt, typename UnaryFunction, typename... Args>
		void sub_handle_execute_function_chunk_3_2(UnaryFunction&& unary_op, ArrayVoid& inp_arr, bool& called, Args&&... args);
		template<DType dt, typename UnaryFunction, typename... Args>
		void sub_handle_execute_function_chunk_3_3(UnaryFunction&& unary_op, ArrayVoid& inp_arr, bool& called, Args&&... args);
		template<DType dt, typename UnaryFunction, typename... Args>
		void sub_handle_execute_function_chunk_2_1(UnaryFunction&& unary_op, ArrayVoid& inp_arr, bool& called, Args&&... args);
		template<DType dt, typename UnaryFunction, typename... Args>
		void sub_handle_execute_function_chunk_2_3(UnaryFunction&& unary_op, ArrayVoid& inp_arr, bool& called, Args&&... args);
		template<DType dt, typename UnaryFunction, typename... Args>
		void sub_handle_execute_function_chunk_2_2(UnaryFunction&& unary_op, ArrayVoid& inp_arr, bool& called, Args&&... args);

		
		template<DType dt, DType... dts, typename UnaryFunction, typename... Args>
		auto cexecute_function(UnaryFunction&& unary_op, Args&&... args) const;
		template<DType dt, DType... dts, typename UnaryFunction, typename... Args>
		auto cexecute_function(UnaryFunction&& unary_op, const ArrayVoid& inp_arr, Args&&... args) const;
		template<DType dt, DType... dts>
		auto cexecute_function(bool throw_error=true, const char* func_name = __builtin_FUNCTION()) const;
		template<DType dt, typename UnaryFunction, typename Output, typename... Args>
		void sub_handle_cexecute_function(UnaryFunction&& unary_op, Output& v, bool& called, const ArrayVoid& inp_arr, Args&&... args) const;
		template<DType dt, typename UnaryFunction, typename Output, typename... Args>
		void sub_handle_cexecute_function(UnaryFunction&& unary_op, Output& v, bool& called, Args&&... args) const;
		template<DType dt, typename UnaryFunction, typename... Args>
		void sub_handle_cexecute_function_void(UnaryFunction&& unary_op, bool& called, const ArrayVoid& inp_arr, Args&&... args) const;
		template<DType dt, typename UnaryFunction, typename... Args>
		void sub_handle_cexecute_function_void(UnaryFunction&& unary_op, bool& called, Args&&... args) const;

		template<typename WrappedTypes, typename UnaryFunction, typename... Args, std::enable_if_t<
			DTypeFuncs::is_wrapped_dtype<WrappedTypes>::value
			&& !DTypeFuncs::is_wrapped_dtype<UnaryFunction>::value, bool> = true>
		auto cexecute_function(UnaryFunction&& unary_op, Args&&... args) const;
		template<typename WrappedTypes, typename UnaryFunction, typename... Args>
		auto cexecute_function(UnaryFunction&& unary_op, const ArrayVoid& inp_arr, Args&&... args) const; 
		template<typename WrappedTypes, std::enable_if_t<DTypeFuncs::is_wrapped_dtype<WrappedTypes>::value, bool> = true>
		auto cexecute_function(bool throw_error=true, const char* func_name = __builtin_FUNCTION()) const; 
		
		template<class UnaryOperator, class... Args>
		auto execute_function(UnaryOperator&& unary_op, Args&&... arg);
		template<class UnaryOperator, class... Args>
		auto execute_function_nbool(UnaryOperator&& unary_op, Args&&... arg);
		template<class UnaryOperator, class... Args>
		auto cexecute_function(UnaryOperator&& unary_op, Args&&... arg) const;
		template<class UnaryOperator, class... Args>
		auto cexecute_function_nbool(UnaryOperator&& unary_op, Args&&... arg) const;
		template<class UnaryOperator, class... Args>
		auto execute_function(UnaryOperator&& unary_op, ArrayVoid& inp_arr, Args&&... arg);
		template<class UnaryOperator, class... Args>
		auto cexecute_function(UnaryOperator&& unary_op, const ArrayVoid& inp_arr, Args&&... arg) const;
		template<class UnaryOperator, class... Args>
		auto execute_function_nbool(UnaryOperator&& unary_op, ArrayVoid& inp_arr, Args&&... arg);
		template<class UnaryOperator, class... Args>
		auto cexecute_function_nbool(UnaryOperator&& unary_op, const ArrayVoid& inp_arr, Args&&... arg) const;


		template<DType... dts, class OutputIt, class UnaryOperation, std::enable_if_t<!std::is_same_v<OutputIt, bool>, bool> = true>
		OutputIt transform_function(UnaryOperation&& unary_op, OutputIt d_first, bool throw_error = true, const char* str = __builtin_FUNCTION()) const;
		template<DType dt, class OutputIt, class UnaryOperation>
		void sub_transform_function(UnaryOperation&& unary_op, OutputIt& d_first) const;

		template<typename WrappedTypes, class OutputIt, class UnaryOperation, std::enable_if_t<
			DTypeFuncs::is_wrapped_dtype<WrappedTypes>::value
			&& !DTypeFuncs::is_wrapped_dtype<UnaryOperation>::value
			&& !DTypeFuncs::is_wrapped_dtype<OutputIt>::value, bool> = true>
		OutputIt transform_function(UnaryOperation&& unary_op, OutputIt d_first, bool throw_error = true, const char* str = __builtin_FUNCTION()) const;


		template<class OutputIt, class UnaryOperation>
		OutputIt transform_function(UnaryOperation&& unary_op, OutputIt d_first) const;
		template<class OutputIt, class UnaryOperation>
		OutputIt transform_function_nbool(UnaryOperation&& unary_op, OutputIt d_first) const;


		template<DType... dts, class InputIt2, class OutputIt, class UnaryOperation, 
			std::enable_if_t<!std::is_same_v<OutputIt, bool> 
				&& !std::is_same_v<OutputIt, const char*>
				&& !std::is_same_v<InputIt2, bool> 
				&& !std::is_same_v<InputIt2, const char*>, bool> = true>
		OutputIt transform_function(UnaryOperation&& unary_op, InputIt2 inp2, OutputIt d_first, bool throw_error=true, const char* str=__builtin_FUNCTION()) const;
		template<DType dt, class InputIt2, class OutputIt, class UnaryOperation>
		void sub_transform_function(UnaryOperation&& unary_op, InputIt2& inp2, OutputIt& d_first) const;
		template<class InputIt2, class OutputIt, class UnaryOperation>
		OutputIt transform_function(UnaryOperation&& unary_op, InputIt2 inp2, OutputIt d_first) const;
		template<class InputIt2, class OutputIt, class UnaryOperation>
		OutputIt transform_function_nbool(UnaryOperation&& unary_op, InputIt2 inp2, OutputIt d_first) const;

		template<typename WrappedTypes, class InputIt2, class OutputIt, class UnaryOperation, std::enable_if_t<
			DTypeFuncs::is_wrapped_dtype<WrappedTypes>::value
				&& !std::is_same_v<OutputIt, bool> 
				&& !std::is_same_v<OutputIt, const char*>
				&& !std::is_same_v<InputIt2, bool> 
				&& !std::is_same_v<InputIt2, const char*>
			, bool> = true>	
		OutputIt transform_function(UnaryOperation&& unary_op, InputIt2 inp2, OutputIt d_first, bool throw_error=true, const char* str=__builtin_FUNCTION()) const;

		template<DType... dts, class OutputIt, class UnaryOperation, std::enable_if_t<!std::is_same_v<OutputIt, bool>, bool> = true>
		OutputIt transform_function(UnaryOperation&& unary_op, const ArrayVoid& inp_arr, OutputIt d_first, bool throw_error = true, const char* str = __builtin_FUNCTION()) const;
		template<DType dt, class OutputIt, class UnaryOperation>
		void sub_transform_function(UnaryOperation&& unary_op, const ArrayVoid& inp_arr, OutputIt& d_first) const;
		template<class OutputIt, class UnaryOperation>
		OutputIt transform_function(UnaryOperation&& unary_op, const ArrayVoid& inp_arr, OutputIt d_first) const;
		template<class OutputIt, class UnaryOperation>
		OutputIt transform_function_nbool(UnaryOperation&& unary_op, const ArrayVoid& inp_arr, OutputIt d_first) const;

		template<typename WrappedTypes, class OutputIt, class UnaryOperation, std::enable_if_t<
			DTypeFuncs::is_wrapped_dtype<WrappedTypes>::value && !std::is_same_v<OutputIt, bool>, bool> = true>
		OutputIt transform_function(UnaryOperation&& unary_op, const ArrayVoid& inp_arr, OutputIt d_first, bool throw_error = true, const char* str = __builtin_FUNCTION()) const;


		template<DType dt, DType... dts, class UnaryOperation>
		void transform_function(UnaryOperation&& unary_op, const ArrayVoid& inp_arr, bool throw_error=true, const char* str = __builtin_FUNCTION()); //done
		template<DType dt, class UnaryOperation>
		void sub_transform_function(UnaryOperation&& unary_op, const ArrayVoid& inp_arr);
		template<class UnaryOperation>
		void transform_function(UnaryOperation&& unary_op, const ArrayVoid& inp_arr); //done
		template<class UnaryOperation>
		void transform_function_nbool(UnaryOperation&& unary_op, const ArrayVoid& inp_arr);
	
		template<typename WrappedTypes, class UnaryOperation, std::enable_if_t< DTypeFuncs::is_wrapped_dtype<WrappedTypes>::value, bool> = true>
		void transform_function(UnaryOperation&& unary_op, const ArrayVoid& inp_arr, bool throw_error=true, const char* str = __builtin_FUNCTION());



		template<DType...dts, class UnaryOperation>
		void for_ceach(UnaryOperation&& unary_op, bool throw_error=true, const char* str=__builtin_FUNCTION()) const; //done
		template<DType...dts, class UnaryOperation>
		void for_each(UnaryOperation&& unary_op, bool throw_error=true, const char* str=__builtin_FUNCTION()); //done 
		template<class UnaryOperation>
		void for_ceach(UnaryOperation&& unary_op) const; //done
		template<class UnaryOperation>
		void for_each(UnaryOperation&& unary_op); //done 
		template<class UnaryOperation>
		void for_ceach_nbool(UnaryOperation&& unary_op) const;
		template<class UnaryOperation>
		void for_each_nbool(UnaryOperation&& unary_op);
		template<DType dt, class UnaryOperation>
		void sub_for_each(UnaryOperation&& unary_op);
		template<DType dt, class UnaryOperation>
		void sub_for_ceach(UnaryOperation&& unary_op) const;

		template<typename WrappedTypes, class UnaryOperation, std::enable_if_t< DTypeFuncs::is_wrapped_dtype<WrappedTypes>::value, bool> = true>
		void for_ceach(UnaryOperation&& unary_op, bool throw_error=true, const char* str=__builtin_FUNCTION()) const;
		template<typename WrappedTypes, class UnaryOperation, std::enable_if_t< DTypeFuncs::is_wrapped_dtype<WrappedTypes>::value, bool> = true>
		void for_each(UnaryOperation&& unary_op, bool throw_error=true, const char* str=__builtin_FUNCTION());


};
}


// Specialization of std::swap for nt::ArrayVoid
namespace std {
    inline void swap(::nt::ArrayVoid& lhs, ::nt::ArrayVoid& rhs) {
        lhs.swap(rhs); // Call your custom swap function
    }
}

#endif
