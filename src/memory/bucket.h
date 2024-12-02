#ifndef _NT_BUCKET_H_
#define _NT_BUCKET_H_

#include "../intrusive_ptr/intrusive_ptr.hpp"
#include "device.h"
#include <memory>
/* #include "../dtype/DType.h" */
#include "../utils/utils.h"
#include <vector>
#include "../dtype/DType_enum.h"
#include <functional>
#include "iterator.h"
#include <type_traits>
/* #include "../dtype/ArrayVoid.h" */

namespace nt{

class Bucket;

//46-4-5 (4) unredo
//"redo": ["64-1-3 (9)", 
//"68-1-1 (2)", 
//"68-1-1 (7)", 
//"77-1-9 (6)", 
//"77-1-9 (7)", 
//"56-4-1 (1)", 
//"54-4-8 (2)", 
//"54-4-8 (3)", 
//"56-2-2 (7)", 
//"54-5-3 (7)", 
//"54-5-4 (7)"]
class Bucket{
	nt::intrusive_ptr<DeviceHolder> buckets_; //buckets of contiguous memory
	nt::intrusive_ptr<void*[]> strides_; //void* to store beggining and end
					   //or to store in terms of just pointing to specific pointers
					   // (more memory efficient for more random indexing)
	const int64_t stride_size; // holds the size of the strides_
	const int64_t bs; // holds the amount of buckets_
	//const int64_t total_size; <- maybe add this, I think it was a bad idea to make the Size() function take so damn long potentially
	//if I just make an inline function that returns this I could do that with minimal changes
	//I wouldn't even have to change the constructors if I kept the way I aquire the size the same, I may still change that though
	//
	bool strides_blocked; 
		// holds if the strides are blocked off into buckets, 
		// or if it is just void** holding onto one point at a time, 
		// the latter is more efficient with operations like transpose(-1,-2) or [Tensor == (num)]
		// the first is more efficient with operations like transpose of higher order, or just in general
		// less memory overhead for the first if it can be allowed
		// default this is true unless the buckets are split up
		// I may have the constructor directly below have a total_size and when it is (-1) use the function that calculates the entire size up

	Bucket(intrusive_ptr<DeviceHolder> buckets, intrusive_ptr<void*[]> strides, int64_t strideS, int64_t bS, bool blocked, DType dt);
	int64_t blocked_stride_size() const;
	Bucket blocked_strides_clone() const;
	Bucket strided_clone() const;
	//these are special cases because they also cone the tensors themselves
	Bucket blocked_strides_clone_tensor() const;
	Bucket strided_clone_tensor() const;

	uint64_t getBucketSize(const uint64_t bucket_index) const;

	template<typename Buck>
	static void processCatData(const Buck& b, std::vector<std::reference_wrapper<const intrusive_ptr<Device> >>& nData, nt::intrusive_ptr<void*[]> nStrides, uint64_t& stride_index);
	template<typename First>
	static void processCatDataHelper(std::vector<std::reference_wrapper<const intrusive_ptr<Device> >>& nData, nt::intrusive_ptr<void*[]>& nStrides,  uint64_t& stride_index, const First& first);
	template<typename First, typename... Rest>
	static void processCatDataHelper(std::vector<std::reference_wrapper<const intrusive_ptr<Device> >>& nData, nt::intrusive_ptr<void*[]>& nStrides, uint64_t& stride_index, const First& first, const Rest&... rest);

	static Bucket catV(const std::vector<Bucket>& buckets);
	static Bucket catV(const std::vector<std::reference_wrapper<const Bucket> >& buckets);
	
	template<typename Buck>
	static void processCatStrideSize(int64_t& st, const Buck& b);
	static void processCatStrideSizeHelper(int64_t& st);
	template<typename First, typename... Rest>
	static void processCatStrideSizeHelper(int64_t& st, const First& first, const Rest&... rest);

	//this already assumes and should not be used till verified they are all the same
	template<typename First, typename... Rest>
	static bool processCatBlockType(const First& first, const Rest&... rest);
	template<typename First>
	static bool dont_convert_strides(const First& bf);
	template<typename First, typename Second, typename... Rest>
	static bool dont_convert_strides(const First& bf, const Second& bs, const Rest&... rest);
	static void convertBucketsHelper(std::vector<Bucket>& buckets, uint32_t& index){}
	template<typename First, typename... Rest>
	static void convertBucketsHelper(std::vector<Bucket>& buckets, uint32_t& index, const First& bf, const Rest&... rest);
	template<typename... Buckets>
	static std::vector<Bucket> convertBuckets(const Buckets&... buckets);
	inline static DType processCatDType(const Bucket& b) {return b.dtype;}
	template<typename First, typename... Rest>
	static DType processCatDType(const First& first, const Rest&... rest);
	
	uint64_t bucket_index(uint64_t& index) const;
	/* void arrange_contiguous(); */
	template<typename T>
	T split_strided_(uint64_t splitting) const;
	template<typename T>
	T split_contiguous_(uint64_t splitting) const;
	template<typename T>
	T split_bucketed_(uint64_t splitting) const;
	static Bucket makeCopyBucket(DType dt, const intrusive_ptr<DeviceHolder>& bucks, bool blocked, int64_t bS, int64_t stride_size=0);
	public:
		DType dtype;
		Bucket(const int64_t size, DType dt, DeviceType device_type = dCPU);
		Bucket(const int64_t size, DType dt, void* ptr, DeleterFnPtr func);
		Bucket();
		Bucket(const Bucket& b);
		Bucket(Bucket&& b);
		Bucket& operator=(const Bucket& b);
		Bucket& operator=(Bucket&& b);

		static Bucket makeNullBucket(DType dt = DType::Float32, int64_t stride_size=0); //dangerous to use if not immediately initialized right after
		inline const int64_t& buckets_amt() const {return bs;}
		inline const int64_t& stride_amt() const {return stride_size;}
		inline const DeviceType& device_type() const noexcept {return buckets_[0]->get_device_type();}
		inline void nullify(){
			buckets_.nullify();
			strides_.nullify();
			const_cast<int64_t&>(stride_size) = 0;
			const_cast<int64_t&>(bs) = 0;
			strides_blocked = true;

		}
		/* ~Bucket(); */
		
		inline uint32_t iterator_type() const {
			/* std::cout << "stride_size: "<<stride_size<<std::endl; */
			if(!strides_blocked){return 3;}
			if(strides_blocked && is_contiguous()){return 1;}
			if(strides_blocked) {return 2;}
			return 3;
		}
		template<typename T>
		inline T* begin_contiguous(){
			nt::utils::throw_exception(iterator_type() == 1, "Expected data to be contiguous to use contiguous iterator");
			return reinterpret_cast<T*>(data_ptr());
		}
		template<typename T>
		inline T* end_contiguous(){
			nt::utils::throw_exception(iterator_type() == 1, "Expected data to be contiguous to use contiguous iterator");
			return reinterpret_cast<T*>(data_ptr_end());
		}
		template<typename T>
		inline BucketIterator_blocked<T> begin_blocked(){
			nt::utils::throw_exception(iterator_type() == 2, "Expected data to be blocked to use blocked iterator");
			return BucketIterator_blocked<T>(reinterpret_cast<T**>(stride_begin()), stride_size/2, 0); // bs is just stride_size / 2 (should be) 
		}
		template<typename T>
		inline BucketIterator_blocked<T> end_blocked(){
			nt::utils::throw_exception(iterator_type() == 2, "Expected data to be blocked to use blocked iterator");
			if(stride_size == 0){return BucketIterator_blocked<T>(reinterpret_cast<T**>(stride_end()), stride_size/2, stride_size/2);}
			return BucketIterator_blocked<T>(reinterpret_cast<T**>(stride_begin()) + stride_size-1, stride_size/2, stride_size/2); // bs is just stride_size / 2 (should be) 
		}
		template<typename T>
		inline BucketIterator_list<T> begin_list(){
			nt::utils::throw_exception(iterator_type() == 3, "Expected data to be entirely bucketed to use list iterator");
			return BucketIterator_list<T>(reinterpret_cast<T**>(stride_begin()));
		}
		template<typename T>
		inline BucketIterator_list<T> end_list(){
			nt::utils::throw_exception(iterator_type() == 3, "Expected data to be entirely bucketed to use list iterator");
			return BucketIterator_list<T>(reinterpret_cast<T**>(stride_end()));
		}

		template<typename T>
		inline const T* cbegin_contiguous() const{
			nt::utils::throw_exception(iterator_type() == 1, "Expected data to be contiguous to use contiguous iterator");
			return reinterpret_cast<const T*>(data_ptr());
		}
		template<typename T>
		inline const T* cend_contiguous() const{
			nt::utils::throw_exception(iterator_type() == 1, "Expected data to be contiguous to use contiguous iterator");
			return reinterpret_cast<const T*>(data_ptr_end());
		}
		template<typename T>
		inline BucketIterator_blocked<const T> cbegin_blocked() const{
			nt::utils::throw_exception(iterator_type() == 2, "Expected data to be blocked to use blocked iterator");
			return BucketIterator_blocked<const T>(reinterpret_cast<T**>(stride_begin()), stride_size/2, 0); // bs is just stride_size / 2  
		}
		template<typename T>
		inline BucketIterator_blocked<const T> cend_blocked() const{
			nt::utils::throw_exception(iterator_type() == 2, "Expected data to be blocked to use blocked iterator");
			if(stride_size == 0){return BucketIterator_blocked<const T>(reinterpret_cast<T**>(stride_end()), stride_size/2, stride_size/2);}
			return BucketIterator_blocked<const T>(reinterpret_cast<T**>(stride_begin()) + stride_size-1, stride_size/2, stride_size/2); // bs is just stride_size / 2  
		}
		template<typename T>
		inline BucketIterator_list<const T> cbegin_list() const {
			nt::utils::throw_exception(iterator_type() == 3, "Expected data to be entirely bucketed to use list iterator");
			return BucketIterator_list<const T>(reinterpret_cast<T**>(stride_begin()));
		}
		template<typename T>
		inline BucketIterator_list<const T> cend_list() const {
			nt::utils::throw_exception(iterator_type() == 3, "Expected data to be entirely bucketed to use list iterator");
			return BucketIterator_list<const T>(reinterpret_cast<T**>(stride_end()));
		}
		template<size_t i, typename T, typename = std::enable_if_t<i == 1> >
		T* begin(){
			return reinterpret_cast<T*>(data_ptr());
		}
		template<size_t i, typename T, typename = std::enable_if_t<i == 1> >
		const T* cbegin() const{
			return reinterpret_cast<const T*>(data_ptr());
		}
		template<size_t i, typename T, typename = std::enable_if_t<i == 1> >
		T* end(){
			return reinterpret_cast<T*>(data_ptr_end());
		}
		template<size_t i, typename T, typename = std::enable_if_t<i == 1> >
		const T* cend() const{
			return reinterpret_cast<const T*>(data_ptr_end());
		}
		template<size_t i, typename T, typename = std::enable_if_t<i == 2> >
		BucketIterator_blocked<T> begin(){
			return BucketIterator_blocked<T>(reinterpret_cast<T**>(stride_begin()), stride_size/2, 0);;
		}
		template<size_t i, typename T, typename = std::enable_if_t<i == 2> >
		BucketIterator_blocked<const T> cbegin() const{
			return BucketIterator_blocked<const T>(reinterpret_cast<T**>(stride_begin()), stride_size/2, 0);
		}
		//stride_size / 2 is the amount of contiguous buckets there are
		template<size_t i, typename T, typename = std::enable_if_t<i == 2> >
		BucketIterator_blocked<T> end(){
			if(stride_size == 0){return BucketIterator_blocked<T>(reinterpret_cast<T**>(stride_end()), stride_size/2, stride_size/2);}
			return BucketIterator_blocked<T>(reinterpret_cast<T**>(stride_begin()) + stride_size-1, stride_size/2, stride_size/2);
		}
		template<size_t i, typename T, typename = std::enable_if_t<i == 2> >
		BucketIterator_blocked<const T> cend() const{
			if(stride_size == 0){return BucketIterator_blocked<const T>(reinterpret_cast<T**>(stride_end()), stride_size/2, stride_size/2);}
			return BucketIterator_blocked<const T>(reinterpret_cast<T**>(stride_begin()) + stride_size-1, stride_size/2, stride_size/2);
		}
		template<size_t i, typename T, typename = std::enable_if_t<i == 3> >
		BucketIterator_list<T> begin(){
			return BucketIterator_list<T>(reinterpret_cast<T**>(stride_begin()));;
		}
		template<size_t i, typename T, typename = std::enable_if_t<i == 3> >
		BucketIterator_list<const T> cbegin() const{
			return BucketIterator_list<const T>(reinterpret_cast<T**>(stride_begin()));
		}
		template<size_t i, typename T, typename = std::enable_if_t<i == 3> >
		BucketIterator_list<T> end(){
			return BucketIterator_list<T>(reinterpret_cast<T**>(stride_end()));
		}
		template<size_t i, typename T, typename = std::enable_if_t<i == 3> >
		BucketIterator_list<const T> cend() const{
			return BucketIterator_list<const T>(reinterpret_cast<T**>(stride_end()));
		}



		int64_t size() const;
		Bucket contiguous() const;
		Bucket clone() const;
		bool is_contiguous() const;
		inline int64_t use_count() const {return buckets_[0].use_count();}
		Bucket new_bounds(uint64_t start, uint64_t end) const;
		bool can_force_contiguity() const;
		bool can_force_contiguity_bytes(const int64_t& bytes) const;
		int64_t force_contig_size() const;
		Bucket bound_force_contiguity_bucket() const;
		Bucket force_contiguity_and_bucket() const; //this forces contiguity and buckets all indices
		Bucket force_contiguity(int64_t) const; //this function disregards the strided view, basically lets say that you performed a transpose(-1,-2)
							//this resulted in a strided view, meaning is_strided() == true
							//this disregards that, and just makes this into a single bucket based on the start of data_ptr()
							//and the size given for this functions argument
		inline const bool is_null() const {return bs == 0 || stride_size == 0;}
		inline void* data_ptr() noexcept {if(is_null()){return nullptr;}return strides_[0];}
			//can be dangerous if not contiguous
		inline void* data_ptr_end() noexcept {
			if(is_null()){return nullptr;}
			return strides_[stride_size-1];
		} 
		inline const void* data_ptr() const noexcept {if(is_null()){return nullptr;}return strides_[0];}
		inline const void* data_ptr_end() const noexcept {
			if(is_null()){return nullptr;}
			return strides_[stride_size-1];
		}
		inline void** stride_begin() const {if(is_null()){return nullptr;}return strides_.get();}
		inline void** stride_end() const {if(is_null()){return nullptr;}return strides_.get() + stride_size;}

		inline const bool is_shared() const {return device_type() == dCPUShared;}
		Bucket to_shared() const;
		Bucket to_cpu() const;
		Bucket to_device(DeviceType) const;
/* #ifdef USE_PARALLEL */
/* 		static Bucket FromShared(intrusive_ptr<void[]> ptr, uint64_t s, DType d); */
/* #endif */
		/* void print(){ */
		/* 	std::cout << '{'; */
		/* 	auto mbegin = begin(); */
		/* 	auto mend = end(); */
		/* 	for(;mbegin != mend; ++mbegin) */
		/* 		std::cout << *mbegin << ','; */
		/* 	std::cout << '}' << std::endl; */
		/* } */
		
		template<typename... Buckets>
		static Bucket cat(const Buckets&... buckets);
		
		/* inline float& operator[](uint64_t i){ */
		/* 	return reinterpret_cast<float*>(data[bucket_index(i)].get())[i]; */
		/* } */
		
		inline Bucket operator+(int64_t i) const{
			uint64_t msize = size();
			uint64_t adding = (i < 0) ? msize + i : i;
			return new_bounds(adding, msize);
		}
		template<typename T>
		T split(uint64_t splitting) const;
		void swap(Bucket&);
		Bucket bucket_all_indices() const;
		inline const bool is_strided() const {return strides_blocked == false;}
		//this makes a new bucket with a new stride size, coppies buckets_
		//strides are not initialized, and is inteaded to be filled in by the user
		//it is now going to be assumed that it is no longer blocked
		inline Bucket new_stride_size(int64_t n_stride_size, bool is_blocked=false) const {
			return Bucket(buckets_, nt::intrusive_ptr<void*[]>(n_stride_size), n_stride_size, bs, is_blocked, dtype);
		}
		Bucket copy_strides() const;
};

}


// Specialization of std::swap for nt::Bucket
namespace std {
    inline void swap(::nt::Bucket& lhs, ::nt::Bucket& rhs) {
        lhs.swap(rhs); // Call your custom swap function
    }
}

#include "bucket_cat.h"

#endif // _NT_BUCKET_H_
