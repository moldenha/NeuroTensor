#ifndef NT_BUCKET_CPU_H__
#define NT_BUCKET_CPU_H__

#include "../intrusive_ptr/intrusive_ptr.hpp"
#include "../intrusive_ptr/intrusive_tracked_list.hpp"
#include "device.h"
#include <memory>
/* #include "../dtype/DType.h" */
#include "../utils/utils.h"
#include <vector>
#include "../dtype/DType_enum.h"
#include <functional>
#include "iterator.h"
#include <type_traits>
#include "../utils/api_macro.h"
#include "bucket.h"
/* #include "../dtype/ArrayVoid.h" */
/*
 

        virtual inline intrusive_ptr<Bucket> copy_strides() const = 0;

 */


namespace nt{

class BucketCPU;


//have it hold an intrusive_ptr<intrusive_variable<bool>>
//this will allow it to hold true if memory is made unmodifiable dynamically
//and then just implement some errors for if the memory has been marked const or not by
//any functions that are not marked const
//which is really just the iterator function, the nullify function, swap, and the cat function
//of those, nullify is fine, because the underlying tensor memory will not be modified
//swap is also fine because the underlying tensor memory will not be modified
//for cat, it is a little bit trickier, 
//          it will either throw an error if it should not be modified,
//          or it will make the output memory non-modifiable <- this seems like the best option (maybe print warning)
//          or it will clone that specific memory

class NEUROTENSOR_API BucketCPU : public Bucket {
    template <class TTarget,
              class DeleteOp,
              class NullType,
              class... Args>
    friend intrusive_ptr<TTarget, DeleteOp, NullType>
    make_intrusive(Args&&... args);
	intrusive_ptr<DeviceHolder> buckets_; //buckets of contiguous memory
	intrusive_tracked_list<void*> strides_; //void* to store beggining and end
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

	BucketCPU(intrusive_ptr<DeviceHolder> buckets, intrusive_tracked_list<void*> strides, int64_t strideS, int64_t bS, bool blocked, DType dt);
	int64_t blocked_stride_size() const;
	BucketCPU blocked_strides_clone_cpu() const noexcept;
	BucketCPU strided_clone_cpu() const noexcept;
	//these are special cases because they also cone the tensors themselves
	BucketCPU blocked_strides_clone_tensor() const noexcept;
	BucketCPU strided_clone_tensor() const noexcept;

	uint64_t getBucketCPUSize(const uint64_t bucket_index) const;

	template<typename Buck>
	static void processCatData(const Buck& b, std::vector<std::reference_wrapper<const intrusive_ptr<Device> >>& nData, intrusive_tracked_list<void*> nStrides, uint64_t& stride_index);
	template<typename First>
	static void processCatDataHelper(std::vector<std::reference_wrapper<const intrusive_ptr<Device> >>& nData, intrusive_tracked_list<void*>& nStrides,  uint64_t& stride_index, const First& first);
	template<typename First, typename... Rest>
	static void processCatDataHelper(std::vector<std::reference_wrapper<const intrusive_ptr<Device> >>& nData, intrusive_tracked_list<void*>& nStrides, uint64_t& stride_index, const First& first, const Rest&... rest);

	static BucketCPU catV(const std::vector<BucketCPU>& buckets);
	static BucketCPU catV(const std::vector<std::reference_wrapper<const BucketCPU> >& buckets);
	
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
	static void convertBucketCPUsHelper(std::vector<BucketCPU>& buckets, uint32_t& index){}
	template<typename First, typename... Rest>
	static void convertBucketCPUsHelper(std::vector<BucketCPU>& buckets, uint32_t& index, const First& bf, const Rest&... rest);
	template<typename... BucketCPUs>
	static std::vector<BucketCPU> convertBucketCPUs(const BucketCPUs&... buckets);
	inline static DType processCatDType(const BucketCPU& b) {return b.dtype_;}
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

    template<typename T>
    T range_contiguous_(std::vector<std::pair<int64_t, int64_t>> ranges) const;
    template<typename T>
    T range_strided_(std::vector<std::pair<int64_t, int64_t>> ranges) const;
    template<typename T>
    T range_bucketed_(std::vector<std::pair<int64_t, int64_t>> ranges) const;
	static BucketCPU makeCopyBucketCPU(DType dt, const intrusive_ptr<DeviceHolder>& bucks, bool blocked, int64_t bS, int64_t stride_size=0);
	public:
		BucketCPU(const int64_t size, DType dt, MemoryLayout memory_t = MemoryLayout::Private);
		BucketCPU(const int64_t size, DType dt, void* ptr, DeleterFnPtr func);
		BucketCPU();
		BucketCPU(const BucketCPU& b);
		BucketCPU(BucketCPU&& b);
		BucketCPU(std::nullptr_t);
		BucketCPU& operator=(const BucketCPU& b);
		BucketCPU& operator=(BucketCPU&& b);

		const intrusive_tracked_list<void*>& intrusive_strides() const& noexcept {return strides_;}
		const intrusive_ptr<DeviceHolder>& intrusive_device() const& noexcept {return buckets_;} 

		static BucketCPU makeNullBucketCPU(DType dt = DType::Float32, int64_t stride_size=0); //dangerous to use if not immediately initialized right after
		inline const int64_t& buckets_amt() const noexcept {return bs;}
		inline const int64_t& stride_amt() const noexcept {return stride_size;}
		inline const DeviceType device_type() const noexcept override {return DeviceType::CPU;}
		inline const MemoryLayout memory_layout() const noexcept {return buckets_[0]->get_memory_layout();}
        inline bool is_private() const noexcept override {return this->memory_layout() == MemoryLayout::Private;}
        inline bool is_shared() const noexcept override {return this->memory_lauout() == MemoryLayout::Shared;}
        inline bool is_cpu() const noexcept override {return true;}
		inline void nullify() override {
			buckets_.nullify();
			strides_.nullify();
			const_cast<int64_t&>(stride_size) = 0;
			const_cast<int64_t&>(bs) = 0;
			strides_blocked = true;

		}
		inline bool occupy_same_memory(const BucketCPU& b) const noexcept {
			return b.strides_ == strides_ && buckets_ == b.buckets_;
		}
		/* ~BucketCPU(); */
		
		inline uint32_t iterator_type() const {
			/* std::cout << "stride_size: "<<stride_size<<std::endl; */
			if(!strides_blocked){return 3;}
			if(strides_blocked && is_contiguous()){return 1;}
			if(strides_blocked) {return 2;}
			return 3;
		}
		template<typename T>
		inline T* begin_contiguous(){
			utils::throw_exception(iterator_type() == 1, "Expected data to be contiguous to use contiguous iterator");
			return reinterpret_cast<T*>(data_ptr());
		}
		template<typename T>
		inline T* end_contiguous(){
			utils::throw_exception(iterator_type() == 1, "Expected data to be contiguous to use contiguous iterator");
			return reinterpret_cast<T*>(data_ptr_end());
		}
		template<typename T>
		inline BucketCPUIterator_blocked<T> begin_blocked(){
			utils::throw_exception(iterator_type() == 2, "Expected data to be blocked to use blocked iterator");
			return BucketCPUIterator_blocked<T>(reinterpret_cast<T**>(stride_begin()), 
					                 reinterpret_cast<T*>(data_ptr()), 
							 stride_size/2-1, 0); 
			// bs is just stride_size / 2 (should be) 
		}
		template<typename T>
		inline BucketCPUIterator_blocked<T> end_blocked(){
			utils::throw_exception(iterator_type() == 2, "Expected data to be blocked to use blocked iterator");
			if(stride_size == 0){
				return BucketCPUIterator_blocked<T>(reinterpret_cast<T**>(stride_end()),
					reinterpret_cast<T*>(data_ptr_end()), 
					stride_size/2, stride_size/2);
			}
			return BucketCPUIterator_blocked<T>(reinterpret_cast<T**>(stride_begin()) + stride_size-1, 
						 reinterpret_cast<T*>(data_ptr_end()), 
						stride_size/2-1, stride_size/2-1); 
			// bs is just stride_size / 2 (should be) 
		}
		template<typename T>
		inline BucketCPUIterator_list<T> begin_list(){
			utils::throw_exception(iterator_type() == 3, "Expected data to be entirely bucketed to use list iterator");
			return BucketCPUIterator_list<T>(reinterpret_cast<T**>(stride_begin()));
		}
		template<typename T>
		inline BucketCPUIterator_list<T> end_list(){
			utils::throw_exception(iterator_type() == 3, "Expected data to be entirely bucketed to use list iterator");
			return BucketCPUIterator_list<T>(reinterpret_cast<T**>(stride_end()));
		}

		template<typename T>
		inline const T* cbegin_contiguous() const{
			utils::throw_exception(iterator_type() == 1, "Expected data to be contiguous to use contiguous iterator");
			return reinterpret_cast<const T*>(data_ptr());
		}
		template<typename T>
		inline const T* cend_contiguous() const{
			utils::throw_exception(iterator_type() == 1, "Expected data to be contiguous to use contiguous iterator");
			return reinterpret_cast<const T*>(data_ptr_end());
		}
		template<typename T>
		inline BucketCPUIterator_blocked<const T> cbegin_blocked() const{
			utils::throw_exception(iterator_type() == 2, "Expected data to be blocked to use blocked iterator");
			return BucketCPUIterator_blocked<const T>(reinterpret_cast<T**>(stride_begin()), 
					                 reinterpret_cast<const T*>(data_ptr()), 
							stride_size/2-1, 0); // bs is just stride_size / 2  
		}
		template<typename T>
		inline BucketCPUIterator_blocked<const T> cend_blocked() const{
			utils::throw_exception(iterator_type() == 2, "Expected data to be blocked to use blocked iterator");
			if(stride_size == 0){return BucketCPUIterator_blocked<const T>(
					reinterpret_cast<T**>(stride_end()),
					 reinterpret_cast<const T*>(data_ptr_end()), 
					stride_size/2, stride_size/2);}
			return BucketCPUIterator_blocked<const T>(reinterpret_cast<T**>(stride_begin()) + stride_size-1, 
					 reinterpret_cast<const T*>(data_ptr_end()), 
					stride_size/2-1, stride_size/2-1); // bs is just stride_size / 2  
		}
		template<typename T>
		inline BucketCPUIterator_list<const T> cbegin_list() const {
			utils::throw_exception(iterator_type() == 3, "Expected data to be entirely bucketed to use list iterator");
			return BucketCPUIterator_list<const T>(reinterpret_cast<T**>(stride_begin()));
		}
		template<typename T>
		inline BucketCPUIterator_list<const T> cend_list() const {
			utils::throw_exception(iterator_type() == 3, "Expected data to be entirely bucketed to use list iterator");
			return BucketCPUIterator_list<const T>(reinterpret_cast<T**>(stride_end()));
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
		BucketCPUIterator_blocked<T> begin(){
			return BucketCPUIterator_blocked<T>(reinterpret_cast<T**>(stride_begin()), 
					 reinterpret_cast<T*>(data_ptr()), 
					stride_size/2-1, 0);;
		}
		template<size_t i, typename T, typename = std::enable_if_t<i == 2> >
		BucketCPUIterator_blocked<const T> cbegin() const{
			return BucketCPUIterator_blocked<const T>(reinterpret_cast<T**>(stride_begin()), 
					 reinterpret_cast<const T*>(data_ptr()), 
					stride_size/2-1, 0);
		}
		//stride_size / 2 is the amount of contiguous buckets there are
		template<size_t i, typename T, typename = std::enable_if_t<i == 2> >
		BucketCPUIterator_blocked<T> end(){
			if(stride_size == 0){return BucketCPUIterator_blocked<T>(reinterpret_cast<T**>(stride_end()),
					 reinterpret_cast<T*>(data_ptr_end()), 
					stride_size/2, stride_size/2);}
			return BucketCPUIterator_blocked<T>(reinterpret_cast<T**>(stride_begin()) + stride_size-1, 
					 reinterpret_cast<T*>(data_ptr_end()), 
					stride_size/2-1, stride_size/2-1);
		}
		template<size_t i, typename T, typename = std::enable_if_t<i == 2> >
		BucketCPUIterator_blocked<const T> cend() const{
			if(stride_size == 0){return BucketCPUIterator_blocked<const T>(reinterpret_cast<T**>(stride_end()),
					 reinterpret_cast<const T*>(data_ptr_end()), 
					stride_size/2-1, stride_size/2-1);}
			return BucketCPUIterator_blocked<const T>(reinterpret_cast<T**>(stride_begin()) + stride_size-1, 
					reinterpret_cast<const T*>(data_ptr_end()), 
					stride_size/2-1, stride_size/2-1);
		}
		template<size_t i, typename T, typename = std::enable_if_t<i == 3> >
		BucketCPUIterator_list<T> begin(){
			return BucketCPUIterator_list<T>(reinterpret_cast<T**>(stride_begin()));;
		}
		template<size_t i, typename T, typename = std::enable_if_t<i == 3> >
		BucketCPUIterator_list<const T> cbegin() const{
			return BucketCPUIterator_list<const T>(reinterpret_cast<T**>(stride_begin()));
		}
		template<size_t i, typename T, typename = std::enable_if_t<i == 3> >
		BucketCPUIterator_list<T> end(){
			return BucketCPUIterator_list<T>(reinterpret_cast<T**>(stride_end()));
		}
		template<size_t i, typename T, typename = std::enable_if_t<i == 3> >
		BucketCPUIterator_list<const T> cend() const{
			return BucketCPUIterator_list<const T>(reinterpret_cast<T**>(stride_end()));
		}



		int64_t size() const noexcept;
        inline int64_t numel() const noexcept override {return this->size();}
        int64_t storage_size() const noexcept override;
		BucketCPU contiguous_cpu() const;
        inline intrusive_ptr<Bucket> contiguous() const override {
            return make_intrusive<BucketCPU>(this->contiguous_cpu());
        }
		BucketCPU clone_cpu() const noexcept;
        inline intrusive_ptr<Bucket> clone() const noexcept override {
            return make_intrusive<BucketCPU>(this->clone_cpu());
        }
		bool is_contiguous() const override;
        inline bool is_strided() const noexcept override{return strides_blocked == false;}
        inline bool is_affine() const noexcept override{
            // this is just blocked
            return strides_blocked && !this->is_contiguous();
        }
		inline int64_t use_count() const override {return buckets_[0].use_count();}
		BucketCPU new_bounds_cpu(uint64_t start, uint64_t end) const;
        inline intrusive_ptr<Bucket> new_bounds(int64_t start, int64_t end) const override{
            int64_t num = this->size();
            start = start < 0 ? start + num : start;
            end = end < 0 ? end + num : end;
            return make_intrusive<BucketCPU>(this->new_bounds_cpu(static_cast<uint64_t>(start), static_cast<uint64_t>(end)));
        }
		bool can_force_contiguity() const override;
		bool can_force_contiguity_bytes(const int64_t& bytes) const override;
		int64_t force_contig_size() const;
		intrusive_ptr<Bucket> bound_force_contiguity_bucket() const override;
		intrusive_ptr<Bucket> force_contiguity_and_bucket() const override; //this forces contiguity and buckets all indices
		intrusive_ptr<Bucket> force_contiguity(int64_t) const override; //this function disregards the strided view, basically lets say that you performed a transpose(-1,-2)
							//this resulted in a strided view, meaning is_strided() == true
							//this disregards that, and just makes this into a single bucket based on the start of data_ptr()
							//and the size given for this functions argument
		inline bool is_null() const noexcept override {return bs == 0 || stride_size == 0;}
		inline void* data_ptr() noexcept override {if(is_null()){return nullptr;}return strides_[0];}
			//can be dangerous if not contiguous
		inline void* data_ptr_end() noexcept override {
			if(is_null()){return nullptr;}
			return strides_[stride_size-1];
		} 
		inline const void* data_ptr() const noexcept override {if(is_null()){return nullptr;}return strides_[0];}
		inline const void* data_ptr_end() const noexcept override {
			if(is_null()){return nullptr;}
			return strides_[stride_size-1];
		}
		inline void** stride_begin() const {if(is_null()){return nullptr;}return strides_.get();}
		inline void** stride_end() const {if(is_null()){return nullptr;}return strides_.get() + stride_size;}

		BucketCPU to_shared_cpu() const;
        inline intrusive_ptr<Bucket> to_shared() const override {return make_intrusive<BucketCPU>(this->to_shared_cpu());}
		BucketCPU to_private_cpu() const;
        inline intrusive_ptr<Bucket> to_private() const override {return make_intrusive<BucketCPU>(this->to_private_cpu());}


		//BucketCPU to_device(DeviceType) const;

        // to devices:
        inline intrusive_ptr<Bucket> to_cpu(MemoryLayout mem_t = MemoryLayout::Private) const override { 
            return this->to_memory_layout(mem_t); 
        }
        intrusive_ptr<Bucket> to_mtl(MemoryLayout mem_t = MemoryLayout::Private) const override;

        // the following returns if you have a contiguous block of memory that fits into one of the buckets
        // and is a subset of that bucket of memory
        // [if it is blocked or strided it automatically returns true]
        const bool is_sub_memory() const;
/* #ifdef USE_PARALLEL */
/* 		static BucketCPU FromShared(intrusive_ptr<void[]> ptr, uint64_t s, DType d); */
/* #endif */
		/* void print(){ */
		/* 	std::cout << '{'; */
		/* 	auto mbegin = begin(); */
		/* 	auto mend = end(); */
		/* 	for(;mbegin != mend; ++mbegin) */
		/* 		std::cout << *mbegin << ','; */
		/* 	std::cout << '}' << std::endl; */
		/* } */
		
		template<typename... BucketCPUs>
		static BucketCPU cat(const BucketCPUs&... buckets);
		
		/* inline float& operator[](uint64_t i){ */
		/* 	return reinterpret_cast<float*>(data[bucket_index(i)].get())[i]; */
		/* } */
		
		inline BucketCPU operator+(int64_t i) const{
			uint64_t msize = size();
			uint64_t adding = (i < 0) ? msize + i : i;
			return new_bounds_cpu(adding, msize);
		}
		template<typename T>
		T split(uint64_t splitting) const;
        template<typename T>
        T range(std::vector<std::pair<int64_t, int64_t>> ranges) const;
		void swap(BucketCPU&);
		BucketCPU bucket_all_indices_cpu() const;
        inline intrusive_ptr<Bucket> bucket_all_indices() const override{
            return intrusive_ptr<BucketCPU>(this->bucket_all_indices_cpu());
        }

		//this makes a new bucket with a new stride size, coppies buckets_
		//strides are not initialized, and is inteaded to be filled in by the user
		//it is now going to be assumed that it is no longer blocked
		inline intrusive_ptr<Bucket> new_stride_size(int64_t n_stride_size, bool is_blocked=false) const {
			return make_intrusive<BucketCPU>(buckets_, intrusive_tracked_list<void*>(n_stride_size), n_stride_size, bs, is_blocked, this->dtype_);
		}
		intrusive_ptr<Bucket> copy_strides(bool copy_vals = true) const;
        inline std::vector<uint64_t> storage_id() const noexcept { return this->buckets_->storage_id(); }
};

}


// Specialization of std::swap for nt::BucketCPU
namespace std {
    inline void swap(::nt::BucketCPU& lhs, ::nt::BucketCPU& rhs) {
        lhs.swap(rhs); // Call your custom swap function
    }
}

#include "bucket_cpu_cat.h"

#endif // NT_BUCKET_CPU_H__
