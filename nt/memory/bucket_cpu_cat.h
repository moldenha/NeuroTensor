#ifndef NT_BUCKET_CPU_CAT_H__
#define NT_BUCKET_CPU_CAT_H__
#include "bucket_cpu.h"
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

namespace nt{
// Base template (handles the case when the first argument is not a std::vector)
template<typename First, typename... Rest>
struct IsFirstVectorBucketCPU {
    static constexpr bool value = false;
};

// Specialization for when the first argument is a std::vector
template<typename... Args>
struct IsFirstVectorBucketCPU<std::vector<BucketCPU>, Args...> {
    static constexpr bool value = true;
};

template<typename... Args>
struct IsFirstVectorBucketCPU<std::vector<std::reference_wrapper<const BucketCPU> >, Args...> {
    static constexpr bool value = true;
};


inline intrusive_ptr<DeviceHolder> to_device_holder(const std::vector<std::reference_wrapper<const intrusive_ptr<Device> >>& nData){
	intrusive_ptr<DeviceHolder> output = make_intrusive<DeviceHolder>(nData.size());
	for(uint64_t i = 0; i < nData.size(); ++i){
		const intrusive_ptr<Device>& d = nData[i].get();
		(*output)[i] = d;
	}
	return std::move(output);
}

template<typename Buck>
inline void BucketCPU::processCatData(const Buck& b, std::vector<std::reference_wrapper<const intrusive_ptr<Device> >>& nData, nt::intrusive_tracked_list<void*> nStrides, uint64_t& stride_index){
	static_assert(std::is_same_v<Buck, BucketCPU>, "Expected to only recieve type BucketCPU or ArrayVoid");
	if constexpr (std::is_same_v<Buck, BucketCPU>){
		const auto cur_size = nData.size();
		for(uint64_t i = 0; i < b.bs; ++i){
			bool add = true;
			for(uint64_t j = 0; j < cur_size; ++j){
				if(nData[j].get()->is_same(b.buckets_[i])){add = false; break;}
			}
			if(add){nData.push_back(std::cref((*b.buckets_)[i]));}
		}
		for(uint64_t i = 0; i < b.stride_size; ++i, ++stride_index){
			nStrides[stride_index] = b.strides_[i];
		}
	}
}

template<typename First>
inline void BucketCPU::processCatDataHelper(std::vector<std::reference_wrapper<const intrusive_ptr<Device> >>& nData, nt::intrusive_tracked_list<void*>& nStrides, uint64_t& stride_index, const First& first){
	processCatData<First>(first, nData, nStrides, stride_index);
}

template<typename First, typename... Rest>
inline void BucketCPU::processCatDataHelper(std::vector<std::reference_wrapper<const intrusive_ptr<Device> >>& nData, nt::intrusive_tracked_list<void*>& nStrides, uint64_t& stride_index, const First& first, const Rest&... rest){
	processCatData<First>(first, nData, nStrides, stride_index);
	processCatDataHelper(nData, nStrides, stride_index, rest...);
}

template<typename Buck>
inline void BucketCPU::processCatStrideSize(int64_t& st, const Buck& b){
	static_assert(std::is_same_v<Buck, BucketCPU>, "Expected to only recieve type BucketCPU or ArrayVoid");
	if constexpr (std::is_same_v<Buck, BucketCPU>){st += b.stride_size;}
}
inline void BucketCPU::processCatStrideSizeHelper(int64_t& st){}

template<typename First, typename... Rest>
inline void BucketCPU::processCatStrideSizeHelper(int64_t& st, const First& first, const Rest&... rest){
	static_assert(std::is_same_v<First, BucketCPU>, "Expected to only recieve type BucketCPU or ArrayVoid");
	if constexpr (std::is_same_v<First, BucketCPU>){st += first.stride_size;}
	processCatStrideSize(st, rest...);
}

//this already assumes and should not be used till verified they are all the same
template<typename First, typename... Rest>
inline bool BucketCPU::processCatBlockType(const First& first, const Rest&... rest){
	static_assert(std::is_same_v<First, BucketCPU>, "Expected to only recieve type BucketCPU");
	return first.strides_blocked;
}

template<typename First>
inline bool BucketCPU::dont_convert_strides(const First& bf){ 
	static_assert(std::is_same_v<First, BucketCPU>, "Expected to only recieve type BucketCPU");
	return true;
}


template<typename First, typename Second, typename... Rest>
inline bool BucketCPU::dont_convert_strides(const First& bf, const Second& bs, const Rest&... rest){
	static_assert(std::is_same_v<First, BucketCPU> && std::is_same_v<Second, BucketCPU>, "Expected to only recieve type BucketCPU or ArrayVoid");
	bool store_dont_convert_strides = false;
	if constexpr (std::is_same_v<First, BucketCPU> && std::is_same_v<Second, BucketCPU>){
		store_dont_convert_strides = (bf.strides_blocked == bs.strides_blocked);

	}
	return store_dont_convert_strides && dont_convert_strides(bs, rest...);
}


template<typename First, typename... Rest>
inline void BucketCPU::convertBucketCPUsHelper(std::vector<BucketCPU>& buckets, uint32_t& index, const First& bf, const Rest&... rest){
	static_assert(std::is_same_v<First, BucketCPU>, "Expected to only recieve type BucketCPU");
	if constexpr (std::is_same_v<First, BucketCPU>){
		buckets[index] = bf.bucket_all_indices_cpu();
		++index;
	}
	convertBucketCPUsHelper(buckets, index, rest...);
}

template<typename... BucketCPUs>
inline std::vector<BucketCPU> BucketCPU::convertBucketCPUs(const BucketCPUs&... buckets){
	std::vector<BucketCPU> output(sizeof...(buckets));
	uint32_t index = 0;
	convertBucketCPUsHelper(output, index, buckets...);
	return std::move(output);
}


template<typename First, typename... Rest>
inline DType BucketCPU::processCatDType(const First& first, const Rest&... rest){
	DType outp = first.dtype;
	utils::throw_exception(processCatDType(rest...) == outp, "All buckets need to have the same dtype");
	return outp;
}

inline void verifyAllDevTypes(const BucketCPU& a){;}
inline void verifyAllDevTypes(const BucketCPU& a, const BucketCPU& b){
	utils::throw_exception(a.device_type() == b.device_type(), "Expected to concatenate the same devices but got $ and $", a.device_type(), b.device_type());
}
template<typename... BucketCPUs>
inline void verifyAllDevTypes(const BucketCPU& a, const BucketCPU& b, const BucketCPU& c, const BucketCPUs&... buckets){
	utils::throw_exception(a.device_type() == b.device_type(), "Expected to concatenate the same devices but got $ and $", a.device_type(), b.device_type());
	utils::throw_exception(b.device_type() == c.device_type(), "Expected to concatenate the same devices but got $ and $", a.device_type(), b.device_type());
	verifyAllDevTypes(c, buckets...);
}


template<typename... BucketCPUs>
inline BucketCPU BucketCPU::cat(const BucketCPUs&... buckets){
	static_assert((IsFirstVectorBucketCPU<BucketCPUs...>::value && sizeof...(buckets) == 1) || !IsFirstVectorBucketCPU<BucketCPUs...>::value,
			"Only concatenates one vector of buckets at a time");
	if constexpr(IsFirstVectorBucketCPU<BucketCPUs...>::value && sizeof...(buckets) == 1){
		return BucketCPU::catV(buckets...);
	}
	/* else if constexpr(IsFirstVectorArrayVoid<BucketCPUs...>::value && sizeof...(buckets) == 1){ */
	/* 	return BucketCPU::catV(buckets...); */
	/* } */
	else{
		// std::cout << "going to cat "<<sizeof...(BucketCPUs) << "buckets"<<std::endl;
		int64_t n_stride_size = 0;
		processCatStrideSizeHelper(n_stride_size, buckets...);
		// std::cout << n_stride_size << std::endl;
		utils::throw_exception(n_stride_size > 2, "Expected bucket size to be greater than zero for a cat function");
		bool dontConvert = dont_convert_strides(buckets...);
		verifyAllDevTypes(buckets...);
		if(!dontConvert){
			return BucketCPU::catV(convertBucketCPUs(buckets...));
		}
		bool block_type = processCatBlockType(buckets...);
		nt::intrusive_tracked_list<void*> nStrides(n_stride_size);
		std::vector<std::reference_wrapper<const intrusive_ptr<Device> > > n_data;
		uint64_t stride_index = 0;
		processCatDataHelper(n_data, nStrides, stride_index, buckets...);
		return BucketCPU(to_device_holder(n_data), std::move(nStrides), n_stride_size, n_data.size(), block_type, processCatDType(buckets...));
	}

}


}



#endif // NT_BUCKET_CPU_CAT_H__


