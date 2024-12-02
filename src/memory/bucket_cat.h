#ifndef _NT_BUCKET_CAT_H_
#define _NT_BUCKET_CAT_H_
#include "bucket.h"
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
struct IsFirstVectorBucket {
    static constexpr bool value = false;
};

// Specialization for when the first argument is a std::vector
template<typename... Args>
struct IsFirstVectorBucket<std::vector<Bucket>, Args...> {
    static constexpr bool value = true;
};

template<typename... Args>
struct IsFirstVectorBucket<std::vector<std::reference_wrapper<const Bucket> >, Args...> {
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
inline void Bucket::processCatData(const Buck& b, std::vector<std::reference_wrapper<const intrusive_ptr<Device> >>& nData, nt::intrusive_ptr<void*[]> nStrides, uint64_t& stride_index){
	static_assert(std::is_same_v<Buck, Bucket>, "Expected to only recieve type Bucket or ArrayVoid");
	if constexpr (std::is_same_v<Buck, Bucket>){
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
inline void Bucket::processCatDataHelper(std::vector<std::reference_wrapper<const intrusive_ptr<Device> >>& nData, nt::intrusive_ptr<void*[]>& nStrides, uint64_t& stride_index, const First& first){
	processCatData<First>(first, nData, nStrides, stride_index);
}

template<typename First, typename... Rest>
inline void Bucket::processCatDataHelper(std::vector<std::reference_wrapper<const intrusive_ptr<Device> >>& nData, nt::intrusive_ptr<void*[]>& nStrides, uint64_t& stride_index, const First& first, const Rest&... rest){
	processCatData<First>(first, nData, nStrides, stride_index);
	processCatDataHelper(nData, nStrides, stride_index, rest...);
}

inline Bucket Bucket::catV(const std::vector<Bucket>& buckets){
	DType out_dtype = buckets[0].dtype;
	for(const auto& ref : buckets){
		utils::throw_exception(ref.dtype == out_dtype, "Expected all in cat to be of the same dtype but got $ and $", out_dtype, ref.dtype);
	}
	const DeviceType& dev = buckets[0].device_type();
	for(const auto& ref : buckets){
		utils::throw_exception(ref.device_type() == dev, "Expected all in cat to be of the same DeviceType but got $ and $", dev, ref.device_type());
	}
	bool blocked = buckets[0].strides_blocked;
	bool convert_strides = false;
	for(const auto& ref : buckets){
		if(ref.strides_blocked != blocked){
			convert_strides = true;
			break;
		}
	}
	if(convert_strides){
		//dont reserve, has been shown to cause an allocation error with buckets
		std::vector<Bucket> n_bucks(buckets.size(), Bucket::makeNullBucket(out_dtype));
		for(uint32_t i = 0; i < buckets.size(); ++i){
			n_bucks[i] = buckets[i].bucket_all_indices();
		}
		return Bucket::catV(n_bucks);
	}
	std::vector<std::reference_wrapper<const intrusive_ptr<Device> > > n_data;
	n_data.reserve(buckets[0].bs);
	for(uint64_t i = 0; i < buckets[0].bs; ++i)
		n_data.push_back(std::cref((*buckets[0].buckets_)[i]));
	int64_t nstride_s = 0;
	for(const auto& ref : buckets){
		nstride_s += ref.stride_size;
	}
	nt::intrusive_ptr<void*[]> nStrides(nstride_s);
	uint64_t stride_index = 0;
	for(const auto& ref : buckets){
		Bucket::processCatData(ref, n_data, nStrides, stride_index);
	}
	return Bucket(to_device_holder(n_data), std::move(nStrides), nstride_s, n_data.size(), blocked, out_dtype);
}


//this can be really slow when there are a lot of buckets and a lot of strides
//may want to rethink some of this (this was fixed with the new DeviceHolder construct, that can now only need a few Devices instead of what it was before)
//the longest part is the allocation actually
//
//problem before device holder:
// basically, whenever there was a transpose, to make the transpose faster, it would be split at the highest dimension possible
// this allowed all of the tensors and data below that dimension to automatically be transposed automatically (save some complexity, 
// and therefore time by shape().multiiply(dim) ^ (numel() - shape().range(0,dim).multiply())
// sounds great right?
// well, thats what I thought, and for the most part, yeah it was
// this made it so that the longest part (after optimization of the split function), was this concatenation function here
// (obviously the device holder also sped up the split function, without having to allocate another n intrusive_ptr<void> and all that stuff, instead its just a simple addition of an atomic pointer
// (one that would have happened anyways)
//
// so with this function, back to the previous example, if it was a large tensor, then there were thousands of intrusive_ptr<void>'s that had to be allocated and then set
// instead now, it is just a simple addition of an atomic number
// making the below function much faster
 
inline Bucket Bucket::catV(const std::vector<std::reference_wrapper<const Bucket> >& buckets){
	DType out_dtype = buckets[0].get().dtype;
	for(const auto& ref : buckets){
		utils::throw_exception(ref.get().dtype == out_dtype, "Expected all in cat to be of the same dtype but got $ and $", out_dtype, ref.get().dtype);
	}
	const DeviceType& dev = buckets[0].get().device_type();
	for(const auto& ref : buckets){
		utils::throw_exception(ref.get().device_type() == dev, "Expected all in cat to be of the same DeviceType but got $ and $", dev, ref.get().device_type());
	}
	bool blocked = buckets[0].get().strides_blocked;
	bool convert_strides = false;
	for(const auto& ref : buckets){
		if(ref.get().strides_blocked != blocked){
			convert_strides = true;
			break;
		}
	}	
	if(convert_strides){
		//dont reserve, has been shown to cause an allocation error with buckets
		std::vector<Bucket> n_bucks(buckets.size(), Bucket::makeNullBucket(out_dtype));
		for(uint32_t i = 0; i < buckets.size(); ++i){
			n_bucks[i] = buckets[i].get().bucket_all_indices();
		}
		return Bucket::catV(n_bucks);
	}
	std::vector<std::reference_wrapper<const intrusive_ptr<Device> > > n_data;
	n_data.reserve(buckets[0].get().bs);
	for(uint64_t i = 0; i < buckets[0].get().bs; ++i){
		n_data.push_back(std::cref((*buckets[0].get().buckets_)[i]));
	}
	int64_t nstride_s = 0;
	for(const auto& ref : buckets){
		nstride_s += ref.get().stride_size;
	}
	nt::intrusive_ptr<void*[]> nStrides(nstride_s);
	uint64_t stride_index = 0;
	for(const auto& ref : buckets){
		Bucket::processCatData(ref.get(), n_data, nStrides, stride_index);
	}

	return Bucket(to_device_holder(n_data), std::move(nStrides), nstride_s, n_data.size(), blocked, out_dtype);
}

template<typename Buck>
inline void Bucket::processCatStrideSize(int64_t& st, const Buck& b){
	static_assert(std::is_same_v<Buck, Bucket>, "Expected to only recieve type Bucket or ArrayVoid");
	if constexpr (std::is_same_v<Buck, Bucket>){st += b.stride_size;}
}
inline void Bucket::processCatStrideSizeHelper(int64_t& st){}

template<typename First, typename... Rest>
inline void Bucket::processCatStrideSizeHelper(int64_t& st, const First& first, const Rest&... rest){
	static_assert(std::is_same_v<First, Bucket>, "Expected to only recieve type Bucket or ArrayVoid");
	if constexpr (std::is_same_v<First, Bucket>){st += first.stride_size;}
	processCatStrideSize(st, rest...);
}

//this already assumes and should not be used till verified they are all the same
template<typename First, typename... Rest>
inline bool Bucket::processCatBlockType(const First& first, const Rest&... rest){
	static_assert(std::is_same_v<First, Bucket>, "Expected to only recieve type Bucket");
	return first.strides_blocked;
}

template<typename First>
inline bool Bucket::dont_convert_strides(const First& bf){ 
	static_assert(std::is_same_v<First, Bucket>, "Expected to only recieve type Bucket");
	return true;
}


template<typename First, typename Second, typename... Rest>
inline bool Bucket::dont_convert_strides(const First& bf, const Second& bs, const Rest&... rest){
	static_assert(std::is_same_v<First, Bucket> && std::is_same_v<Second, Bucket>, "Expected to only recieve type Bucket or ArrayVoid");
	bool store_dont_convert_strides = false;
	if constexpr (std::is_same_v<First, Bucket> && std::is_same_v<Second, Bucket>){
		store_dont_convert_strides = (bf.strides_blocked == bs.strides_blocked);

	}
	return store_dont_convert_strides && dont_convert_strides(bs, rest...);
}


template<typename First, typename... Rest>
inline void Bucket::convertBucketsHelper(std::vector<Bucket>& buckets, uint32_t& index, const First& bf, const Rest&... rest){
	static_assert(std::is_same_v<First, Bucket>, "Expected to only recieve type Bucket");
	if constexpr (std::is_same_v<First, Bucket>){
		buckets[index] = bf.bucket_all_indices();
		++index;
	}
	convertBucketsHelper(buckets, index, rest...);
}

template<typename... Buckets>
inline std::vector<Bucket> Bucket::convertBuckets(const Buckets&... buckets){
	std::vector<Bucket> output(sizeof...(buckets));
	uint32_t index = 0;
	convertBucketsHelper(output, index, buckets...);
	return std::move(output);
}


template<typename First, typename... Rest>
inline DType Bucket::processCatDType(const First& first, const Rest&... rest){
	DType outp = first.dtype;
	utils::throw_exception(processCatDType(rest...) == outp, "All buckets need to have the same dtype");
	return outp;
}

inline void verifyAllDevTypes(const Bucket& a){;}
inline void verifyAllDevTypes(const Bucket& a, const Bucket& b){
	utils::throw_exception(a.device_type() == b.device_type(), "Expected to concatenate the same devices but got $ and $", a.device_type(), b.device_type());
}
template<typename... Buckets>
inline void verifyAllDevTypes(const Bucket& a, const Bucket& b, const Bucket& c, const Buckets&... buckets){
	utils::throw_exception(a.device_type() == b.device_type(), "Expected to concatenate the same devices but got $ and $", a.device_type(), b.device_type());
	utils::throw_exception(b.device_type() == c.device_type(), "Expected to concatenate the same devices but got $ and $", a.device_type(), b.device_type());
	verifyAllDevTypes(c, buckets...);
}


template<typename... Buckets>
inline Bucket Bucket::cat(const Buckets&... buckets){
	static_assert((IsFirstVectorBucket<Buckets...>::value && sizeof...(buckets) == 1) || !IsFirstVectorBucket<Buckets...>::value,
			"Only concatenates one vector of buckets at a time");
	if constexpr(IsFirstVectorBucket<Buckets...>::value && sizeof...(buckets) == 1){
		return Bucket::catV(buckets...);
	}
	/* else if constexpr(IsFirstVectorArrayVoid<Buckets...>::value && sizeof...(buckets) == 1){ */
	/* 	return Bucket::catV(buckets...); */
	/* } */
	else{
		std::cout << "going to cat "<<sizeof...(Buckets) << "buckets"<<std::endl;
		int64_t n_stride_size = 0;
		processCatStrideSizeHelper(n_stride_size, buckets...);
		std::cout << n_stride_size << std::endl;
		utils::throw_exception(n_stride_size > 2, "Expected bucket size to be greater than zero for a cat function");
		bool dontConvert = dont_convert_strides(buckets...);
		verifyAllDevTypes(buckets...);
		if(!dontConvert){
			return Bucket::catV(convertBuckets(buckets...));
		}
		bool block_type = processCatBlockType(buckets...);
		nt::intrusive_ptr<void*[]> nStrides(n_stride_size);
		std::vector<std::reference_wrapper<const intrusive_ptr<Device> > > n_data;
		uint64_t stride_index = 0;
		processCatDataHelper(n_data, nStrides, stride_index, buckets...);
		return Bucket(to_device_holder(n_data), std::move(nStrides), n_stride_size, n_data.size(), block_type, processCatDType(buckets...));
	}

}


}



#endif // _NT_BUCKET_CAT_H_


