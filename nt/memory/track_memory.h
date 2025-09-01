#ifndef NT_TRACK_MEMORY_H__
#define NT_TRACK_MEMORY_H__
#include "DeviceEnum.h"
#include "meta_allocator.h"
#include "memory_limits.h"

namespace nt::utils{
namespace memory_details{

inline std::atomic<int64_t>& getCPUAllocated(){
    static std::atomic<int64_t> CPUAllocated = 0;
    return CPUAllocated;
}

inline std::atomic<int64_t>& getCPUSharedAllocated(){
    static std::atomic<int64_t> CPUSharedAllocated = 0;
    return CPUSharedAllocated;
}

}

NT_ALWAYS_INLINE int64_t getAllocatedMemory(DeviceType dt){
	switch(dt){
		case DeviceType::META:
            return memory_details::getMetaAllocated().load(std::memory_order_acquire);
		case DeviceType::CPU:
			return memory_details::getCPUAllocated().load(std::memory_order_acquire);
		case DeviceType::CPUShared:
			return memory_details::getCPUSharedAllocated().load(std::memory_order_acquire);
		default:
            return memory_details::getMetaAllocated().load(std::memory_order_acquire);
	}
}


NT_ALWAYS_INLINE void setAllocatedMemory(DeviceType dt, int64_t size){
	switch(dt){
		case DeviceType::META:
            memory_details::getMetaAllocated().store(size, std::memory_order_relaxed);
            return;
		case DeviceType::CPU:
			memory_details::getCPUAllocated().store(size, std::memory_order_relaxed);
            return;
		case DeviceType::CPUShared:
			memory_details::getCPUSharedAllocated().store(size, std::memory_order_relaxed);
            return;
		default:
            memory_details::getMetaAllocated().store(size, std::memory_order_relaxed);
            return;
	}
}

NT_ALWAYS_INLINE void addAllocatedMemory(DeviceType dt, int64_t size){
	switch(dt){
		case DeviceType::META:
            memory_details::getMetaAllocated().fetch_add(size, std::memory_order_relaxed);
            return;
		case DeviceType::CPU:
			memory_details::getCPUAllocated().fetch_add(size, std::memory_order_relaxed);
            return;
		case DeviceType::CPUShared:
			memory_details::getCPUSharedAllocated().fetch_add(size, std::memory_order_relaxed);
            return;
		default:
            memory_details::getMetaAllocated().fetch_add(size, std::memory_order_relaxed);
            return;
	}
}

NT_ALWAYS_INLINE void subAllocatedMemory(DeviceType dt, int64_t size){
	switch(dt){
		case DeviceType::META:
            memory_details::getMetaAllocated().fetch_sub(size, std::memory_order_acq_rel);
            return;
		case DeviceType::CPU:
			memory_details::getCPUAllocated().fetch_sub(size, std::memory_order_acq_rel);
            return;
		case DeviceType::CPUShared:
			memory_details::getCPUSharedAllocated().fetch_sub(size, std::memory_order_acq_rel);
            return;
		default:
			memory_details::getMetaAllocated().fetch_sub(size, std::memory_order_acq_rel);
            return;
	}
}

NT_ALWAYS_INLINE int64_t MaxMemory(DeviceType dt){
    switch(dt){
		case DeviceType::META:
			return -1;
		case DeviceType::CPU:
			return std::numeric_limits<int64_t>::max();
		case DeviceType::CPUShared:
			return get_shared_memory_max();

	}
}

NT_ALWAYS_INLINE void CheckAllocation(DeviceType dt, int64_t bytes){
	int64_t curMem = getAllocatedMemory(dt);
	int64_t maxMem = MaxMemory(dt);
	throw_exception((maxMem - curMem - bytes) >= 0 || maxMem == -1,
			"Trying to allocate $ bytes of memory on $, but already allocated $ and there is a max of $ bytes to allocate", bytes, dt, curMem, maxMem);
    addAllocatedMemory(dt, bytes);
}

NT_ALWAYS_INLINE void DeallocateMemory(DeviceType dt, int64_t bytes){
	subAllocatedMemory(dt, bytes);
}

}

#endif
