#include "device.h"
#include "../utils/utils.h"
#include "../mtl/utils/mtl_general.h"
#include "../mtl/abstraction.h"
#include "../dtype/DType.h"
#include "../Tensor.h"
#include <iostream>
#include "../intrusive_ptr/intrusive_ptr.hpp"
#include "meta_allocator.h"
#include <string>

#include <cstdlib> // For std::aligned_alloc

#ifdef _WIN32
#include "SharedCPU_win.hpp"
#elif defined(__linux__) || defined(__APPLE__)
#include "SharedCPU_unix.hpp"
#else
#error "Unsupported device"
#endif


namespace nt{

void deleteNothing(void*){;}
void deleteAlignedArray(void* ptr){untracked_free_aligned_alloc(ptr);}


template<DType Dt = DType::Integer>
void* create_cpu_memory(const DType& dt, const int64_t& size){

	if(dt != Dt){return create_cpu_memory<DTypeFuncs::next_dtype_it<Dt>>(dt, size);}
	if(Dt == DType::Bool || Dt == DType::TensorObj){
		return new DTypeFuncs::dtype_to_type_t<Dt>[size];
	}
	//if it is numerical, it will just make it aligned, this would be in terms of extra memory at most the equivalent of 7 doubles for example
	//and when dealing with tensors of thousands, it's not that big of a deal for the speed increase seen in operations such as matrix multiplication
	//AVX instruction sets and the mkl library require alignment, so the alignment here adheres to that without having to copy memory in those other 
	//functions causing additional overhead
	uint64_t amt = static_cast<uint64_t>(size) * sizeof(DTypeFuncs::dtype_to_type_t<Dt>);
	/* if(amt > std::numeric_limits<int64_t>::max()){std::cout << "Potentially going to excede maximum size by allocating "<<amt<<" bytes"<<std::endl;} */
	const std::size_t align_byte = 64;
	if (amt % align_byte != 0) amt += align_byte - (amt % align_byte);
	return untracked_aligned_alloc(align_byte, amt);
}



DeviceCPU::DeviceCPU()
	:memory_(nullptr), end_(nullptr), dealc(&deleteNothing)
	{}

DeviceCPU::~DeviceCPU(){
	release_memory();
}

void DeviceCPU::allocate_memory(const DType dt, const int64_t size){
	release_memory();
	utils::throw_exception(size >= 0, "Cannot allocate negative bytes of memory, tried to allocate $ bytes", size);
    int64_t total_size = (DTypeFuncs::size_of_dtype(dt) * size);

	if(dt == DType::TensorObj){
        utils::CheckAllocation(DeviceType::CPU, size * sizeof(Tensor));
        memory_ = new Tensor[size];
        end_ = reinterpret_cast<Tensor*>(memory_) + size;
		dealc = &untracked_deleteCPPArray<Tensor>;
	}
	else if(dt == DType::Bool){
        utils::CheckAllocation(DeviceType::CPU, size);
		memory_ = new uint_bool_t[size];
		end_ = reinterpret_cast<uint_bool_t*>(memory_) + size;
		dealc = &untracked_deleteCPPArray<uint_bool_t>;
	}
	else{
        utils::CheckAllocation(DeviceType::CPU, total_size);
		memory_ = create_cpu_memory(dt, size);
		end_ = reinterpret_cast<uint8_t*>(memory_) + total_size;
		dealc = &deleteAlignedArray;
	}

	utils::THROW_EXCEPTION(memory_ != nullptr, "Failed to allocate cpu memory");
}

void DeviceCPU::release_memory(){
	if(memory_){
        int64_t total_size = static_cast<int64_t>(
				reinterpret_cast<uint8_t*>(end_) -
				reinterpret_cast<uint8_t*>(memory_));
        utils::DeallocateMemory(DeviceType::CPU, total_size);
        // std::cout << "deleted "<<total_size<<" bytes of memory"<<std::endl;
		dealc(memory_);
	}
	memory_ = nullptr;
	end_ = nullptr;
}

void DeviceCPU::capture_memory(void* mem, void* end){
	release_memory();
	memory_ = mem;
	end_ = end;
}

void DeviceCPU::capture_deleter(DeleterFnPtr func){
	dealc = func;
}



nt::intrusive_ptr<Device> make_device_(const DeviceType dt, const MemoryLayout layout){
	switch(dt){
		case DeviceType::CPU:{
            switch(layout){
			    case MemoryLayout::Private:
                    return make_intrusive<DeviceCPU>();
                case MemoryLayout::Shared:
                    return make_intrusive<DeviceSharedCPU>();
                default:
                    // none
                    utils::throw_exception(false, "Error, got unsupported memory layout for making CPU memory $", layout);
                    return intrusive_ptr<Device>(nullptr);
            }
        }
#ifdef NT_MTL_SUPPORTED
        case DeviceType::MTL:{
            switch(layout){
			    case MemoryLayout::Private:
                    return make_intrusive<DeviceMTLPrivate>();
                case MemoryLayout::Shared:
                    return make_intrusive<DeviceMTLShared>();
                default:
                    // none
                    utils::throw_exception(false, "Error, got unsupported memory layout for making MTL memory $", layout);
                    return intrusive_ptr<Device>(nullptr);
            }
        }
#endif
		default:
            utils::throw_exception(false, "Got unsupported device type $ to be made", dt);
			return make_intrusive<DeviceCPU>(); //by default it will be put on the cpu
	}
}

intrusive_ptr<DeviceHolder> make_device(const DeviceType dt, const MemoryLayout layout){
    intrusive_ptr<DeviceHolder> out = nt::make_intrusive<DeviceHolder>(1);
    out->get(0) = make_device_(dt, layout);
}


DeviceType get_device_type(const intrusive_ptr<DeviceHolder>& ptr){
	if(!bool(ptr)){
		return dMETA;
	}
	return ptr[0]->get_device_type();
}

MemoryLayout get_memory_layout(const intrusive_ptr<DeviceHolder>& ptr){
	if(!bool(ptr)){
		return MemoryLayout::None;
	}
	return ptr[0]->get_memory_layout();
}

}

#ifdef NT_MTL_SUPPORTED
namespace nt{
// if mtl supported on this platform (Apple silicone, then define the device for it)
DeviceMTLPrivate::DeviceMTLPrivate()
    :memory(nullptr), size(0)
{}

DeviceMTLShared::DeviceMTLShared()
    :memory(nullptr), size(0)
{}

void DeviceMTLPrivate::allocate_memory(const DType dt, const int64_t size){
   utils::throw_exception(size >= 0, 
           "Cannot allocate negative bytes of memory,"
           " tried to allocate $ bytes", size);
   utils::throw_exception(mtl::supported(dt),
                "Error, dtype $ is not supported on MTL GPU's",
                dt);
   if(dt == DType::Float32 || dt == DType::Float16){
        // then I want vectorized types to be by default available
        // it is already guarenteed that size > 0 -> bitwise operations can be used
        // n & 3 is equivalent to n % 4 for positive numbers
        int64_t remainder = size & 3; 
        int64_t to_add = (4 - remainder) % 4;
        size += to_add; // if already divisible by 4 -> += 0
   }
   size_t dtype_size = DTypeFuncs::size_of_dtype(dt);
   int64_t total_size = (static_cast<int64_t>(dtype_size) * size);
   this->memory = make_intrusive<mtl::abs::MetalBuffer>(
           mtl::abs::MetalContext::instance().device(),
           total_size, dtype_size, MTL::ResourceStorageModePrivate
   );
   this->size = total_size;
}

void DeviceMTLShared::allocate_memory(const DType dt, const int64_t size){
   utils::throw_exception(size >= 0, 
           "Cannot allocate negative bytes of memory,"
           " tried to allocate $ bytes", size);
   utils::throw_exception(mtl::supported(dt),
                "Error, dtype $ is not supported on MTL GPU's",
                dt);
   if(dt == DType::Float32 || dt == DType::Float16){
        // then I want vectorized types to be by default available
        // it is already guarenteed that size > 0 -> bitwise operations can be used
        // n & 3 is equivalent to n % 4 for positive numbers
        int64_t remainder = size & 3; 
        int64_t to_add = (4 - remainder) % 4;
        size += to_add; // if already divisible by 4 -> += 0
   }
   size_t dtype_size = DTypeFuncs::size_of_dtype(dt);
   int64_t total_size = (static_cast<int64_t>(dtype_size) * size);
   this->memory = make_intrusive<mtl::abs::MetalBuffer>(
       mtl::abs::MetalContext::instance().device(),
           total_size, dtype_size, MTL::ResourceStorageModeShared
   );
   this->size = total_size;
}

void DeviceMTLPrivate::allocate_memory(const int64_t size){
   utils::throw_exception(size >= 0, 
           "Cannot allocate negative bytes of memory,"
           " tried to allocate $ bytes", size);
   this->memory = make_intrusive<mtl::abs::MetalBuffer>(
        mtl::abs::MetalContext::instance().device(),
           size, 1, MTL::ResourceStorageModePrivate
   );
   this->size = size;
}

void DeviceMTLShared::allocate_memory(const int64_t size){
   utils::throw_exception(size >= 0, 
           "Cannot allocate negative bytes of memory,"
           " tried to allocate $ bytes", size);
   this->memory = make_intrusive<mtl::abs::MetalBuffer>(
           mtl::abs::MetalContext::instance().device(),
        size, 1, MTL::ResourceStorageModeShared
   );

   // this->memory = make_intrusive<mtl::abs::MetalBuffer>(
   //         mtl::abs::MetalContext::instance().device()->newBuffer(
   //      size, MTL::ResourceStorageModeShared)
   // );
   this->size = size;
}

void DeviceMTLPrivate::release_memory() {
    this->memory->reset(nullptr);
}

void DeviceMTLShared::release_memory() {
    this->memory->reset(nullptr);
}


// Consider removing these:
namespace mtl{

DeviceMTLPrivate mtl_shared_to_private(const DeviceMTLShared& device, MTL::CommandBuffer* buffer){
    // Copy CPU → GPU buffer
    DeviceMTLPrivate outBuffer;
    outBuffer.allocate_memory(device.Size());
    outBuffer.get_buffer()->typeBytes = device.get_buffer()->typeBytes;
    MTL::BlitCommandEncoder* blit = buffer->blitCommandEncoder();
    blit->copyFromBuffer(device.get_buffer()->buffer, 0, outBuffer.get_buffer()->buffer, 0, device.Size());
    blit->endEncoding();
    // buffer->commit();
    // buffer->WaitUntilComplete();
    return outBuffer;
}

DeviceMTLPrivate& mtl_shared_to_private(const DeviceMTLShared& device, MTL::CommandBuffer* buffer, DeviceMTLPrivate& out){
    // Copy CPU → GPU buffer
    MTL::BlitCommandEncoder* blit = buffer->blitCommandEncoder();
    blit->copyFromBuffer(device.get_buffer()->buffer, 0, out.get_buffer()->buffer, 0, device.Size());
    blit->endEncoding();
    // buffer->commit();
    // buffer->WaitUntilComplete();
    return out;
}

DeviceMTLShared mtl_private_to_shared(const DeviceMTLPrivate& device, MTL::CommandBuffer* buffer){
    // Copy GPU → CPU buffer
    DeviceMTLShared outBuffer;
    outBuffer.allocate_memory(device.Size());
    outBuffer.memory->typeBytes = device.get_buffer()->typeBytes;
    MTL::BlitCommandEncoder* blit = buffer->blitCommandEncoder();
    blit->copyFromBuffer(device.get_buffer()->buffer, 0, outBuffer.get_buffer()->buffer, 0, device.Size());
    blit->endEncoding();
    // buffer->commit();
    // buffer->WaitUntilComplete();
    return outBuffer;
}

DeviceMTLShared& mtl_private_to_shared(const DeviceMTLPrivate& device, MTL::CommandBuffer* buffer, DeviceMTLShared& out){
    // Copy GPU → CPU buffer
    MTL::BlitCommandEncoder* blit = buffer->blitCommandEncoder();
    blit->copyFromBuffer(device.get_buffer()->buffer, 0, out.get_buffer()->buffer, 0, device.Size());
    blit->endEncoding();
    // buffer->commit();
    // buffer->WaitUntilComplete();
    return out;
}

DeviceMTLShared cpu_to_mtl_shared(const DeviceCPU& device){
    DeviceMTLShared shared;
    std::ptrdiff_t total_size = device.get_end_memory() - device.get_memory();
    shared.allocate_memory(total_size);
    outBuffer.get_buffer()->typeBytes = 1;
    std::memcpy(shared.get_memory(), device.get_memory(), total_size);
}

DeviceMTLPrivate cpu_to_mtl_private(const DeviceCPU& device, MTL::CommandBuffer* buffer){
    DeviceMTLShared shared = cpu_to_mtl_shared(device);
    DeviceMTLPrivate output = mtl_shared_to_private(shared, buffer);
    return output;
}

}
}
#endif
