#include "device.h"
#include "../utils/utils.h"
#include "../dtype/DType.h"
#include "../Tensor.h"
#include <iostream>
#include "../intrusive_ptr/intrusive_ptr.hpp"

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
void deleteAlignedArray(void* ptr){std::free(ptr);}


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
	return detail::portable_aligned_alloc(align_byte, amt);
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
	if(dt == DType::TensorObj){
		memory_ = new Tensor[size];
		end_ = reinterpret_cast<Tensor*>(memory_) + size;
		dealc = &deleteCPPArray<Tensor>;
	}
	else if(dt == DType::Bool){
		memory_ = new uint_bool_t[size];
		end_ = reinterpret_cast<uint_bool_t*>(memory_) + size;
		dealc = &deleteCPPArray<uint_bool_t>;
	}
	else{
		memory_ = create_cpu_memory(dt, size);
		end_ = reinterpret_cast<uint8_t*>(memory_) + (DTypeFuncs::size_of_dtype(dt) * size);
		dealc = &deleteAlignedArray;
	}

	utils::THROW_EXCEPTION(memory_ != nullptr, "Failed to allocate cpu memory");
}

void DeviceCPU::release_memory(){
	if(memory_){
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



nt::intrusive_ptr<Device> make_device(const DeviceType dt){
	switch(dt){
		case DeviceType::CPU:
			return make_intrusive<DeviceCPU>();
		case DeviceType::CPUShared:
			return make_intrusive<DeviceSharedCPU>();
		default:
			return make_intrusive<DeviceCPU>(); //by default it will be put on the cpu
	}
}


DeviceType get_device_type(const intrusive_ptr<DeviceHolder>& ptr){
	if(!bool(ptr)){
		return dMETA;
	}
	return ptr[0]->get_device_type();
}


}
