#include "bucket.h"
#include <memory>
#include "../intrusive_ptr/intrusive_ptr.hpp"
#include "../dtype/DType.h"
#include "../Tensor.h"
#include <utility>
#ifdef USE_PARALLEL
	#include <tbb/parallel_for.h>
	#include <tbb/parallel_for_each.h>
	#include <tbb/mutex.h>
#endif


namespace nt{





//the below is extremely important that it is initialized using new[]
//new will actually call an initializer (instead of just filling it with garbage)
//this is important because an std::unique_ptr<uint64_t[]> could be classified as the pointer not being a nullptr, but actually, if you tried to deallocate it or move another bucket into it, it would cause a malloc error of freeing a pointer never allocated in the first place
//another instance is this will fill bucket_sizes with a crap value, you could get lucky and get 0 or you could get unlucky and get a large number
//meaning if you checked for the above issue using bucket size, there is no guarentee that it will be zero
//granted the above is mainly if the dtype you are getting is DType::TensorObj
//However, it is still a dtype and an important one at that, so it must be accounted for

Bucket::Bucket(nt::intrusive_ptr<DeviceHolder> buckets, nt::intrusive_ptr<void*[]> strides, int64_t strideS, int64_t bS, bool blocked, DType dt)
	:buckets_(buckets),
	strides_(strides),
	stride_size(strideS),
	bs(bS),
	strides_blocked(blocked),
	dtype(dt)
{}

int64_t Bucket::blocked_stride_size() const{
	int64_t size = 0;
	/* std::cout << "doing blocked_stride_size"<<std::endl; */
	std::size_t dtype_s = DTypeFuncs::size_of_dtype(dtype);
	for(uint64_t i = 0; i < stride_size; ++i){
		const uint8_t* begin = reinterpret_cast<const uint8_t*>(strides_[i]);
		++i;
		const uint8_t* end = reinterpret_cast<const uint8_t*>(strides_[i]);
		/* std::cout << "inner block size: "<<(end-begin)<<std::endl; */
		size += (end - begin);
        // std::cout << "size is currently "<<size / dtype_s << std::endl;
	}
	return size / dtype_s;
}

Bucket Bucket::blocked_strides_clone() const {
	Bucket output(size(), dtype, device_type());
	uint8_t* begin = reinterpret_cast<uint8_t*>((*output.buckets_)[0]->get_memory());
	for(uint64_t i = 0; i < stride_size; ++i){
		const uint8_t* sBegin = reinterpret_cast<const uint8_t*>(strides_[i]);
		++i;
		const uint8_t* sEnd = reinterpret_cast<const uint8_t*>(strides_[i]);
		std::ptrdiff_t distance = (sEnd - sBegin);
		std::copy(sBegin, sEnd, begin);
		begin += distance;
	}
	return std::move(output);
}

Bucket Bucket::blocked_strides_clone_tensor() const {
	int64_t nsize = size();
	nt::intrusive_ptr<DeviceHolder> nData = make_intrusive<DeviceHolder>(1);
	(*nData)[0] = make_device(dCPU);
	(*nData)[0]->allocate_memory(DType::TensorObj, nsize);
	nt::intrusive_ptr<void*[]> nStrides(2);
	nStrides[0] = (*nData)[0]->get_memory();
	nStrides[1] = reinterpret_cast<Tensor*>((*nData)[0]->get_memory()) + nsize;
	Tensor* begin = reinterpret_cast<Tensor*>((*nData)[0]->get_memory());
	for(uint64_t i = 0; i < stride_size; ++i){
		const Tensor* sBegin = reinterpret_cast<const Tensor*>(strides_[i]);
		++i;
		const Tensor* sEnd = reinterpret_cast<const Tensor*>(strides_[i]);
		for(;sBegin != sEnd; ++sBegin, ++begin){
			*begin = sBegin->clone();
		}
	}
	return Bucket(std::move(nData), std::move(nStrides), 2, 1, true, dtype);
}


Bucket Bucket::strided_clone() const{
	Bucket output(size(), dtype, device_type());
	const std::size_t dtype_s = DTypeFuncs::size_of_dtype(dtype);
	uint8_t* begin = reinterpret_cast<uint8_t*>((*output.buckets_)[0]->get_memory());
	for(uint64_t i = 0; i < stride_size; ++i){
		const uint8_t* sBegin = reinterpret_cast<const uint8_t*>(strides_[i]);
		std::copy(sBegin, sBegin + dtype_s, begin);
		begin += dtype_s;
	}
	return std::move(output);
}

Bucket Bucket::strided_clone_tensor() const {
	int64_t nsize = size();
	nt::intrusive_ptr<DeviceHolder> nData = make_intrusive<DeviceHolder>(1);
	(*nData)[0] = make_device(dCPU);
	(*nData)[0]->allocate_memory(DType::TensorObj, nsize);
	nt::intrusive_ptr<void*[]> nStrides(2);
	nStrides[0] = (*nData)[0].get();
	nStrides[2] = reinterpret_cast<Tensor*>( (*nData)[0]->get_memory()) + nsize;
	Tensor* begin = reinterpret_cast<Tensor*>( (*nData)[0]->get_memory());
	Tensor** s_begin = reinterpret_cast<Tensor**>(stride_begin());
	Tensor** s_end = reinterpret_cast<Tensor**>(stride_end());
	for(;s_begin != s_end; ++s_begin, ++begin){
		*begin = (*s_begin)->clone();
	}
	return Bucket(std::move(nData), std::move(nStrides), 2, 1, true, dtype);
}

uint64_t Bucket::getBucketSize(const uint64_t bucket_index) const{
	if(!strides_blocked){return 1;}
	const uint32_t dtype_s = DTypeFuncs::size_of_dtype(dtype);
	return (reinterpret_cast<const uint8_t*>(strides_[bucket_index * 2 + 1]) - reinterpret_cast<const uint8_t*>(strides_[bucket_index * 2])) / dtype_s;
}

uint64_t Bucket::bucket_index(uint64_t& index) const {
	const std::size_t dtype_s = DTypeFuncs::size_of_dtype(dtype);
	for(uint64_t i = 0; i < uint64_t(stride_size/2); ++i){
		std::ptrdiff_t distance =
		(reinterpret_cast<const uint8_t*>(strides_[i*2+1]) - reinterpret_cast<const uint8_t*>(strides_[i*2]))
			/ dtype_s; 
		if(distance > index){return i;}
		index -= distance;
	}
	return uint64_t(stride_size/2);
}

Bucket::Bucket(const int64_t size, DType dt, DeviceType device_type)
	:buckets_(make_intrusive<DeviceHolder>(1)),
	strides_(2),
	bs(1),
	stride_size(2),
	strides_blocked(true),
	dtype(dt)
{
	(*buckets_)[0] = make_device(device_type);
	(*buckets_)[0]->allocate_memory(dt, size);
	strides_[0] = (*buckets_)[0]->get_memory();
	strides_[1] = reinterpret_cast<void*>(reinterpret_cast<uint8_t*>((*buckets_)[0]->get_memory()) + (size * DTypeFuncs::size_of_dtype(dt)));
}


Bucket::Bucket(const int64_t size, DType dt, void* ptr, DeleterFnPtr func)
	:buckets_(make_intrusive<DeviceHolder>(1)),
	strides_(2),
	bs(1),
	stride_size(2),
	strides_blocked(true),
	dtype(dt)
{
	utils::throw_exception(size >= 0, "Cannot capture negative bytes of memory but got $ bytes", size);
	(*buckets_)[0] = make_device(dCPU);
	size_t d_size = DTypeFuncs::size_of_dtype(dt);
	const uint64_t p_size = size * d_size;
	dynamic_cast<DeviceCPU*>((*buckets_)[0].get())->capture_memory(ptr, 
			reinterpret_cast<uint8_t*>(ptr) + p_size);
	dynamic_cast<DeviceCPU*>((*buckets_)[0].get())->capture_deleter(func);
	strides_[0] = (*buckets_)[0]->get_memory();
	strides_[1] = reinterpret_cast<void*>(reinterpret_cast<uint8_t*>((*buckets_)[0]->get_memory()) + p_size);	
}



//just going to contain a single float variable

Bucket::Bucket()
	:buckets_(make_intrusive<DeviceHolder>(1)),
	strides_(2),
	bs(1),
	stride_size(2),
	strides_blocked(true),
	dtype(DType::Float32)
{
	
	/* utils::THROW_EXCEPTION(1 == 0, "Bucket() default, data_nullptr: $ size_nullptr: $ bucket_sizes: $", data == nullptr, sizes == nullptr, bucket_sizes); */
	(*buckets_)[0] = make_device(dCPU); //by default on the cpu
	(*buckets_)[0]->allocate_memory(dtype, 1);
	strides_[0] = (*buckets_)[0]->get_memory();
	strides_[1] = reinterpret_cast<void*>(reinterpret_cast<float*>((*buckets_)[0]->get_memory()) + 1);
}



Bucket::Bucket(const Bucket& other)
	:buckets_(other.buckets_),
	strides_(other.strides_),
	stride_size(other.stride_size),
	bs(other.bs),
	strides_blocked(other.strides_blocked),
	dtype(other.dtype)
{}

Bucket::Bucket(Bucket&& other)
	:buckets_(std::move(other.buckets_)),
	strides_(std::move(other.strides_)),
	stride_size(std::exchange(const_cast<int64_t&>(other.stride_size), 0)),
	bs(std::exchange(const_cast<int64_t&>(other.bs), 0)),
	strides_blocked(other.strides_blocked),
	dtype(other.dtype)
{}

Bucket::Bucket(std::nullptr_t)
	:buckets_(nullptr),
	strides_(nullptr),
	stride_size(0),
	bs(0),
	strides_blocked(true),
	dtype(DType::Float32)
{}


bool Bucket::is_contiguous() const {
	if(strides_blocked && stride_size == 2){return true;}
	if(strides_blocked){return false;}
	const std::size_t type_size = DTypeFuncs::size_of_dtype(dtype);
	void** arr = strides_.get();
	for(uint64_t i = 1; i < stride_size; ++i){
		if(reinterpret_cast<uint8_t*>(arr[i]) != reinterpret_cast<uint8_t*>(arr[i-1]) + type_size)
			return false;
	}
	return true;
}

Bucket& Bucket::operator=(const Bucket& b){
	buckets_ = b.buckets_;
	strides_ = b.strides_;
	const_cast<int64_t&>(stride_size) = b.stride_size;
	const_cast<int64_t&>(bs) = b.bs;
	strides_blocked = b.strides_blocked;
	dtype = b.dtype;
	return *this;
}

Bucket& Bucket::operator=(Bucket&& b){
	buckets_ = std::move(b.buckets_);
	strides_ = std::move(b.strides_);
	const_cast<int64_t&>(stride_size) = std::exchange(const_cast<int64_t&>(b.stride_size), 0);
	const_cast<int64_t&>(bs) = std::exchange(const_cast<int64_t&>(b.bs), 0);
	strides_blocked = b.strides_blocked;
	dtype = b.dtype;
	return *this;
}


int64_t Bucket::size() const {
	if(strides_blocked){return blocked_stride_size();}
	return stride_size;
}


Bucket Bucket::clone() const {
	if(strides_blocked){return (dtype == DType::TensorObj) ? blocked_strides_clone_tensor() : blocked_strides_clone();}
	return strided_clone();
}

Bucket Bucket::contiguous() const{
	if(dtype == DType::TensorObj){
		if(!is_contiguous()){return clone();}
		uint32_t type = iterator_type();
		if(type == 1){
			auto begin = cbegin_contiguous<Tensor>();
			auto end = cend_contiguous<Tensor>();
			for(;begin != end; ++begin){
				if(!begin->is_contiguous()){return clone();}
			}
			return *this;
		}
		else if(type == 2){
			auto begin = cbegin_blocked<Tensor>();
			auto end = cend_blocked<Tensor>();
			for(;begin != end; ++begin){
				if(!begin->is_contiguous()){return clone();}
			}
			return *this;
		}
		else if(type == 3){
			auto begin = cbegin_list<Tensor>();
			auto end = cend_list<Tensor>();
			for(;begin != end; ++begin){
				if(!begin->is_contiguous()){return clone();}
			}
			return *this;		
		}
		return *this;
	}
	if(is_contiguous()){return *this;}
	return clone();
}






void set_correct_block_force_contiguity(
		std::vector<std::pair<uint8_t*, uint8_t*> >& blocks,
		uint8_t* begin, uint8_t* end,
		const nt::intrusive_ptr<DeviceHolder>& devices,
		const int64_t& bs_){

	for(int64_t i = 0; i < bs_; ++i){
		if((*devices)[i]->in_block(begin)){
			if(blocks[i].first == nullptr){
				blocks[i].first = begin;
				blocks[i].second = end;
				return;
			}
			if(blocks[i].first > begin){
				blocks[i].first = begin;
			}
			if(blocks[i].second < end){
				blocks[i].second = end;
			}
			return;
		}
	}
}

void set_correct_block_force_contiguity(
		std::vector<std::pair<uint8_t*, uint8_t*> >& blocks,
		void** o_begin,
		const nt::intrusive_ptr<DeviceHolder>& devices,
		const int64_t& bs_){
	
	uint8_t* cur_s = reinterpret_cast<uint8_t*>(*o_begin);
	/* std::cout << "checking "<<*reinterpret_cast<float*>(cur_s)<<std::endl; */
	for(int64_t i = 0; i < bs_; ++i){
		if((*devices)[i]->in_block(cur_s)){
			/* std::cout << "in block "<<i<<std::endl; */
			if(blocks[i].first == nullptr){
				blocks[i].first = cur_s;
				blocks[i].second = cur_s;
				return;
			}
			if(blocks[i].first > cur_s){
				/* std::cout << "greater than"<<std::endl; */
				blocks[i].first = cur_s;
			}
			if(blocks[i].second < cur_s){
				/* std::cout << "less than"<<std::endl; */
				blocks[i].second = cur_s;
			}
			return;
		}
	}
}

/* void print_blocks_remake(std::vector<std::pair<uint8_t*, uint8_t*> >& blocks){ */
/* 	for(size_t i = 0; i < blocks.size(); ++i){ */
/* 		std::cout << i << ": "; */
/* 		if(blocks[i].first == nullptr){ */
/* 			std::cout << "{ nullptr, "; */
/* 		}else{ */
/* 			std::cout << "{ "<<*reinterpret_cast<float*>(blocks[i].first)<<", "; */
/* 		} */
/* 		if(blocks[i].second == nullptr){ */
/* 			std::cout << "nullptr}"<<std::endl; */
/* 		}else{ */
/* 			std::cout << *reinterpret_cast<float*>(blocks[i].second)<<"}"<<std::endl; */
/* 		} */

/* 	} */
/* } */

Bucket Bucket::force_contiguity_and_bucket() const{
	//this function looks at all the different iteraor types
	//and forces contiguity based on those
	//and it then buckets all the indices
	uint32_t i_type = iterator_type();
	//off the bat, if it is already contiguous, then just bucket all the indices
	if(i_type == 1){return bucket_all_indices();}
	//this is if it is broken into buckets already
	if(i_type == 2){
		//this means that it is all in the same contiguous block of memory
		if(bs == 1){
			size_t d_size = DTypeFuncs::size_of_dtype(dtype);

			uint8_t* begin = reinterpret_cast<uint8_t*>(*stride_begin());
			uint8_t* end = reinterpret_cast<uint8_t*>(*(stride_end()-1));
			void** s_begin = stride_begin();
			void** s_end = stride_end();
			for(;s_begin !=  s_end; ++s_begin){
				if(s_begin == s_end){break;}
				if(reinterpret_cast<uint8_t*>(s_begin) > end){
					end = reinterpret_cast<uint8_t*>(s_begin);
				}
			}
			utils::THROW_EXCEPTION(end > begin, "Cannot force contiguity");
			int64_t n_size = (end - begin) / d_size;
			intrusive_ptr<void*[]> nStrides(n_size);
			void** o_begin = nStrides.get();
			for(;begin < end; begin += d_size, ++o_begin)
				*o_begin = begin;
			return Bucket(buckets_, std::move(nStrides), n_size, bs, false, dtype);
		}
		//this is for when there is a contiguous block of memory for every block
		if(bs == (stride_size/2)){
			return bucket_all_indices();
		}
		//the last case is when there is the potential to be multiple buckets inside a single block
		//and there are multiply blocks
		//this vector is going to hold all of the beginings and ends per block of memory
		//then each individual block will be bucketed
		std::vector<std::pair<uint8_t*, uint8_t*> > blocks(bs);
		//initialize them all to nullptr's
		for(int64_t i = 0; i < bs; ++i){
			blocks[i].first = nullptr; blocks[i].second = nullptr;
		}
		//assign each pointer to being either begin or end of blocks if it fits
		//in the corresponding block of memory
		void** o_begin = stride_begin();
		void** o_end = stride_end();
		for(;o_begin != o_end; ++o_begin){
			uint8_t* p_begin = reinterpret_cast<uint8_t*>(*o_begin);
			++o_begin;
			uint8_t* p_end = reinterpret_cast<uint8_t*>(o_begin);
			set_correct_block_force_contiguity(blocks, p_begin, p_end, buckets_, bs);
		}
		//get the exact size of the new blocks of memory
		size_t d_size = DTypeFuncs::size_of_dtype(dtype);
		int64_t n_size = 0;
		for(int64_t i = 0; i < bs; ++i){
			n_size += (blocks[i].second - blocks[i].first);
		}
		n_size /= d_size;
		//now create the strides;
		intrusive_ptr<void*[]> nStrides(n_size);
		//now bucket all the strides
		void** n_begin = nStrides.get();
		for(int64_t i = 0; i < bs; ++i){
			uint8_t* begin = blocks[i].first;
			uint8_t* end = blocks[i].second;
			for(;begin != end; begin += d_size, ++n_begin)
				*n_begin = begin;	
		}
		return Bucket(buckets_, std::move(nStrides), n_size, bs, false, dtype);
	}
	//last is if they are already bucketed 
	//this one is a little more tricky
	/* std::cout << "forcing contiguous for already blocked" << std::endl; */
	if(is_contiguous())
		return *this;
	if(bs == 1){
		//then they are already guarenteed to be in the same block of memory
		uint8_t* begin = reinterpret_cast<uint8_t*>(*stride_begin());
		uint8_t* end = reinterpret_cast<uint8_t*>(*(stride_end()-1));
		void** n_begin = stride_begin();
		void** n_end = stride_end();
		for(;n_begin != n_end; ++n_begin){
			if(reinterpret_cast<uint8_t*>(*n_begin) > end){
				end = reinterpret_cast<uint8_t*>(*n_begin); 
			}
			if(reinterpret_cast<uint8_t*>(*n_begin) < begin){
				begin = reinterpret_cast<uint8_t*>(*n_begin); 
			}
		}
		size_t d_size = DTypeFuncs::size_of_dtype(dtype);
		int64_t n_size = (end - begin) / d_size;
		intrusive_ptr<void*[]> nStrides(n_size);
		void** o_begin = nStrides.get();
		for(;begin != end; begin += d_size, ++o_begin)
			*o_begin = begin;	
		return Bucket(buckets_, std::move(nStrides), n_size, bs, false, dtype);
	}

	//last one is if there are multiple device blocks to take care of
	//this is going to be similar to iterator_2 version
	std::vector<std::pair<uint8_t*, uint8_t*> > blocks(bs);
	for(int64_t i = 0; i < bs; ++i){
		blocks[i].first = nullptr; blocks[i].second = nullptr;
	}
	void** o_begin = stride_begin();
	void** o_end = stride_end();
	for(;o_begin != o_end; ++o_begin){
		/* std::cout << "now checking for contiguity for: "<<*reinterpret_cast<float*>(*o_begin)<<std::endl; */
		set_correct_block_force_contiguity(blocks, o_begin, buckets_, bs);
		/* print_blocks_remake(blocks); */

	}	

	size_t d_size = DTypeFuncs::size_of_dtype(dtype);
	int64_t n_size = 0;
	for(int64_t i = 0; i < bs; ++i){
		if(blocks[i].first == blocks[i].second){
			n_size += d_size;
			continue;
		}
		//add one that signifies the end
		n_size += (blocks[i].second - blocks[i].first) + d_size;
	}
	n_size /= d_size;
	intrusive_ptr<void*[]> nStrides(n_size);
	void** n_begin = nStrides.get();
	for(int64_t i = 0; i < bs; ++i){
		uint8_t* begin = blocks[i].first;
		uint8_t* end = blocks[i].second;
		if(begin == end){
			*n_begin = begin;
			++n_begin;
			continue;
		}
		for(;begin != end; begin += d_size, ++n_begin)
			*n_begin = begin;
		*n_begin = begin;
		++n_begin;
	}
	/* std::cout << n_size << " is n_size"<<std::endl; */
	return Bucket(buckets_, std::move(nStrides), n_size, bs, false, dtype);
}


//this is a function that basically finds the minimum element in memory
//(in terms of if the memory was contiguous and the buckets were first)
//and the last point in memory
//and the contigitizes it based on that
//and then buckets it
Bucket Bucket::bound_force_contiguity_bucket() const{
	if(bs == 1){return force_contiguity_and_bucket();}
	uint32_t i_type = iterator_type();
	std::vector<uint8_t*> blocks(2);
	uint8_t* cpy_first_bucket_l = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>((*buckets_)[0]->get_end_memory()));
	uint8_t* cpy_last_bucket_f = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>((*buckets_)[bs-1]->get_memory()));
	blocks[0] = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>((*buckets_)[0]->get_end_memory()));
	blocks[1] = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>((*buckets_)[bs-1]->get_memory()));
	
	if(i_type == 2){
		void** s_begin = stride_begin();
		void** s_end = stride_end();
		for(;s_begin != s_end; ++s_begin){
			uint8_t* begin = reinterpret_cast<uint8_t*>(*s_begin);
			++s_begin;
			uint8_t* end = reinterpret_cast<uint8_t*>(*s_begin);
			if((*buckets_)[0]->in_block(begin)){
				if(blocks[0] > begin)
					blocks[0] = begin;
			}
			else if((*buckets_)[bs-1]->in_block(end)){
				if(blocks[1] < end)
					blocks[1] = end;
			}
		}
	}
	else if(i_type == 3){
		void** s_begin = stride_begin();
		void** s_end = stride_end();
		for(;s_begin != s_end; ++s_begin){
			uint8_t* ptr = reinterpret_cast<uint8_t*>(*s_begin);
			if((*buckets_)[0]->in_block(ptr)){
				if(blocks[0] > ptr)
					blocks[0] = ptr;
			}
			else if((*buckets_)[bs-1]->in_block(ptr)){
				if(blocks[1] < ptr)
					blocks[1] = ptr;
			}
		}
	}
	size_t d_size = DTypeFuncs::size_of_dtype(dtype);
	int64_t n_size = 0;
	//first bucket
	n_size += cpy_first_bucket_l - blocks[0];
	//middle buckets
	for(int i = 1; i < bs-1; ++i){
		n_size += reinterpret_cast<const uint8_t*>((*buckets_)[i]->get_end_memory())
			- reinterpret_cast<const uint8_t*>((*buckets_)[i]->get_memory());
	}
	//last bucket
	n_size += blocks[1] - cpy_last_bucket_f;
	n_size /= d_size;
	intrusive_ptr<void*[]> nStrides(n_size);
	void** o_begin = nStrides.get();

	//do the first bucket:
	for(;blocks[0] != cpy_first_bucket_l; blocks[0] += d_size, ++o_begin){
		*o_begin = blocks[0];
	}
	//do middle buckets
	for(int i = 1; i < bs-1; ++i){
		uint8_t* begin = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>((*buckets_)[i]->get_memory()));
		uint8_t* end = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>((*buckets_)[i]->get_end_memory()));
		for(;begin != end; begin += d_size, ++o_begin)
			*o_begin = begin;
	}
	//do last bucket
	for(;cpy_last_bucket_f != blocks[1]; cpy_last_bucket_f += d_size, ++o_begin)
		*o_begin = cpy_last_bucket_f;

	return Bucket(buckets_, std::move(nStrides), n_size, bs, false, dtype);
}


//this is if the following function can safely be completed
bool Bucket::can_force_contiguity() const{
	return bs == 1;
}
int64_t Bucket::force_contig_size() const {
	return reinterpret_cast<const uint8_t*>(data_ptr_end()) - reinterpret_cast<const uint8_t*>(data_ptr());
}

bool Bucket::can_force_contiguity_bytes(const int64_t& bytes) const{
	if(bs == 1){return bytes <= force_contig_size();}
	const uint8_t* begin = reinterpret_cast<const uint8_t*>(data_ptr());
	const uint8_t* end = begin + bytes;
	int64_t i;
	for(i = 0; i < bs; ++i){
		if((*buckets_)[i]->in_block(begin)){break;}
	}
	if(i == bs){return false;} //this should be impossible and would be a really big issue with how memory is handeled
	return reinterpret_cast<const uint8_t*>((*buckets_)[i]->get_end_memory()) >= end;
}


Bucket Bucket::force_contiguity(int64_t n_size) const{
	n_size *= DTypeFuncs::size_of_dtype(dtype);
	/* intrusive_ptr<intrusive_ptr<void>> nData(1); */ //it used to be a new data would be made
							   //however, with the device adaptation, the same intrusive_ptr is just coppied over
							   //which is much less memory-intensive and faster than the old version
	intrusive_ptr<void*[]> nStrides(2);
	//get difference between current strides at 0:
	
	/* uint64_t start = (reinterpret_cast<const uint8_t*>(strides_[0]) - reinterpret_cast<const uint8_t*>((*buckets_)[0].get())); */
	/* nData[0] = (*buckets_)[0] + start; */
	nStrides[0] = *stride_begin();
	nStrides[1] = reinterpret_cast<uint8_t*>(nStrides[0]) + n_size;
	return Bucket(buckets_, std::move(nStrides), 2, bs, true, dtype);
}

Bucket Bucket::to_shared() const{
	if(device_type() == dCPUShared)
		return *this;
	Bucket output(size(), dtype, dCPUShared);

	uint8_t* o_begin = reinterpret_cast<uint8_t*>(output.data_ptr());
	uint32_t type = iterator_type();
	if(type == 1){
		const uint8_t* begin = reinterpret_cast<const uint8_t*>(data_ptr());
		const uint8_t* end = reinterpret_cast<const uint8_t*>(data_ptr_end());
		std::memcpy(o_begin, begin, end-begin);
		return std::move(output);
	}
	else if(type == 2){
		void** s_begin = stride_begin();
		void** s_end = stride_end();
		for(;s_begin != s_end; ++s_begin){
			const uint8_t* sBegin = reinterpret_cast<const uint8_t*>(*s_begin);
			++s_begin;
			const uint8_t* sEnd = reinterpret_cast<const uint8_t*>(*s_begin);
			std::ptrdiff_t distance = (sEnd - sBegin);
			std::memcpy(o_begin, sBegin, distance);
			o_begin += distance;
			if(s_begin == s_end){break;}
		}
		return std::move(output);
	}
	else if(type == 3){

		void** s_begin = stride_begin();
		void** s_end = stride_end();
		const std::size_t d_size = DTypeFuncs::size_of_dtype(dtype);
		for(;s_begin != s_end; ++s_begin){
			const uint8_t* sBegin = reinterpret_cast<const uint8_t*>(*s_begin);
			std::memcpy(o_begin, sBegin, d_size);
			o_begin += d_size;
		}
		return std::move(output);
	}
	return output;
}
Bucket Bucket::to_cpu() const{
	if(device_type() == dCPU)
		return *this;
	Bucket output(size(), dtype, dCPU);
	uint8_t* o_begin = reinterpret_cast<uint8_t*>(output.data_ptr());
	uint32_t type = iterator_type();
	if(type == 1){
		const uint8_t* begin = reinterpret_cast<const uint8_t*>(data_ptr());
		const uint8_t* end = reinterpret_cast<const uint8_t*>(data_ptr_end());
		std::memcpy(o_begin, begin, end-begin);
		return std::move(output);
	}
	else if(type == 2){
		void** s_begin = stride_begin();
		void** s_end = stride_end();
		for(;s_begin != s_end; ++s_begin){
			const uint8_t* sBegin = reinterpret_cast<const uint8_t*>(*s_begin);
			++s_begin;
			const uint8_t* sEnd = reinterpret_cast<const uint8_t*>(*s_begin);
			std::ptrdiff_t distance = (sEnd - sBegin);
			std::memcpy(o_begin, sBegin, distance);
			o_begin += distance;
			if(s_begin == s_end){break;}
		}
		return std::move(output);
	}
	else if(type == 3){

		void** s_begin = stride_begin();
		void** s_end = stride_end();
		const std::size_t d_size = DTypeFuncs::size_of_dtype(dtype);
		for(;s_begin != s_end; ++s_begin){
			const uint8_t* sBegin = reinterpret_cast<const uint8_t*>(*s_begin);
			std::memcpy(o_begin, sBegin, d_size);
			o_begin += d_size;
		}
		return std::move(output);
	}
	return output;
}

Bucket Bucket::to_device(DeviceType dt) const {
	switch(dt){
		case DeviceType::META:
			utils::throw_exception(dt != DeviceType::META, "Cannot convert to meta");
		case DeviceType::CPU:
			return to_cpu();
		case DeviceType::CPUShared:
			return to_shared();
		default:
			return *this;
	}
}

Bucket Bucket::new_bounds(uint64_t start, uint64_t end) const{
	nt::utils::THROW_EXCEPTION(start < end, "Expected start to be less than end but got $ and $", start, end);
	nt::utils::THROW_EXCEPTION(end <= size(), "Expected end to be less than or equal to $ but got $", size(), end);
	const std::size_t dtype_s = DTypeFuncs::size_of_dtype(dtype);
    if(!strides_blocked){
        // std::cout << "non-blocked new bounds"<<std::endl;
		return Bucket(buckets_, strides_ + start, end-start, bs, false, dtype);
	}
	if(is_contiguous()){
        // std::cout << "contiguous new bounds"<<std::endl;
		/* nt::intrusive_ptr<nt::intrusive_ptr<void>> nData(1); */
		/* nData[0] = (*buckets_)[0] + (start * dtype_s); */
		nt::intrusive_ptr<void*[]> nStrides(2);
		nStrides[0] = reinterpret_cast<uint8_t*>(*(stride_begin())) + (start * dtype_s);
		nStrides[1] = reinterpret_cast<uint8_t*>(*(stride_begin())) + (end   * dtype_s);
		return Bucket(buckets_, std::move(nStrides), 2, 1, true, dtype);
	}
    // std::cout << "blocked new bounds"<<std::endl;
    // std::cout << "start was "<<start<<std::endl;
    uint64_t wanted_distance = (end-start) * dtype_s;
    // uint64_t strides_checked = 0;
    // for(uint64_t i =0; i < stride_size; ++i){
        
    // }
	uint64_t checked_begin = 0;
	/* uint64_t tSize = 0; */
	for(uint64_t i = 0; i < stride_size; ++i){
        if(start == 0) break;
		const uint8_t* begin = reinterpret_cast<const uint8_t*>(strides_[i]);
		++i;
		const uint8_t* stop = reinterpret_cast<const uint8_t*>(strides_[i]);
		std::ptrdiff_t distance = ((stop - begin) / dtype_s);
		if(start > distance){start -= distance; end -= distance; checked_begin += 2; continue;}
		if(start == distance){
			start = 0; 
			end -= distance; 
			checked_begin += 2;
		}
		break;
	}
    // std::cout << "start, checked begin: "<<start<<','<<checked_begin<<std::endl;
	uint64_t checked_end = checked_begin;
	uint64_t tSize = (reinterpret_cast<const uint8_t*>(strides_[checked_begin + 1]) - reinterpret_cast<const uint8_t*>(strides_[checked_begin])) / dtype_s;
	uint64_t ntSize = tSize;
    // std::cout << "ntSize: "<<ntSize<<std::endl;
	if(ntSize >= end){
		/* nt::intrusive_ptr<nt::intrusive_ptr<void>> nData(1); */
		/* nData[0] = (*buckets_)[checked_begin] + (start * dtype_s); */
		nt::intrusive_ptr<void*[]> nStrides(2);
		nStrides[0] = reinterpret_cast<uint8_t*>(*(stride_begin() + checked_begin)) + (start * dtype_s);
		nStrides[1] = reinterpret_cast<uint8_t*>(*(stride_begin() + checked_begin)) + (end   * dtype_s);
        return Bucket(buckets_, std::move(nStrides), 2, 1, true, dtype);
	}
	int64_t ntSizeCpy = ntSize;
	ntSize -= start;
	checked_end += 2;
	for(;checked_end < stride_size; checked_end += 2){
		ntSize += getBucketSize(checked_end / 2);
		if(end <= ntSize){break;}
	}
    // std::cout << "checked end: "<<checked_end<<std::endl;
    // std::cout << "ntSize: "<<ntSize<<std::endl;
	checked_end = (checked_end == stride_size) ? stride_size : checked_end+2;
    if(ntSize == end){
        nt::intrusive_ptr<void*[]> nStrides(checked_end - checked_begin);
        const int64_t num_buckets = (checked_end-checked_begin) / 2;
        nStrides[0] = reinterpret_cast<uint8_t*>(*(stride_begin() + 0 + checked_begin)) + (start * dtype_s);
        nStrides[1] = reinterpret_cast<uint8_t*>(*(stride_begin() + 1 + checked_begin));
        for(uint64_t i = 0; i < num_buckets * 2; ++i){
            nStrides[i] = reinterpret_cast<uint8_t*>(*(stride_begin() + i + checked_begin));
            ++i;
            nStrides[i] = reinterpret_cast<uint8_t*>(*(stride_begin() + i + checked_begin));
        }
        return Bucket(buckets_, std::move(nStrides), checked_end-checked_begin,
                                num_buckets, true, dtype);
    }
    //the only other possibility is ntSize < end
    //TODO: copy the above, but the last stride will be the difference between (ntSize - getBucketSize(checked_end - 2 / 2)) and end
    //similar to how start was used in the first stride
	uint64_t nstride_size = checked_end - checked_begin;
	nt::utils::THROW_EXCEPTION(nstride_size > 0, "nBS was 0 for new_bounds");
	nt::intrusive_ptr<void*[]> nStrides(nstride_size);
	start *= dtype_s;
	end *= dtype_s;
	/* nData[0] = (*buckets_)[checked_begin] + start; */
	void** arr = nStrides.get(); // start
	void** arr_end = nStrides.get() + nstride_size;
	void** my_strides_begin = stride_begin() + checked_begin; //start
	*arr = reinterpret_cast<uint8_t*>(*my_strides_begin) + start;
	++arr; //next end
	++my_strides_begin; // next end
	uint64_t current_size = ((reinterpret_cast<uint8_t*>(*my_strides_begin) - reinterpret_cast<uint8_t*>(*(my_strides_begin-1)))) - start;
	*arr = *my_strides_begin;
	end -= (current_size + start);

	/* *arr = stride_begin() + (checked_begin * 2 + 1); */
	++arr; //now at next start
	++my_strides_begin; //now at next start
	for(;arr != arr_end;){
		*arr = *my_strides_begin;
		++arr;//now at next end
        if(arr + 1 == arr_end){break;}
		++my_strides_begin;//now at next end
		uint64_t current_size = ((reinterpret_cast<uint8_t*>(*(my_strides_begin+1)) - reinterpret_cast<uint8_t*>(*my_strides_begin)));

		*arr = *my_strides_begin;
		end -= current_size;
		++arr;//now at next end
		++my_strides_begin;//now at next end
	}
	*arr = reinterpret_cast<uint8_t*>(*my_strides_begin) + (end);
	return Bucket(buckets_, std::move(nStrides), nstride_size, bs, true, dtype);
}


Bucket Bucket::makeNullBucket(DType dt, int64_t stride_size){
	intrusive_ptr<DeviceHolder> nData;
	intrusive_ptr<void*[]> nStrides(stride_size);
	return Bucket(std::move(nData), std::move(nStrides), stride_size, 0, false, dt);
}

Bucket Bucket::makeCopyBucket(DType dt, const intrusive_ptr<DeviceHolder>& bucks, bool blocked, int64_t bS, int64_t stride_size){
	return Bucket(bucks, intrusive_ptr<void*[]>(stride_size), stride_size, bS, blocked, dt);
}

//this assumes all the data is just a contiguous dataset

template <>
std::vector<Bucket> Bucket::split_contiguous_<std::vector<Bucket>>(uint64_t splitting) const{
	int64_t msize = size();
	if(splitting > msize){
		return std::vector<Bucket > {*this};
	}
	uint64_t div = msize / splitting;
	uint64_t remainder = msize % splitting;
	bool r = remainder > 0;
	std::vector<Bucket > fb( (r) ? div + 1 : div, makeCopyBucket(dtype, buckets_, true, bs, 2));
	
	const std::size_t dtype_s = DTypeFuncs::size_of_dtype(dtype);
	uint64_t current = 0;
	const int64_t adding = dtype_s * splitting;
	for(uint64_t i = 0; i < div; ++i){
		fb[i].strides_[0] = reinterpret_cast<uint8_t*>(*stride_begin())  + current;
		fb[i].strides_[1] = reinterpret_cast<uint8_t*>(fb[i].strides_[0]) + adding;
		current += adding;
	}
	if(r){
		fb.back().strides_[0] = reinterpret_cast<uint8_t*>(*stride_begin())      + current;
		fb.back().strides_[1] = reinterpret_cast<uint8_t*>(fb.back().strides_[0]) + (remainder * dtype_s);
	}
	return std::move(fb);
}


//this assumes all the data is just a contiguous dataset

#ifndef USE_PARALLEL
template <>
Tensor Bucket::split_contiguous_<Tensor>(uint64_t splitting) const{
	int64_t msize = size();
	if(splitting > msize){
		Tensor output = Tensor::makeNullTensorArray(1);
		Tensor& t = output.item<Tensor>();
		t._vals.bucket = *this;
		t._vals.dtype = dtype;
		t._vals.size = size();
		t._total_size = size();
		t.dtype = dtype;
		t._size = SizeRef({t._total_size});
		return output;
	}
	uint64_t div = msize / splitting;
	uint64_t remainder = msize % splitting;
	bool r = remainder > 0;
	Tensor array = Tensor::makeNullTensorArray((r) ? div + 1 : div);
	
	const std::size_t dtype_s = DTypeFuncs::size_of_dtype(dtype);
	uint64_t current = 0;
	const int64_t adding = dtype_s * splitting;
	Tensor* begin = reinterpret_cast<Tensor*>(array.data_ptr());

	for(uint64_t i = 0; i < div; ++i, ++begin){
		begin->_vals.bucket.buckets_ = buckets_;
		begin->_vals.bucket.strides_ = intrusive_ptr<void*[]>(2);
		begin->_vals.bucket.strides_[0] = reinterpret_cast<uint8_t*>(*stride_begin())                + current;
		begin->_vals.bucket.strides_[1] = reinterpret_cast<uint8_t*>(begin->_vals.bucket.strides_[0]) + adding;

		const_cast<int64_t&>(begin->_vals.bucket.stride_size) = 2;
		const_cast<int64_t&>(begin->_vals.bucket.bs) = bs;
		begin->_vals.bucket.strides_blocked = true;
		begin->dtype = dtype;
		begin->_vals.dtype = dtype;
		begin->_vals.bucket.dtype = dtype;
		begin->_total_size = splitting;
		begin->_vals.size = splitting;
		current += adding;
	}
	if(r){
		
		begin->_vals.bucket.buckets_ = buckets_;
		begin->_vals.bucket.strides_ = intrusive_ptr<void*[]>(2);
		begin->_vals.bucket.strides_[0] = reinterpret_cast<uint8_t*>(*stride_begin())                + current;
		begin->_vals.bucket.strides_[1] = *stride_end();

		const_cast<int64_t&>(begin->_vals.bucket.stride_size) = 2;
		const_cast<int64_t&>(begin->_vals.bucket.bs) = bs;
		begin->_vals.bucket.strides_blocked = true;
		begin->dtype = dtype;
		begin->_vals.dtype = dtype;
		begin->_vals.bucket.dtype = dtype;
		begin->_total_size = remainder;
		begin->_vals.size = remainder;
	}
	return std::move(array);
}

#else
template <>
Tensor Bucket::split_contiguous_<Tensor>(uint64_t splitting) const{
	int64_t msize = size();
	if(splitting > msize){
		Tensor output = Tensor::makeNullTensorArray(1);
		Tensor& t = output.item<Tensor>();
		t._vals.bucket = *this;
		t._vals.dtype = dtype;
		t._vals.size = size();
		t._total_size = size();
		t.dtype = dtype;
		t._size = SizeRef({t._total_size});
		return output;
	}
	uint64_t div = msize / splitting;
	uint64_t remainder = msize % splitting;
	bool r = remainder > 0;
	Tensor array = Tensor::makeNullTensorArray((r) ? div + 1 : div);
	
	const std::size_t dtype_s = DTypeFuncs::size_of_dtype(dtype);
	const int64_t adding = dtype_s * splitting;
	tbb::parallel_for(utils::calculateGrainSize1D(0, div), [&](const auto range){
	Tensor* begin = reinterpret_cast<Tensor*>(array.data_ptr()) + range.begin();
	uint64_t current = adding * range.begin();
	Tensor* end = begin + (range.end() - range.begin());
	for(;begin != end; ++begin){
		begin->_vals.bucket.buckets_ = buckets_;
		begin->_vals.bucket.strides_ = intrusive_ptr<void*[]>(2);
		begin->_vals.bucket.strides_[0] = reinterpret_cast<uint8_t*>(*stride_begin())                + current;
		begin->_vals.bucket.strides_[1] = reinterpret_cast<uint8_t*>(begin->_vals.bucket.strides_[0]) + adding;

		const_cast<int64_t&>(begin->_vals.bucket.stride_size) = 2;
		const_cast<int64_t&>(begin->_vals.bucket.bs) = bs;
		begin->_vals.bucket.strides_blocked = true;
		begin->dtype = dtype;
		begin->_vals.dtype = dtype;
		begin->_vals.bucket.dtype = dtype;
		begin->_total_size = splitting;
		begin->_vals.size = splitting;
		current += adding;
	}
	});
	if(r){
		Tensor* begin = reinterpret_cast<Tensor*>(array.data_ptr()) + div;
		begin->_vals.bucket.buckets_ = buckets_;
		begin->_vals.bucket.strides_ = intrusive_ptr<void*[]>(2);
		begin->_vals.bucket.strides_[0] = reinterpret_cast<uint8_t*>(*stride_begin()) + (div * splitting);
		begin->_vals.bucket.strides_[1] = *stride_end();

		const_cast<int64_t&>(begin->_vals.bucket.stride_size) = 2;
		const_cast<int64_t&>(begin->_vals.bucket.bs) = bs;
		begin->_vals.bucket.strides_blocked = true;
		begin->dtype = dtype;
		begin->_vals.dtype = dtype;
		begin->_vals.bucket.dtype = dtype;
		begin->_total_size = remainder;
		begin->_vals.size = remainder;
	}
	return std::move(array);
}
#endif



//this is when strides are not broken up into blocks
//with the devices adaptation really nothing to change here which is quite nice
template <>
std::vector<Bucket> Bucket::split_strided_<std::vector<Bucket>>(uint64_t splitting) const{
	int64_t msize = size();
	if(splitting > msize){
		return std::vector<Bucket > {*this};
	}
	uint64_t div = msize / splitting;
	uint64_t remainder = msize % splitting;
	bool r = remainder > 0;
	
	std::vector<Bucket > fb( (r) ? div + 1 : div, Bucket::makeNullBucket(dtype));
	uint64_t current = 0;
	/* const int64_t adding = dtype_s * splitting; */

	for(uint64_t i = 0; i < div; ++i){
		fb[i].buckets_ = buckets_;
		fb[i].strides_ = strides_ + current;
		const_cast<int64_t&>(fb[i].stride_size) = splitting;
		const_cast<int64_t&>(fb[i].bs) = bs;
		fb[i].strides_blocked = false;
		current += splitting;
	}
	if(r){
		int64_t end_size = msize - current;
		if(end_size == 0){return std::move(fb);}
		utils::THROW_EXCEPTION(end_size > 0, "Expected end size $ to be greater than zero, must be calculation issue", end_size);
		fb.back().buckets_ = buckets_;
		fb.back().strides_ = strides_ + current;
		const_cast<int64_t&>(fb.back().stride_size) = end_size;
		const_cast<int64_t&>(fb.back().bs) = bs;
		fb.back().strides_blocked = false;
	}

	return std::move(fb);
}

#ifndef USE_PARALLEL

template <>
Tensor Bucket::split_strided_<Tensor>(uint64_t splitting) const{
	int64_t msize = size();
	if(splitting > msize){
		Tensor output = Tensor::makeNullTensorArray(1);
		Tensor& t = output.item<Tensor>();
		t._vals.bucket = *this;
		t._vals.dtype = dtype;
		t._vals.size = size();
		t._total_size = size();
		t.dtype = dtype;
		t._size = SizeRef({t._total_size});
		return output;
	}
	uint64_t div = msize / splitting;
	uint64_t remainder = msize % splitting;
	bool r = remainder > 0;
	
	Tensor array = Tensor::makeNullTensorArray((r) ? div + 1 : div);
	uint64_t current = 0;
	/* const int64_t adding = dtype_s * splitting; */
	Tensor* begin = reinterpret_cast<Tensor*>(array.data_ptr());
	for(uint64_t i = 0; i < div; ++i, ++begin){
		begin->_vals.bucket.buckets_ = buckets_;
		begin->_vals.bucket.strides_ = strides_ + current;
		const_cast<int64_t&>(begin->_vals.bucket.stride_size) = splitting;
		const_cast<int64_t&>(begin->_vals.bucket.bs) = bs;
		begin->_vals.bucket.strides_blocked = false;
		begin->dtype = dtype;
		begin->_vals.dtype = dtype;
		begin->_vals.bucket.dtype = dtype;
		begin->_total_size = splitting;
		begin->_vals.size = splitting;
		current += splitting;
	}
	if(r){
		int64_t end_size = msize - current;
		if(end_size == 0){return std::move(array);}
		utils::THROW_EXCEPTION(end_size > 0, "Expected end size $ to be greater than zero, must be calculation issue", end_size);
		begin->_vals.bucket.buckets_ = buckets_;
		begin->_vals.bucket.strides_ = strides_ + current;
		const_cast<int64_t&>(begin->_vals.bucket.stride_size) = end_size;
		const_cast<int64_t&>(begin->_vals.bucket.bs) = bs;
		begin->_vals.bucket.strides_blocked = false;
		begin->dtype = dtype;
		begin->_vals.dtype = dtype;
		begin->_vals.bucket.dtype = dtype;
		begin->_total_size = end_size; 
		begin->_vals.size = end_size;

	}

	return std::move(array);
}

#else

template <>
Tensor Bucket::split_strided_<Tensor>(uint64_t splitting) const{
	int64_t msize = size();
	if(splitting > msize){
		Tensor output = Tensor::makeNullTensorArray(1);
		Tensor& t = output.item<Tensor>();
		t._vals.bucket = *this;
		t._vals.dtype = dtype;
		t._vals.size = size();
		t._total_size = size();
		t.dtype = dtype;
		t._size = SizeRef({t._total_size});
		return output;
	}
	uint64_t div = msize / splitting;
	uint64_t remainder = msize % splitting;
	bool r = remainder > 0;
	
	Tensor array = Tensor::makeNullTensorArray((r) ? div + 1 : div);
	tbb::parallel_for(utils::calculateGrainSize1D(0, div), [&](const auto range){
	uint64_t current = splitting * range.begin();
	/* const int64_t adding = dtype_s * splitting; */
	Tensor* begin = reinterpret_cast<Tensor*>(array.data_ptr()) + range.begin();
	Tensor* end = begin + (range.end() - range.begin());
	for(;begin != end; ++begin){
		begin->_vals.bucket.buckets_ = buckets_;
		begin->_vals.bucket.strides_ = strides_ + current;
		const_cast<int64_t&>(begin->_vals.bucket.stride_size) = splitting;
		const_cast<int64_t&>(begin->_vals.bucket.bs) = bs;
		begin->_vals.bucket.strides_blocked = false;
		begin->dtype = dtype;
		begin->_vals.dtype = dtype;
		begin->_vals.bucket.dtype = dtype;
		begin->_total_size = splitting;
		begin->_vals.size = splitting;
		current += splitting;
	}
	});

	if(r){
		uint64_t current = splitting * div;
		int64_t end_size = msize - current;
		if(end_size == 0){return std::move(array);}
		utils::THROW_EXCEPTION(end_size > 0, "Expected end size $ to be greater than zero, must be calculation issue", end_size);

		Tensor* begin = reinterpret_cast<Tensor*>(array.data_ptr()) + div;
		begin->_vals.bucket.buckets_ = buckets_;
		begin->_vals.bucket.strides_ = strides_ + current;

		const_cast<int64_t&>(begin->_vals.bucket.stride_size) = end_size;
		const_cast<int64_t&>(begin->_vals.bucket.bs) = bs;
		begin->_vals.bucket.strides_blocked = false;
		begin->dtype = dtype;
		begin->_vals.dtype = dtype;
		begin->_vals.bucket.dtype = dtype;
		begin->_total_size = end_size; 
		begin->_vals.size = end_size;
	}
	return std::move(array);
}

#endif


//this version is for when the bucket is broken up into buckets but not contiguous

template <>
std::vector<Bucket> Bucket::split_bucketed_<std::vector<Bucket>>(uint64_t splitting) const{
	int64_t msize = size();
	if(splitting > msize){
		return std::vector<Bucket > {*this};
	}
	uint64_t div = msize / splitting;
	uint64_t remainder = msize % splitting;
	bool r = remainder > 0;
	const std::size_t dtype_s = DTypeFuncs::size_of_dtype(dtype);
	
	std::vector<Bucket > fb( (r) ? div + 1 : div, makeCopyBucket(dtype, buckets_, true, bs, 0)); //the 0 ensures that the fb[i].strides_ are holding nullptr's so no memory is allocated
	splitting *= dtype_s;
	remainder *= dtype_s;

	void** c_strides = stride_begin();
	uint8_t* begin = reinterpret_cast<uint8_t*>(*c_strides);
	++c_strides;
	uint8_t* end = reinterpret_cast<uint8_t*>(*c_strides);

	for(uint64_t i = 0; i < fb.size(); ++i){
		std::ptrdiff_t distance = (end - begin);
		if(r && i == fb.size()-1){splitting = remainder;} //this accounts for the if(r)
		if(distance > splitting){
			fb[i].strides_ = intrusive_ptr<void*[]>(2);
			fb[i].strides_[0] = begin;
			begin += splitting;
			fb[i].strides_[1] = begin;
			const_cast<int64_t&>(fb[i].stride_size) = 2;
			/* const_cast<int64_t&>(fb[i].bs) = bs; */
			fb[i].strides_blocked = true;
			/* current += splitting; */
			continue;
		}
		if(distance == splitting){
			fb[i].strides_ = intrusive_ptr<void*[]>(2);
			fb[i].strides_[0] = begin;
			fb[i].strides_[1] = end;
			const_cast<int64_t&>(fb[i].stride_size) = 2;
			/* const_cast<int64_t&>(fb[i].bs) = bs; */
			fb[i].strides_blocked = true;
			if((i + 1) == fb.size()){break;}
			++c_strides;
			begin = reinterpret_cast<uint8_t*>(*c_strides);
			++c_strides;
			end = reinterpret_cast<uint8_t*>(*c_strides);
			/* current = 0; */
			continue;
		}
		//distance < splitting
		//new logic (feels cleaner and more exact):
		//fb[i].buckets_ = buckets_;
		//this so far is to keep track of the original information
		//the newStrides is to decide exactly how much memory in terms of new strides to allocate
		int64_t newStrides = 2;
		uint64_t left = splitting - distance;
		void** c_stride_cpy = c_strides;
		uint8_t* original_begin = begin;
		uint8_t* original_end = end;
		//the point of this loop is to iterate begin and end until it reaches the distance between the original begin and this end is greater than or equal to splitting
		//this allows for an easier way to keep track of where everything is in terms of iteration
		//and makes implementing an easy to follow and imoplement loop for whenever original_begin != begin
		while(distance < splitting){
			++c_stride_cpy;
			begin = reinterpret_cast<uint8_t*>(*c_stride_cpy);
			++c_stride_cpy;
			end = reinterpret_cast<uint8_t*>(*c_stride_cpy);
			newStrides += 2;
			distance += (end-begin);
			if(distance < splitting){left -= (end-begin);}
		}
		fb[i].strides_ = intrusive_ptr<void*[]>(newStrides); //make the new strides for the output bucked
		const_cast<int64_t&>(fb[i].stride_size) = newStrides; // log the size of the strides
		void** out_strides = fb[i].strides_.get(); // keep track of the strides of the output bucket (first begin)
		while(original_begin != begin){
			*out_strides = original_begin;
			++out_strides; //next end
			*out_strides = original_end;
			++out_strides; //next begin;
			++c_strides; //next begin;
			original_begin = reinterpret_cast<uint8_t*>(*c_strides);
			++c_strides; //next end
			original_end = reinterpret_cast<uint8_t*>(*c_strides);
			//on the end of this iteration, if original_begin == begin, then it just ends, and then begin is out_strides, and begin + left is (out_strides)
		}
		utils::throw_exception(c_strides == c_stride_cpy, "Expected c_strides and c_stride_cpy to be the same, but did math logic incorrect"); //this will be taken out, just for now
		*out_strides = begin;
		++out_strides; //next end
		if(distance == splitting){
			*out_strides = end;
			if((i+1) == fb.size()){break;}
			++c_strides;
			begin = reinterpret_cast<uint8_t*>(*c_strides);
			++c_strides;
			end = reinterpret_cast<uint8_t*>(*c_strides);
		}
		else{
			begin += left;
			*out_strides = begin;
		}

	}
	return std::move(fb);
}

#ifndef USE_PARALLEL

template <>
Tensor Bucket::split_bucketed_<Tensor>(uint64_t splitting) const{
	int64_t msize = size();
	if(splitting > msize){
		Tensor output = Tensor::makeNullTensorArray(1);
		Tensor& t = output.item<Tensor>();
		t._vals.bucket = *this;
		t._vals.dtype = dtype;
		t._vals.size = size();
		t._total_size = size();
		t.dtype = dtype;
		t._size = SizeRef({t._total_size});
		return output;
	}
	uint64_t div = msize / splitting;
	uint64_t remainder = msize % splitting;
	bool r = remainder > 0;
	const std::size_t dtype_s = DTypeFuncs::size_of_dtype(dtype);
	
	Tensor array = Tensor::makeNullTensorArray((r) ? div + 1 : div);
    const uint64_t splitting_tsize = splitting;
	splitting *= dtype_s;
	remainder *= dtype_s;
	
	void** c_strides = stride_begin();
	uint8_t* begin = reinterpret_cast<uint8_t*>(*c_strides);
	++c_strides;
	uint8_t* end = reinterpret_cast<uint8_t*>(*c_strides);

	Tensor* begin_t = reinterpret_cast<Tensor*>(array.data_ptr());
	Tensor* end_t = begin_t + array.numel();
	/* uint32_t counter = 0; */
	for(;begin_t != end_t; ++begin_t){
		/* std::cout << "for counter "<<counter<< " the current is "<<current<<" and the (bucket,stride) is "<<current_bucket_index<<','<<current_stride_index<<std::endl; */
		/* ++counter; */
		std::ptrdiff_t distance = (end - begin);
		if(r && begin_t + 1 == end_t){
            splitting = remainder; 
            splitting_tsize = remainder / dtype_s
        } //this accounts for the if(r)
		//the below lines have to be run no matter what for every iteration of the tensors, made sense to put them up front for readability
		//also would be easiest to change them only once
		begin_t->_vals.bucket.buckets_ = buckets_;
		begin_t->_total_size = splitting_tsize;
		begin_t->_vals.size = splitting_tsize;
		begin_t->dtype = dtype;
		begin_t->_vals.dtype = dtype;
		begin_t->_vals.bucket.dtype = dtype;
		begin_t->_vals.bucket.strides_blocked = true;
		const_cast<int64_t&>(begin_t->_vals.bucket.bs) = bs;
	
		if(distance > splitting){
			begin_t->_vals.bucket.strides_ = intrusive_ptr<void*[]>(2);
			begin_t->_vals.bucket.strides_[0] = begin;
			begin += splitting;
			begin_t->_vals.bucket.strides_[1] = begin;
			const_cast<int64_t&>(begin_t->_vals.bucket.stride_size) = 2;
			/* current += splitting; */
			continue;
		}
		if(distance == splitting){
			begin_t->_vals.bucket.strides_ = intrusive_ptr<void*[]>(2);
			begin_t->_vals.bucket.strides_[0] = begin;
			begin_t->_vals.bucket.strides_[1] = end;
			const_cast<int64_t&>(begin_t->_vals.bucket.stride_size) = 2;
			if((begin_t + 1) == end_t){break;}
			++c_strides;
			begin = reinterpret_cast<uint8_t*>(*c_strides);
			++c_strides;
			end = reinterpret_cast<uint8_t*>(*c_strides);
			/* current = 0; */
			continue;
		}
		//distance < splitting
		//
		//
		//new logic (feels cleaner and more exact):
		//this so far is to keep track of the original information
		//the newStrides is to decide exactly how much memory in terms of new strides to allocate
		int64_t newStrides = 2;
		uint64_t left = splitting - distance;
		void** c_stride_cpy = c_strides;
		uint8_t* original_begin = begin;
		uint8_t* original_end = end;
		//the point of this loop is to iterate begin and end until it reaches the distance between the original begin and this end is greater than or equal to splitting
		//this allows for an easier way to keep track of where everything is in terms of iteration
		//and makes implementing an easy to follow and imoplement loop for whenever original_begin != begin
		while(distance < splitting){
			++c_stride_cpy;
			begin = reinterpret_cast<uint8_t*>(*c_stride_cpy);
			++c_stride_cpy;
			end = reinterpret_cast<uint8_t*>(*c_stride_cpy);
			newStrides += 2;
			distance += (end-begin);
			if(distance < splitting){left -= (end-begin);}
		}
		begin_t->_vals.bucket.strides_ = intrusive_ptr<void*[]>(newStrides); //make the new strides for the output bucked
		const_cast<int64_t&>(begin_t->_vals.bucket.stride_size) = newStrides; // log the size of the strides
		void** out_strides = begin_t->_vals.bucket.strides_.get(); // keep track of the strides of the output bucket (first begin)
		while(original_begin != begin){
			*out_strides = original_begin;
			++out_strides; //next end
			*out_strides = original_end;
			++out_strides; //next begin;
			++c_strides; //next begin;
			original_begin = reinterpret_cast<uint8_t*>(*c_strides);
			++c_strides; //next end
			original_end = reinterpret_cast<uint8_t*>(*c_strides);
			//on the end of this iteration, if original_begin == begin, then it just ends, and then begin is out_strides, and begin + left is (out_strides)
		}
		utils::throw_exception(c_strides == c_stride_cpy, "Expected c_strides and c_stride_cpy to be the same, but did math logic incorrect"); //this will be taken out, just for now
		*out_strides = begin;
		++out_strides; //next end
		if(distance == splitting){
			*out_strides = end;
			if((begin_t + 1) == end_t){break;}
			++c_strides;
			begin = reinterpret_cast<uint8_t*>(*c_strides);
			++c_strides;
			end = reinterpret_cast<uint8_t*>(*c_strides);
		}
		else{
			begin += left;
			*out_strides = begin;
		}
		

	}
	return std::move(array);
}

#else


template <>
Tensor Bucket::split_bucketed_<Tensor>(uint64_t splitting) const{
	int64_t msize = size();
	if(splitting > msize){
		Tensor output = Tensor::makeNullTensorArray(1);
		Tensor& t = output.item<Tensor>();
		t._vals.bucket = *this;
		t._vals.dtype = dtype;
		t._vals.size = size();
		t._total_size = size();
		t.dtype = dtype;
		t._size = SizeRef({t._total_size});
		return output;
	}
	const uint64_t div = msize / splitting;
	uint64_t remainder = msize % splitting;
	const bool r = remainder > 0;
	const std::size_t dtype_s = DTypeFuncs::size_of_dtype(dtype);
	
	Tensor array = Tensor::makeNullTensorArray((r) ? div + 1 : div);
    Tensor* array_end = reinterpret_cast<Tensor*>(array.data_ptr_end());

    const uint64_t splitting_tsize = splitting;
    const uint64_t remainder_tsize = remainder;
	splitting *= dtype_s;
	remainder *= dtype_s;
	static tbb::mutex printMutex;
	tbb::parallel_for(utils::calculateGrainSize1D(0, array.numel()), [&](const auto range){
	
	uint64_t current = splitting * range.begin();

	void** c_strides = stride_begin();
	uint8_t* begin = reinterpret_cast<uint8_t*>(*c_strides);
	++c_strides;
	uint8_t* end = reinterpret_cast<uint8_t*>(*c_strides);

	std::ptrdiff_t distance = (end-begin);
	while(distance < current){
		if(distance >= current)
			break;
		current -= distance;
		++c_strides;
		begin = reinterpret_cast<uint8_t*>(*c_strides);
		++c_strides;
		end = reinterpret_cast<uint8_t*>(*c_strides);
		distance = (end - begin);
	}
	if(distance == current){
		++c_strides;
		begin = reinterpret_cast<uint8_t*>(*c_strides);
		++c_strides;
		end = reinterpret_cast<uint8_t*>(*c_strides);	
	}
	else{
		begin += current;
	}

	Tensor* begin_t = reinterpret_cast<Tensor*>(array.data_ptr()) + range.begin();
	Tensor* end_t = begin_t + (range.end() - range.begin());
	for(;begin_t != end_t; ++begin_t){
		std::ptrdiff_t distance = (end - begin);
        int64_t cur_splitting = (r && begin_t + 1 == array_end) ? 
                remainder : splitting;
        int64_t cur_splitting_t = (r && begin_t + 1 == array_end) ? 
                remainder_tsize : splitting_tsize;
        //this accounts for the if(r)



		//the below lines have to be run no matter what for every iteration of the tensors, made sense to put them up front for readability
		//also would be easiest to change them only once
		begin_t->_vals.bucket.buckets_ = buckets_;
		begin_t->_total_size = cur_splitting_t;
		begin_t->_vals.size = cur_splitting_t;
		begin_t->dtype = dtype;
		begin_t->_vals.dtype = dtype;
		begin_t->_vals.bucket.dtype = dtype;
		begin_t->_vals.bucket.strides_blocked = true;
		const_cast<int64_t&>(begin_t->_vals.bucket.bs) = bs;
	
		if(distance > cur_splitting){
			begin_t->_vals.bucket.strides_ = intrusive_ptr<void*[]>(2);
			begin_t->_vals.bucket.strides_[0] = begin;
			begin += cur_splitting;
			begin_t->_vals.bucket.strides_[1] = begin;
			const_cast<int64_t&>(begin_t->_vals.bucket.stride_size) = 2;
			/* current += splitting; */
			continue;
		}
		if(distance == cur_splitting){
			begin_t->_vals.bucket.strides_ = intrusive_ptr<void*[]>(2);
			begin_t->_vals.bucket.strides_[0] = begin;
			begin_t->_vals.bucket.strides_[1] = end;
			const_cast<int64_t&>(begin_t->_vals.bucket.stride_size) = 2;
			if((begin_t + 1) == end_t){break;}
			++c_strides;
			begin = reinterpret_cast<uint8_t*>(*c_strides);
			++c_strides;
			end = reinterpret_cast<uint8_t*>(*c_strides);
			/* current = 0; */
			continue;
		}
		//distance < splitting
		//
		//
		//new logic (feels cleaner and more exact):
		//this so far is to keep track of the original information
		//the newStrides is to decide exactly how much memory in terms of new strides to allocate
		int64_t newStrides = 2;
		uint64_t left = cur_splitting - distance;
		void** c_stride_cpy = c_strides;
		uint8_t* original_begin = begin;
		uint8_t* original_end = end;
		//the point of this loop is to iterate begin and end until it reaches the distance between the original begin and this end is greater than or equal to splitting
		//this allows for an easier way to keep track of where everything is in terms of iteration
		//and makes implementing an easy to follow and imoplement loop for whenever original_begin != begin
		while(distance < cur_splitting){
			// Update 'begin' and 'end' to the next strides
			begin = reinterpret_cast<uint8_t*>(*(++c_stride_cpy));
			end = reinterpret_cast<uint8_t*>(*(++c_stride_cpy));

			// Update stride count and total distance
			newStrides += 2;
			distance += (end - begin);

			//update the remaining distance
			if(distance < cur_splitting){left -= (end-begin);}
		}
		begin_t->_vals.bucket.strides_ = intrusive_ptr<void*[]>(newStrides); //make the new strides for the output bucked
		const_cast<int64_t&>(begin_t->_vals.bucket.stride_size) = newStrides; // log the size of the strides
		void** out_strides = begin_t->_vals.bucket.strides_.get(); // keep track of the strides of the output bucket (first begin)
		while (original_begin != begin) {
			// Set current stride start and end
			*out_strides++ = original_begin;
			*out_strides++ = original_end;

			// Move to the next pair of strides
			original_begin = reinterpret_cast<uint8_t*>(*(++c_strides));
			original_end = reinterpret_cast<uint8_t*>(*(++c_strides));

			// Comment: If original_begin == begin, the loop will exit,
			// and begin will be set to out_strides with (out_strides + left) as its end
		}
		//for debugging purposes
		utils::throw_exception(*c_strides == *c_stride_cpy, "Expected c_strides and c_stride_cpy to be the same, but did math logic incorrect"); //this will be taken out, just for now
		*out_strides = begin;
		++out_strides; //next end
		if(distance == cur_splitting){
			*out_strides = end;
			if((begin_t + 1) == end_t){break;}
			begin = reinterpret_cast<uint8_t*>(*(++c_strides));
			end = reinterpret_cast<uint8_t*>(*(++c_strides));
		}
		else{
			begin += left;
			*out_strides = begin;
		}	

	}});
	return std::move(array);
}

#endif

template <>
std::vector<Bucket> Bucket::split<std::vector<Bucket>>(uint64_t splitting) const {
	if(!strides_blocked){return split_strided_<std::vector<Bucket>>(splitting);}
	if(is_contiguous()){return split_contiguous_<std::vector<Bucket>>(splitting);}
	return split_bucketed_<std::vector<Bucket>>(splitting);
}

template <>
Tensor Bucket::split<Tensor>(uint64_t splitting) const {
	if(!strides_blocked){return split_strided_<Tensor>(splitting);}
	if(is_contiguous()){return split_contiguous_<Tensor>(splitting);}
	return split_bucketed_<Tensor>(splitting);
}

void Bucket::swap(Bucket& other){
	buckets_.swap(other.buckets_);
	strides_.swap(other.strides_);
	std::swap(const_cast<int64_t&>(bs), const_cast<int64_t&>(other.bs));
	std::swap(const_cast<int64_t&>(stride_size), const_cast<int64_t&>(other.stride_size));
	std::swap(strides_blocked, other.strides_blocked);
	std::swap(dtype, other.dtype);
	
}

Bucket Bucket::bucket_all_indices() const{
	if(!strides_blocked){return *this;}
	const uint32_t dtype_s = DTypeFuncs::size_of_dtype(dtype);
	const int64_t n_stride_size = size();
	nt::intrusive_ptr<void*[]> nStrides(n_stride_size);
	void** arr = nStrides.get();
	for(uint64_t i = 0; i < stride_size; ++i){
		uint8_t* begin = reinterpret_cast<uint8_t*>(*(stride_begin() + i));
		++i;
		uint8_t* end = reinterpret_cast<uint8_t*>(*(stride_begin() +i));
		for(;begin != end; begin += dtype_s, ++arr){
			*arr = (void*)begin;
		}
	}
	return Bucket(buckets_, std::move(nStrides), n_stride_size, bs, false, dtype);
}

Bucket Bucket::copy_strides() const {
	if(strides_blocked){return bucket_all_indices();}
	const int64_t n_stride_size = size();
	nt::intrusive_ptr<void*[]> nStrides(n_stride_size);
	void** arr = nStrides.get();
	void** current_strides = stride_begin();
	void** current_end = stride_end();
	for(;current_strides != current_end; ++current_strides, ++arr){
		uint8_t* begin = reinterpret_cast<uint8_t*>(*current_strides);
		*arr = (void*)begin;
	}
	return Bucket(buckets_, std::move(nStrides), n_stride_size, bs, false, dtype);
}



Bucket Bucket::catV(const std::vector<Bucket>& buckets){
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
 
Bucket Bucket::catV(const std::vector<std::reference_wrapper<const Bucket> >& buckets){
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


/* Bucket Bucket::FromShared(intrusive_ptr<void[]> ptr, uint64_t s, DType dt){ */
/* 	Bucket output(s, dt, dCPU); */
/* 	void* mem = output.data_ptr(); */
/* 	std::memcpy(mem, ptr.get(), s * DTypeFuncs::size_of_dtype(dt)); */
/* 	return std::move(output); */
/* } */

}
