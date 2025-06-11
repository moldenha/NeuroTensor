#ifndef _NT_ITERATOR_PARALLEL_FOR_NO_PARALLEL_HPP_
#define _NT_ITERATOR_PARALLEL_FOR_NO_PARALLEL_HPP_

namespace nt{
namespace threading{
namespace detail{
//NOTE: this assumes going into it, that it is already at the start position
template<typename UnaryFunction, typename T>
inline BucketIterator_blocked<T> iterator_parallel_for_one_stride(UnaryFunction&& in_func, const int64_t& start, const int64_t& end, BucketIterator_blocked<T>& it){
	BucketIterator_blocked<T> begin = it;
	BucketIterator_blocked<T> finish = it + (end);
	uint64_t diff = block_diff(begin, finish);
	std::vector<std::pair<T*, T*>> bounds;
	bounds.reserve(diff);
	while(!same_block(begin, finish)){
		bounds.push_back({(T*)begin, begin.block_end()});
		begin.iterate_next_block();
	}
	bounds.push_back({(T*)begin, finish.block_end()});

	for (const auto& [block_start, block_end] : bounds) {
		T* begin_ptr = block_start;
		uint64_t block_start_index = block_start - (T*)it;
		uint64_t block_end_index = block_end - (T*)it;
		uint64_t block_size = block_end_index - block_start_index;
		in_func(blocked_range<1>::make_range(block_start_index, block_end_index), begin_ptr);
	}
	return finish;
}


template<typename UnaryFunction, typename T>
inline void iterator_parallel_for_1d_send_pointers(UnaryFunction&& in_func, uint64_t& block_start_index, const uint64_t& block_size, const int64_t& stride, T* start_ptr){
	uint64_t block_end_index = block_start_index + block_size;
	in_func(blocked_range<1>::make_range(block_start_index, block_end_index), start_ptr);
	block_start_index = block_end_index;
	return;
}

}


//min_block_size is held as an argument for the non parallel version purely because of parallel to non-parallel compatibility
template<typename UnaryFunction, typename T>
inline T* iterator_parallel_for(UnaryFunction&& in_func, const int64_t start, const int64_t end, T* it, int64_t stride=1, uint64_t min_block_size=20){
	in_func(blocked_range<1>::make_range(start, end), it);
	return it + ((end - start) * stride);

}

template<typename UnaryFunction, typename T>
inline BucketIterator_list<T> iterator_parallel_for(UnaryFunction&& in_func, const int64_t start, const int64_t end, BucketIterator_list<T>& it, int64_t stride=1, uint64_t min_block_size=20){
	in_func(blocked_range<1>::make_range(start, end), it);
	return it + ((end - start) * stride);
}

template<typename UnaryFunction, typename T>
inline BucketIterator_blocked<T> iterator_parallel_for(UnaryFunction&& in_func, const int64_t start, const int64_t end, BucketIterator_blocked<T>& it, int64_t stride=1, uint64_t min_block_size=20){
	if(it.block_size() >= (end - start) * stride){
		iterator_parallel_for(std::forward<UnaryFunction&&>(in_func), start, end, (T*)it, stride);
		return it + (end * stride);
	}
	if(stride == 1){
		return detail::iterator_parallel_for_one_stride(std::forward<UnaryFunction&&>(in_func), start, end, it);
	}

	/* int64_t max_concurrency = std::max<int64_t>(1, (end-start) / (utils::getThreadsPerCore() * 10)); */
	BucketIterator_blocked<T> begin = it;
	BucketIterator_blocked<T> finish = it + (end * stride);
	uint64_t diff = block_diff(begin, finish);
	std::vector<std::pair<T*, T*>> bounds;
	std::vector<BucketIterator_blocked<T>> iterators;
	bounds.reserve(diff);
	iterators.reserve(diff);
	while(!same_block(begin, finish)){
		iterators.push_back(begin);
		bounds.push_back({(T*)begin, begin.block_end()});
		begin.iterate_next_block();
	}
	bounds.push_back({(T*)begin, finish.block_end()});
	iterators.push_back(finish);

	uint64_t last_start = 0;
	uint64_t block_start_index = start;
	for(uint64_t i = 0; i < iterators.size(); ++i){
		const auto& [begin_ptr, end_ptr] = bounds[i];
		std::ptrdiff_t ptr_diff = end_ptr - begin_ptr;
		if(last_start > 0){
			if(ptr_diff <= last_start){last_start -= ptr_diff;continue;}
		}
		T* start_ptr = begin_ptr + last_start;
		ptr_diff = end_ptr - start_ptr;
		uint64_t r = (ptr_diff  % stride);
		/* uint64_t block_size = (r == 0) ? 0 : 1; //if the remainder is zero then it will iterate through all the possible strides, */
						   //otherwise there will be the 0th case not accounted for
		/* last_start = (r == 0) ? 0 : stride - r; */
		uint64_t block_size = ptr_diff / stride;
		detail::iterator_parallel_for_1d_send_pointers(in_func, block_start_index, block_size, stride, start_ptr);
		if(r == 0){last_start = 0; continue;}
		//this is for when there is one left and a blocked iterator is needed
		BucketIterator_blocked<T> bucket = iterators[i] + (block_size * stride) + last_start;
		in_func(blocked_range<1>::make_range(block_start_index, block_start_index + 1), bucket);
		block_start_index += 1;
		last_start = stride - r;
	}
	return finish;
}


}
}

#endif //_NT_ITERATOR_PARALLEL_FOR_NO_PARALLEL_HPP_ 
