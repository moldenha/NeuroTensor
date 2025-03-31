/*
 *
 * the entire point of this header file is to take an iterator pointing to a multidimensional object (a tensor) and optimize for loops on it
 * because this library has 3 different iterator types, and of course different strides are supported
 *  -regular pointer (T*)
 *  -list all bucketed pointer (BucketIterator_list<T>, T**)
 *  -and an iterator when it is multiple buckets of contiguous memory (BucketIterator_blocked<T>)
 *
 * BucketIterator_blocked<T> has a lot of overhead when compared to T*,
 * this is meant to reduce that
 * a lot of this logic can be translated to handling cuda kernels especially when memory is bucketed
 * will probably make a similar wrapper of this when cuda support is added
 * will make an std only wrapper (which is just having to make a task group as I already have an std::parallel_for equivalent to tbb::parallel_for
 * also going to make an accelerate wrapper for Accelerate, for Apple ARM cores
 *
 * In the future I will adapt this to add 2d, 3d, and 4d support (you can see commented out lines where there was untested 2d support)
 *
 * I would also like to add dual iterator support
 * this just happens to not be at the top of my to-do list right now
 * and this was honestly a little one day code side quest
 *
 *
 */



#ifndef _NT_ITERATOR_PARALLEL_FOR_PARALLEL_HPP_
#define _NT_ITERATOR_PARALLEL_FOR_PARALLEL_HPP_
#include "Threading.h"


namespace nt{
namespace threading{

namespace detail{


//NOTE: this assumes going into it, that it is already at the start position
template<typename UnaryFunction, typename T>
inline BucketIterator_blocked<T> iterator_parallel_for_one_stride(UnaryFunction&& in_func, const int64_t& start, const int64_t& end, BucketIterator_blocked<T>& it, uint64_t min_block_size=20){
	int64_t max_concurrency = std::max<int64_t>(1, (end-start) / (utils::getThreadsPerCore() * 10));
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

	/* int64_t max_concurrency = tbb::this_task_arena::max_concurrency(); */
	tbb::task_group tg;
	for (const auto& [block_start, block_end] : bounds) {
		T* begin_ptr = block_start;
		uint64_t block_start_index = block_start - (T*)it;
		uint64_t block_end_index = block_end - (T*)it;
		uint64_t block_size = block_end_index - block_start_index;
		if(block_end_index - block_start_index <= min_block_size){
			tg.run([&in_func, begin_ptr, block_start_index, block_end_index] {
			    in_func(blocked_range<1>::make_range(block_start_index, block_end_index), begin_ptr);
			});
			continue;
		}
		uint64_t num_chunks = (min_block_size != 0) ?
			std::max<uint64_t>(1, std::min<uint64_t>(block_size / min_block_size, max_concurrency))
			: std::max<int64_t>(1, int64_t(block_size / max_concurrency));
		max_concurrency -= num_chunks;
		uint64_t chunk_size = std::max(min_block_size, block_size / num_chunks);
		for (uint64_t i = 0; i < block_size; i += chunk_size) {
			uint64_t chunk_end = std::min(i + chunk_size, block_size);
			tg.run([&in_func, begin_ptr, i, chunk_end] {
				in_func(blocked_range<1>::make_range(i, chunk_end), begin_ptr + i);
			});
		}
	}
	tg.wait();
	return finish;
}



template<typename UnaryFunction, typename T>
inline void iterator_parallel_for_1d_send_pointers(UnaryFunction&& in_func, uint64_t& block_start_index, const uint64_t& block_size, const int64_t& stride, T* start_ptr, const uint64_t& min_block_size, tbb::task_group& tg, const uint64_t& max_concurrency){
	uint64_t block_end_index = block_start_index + block_size;
	if(block_size <= min_block_size){
		tg.run([&in_func, block_start_index, block_end_index, start_ptr] {
		    in_func(blocked_range<1>::make_range(block_start_index, block_end_index), start_ptr);
		});
		block_start_index = block_end_index;
		return;
	}
	uint64_t num_chunks = (min_block_size != 0) ?
		std::max<uint64_t>(1, std::min<uint64_t>(block_size / min_block_size, max_concurrency))
		: std::max<int64_t>(1, int64_t(block_size / max_concurrency));

	uint64_t chunk_size = std::max(min_block_size, block_size / num_chunks);
	uint64_t individual_adding = chunk_size * stride;
	//num_chunks = block_size / chunk_size
	/* max_concurrency = std::max<int64_t>(0, max_concurrency - (block_size / chunk_size)); */

	for (uint64_t i = 0; i < block_size; i += chunk_size) {
		uint64_t chunk_end = std::min(i + chunk_size, block_size);
		tg.run([&in_func, start_ptr, i, chunk_end] {
			in_func(blocked_range<1>::make_range(i, chunk_end), start_ptr);
		});
		if(chunk_end < block_size)
			start_ptr += individual_adding;
	}
	block_start_index = block_end_index;
}


/* template<typename UnaryFunction, typename T> */
/* inline uint64_t iterator_parallel_for_2d_send_pointers_1d_n_2d_0(UnaryFunction&& in_func, uint64_t& _1d_block_start_index, const uint64_t& _2d_block_start_index */ 
/* 					const uint64_t _1d_block_size, //const uint64_t& _2d_block_size = 0 */
/* 					const int64_t& stride_1d, T* start_ptr, const uint64_t& min_block_size, tbb::task_group& tg, const uint64_t& max_concurrency){ */
/* 	uint64_t _1d_block_end_index = _1d_block_start_index + _1d_block_size; */
/* 	uint64_t _2d_block_end_index = _2d_block_start_index + 1; //just to allow the for loop to move forward */
/* 								  //as this is meant to be used with for loops */
/* 	const uint64_t block_size = _1d_block_size; */
/* 	if(block_size <= min_block_size){ */
/* 		tg.run([&in_func, _1d_block_start_index, _1d_block_end_index, */ 
/* 				_2d_block_start_index, _2d_block_end_index, start_ptr] { */
/* 		    in_func(blocked_range<2>::make_range(_2d_block_start_index, _2d_block_end_index, _1d_block_start_index, _1d_block_end_index), start_ptr); */
/* 		}); */
/* 		_1d_block_start_index = _1d_block_end_index; */
/* 		return block_size; */
/* 	} */
/* 	uint64_t num_chunks = (min_block_size != 0) ? */
/* 		std::max<uint64_t>(1, std::min<uint64_t>(block_size / min_block_size, max_concurrency)) */
/* 		: std::max<int64_t>(1, int64_t(block_size / max_concurrency)); */

/* 	uint64_t chunk_size = std::max(min_block_size, block_size / num_chunks); */
/* 	uint64_t individual_adding = chunk_size * stride_1d; */
/* 	//num_chunks = block_size / chunk_size */
/* 	/1* max_concurrency = std::max<int64_t>(0, max_concurrency - (block_size / chunk_size)); *1/ */

/* 	for (uint64_t i = 0; i < block_size; i += chunk_size) { */
/* 		uint64_t chunk_end = std::min(i + chunk_size, block_size); */
/* 		tg.run([&in_func, start_ptr, i, chunk_end, _2d_block_start_index] { */
/* 			in_func(blocked_range<2>::make_range(_2d_block_start_index, _2d_block_start_index + 1, i, chunk_end), start_ptr); */
/* 		}); */
/* 		if(chunk_end < block_size) */
/* 			start_ptr += individual_adding; */
/* 	} */
/* 	_1d_block_start_index = _1d_block_end_index; */
/* 	return block_size; */
/* } */

/* //this is meant to handle multiple 2d buckets */
/* //this assumes that all of the previous and outlying 1d pointers were handeled before or will be handeled after */
/* template<typename UnaryFunction, typename T> */
/* inline uint64_t iterator_parallel_for_2d_send_pointers_1d_n_2d_n(UnaryFunction&& in_func, const int64_t start_1d, */ 
/* 		const uint64_t& _1d_block_end_index, const uint64_t& _2d_block_start_index */ 
/* 					const uint64_t& _1d_block_size, const uint64_t& _2d_block_size, */
/* 					const int64_t& stride_1d, const int64_t& stride_2d, T* start_ptr, const uint64_t& min_block_size, tbb::task_group& tg, const uint64_t& max_concurrency){ */
/* 	uint64_t _1d_size = (_1d_block_end_index - start_1d); */
/* 	uint64_t _1d_block_start_index = start_1d; */
/* 	uint64_t _2d_block_end_index = _2d_block_start_index + _2d_block_size; //just to allow the for loop to move forward */
/* 								  //as this is meant to be used with for loops */
/* 	const uint64_t block_size = _2d_block_size * _1d_size; */
/* 	if(block_size <= min_block_size){ */
/* 		tg.run([&in_func, _1d_block_start_index, _1d_block_end_index, */ 
/* 				_2d_block_start_index, _2d_block_end_index, start_ptr] { */
/* 		    in_func(blocked_range<2>::make_range(_2d_block_start_index, _2d_block_end_index, _1d_block_start_index, _1d_block_end_index), start_ptr); */
/* 		}); */
/* 		_2d_block_start_index = _2d_block_end_index; */
/* 		return block_size; */
/* 	} */
/* 	uint64_t num_chunks = (min_block_size != 0) ? */
/* 		std::max<uint64_t>(1, std::min<uint64_t>(block_size / min_block_size, max_concurrency)) */
/* 		: std::max<int64_t>(1, int64_t(block_size / max_concurrency)); */
	
/* 	//I need the number of chunks to be such that block_size / num_chunks is divisible by _1d_block_end_index */
/* 	uint64_t chunk_size = std::max(min_block_size, block_size / num_chunks); */
/* 	if(chunk_size % _1d_size != 0){ */
/* 		if(chunk_size < _1d_block_end_index){chunk_size = _1d_size;} */
/* 		else{chunk_size += (chunk_size % _1d_size;} */
/* 	} */
/* 	uint64_t individual_adding = (chunk_size / _1d_size) * stride_2d; */
/* 	//num_chunks = block_size / chunk_size */
/* 	/1* max_concurrency = std::max<int64_t>(0, max_concurrency - (block_size / chunk_size)); *1/ */
/* 	uint64_t 2d_adds = (chunk_size / _1d_size); //this is the amount to iterate _2d_block_start_index by and then end per iteration */
/* 	uint64_t individual_adding = 2d_ads * stride_2d; //this is the amount to iterate start_ptr by */
/* 	uint64_t 2d_start = _2d_block_start_index; */
/* 	uint64_t 2d_end = 2d_start + 2d_adds; */
/* 	for(uint64_t i = 0; i < block_size; i += chunk_size){ */
/* 		tg.run([&in_func, start_ptr, 2d_start, 2d_end, _1d_block_end_index]{ */
/* 			in_func(blocked_range<2>::make_range(2d_start, 2d_end, 0, _1d_block_end_index), start_ptr); */
/* 		}); */
/* 		if(i + chuk_size < block_size){ */
/* 			start_ptr += individual_adding; */
/* 		} */
/* 	} */
/* 	_2d_block_start_index = _2d_block_end_index; */
/* 	return block_size; */
/* } */




}

//the stride is basically how many iterator elements are between each individual index
//for example:
//for(int i = 0; i < 10; ++i, it += stride)
//this is extremely common,
//but will be significantly more useful when it gets into 2D and 3D versions of this
template<typename UnaryFunction, typename T>
inline T* iterator_parallel_for(UnaryFunction&& in_func, const int64_t start, const int64_t end, T* it, int64_t stride=1, uint64_t min_block_size=20){
	int64_t grain_size = std::max<int64_t>(1, (end-start) / (utils::getThreadsPerCore() * 10));
	tbb::parallel_for(tbb::blocked_range<int64_t>(start, end, grain_size),
				[&in_func, &it, &stride](const tbb::blocked_range<int64_t>& r){
					in_func(blocked_range<1>::make_range(r), (it + r.begin() * stride));
				});
	return it + ((end - start) * stride);

}

template<typename UnaryFunction, typename T>
inline BucketIterator_list<T> iterator_parallel_for(UnaryFunction&& in_func, const int64_t start, const int64_t end, BucketIterator_list<T>& it, int64_t stride=1, uint64_t min_block_size=20){
	int64_t grain_size = std::max<int64_t>(1, (end-start) / (utils::getThreadsPerCore() * 10));
	tbb::parallel_for(tbb::blocked_range<int64_t>(start, end),
				[&in_func, &it, &stride](const tbb::blocked_range<int64_t>& r){
					in_func(blocked_range<1>::make_range(r), (it + r.begin() * stride));
				});
	return it + ((end - start) * stride);
}


/*

template<typename UnaryFunction, typename T>
void iterator_parallel_for(UnaryFunction&& in_func, int64_t start, int64_t end, T* it, int64_t stride=1){
    tbb::task_group tg;
    int64_t max_concurrency = tbb::this_task_arena::max_concurrency();
    int64_t chunk_size = (end - start) / max_concurrency;
    for (int64_t i = start; i < end; i += chunk_size) {
        int64_t chunk_end = std::min(i + chunk_size, end);
        tg.run([&in_func, &it, stride, i, chunk_end] {
            in_func(tbb::blocked_range<int64_t>(i, chunk_end), it + i * stride);
        });
    }
    tg.wait();
}

 */
//inline void iterator_parallel_for_1d_send_pointers(UnaryFunction&& in_func, uint64_t& block_start_index, const int64_t& block_size, const int64_t& stride, T* start_ptr, const uint64_t& min_block_size, tbb::task_group& tg, const uint64_t& max_concurrency){

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

	int64_t max_concurrency = tbb::this_task_arena::max_concurrency();
	tbb::task_group tg;
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
		detail::iterator_parallel_for_1d_send_pointers(in_func, block_start_index, block_size, stride, start_ptr, min_block_size, tg, max_concurrency);
		if(r == 0){last_start = 0; continue;}
		//this is for when there is one left and a blocked iterator is needed
		BucketIterator_blocked<T> bucket = iterators[i] + (block_size * stride) + last_start;
		tg.run([&in_func, bucket, block_start_index] {
			in_func(blocked_range<1>::make_range(block_start_index, block_start_index+1), bucket);
		});
		block_start_index += 1;
		last_start = stride - r;
	}
	tg.wait();
	return finish;
}


/*//rows_begin, rows_end, cols_begin, cols_end*/
/*template<typename UnaryFunction, typename T>*/
/*inline void iterator_parallel_for_2d(UnaryFunction&& in_func, const int64_t start_2d, const int64_t end_2d, const int64_t start_1d, const int64_t end_1d, T* it, int64_t stride_2d, int64_t stride_1d, uint64_t min_block_size=20){*/
/*	 int64_t grain_size = std::max<int64_t>(1, (end-start) / (utils::getThreadsPerCore() * 10)); */
/*	utils::THROW_EXCEPTION(stride_1d * (end_1d - start_1d) == stride_2d, "When iterating parallel for a 2d stride, and the size of the collumns is $ from $ to $ when multiplied by the 1d stride $ such is supposed to be equal to the 2d stride ($) but got ($ * $) = $ != $", (end_1d-start_1d) start_1d, end_1d, stride_1d, stride_2d, (end_1d-start_1d), stride_1d, stride_1d * (end_1d - start_1d), stride_2d);*/

/*	tbb::parallel_for(utils::calculateGrainSize2D(start_2d, end_2d, start_1d, end_1d),*/
/*				[&in_func, &it, &stride_2d, &stride_1d](const tbb::blocked_range2d<int64_t>& r){*/
					
/*					in_func(blocked_range<2>::make_range(r), (it + ((r.rows().begin() * stride_2d) + r.cols().begin() * stride_1d)));*/
/*				});*/

/*}*/

/*//rows_begin, rows_end, cols_begin, cols_end*/
/*template<typename UnaryFunction, typename T>*/
/*inline void iterator_parallel_for_2d(UnaryFunction&& in_func, const int64_t start_2d, const int64_t end_2d, const int64_t start_1d, const int64_t end_1d, BucketIterator_list<T>& it, int64_t stride_2d, int64_t stride_1d, uint64_t min_block_size=20){*/
/*	 int64_t grain_size = std::max<int64_t>(1, (end-start) / (utils::getThreadsPerCore() * 10)); */
/*	utils::THROW_EXCEPTION(stride_1d * (end_1d - start_1d) == stride_2d, "When iterating parallel for a 2d stride, and the size of the collumns is $ from $ to $ when multiplied by the 1d stride $ such is supposed to be equal to the 2d stride ($) but got ($ * $) = $ != $", (end_1d-start_1d) start_1d, end_1d, stride_1d, stride_2d, (end_1d-start_1d), stride_1d, stride_1d * (end_1d - start_1d), stride_2d);*/
/*	tbb::parallel_for(utils::calculateGrainSize2D(start_2d, end_2d, start_1d, end_1d),*/
/*				[&in_func, &it, &stride_2d, &stride_1d](const tbb::blocked_range2d<int64_t>& r){*/
/*					in_func(blocked_range<2>::make_range(r), ((r.rows().begin() * stride_2d) + r.cols().begin() * stride_1d));*/
/*				});*/

/*}*/

/**/
/* * inline uint64_t iterator_parallel_for_2d_send_pointers_1d_n_2d_0(UnaryFunction&& in_func, uint64_t& _1d_block_start_index, const uint64_t& _2d_block_start_index*/ 
/*					const uint64_t _1d_block_size, //const uint64_t& _2d_block_size = 0*/
/*					const int64_t& stride_1d, T* start_ptr, const uint64_t& min_block_size, tbb::task_group& tg, const uint64_t& max_concurrency){*/
/**/

/*template<typename UnaryFunction, typename T>*/
/*inline void iterator_parallel_for_2d(UnaryFunction&& in_func, const int64_t start_2d, const int64_t end_2d, const int64_t start_1d, const int64_t end_1d, BucketIterator_blocked<T>& it, int64_t stride_2d, int64_t stride_1d, uint64_t min_block_size=20){*/
/*	utils::THROW_EXCEPTION(stride_1d * (end_1d - start_1d) == stride_2d, "When iterating parallel for a 2d stride, and the size of the collumns is $ from $ to $ when multiplied by the 1d stride $ such is supposed to be equal to the 2d stride ($) but got ($ * $) = $ != $", (end_1d-start_1d) start_1d, end_1d, stride_1d, stride_2d, (end_1d-start_1d), stride_1d, stride_1d * (end_1d - start_1d), stride_2d);*/
/*	int64_t end_index = (end_2d - start_2d) * stride_2d; //this assumes it is just encapsulating stride_1d as well*/
/*							     //may make a version of this where there is an end iterator inserted into the function*/
/*	if(it.block_size() >= std::abs(end_index)){*/
/*		iterator_parallel_for_2d(std::forward<UnaryFunction&&>(in_func), start_2d, end_2d, start_1d, end_1d, (T*)it, stride_2d, stride_1d, min_block_size);*/
/*		return;*/
/*	}*/
/*	min_block_size = (min_block_size < end_1d) ? end_1d : min_block_size + (min_block_size % end_1d); //min_block_size needs to be divisible by end_1d*/
/*													  //look at pointer 2d_n_1d_n for the logic as to why*/

/*	BucketIterator_blocked<T> begin = it;*/
/*	BucketIterator_blocked<T> finish = it + end_index;*/

/*	uint64_t diff = block_diff(begin, finish);*/
/*	std::vector<std::pair<T*, T*>> bounds;*/
/*	std::vector<BucketIterator_blocked<T>> iterators;*/
/*	bounds.reserve(diff);*/
/*	iterators.reserve(diff);*/
/*	while(!same_block(begin, finish)){*/
/*		iterators.push_back(begin);*/
/*		bounds.push_back({(T*)begin, begin.block_end()});*/
/*		begin.iterate_next_block();*/
/*	}*/
/*	bounds.push_back({(T*)begin, finish.block_end()});*/
/*	iterators.push_back(finish);*/

/*	int64_t max_concurrency = tbb::this_task_arena::max_concurrency();*/
/*	tbb::task_group tg;*/
/*	//have to find a way to take into account the amount left of the 1d iterator before moving onto the second one*/
/*	uint64_t last_start = 0;*/
/*	uint64_t _1d_block_start_index = start_1d;*/
/*	uint64_t _2d_block_start_index = start_2d;*/
/*	for(uint64_t i = 0; i < iterators.size(); ++i){*/
/*		if(_2d_block_start_index == end_2d){break;}*/
/*		const auto& [begin_ptr, end_ptr] = bounds[i];*/
/*		std::ptrdiff_t ptr_diff = end_ptr - begin_ptr;*/
/*		if(last_start > 0){*/
/*			if(ptr_diff <= last_start){last_start -= ptr_diff;continue;}*/
/*		}*/
/*		T* start_ptr = begin_ptr + last_start;*/
/*		ptr_diff = end_ptr - start_ptr;*/
/*		uint64_t r = (ptr_diff  % stride_1d);*/
/*		uint64_t _1d_block_size = ptr_diff / stride_1d;*/
/*		uint64_t _2d_block_size = ptr_diff / stride_2d;*/
/*		if(_2d_block_size == 0){*/
/*			if((_1d_block_start_index + _1d_block_size) < end_1d){*/
/*				detail::iterator_parallel_for_2d_send_pointers_1d_n_2d_0(in_func, _1d_block_start_index, _2d_block_start_index,*/
/*						_1d_block_size, start_ptr, min_block_size, tg, max_concurrency);*/
/*			}*/
/*			else if((_1d_block_start_index + _1d_block_size) == end_1d){*/
/*				detail::iterator_parallel_for_2d_send_pointers_1d_n_2d_0(in_func, _1d_block_start_index, _2d_block_start_index,*/
/*						_1d_block_size, start_ptr, min_block_size, tg, max_concurrency);*/
/*				++_2d_block_start_index;*/
/*			}*/
/*			else if((_1d_block_start_index + _1d_block_size) > end_1d){*/
/*				uint64_t left = (_1d_block_start_index + _1d_block_size) - end_1d;*/
/*				detail::iterator_parallel_for_2d_send_pointers_1d_n_2d_0(in_func, _1d_block_start_index, _2d_block_start_index,*/
/*						_1d_block_size-left, start_ptr, min_block_size, tg, max_concurrency);*/
/*				++_2d_block_start_index;*/
/*				detail::iterator_parallel_for_2d_send_pointers_1d_n_2d_0(in_func, _1d_block_start_index, _2d_block_start_index,*/
/*						left, start_ptr, min_block_size, tg, max_concurrency);*/
/*			}*/
/*			if(r == 0){last_start = 0;continue;}*/
/*			BucketIterator_blocked<T> bucket = iterators[i] + (_1d_block_size * stride_1d) + last_start;*/
/*			tg.run([&in_func, bucket, _1d_block_start_index, _2d_block_start_index]{*/
/*					in_func(blocked_range<2>::make_range(_2d_block_start_index, _2d_block_start_index + 1, _1d_block_start_index, _1d_block_start_index + 1), bucket);*/
/*				});*/
/*			_1d_block_start_index += 1;*/
/*			if(_1d_block_start_index == end_1d){*/
/*				_1d_block_start_index = 0;*/
/*				++_2d_block_start_index;*/
/*			}*/
/*			last_start = stride - r;*/
/*			continue;*/
/*		}*/
/*		if(_1d_block_start_index != start_1d){*/
/*			//_1d_block_size + _1d_block_start_index > end_1d (this is use t _2d_block_size > 0)*/
/*			uint64_t left = end_1d - _1d_block_start_index;*/
/*			detail::iterator_parallel_for_2d_send_pointers_1d_n_2d_0(in_func, _1d_block_start_index, _2d_block_start_index,*/
/*					left, start_ptr, min_block_size, tg, max_concurrency);*/
/*			start_ptr += left * stride_1d;*/
/*			++_2d_block_start_index;*/
/*			_1d_block_size -= left;*/
/*		}*/
/*		_2d_block_size = _1d_block_size / (end_1d - start_1d);*/
/*		if(_2d_block_size > 0){*/
/*			detail::iterator_parallel_for_2d_send_pointers_1d_n_2d_n(in_func, start_1d, end_1d, _2d_block_start_index,*/ 
/*					_1d_block_size, _2d_block_size, stride_1d, stride_2d, start_ptr, min_block_size, tg, max_concurrency);*/
/*			_1d_block_start_index = start_1d;*/
/*		}*/
		
/*		if(_1d_block_start_index != start_1d){*/
/*			//_1d_block_size + _1d_block_start_index > end_1d (this is use t _2d_block_size > 0)*/
/*			uint64_t left = end_1d - _1d_block_start_index;*/
/*			detail::iterator_parallel_for_2d_send_pointers_1d_n_2d_0(in_func, _1d_block_start_index, _2d_block_start_index,*/
/*					left, start_ptr, min_block_size, tg, max_concurrency);*/
/*			start_ptr += left * stride_1d;*/
/*			++_2d_block_start_index;*/
/*			_1d_block_size -= left;*/
/*		}*/
/*		if(r == 0){last_start = 0;continue;}*/
/*		BucketIterator_blocked<T> bucket = iterators[i] + (_1d_block_size * stride_1d) + last_start;*/
/*		tg.run([&in_func, bucket, _1d_block_start_index, _2d_block_start_index]{*/
/*				in_func(blocked_range<2>::make_range(_2d_block_start_index, _2d_block_start_index + 1, _1d_block_start_index, _1d_block_start_index + 1), bucket);*/
/*			});*/
/*		_1d_block_start_index += 1;*/
/*		if(_1d_block_start_index == end_1d){*/
/*			_1d_block_start_index = 0;*/
/*			++_2d_block_start_index;*/
/*		}*/
/*		last_start = stride - r;*/

/*	}*/
/*	tg.wait();*/
	
/*}*/


}
}


#endif
