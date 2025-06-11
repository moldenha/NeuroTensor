#include "Threading.h"

#include <array>
#include <cstddef>
#include <cstdlib>
#include <sys/types.h>
#include <type_traits>
#include <algorithm>
#include <iostream>
#include "../utils/utils.h"

namespace nt{
namespace threading{

#ifdef USE_PARALLEL
template<size_t N>
const int64_t block_ranges<N>::total_nums() const{
	int64_t total = 0;
	for(int64_t i = 0; i < N; ++i)
		total += (pairs[i].second - pairs[i].first);
	return total;
}

template<size_t N>
const bool block_ranges<N>::at_end(const std::array<std::pair<int64_t, int64_t>, N>& b) const{
	for(int64_t i = 0; i < N; ++i){
		if(b[i].second != pairs[i].second)
			return false;
	}
	return true;
}

template<size_t N>
void block_ranges<N>::split_all_by_one(const int64_t& nums){
	//this would be less than or ewaul to the max number of threads per core
	blocks.resize(nums);
	std::array<std::pair<int64_t, int64_t>, N> block;
	for(int64_t i = 0; i < N; ++i){
		block[i].first = 0;
		block[i].second = 1;
	}
	int64_t current_block = 1;
	int64_t current_index = N-1;
	blocks[0] = blocked_range<N>(block);
	while(!at_end(block) && current_block < nums){
		while(block[current_index].second == pairs[current_index].second && current_index > 0){
			block[current_index].first = 0;
			block[current_index].second = 1;
			--current_index;
		}
		++block[current_index].first;
		++block[current_index].second;
		current_index = N-1;
		blocks[current_block] = blocked_range<N>(block);
	}
}

template<size_t N>
void block_ranges<N>::add_n(std::array<std::pair<int64_t, int64_t>, N> &block, const int64_t &n, int64_t current_index, const int64_t &restart_st){
	int64_t counter = n;
	if(block[current_index].second == pairs[current_index].second){
		block[current_index].first = pairs[current_index].first;
		block[current_index].second = restart_st;
		--current_index;
		//counter -= (restart_st - pairs[current_index].first);
	}
	while(block[current_index].second + counter > pairs[current_index].second && current_index > 0){
		if(block[current_index].second == pairs[current_index].second && block[current_index].first == 0 && pairs[current_index].second < counter){
			--current_index;
			continue;
		}
		if(block[current_index].second == pairs[current_index].second){
			block[current_index].first = pairs[current_index].first;
			block[current_index].second = counter;
			if(block[current_index].second > pairs[current_index].second){
				counter -= (pairs[current_index].second - pairs[current_index].first);
				--current_index;
			}
			else{counter = 0;}
			continue;
		}
		block[current_index].first = block[current_index].second;
		counter -= (pairs[current_index].second - block[current_index].second);
		block[current_index].second = pairs[current_index].second;
		--current_index;
	}
	if(counter > 0 && block[current_index].second != pairs[current_index].second){
		block[current_index].first = block[current_index].second;
		block[current_index].second += counter;
		if(block[current_index].second > pairs[current_index].second)
			block[current_index].second = pairs[current_index].second;
	}
}

template<size_t N>
void block_ranges<N>::split_all_by_n(const int64_t& nums, const int64_t& n){
	//this would be less than or ewaul to the max number of threads per core
	blocks.resize(nums);
	std::array<std::pair<int64_t, int64_t>, N> block;
	int64_t counter = n;
	bool current_set = false;
	int64_t current_index;
	for(int32_t i = N-1; i >= 0; --i){
		block[i].first = 0;
		block[i].second = (counter < pairs[i].second) ? counter : pairs[i].second; 
		if(!current_set){counter -= block[i].second;}
		if(counter == 0){
			current_set = true;
			current_index = i;
			counter = 1;
		}
	}

	int64_t current_block = 1;
	const int64_t restart_st = block[current_index].second;
	
	blocks[0] = blocked_range<N>(block);
	while(!at_end(block) && current_block < nums){
		add_n(block, n, current_index, restart_st);
		blocks[current_block] = blocked_range<N>(block, current_block);
		++current_block;
	}
}



template<size_t N>
void block_ranges<N>::first_split(const int64_t& num_blocks){
	if(num_blocks < (pairs[0].second - pairs[0].first)){
		const int64_t splitting = (pairs[0].second - pairs[0].first) / num_blocks;
		blocks.resize(num_blocks);
		std::array<std::pair<int64_t, int64_t>, N> block = pairs;
		block[0].second = block[0].first + splitting;
		blocks[0] = blocked_range<N>(block, 0);
		for(int64_t i = 1; i < num_blocks-1; ++i){
			block[0].first = block[0].second;
			block[0].second += splitting;
			blocks[i] = blocked_range<N>(block, i);
		}
		block[0].first = block[0].second;
		block[0].second = pairs[0].second;
		blocks.back() = blocked_range<N>(block);
		return;
	}
	int64_t total = pairs[0].second - pairs[0].first;
	int64_t current = total;
	int64_t current_index = 0;
	/* std::vector<int64_t> adds(max_blocks); */
	std::vector<std::array<std::pair<int64_t, int64_t>, N>> b_blocks(1);
	b_blocks[0] = pairs;
	b_blocks.reserve(num_blocks);
	//basically, while it is less, split it over and over
	int64_t last_split = b_blocks.size() + 1;
	while(b_blocks.size() < num_blocks){
		int64_t can_split = b_blocks.size() + 1;
		int64_t distance;
		for(int64_t i = 0; i < b_blocks.size(); ++i){
			if(distance = b_blocks[i][current_index].second - b_blocks[i][current_index].first; distance > 1){
				can_split = i;
				if(can_split == last_split)
					continue;
				break;
			}
		}
		last_split = can_split;
		if(can_split == (b_blocks.size() + 1)){
			++current_index;
			continue;
		}
		std::array<std::pair<int64_t, int64_t>, N> ba = b_blocks[can_split];
		std::array<std::pair<int64_t, int64_t>, N> bb = b_blocks[can_split];
		ba[current_index].second -= (distance / 2);
		bb[current_index].first = ba[current_index].second;
		b_blocks[can_split] = ba;
		b_blocks.push_back(bb);
	}

	//next they are going to be sorted from first to last range
	std::array<int64_t, N-1> distances;
	if constexpr(N > 2){
		int64_t mult = 1;
		for(long i = N-2; i >= 0; --i){
			distances[i] = (pairs[i+1].second - pairs[i+1].first) * mult;
			mult *= distances[i];
		}
	}
	std::sort(b_blocks.begin(), b_blocks.end(), [&distances](std::array<std::pair<int64_t, int64_t>, N>& a,
								std::array<std::pair<int64_t, int64_t>, N>& b){
		int64_t d_a = a[N-1].second;
		int64_t d_b = b[N-1].second;
		if constexpr( N > 1 ){
			for(long i = N-2; i >= 0; --i){
				d_a += (a[i].second) * distances[i];
				d_b += (b[i].second) * distances[i];
			}
		}
		return d_a < d_b;
	});

	blocks.resize(num_blocks);
	for(int64_t i = 0; i < num_blocks; ++i)
		blocks[i] = blocked_range<N>(b_blocks[i], i);

}

template<size_t N>
void block_ranges<N>::perform_block(){
	const int64_t max_blocks = utils::getThreadsPerCore();
	const int64_t nums = total_nums();
	if(nums < max_blocks){
		split_all_by_one(nums);
		return;
	}
	/* const int64_t n = (nums / max_blocks) * N + 1; */
	/* split_all_by_n(max_blocks, n); */
	first_split(max_blocks);
	for(int64_t i = 0; i < blocks.size(); ++i)
		blocks[i].index = i;
}


//the ctr+E thing the color was slightly off, have to make it where(t != 0) = [0,0,1]


template class block_ranges<1>;
template class block_ranges<2>;
template class block_ranges<3>;
template class block_ranges<4>;
template class block_ranges<5>;
template class block_ranges<6>;
template class block_ranges<7>;
template class block_ranges<8>;
template class block_ranges<9>;
template class block_ranges<10>;


#endif
}}
