#include "permute_old.h"


#include <cstdint>
#include <iostream>
#include <numeric>
#include <vector>
#include <memory.h>
#include "../utils/utils.h"
#include <memory>
#include <algorithm>

namespace nt{
namespace permute{

PermIndexItND::PermIndexItND(std::vector<int64_t> _in, std::vector<int64_t>::const_iterator _adds, const int64_t st, const int64_t ci)
	:increment_nums(std::move(_in)),
	start(st),
	current_index(ci),
	adds(_adds)
	{}


inline PermIndexItND& PermIndexItND::operator++(){
	++start;
	if(start % increment_nums[1] != 0){current_index += increment_nums[0];return *this;}
	std::vector<int64_t>::const_iterator adds_1 = adds;
	int64_t current_div = start;
	current_index = 0;
	for(std::uint_fast8_t n = increment_nums.size()-1; n > 0; --n, ++adds_1){
		current_index += (current_div / increment_nums[n]) * (*adds_1);
		current_div = (current_div % increment_nums[n]);
		if(current_div == 0)
			return *this;
	}
	return *this;
}

PermIndexItND PermIndexItND::operator++(int){
	PermIndexItND tmp = *this;
	++(*this);
	return tmp;
}


inline PermIndexItND& PermIndexItND::operator+=(int64_t ad){
	start += ad;
	current_index = 0;
	std::vector<int64_t>::const_iterator adds_1 = adds;
	int64_t current_div = start;
	for(std::uint_fast8_t n = increment_nums.size()-1; n > 0; --n, ++adds_1){
		current_index += (current_div / increment_nums[n]) * (*adds_1);
		current_div = (current_div % increment_nums[n]);
		if(current_div == 0)
			break;
	}
	current_index += (increment_nums[0] * (start % increment_nums[1]));
	return *this;
}

PermIndexItND PermIndexItND::operator+(int64_t ad){
	PermIndexItND tmp = *this;
	(*this) += ad;
	return tmp;
}

bool operator==(const PermIndexItND& a, const PermIndexItND& b){
	return a.start == b.start;
}

bool operator!=(const PermIndexItND& a, const PermIndexItND& b){
	return a.start != b.start;
}



inline PermIndexItND_contig& PermIndexItND_contig::operator++(){
	++start;
	++current_index;
	return *this;
}

inline PermIndexItND_contig& PermIndexItND_contig::operator+=(int64_t ad){
	start += ad;
	current_index += ad;
	return *this;
}

bool operator==(const PermIndexItND_contig& a, const PermIndexItND_contig& b){
	return a.start == b.start;
}

bool operator!=(const PermIndexItND_contig& a, const PermIndexItND_contig& b){
	return a.start != b.start;
}

PermIndexItND_contig::PermIndexItND_contig(std::vector<int64_t> _in, std::vector<int64_t>::const_iterator _adds, const int64_t st, const int64_t ci)
	:PermIndexItND(std::move(_in), _adds, st, ci)
{}

PermIndexIt2D::PermIndexIt2D(std::vector<int64_t> _in, std::vector<int64_t>::const_iterator _adds, const int64_t st, const int64_t ci)
	:PermIndexItND(std::move(_in), _adds, st, ci)
{}

inline PermIndexIt2D& PermIndexIt2D::operator++(){
	++start;
	if(start % increment_nums[1] == 0){
		current_index = (start / increment_nums[1]) * (*adds);
		return *this;
	}
	current_index += increment_nums[0];
	return *this;
}

inline PermIndexIt2D& PermIndexIt2D::operator+=(int64_t ad){
	start += ad;
	current_index = ((start / increment_nums[1]) * (*adds)) + (increment_nums[0] * (start % increment_nums[1]));
	return *this;
}


bool operator==(const PermIndexIt2D& a, const PermIndexIt2D& b){
	return a.start == b.start;
}

bool operator!=(const PermIndexIt2D& a, const PermIndexIt2D& b){
	return a.start != b.start;
}

PermIndexIt3D::PermIndexIt3D(std::vector<int64_t> _in, std::vector<int64_t>::const_iterator _adds, const int64_t st, const int64_t ci)
	:PermIndexItND(std::move(_in), _adds, st, ci)
{}

inline PermIndexIt3D& PermIndexIt3D::operator++(){
	++start;
	if(start % increment_nums[1] != 0){current_index += increment_nums[0];return *this;}
	std::vector<int64_t>::const_iterator adds_1 = adds;
	int64_t current_div = start;
	current_index = ((current_div / increment_nums[2]) * (*adds_1));
	current_div = current_div % increment_nums[2];
	if(current_div == 0) return *this;
	++adds_1;
	current_index += ((current_div / increment_nums[1]) * *(adds_1));
	return *this;
}

inline PermIndexIt3D& PermIndexIt3D::operator+=(int64_t ad){
	start += ad;
	current_index = ((start / increment_nums[2]) * *(adds)) + (((start % increment_nums[2]) / increment_nums[1]) * *(adds + 1)) + (increment_nums[0] * (start % increment_nums[1]));
	return *this;
}


bool operator==(const PermIndexIt3D& a, const PermIndexIt3D& b){
	return a.start == b.start;
}

bool operator!=(const PermIndexIt3D& a, const PermIndexIt3D& b){
	return a.start != b.start;
}

PermIndexIt4D::PermIndexIt4D(std::vector<int64_t> _in, std::vector<int64_t>::const_iterator _adds, const int64_t st, const int64_t ci)
	:PermIndexItND(std::move(_in), _adds, st, ci)
{}

inline PermIndexIt4D& PermIndexIt4D::operator++(){
	++start;
	if(start % increment_nums[1] != 0){current_index += increment_nums[0];return *this;}
	std::vector<int64_t>::const_iterator adds_1 = adds;
	int64_t current_div = start;
	current_index = ((current_div / increment_nums[3]) * (*adds));
	current_div = current_div % increment_nums[3];
	if(current_div == 0) return *this;
	++adds_1;
	current_index += ((current_div / increment_nums[2]) * *(adds_1));
	current_div = current_div % increment_nums[2];
	if(current_div == 0) return *this;
	++adds_1;
	current_index += ((current_div / increment_nums[1]) * *(adds_1));
	return *this;
}

inline PermIndexIt4D& PermIndexIt4D::operator+=(int64_t ad){
	start += ad;
	current_index = ((start / increment_nums[3]) * *(adds)) + (((start % increment_nums[3]) / increment_nums[2]) * *(adds + 1)) + ((((start % increment_nums[3]) % increment_nums[2]) / increment_nums[1]) * *(adds + 2)) + (increment_nums[0] * (start % increment_nums[1]));
	return *this;
}


bool operator==(const PermIndexIt4D& a, const PermIndexIt4D& b){
	return a.start == b.start;
}

bool operator!=(const PermIndexIt4D& a, const PermIndexIt4D& b){
	return a.start != b.start;
}

PermIndexIt5D::PermIndexIt5D(std::vector<int64_t> _in, std::vector<int64_t>::const_iterator _adds, const int64_t st, const int64_t ci)
	:PermIndexItND(std::move(_in), _adds, st, ci)
{}

inline PermIndexIt5D& PermIndexIt5D::operator++(){
	/* std::vector<int64_t> increment_nums = {_strides[4], _shape[4], _shape[4] * _shape[3], _shape[4] * _shape[3] * _shape[2], _shape[4] * _shape[3] * _shape[2] * _shape[1]}; */
	++start;
	if(start % increment_nums[1] != 0){current_index += increment_nums[0];return *this;}
	std::vector<int64_t>::const_iterator adds_1 = adds;
	int64_t current_div = start;
	current_index = ((current_div / increment_nums[4]) * (*adds));
	current_div = current_div % increment_nums[4];
	if(current_div == 0) return *this;
	++adds_1;
	current_index += ((current_div / increment_nums[3]) * *(adds_1));
	current_div = current_div % increment_nums[3];
	if(current_div == 0) return *this;
	++adds_1;
	current_index += ((current_div / increment_nums[2]) * *(adds_1));
	current_div = current_div % increment_nums[2];
	if(current_div == 0) return *this;
	++adds_1;
	current_index += ((current_div / increment_nums[1]) * *(adds_1));
	return *this;
}

inline PermIndexIt5D& PermIndexIt5D::operator+=(int64_t ad){
	start += ad;
	current_index = ((start / increment_nums[4]) * *(adds)) + (((start % increment_nums[4]) / increment_nums[3]) * *(adds + 1)) + ((((start % increment_nums[4]) % increment_nums[3]) / increment_nums[2]) * *(adds + 2)) + (((((start % increment_nums[4]) % increment_nums[3]) % increment_nums[2]) / increment_nums[1]) * *(adds + 3)) + (increment_nums[0] * (start % increment_nums[1]));
	return *this;
}


bool operator==(const PermIndexIt5D& a, const PermIndexIt5D& b){
	return a.start == b.start;
}

bool operator!=(const PermIndexIt5D& a, const PermIndexIt5D& b){
	return a.start != b.start;
}



PermND::PermND(const std::vector<int64_t> &str, const std::vector<int64_t> &shp)
			:_strides(str),
			_shape(shp),
			start(0)
		{}

inline bool PermND::is_contiguous() const {return std::is_sorted(_strides.cbegin(), _strides.cend());}
inline int64_t PermND::get_index(const int64_t i) const {
	if(i == 0) return 0;
	int64_t current_index = 0;
	int64_t current_div = i;
	std::vector<int64_t> increment_nums(_strides.size());
	increment_nums[0] = _strides.back();
	increment_nums[1] = _shape.back(); //increment_num_1
	for(std::uint_fast8_t j = 2; j < _strides.size(); ++j)
		increment_nums[j] = increment_nums[j-1] * _shape[_strides.size() - (j)];
	auto adds = _strides.cbegin();
	for(std::uint_fast8_t n = _strides.size() - 1; n > 0; --n, ++adds){
		current_index += (current_div / increment_nums[n]) * (*adds);
		current_div = (current_div % increment_nums[n]);
		if(current_div == 0)
			break;
	}
	return current_index + (increment_nums[0] * (i % increment_nums[1]));
}

std::shared_ptr<PermIndexItND> PermND::begin(int64_t i) const{
	std::vector<int64_t> increment_nums(_strides.size());
	increment_nums[0] = _strides.back();
	increment_nums[1] = _shape.back(); //increment_num_1
	for(std::uint_fast8_t j = 2; j < _strides.size(); ++j)
		increment_nums[j] = increment_nums[j-1] * _shape[_strides.size() - (j)];
	if(is_contiguous())
		return std::make_unique<PermIndexItND_contig>(std::move(increment_nums),
				_strides.cbegin(), i, i);
	return std::make_unique<PermIndexItND>(std::move(increment_nums),
				_strides.cbegin(), i, get_index(i));
}

std::shared_ptr<PermIndexItND> PermND::end() const{
	int64_t total_size = std::accumulate(_shape.cbegin(), _shape.cend(), 1, std::multiplies<int64_t>());
	return begin(total_size);
}

std::vector<int64_t> PermND::return_indexes() const{
	int64_t total_size = std::accumulate(_shape.cbegin(), _shape.cend(), 1, std::multiplies<int64_t>());
	if(is_contiguous()){
		std::vector<int64_t> outp(total_size);
		std::iota(outp.begin(), outp.end(), 0);
		return std::move(outp);
	}
	std::vector<int64_t> indexes(total_size, 0);
	std::vector<int64_t> increment_nums(_strides.size());
	increment_nums[0] = _strides.back();
	increment_nums[1] = _shape.back(); //increment_num_1
	int64_t current_div;
	for(std::uint_fast8_t j = 2; j < _strides.size(); ++j)
		increment_nums[j] = increment_nums[j-1] * _shape[_strides.size() - (j)];
	for(int64_t i = 1; i < total_size; ++i){
		if(i % increment_nums[1] != 0){indexes[i] = indexes[i-1] + increment_nums[0];continue;}
		auto adds = _strides.cbegin();
		current_div = i;
		for(std::uint_fast8_t n = _strides.size()-1; n > 0; --n, ++adds){
			indexes[i] += (current_div / increment_nums[n]) * (*adds);
			current_div = (current_div % increment_nums[n]);
			if(current_div == 0)
				break;
		}
	}
	return std::move(indexes);
}

std::vector<int64_t> PermND::return_indexes(std::vector<int64_t>::const_iterator begin_a) const{
	int64_t total_size = std::accumulate(_shape.cbegin(), _shape.cend(), 1, std::multiplies<int64_t>());
	if(is_contiguous())
		return std::vector<int64_t>(begin_a, begin_a + total_size);
	std::vector<int64_t> indexes(total_size);
	std::vector<int64_t> increment_nums(_strides.size());
	increment_nums[0] = _strides.back();
	increment_nums[1] = _shape.back(); //increment_num_1
	int64_t current_div, current_index;
	for(std::uint_fast8_t j = 2; j < _strides.size(); ++j)
		increment_nums[j] = increment_nums[j-1] * _shape[_strides.size() - (j)];
	indexes[0] = *(begin_a);
	for(int64_t i = 1; i < total_size; ++i){
		if(i % increment_nums[1] != 0){current_index += increment_nums[0];}
		else{
			auto adds = _strides.cbegin();
			current_div = i;
			current_index = 0;
			for(std::uint_fast8_t n = _strides.size()-1; n > 0; --n, ++adds){
				current_index += (current_div / increment_nums[n]) * (*adds);
				current_div = (current_div % increment_nums[n]);
				if(current_div == 0)
					break;
			}
		}
		indexes[i] = *(begin_a + current_index);
	}
	return std::move(indexes);
}



std::vector<int64_t> PermND::return_indexes(const int64_t* begin_a) const{
	int64_t total_size = std::accumulate(_shape.cbegin(), _shape.cend(), 1, std::multiplies<int64_t>());
	if(is_contiguous())
		return std::vector<int64_t>(begin_a, begin_a + total_size);
	std::vector<int64_t> indexes(total_size);
	std::vector<int64_t> increment_nums(_strides.size());
	increment_nums[0] = _strides.back();
	increment_nums[1] = _shape.back(); //increment_num_1
	int64_t current_div, current_index;
	for(std::uint_fast8_t j = 2; j < _strides.size(); ++j)
		increment_nums[j] = increment_nums[j-1] * _shape[_strides.size() - (j)];
	indexes[0] = *(begin_a);
	for(int64_t i = 1; i < total_size; ++i){
		if(i % increment_nums[1] != 0){current_index += increment_nums[0];}
		else{
			auto adds = _strides.cbegin();
			current_div = i;
			current_index = 0;
			for(std::uint_fast8_t n = _strides.size()-1; n > 0; --n, ++adds){
				current_index += (current_div / increment_nums[n]) * (*adds);
				current_div = (current_div % increment_nums[n]);
				if(current_div == 0)
					break;
			}
		}
		indexes[i] = *(begin_a + current_index);
	}
	return std::move(indexes);
}

void PermND::perm_in_place(void** original, void** outp) const{
	auto begin = this->begin();
	auto end = this->end();
	for(;*begin != *end; ++(*begin), ++outp){
		*outp = const_cast<void*>(original[*(*begin)]);
	}
}

void PermND::perm_in_place(int64_t* first, int64_t* last, const int64_t& total) const{
	std::vector<int64_t> indexes = return_indexes(first);
	std::copy(indexes.cbegin(), indexes.cend(), first);
}


Perm2D::Perm2D(const std::vector<int64_t> &str, const std::vector<int64_t> &shp)
	:PermND(str, shp)
	{utils::throw_exception(_strides.size() == 2 && _shape.size() == 2,
			"\nRuntime Error: Unable to permute 2D with strides of size $ and shape of size $", _strides.size(), _shape.size());}


inline int64_t Perm2D::get_index(const int64_t i) const {return ((i / _shape.back()) * _strides[0]) + (_strides[1] * (i % _shape[1]));}

std::shared_ptr<PermIndexItND> Perm2D::begin(int64_t i) const{
	std::vector<int64_t> increment_nums = {_strides.back(), _shape.back()};
	if(is_contiguous())
		return std::make_unique<PermIndexItND_contig>(std::move(increment_nums),
				_strides.cbegin(), i, i);
	return std::make_unique<PermIndexIt2D>(std::move(increment_nums),
				_strides.cbegin(), i, get_index(i));
}

std::vector<int64_t> Perm2D::return_indexes() const{
	int64_t total_size = std::accumulate(_shape.cbegin(), _shape.cend(), 1, std::multiplies<int64_t>());
	std::vector<int64_t> indexes(total_size, 0);
	const int64_t& increment = _strides.back();
	const int64_t& increment_num = _shape.back();
	const int64_t& add = _strides[0];
	int64_t current_add = 0;
	for(int64_t j = 1; j < total_size; ++j){
		if(j % increment_num == 0){
			indexes[j] = current_add * add;
			++current_add;
			continue;
		}
		indexes[j] = indexes[j-1] + increment;
	}
	return std::move(indexes);	
}

std::vector<int64_t> Perm2D::return_indexes(std::vector<int64_t>::const_iterator begin_a) const{
	int64_t total_size = std::accumulate(_shape.cbegin(), _shape.cend(), 1, std::multiplies<int64_t>());
	std::vector<int64_t> indexes(total_size);
	const int64_t& increment = _strides.back();
	const int64_t& increment_num = _shape.back();
	const int64_t& add = _strides[0];
	int64_t current_add = 0;
	int64_t current_index = 0;
	indexes[0] = *(begin_a);
	for(int64_t j = 1; j < total_size; ++j){

		if(j % increment_num == 0){
			current_index = current_add * add;
			indexes[j] = *(begin_a + current_index);
			++current_add;
			continue;
		}
		current_index += increment;
		indexes[j] = *(begin_a + current_index);
	}
	return std::move(indexes);	
}

std::vector<int64_t> Perm2D::return_indexes(const int64_t* begin_a) const{
	int64_t total_size = std::accumulate(_shape.cbegin(), _shape.cend(), 1, std::multiplies<int64_t>());
	std::vector<int64_t> indexes(total_size);
	const int64_t& increment = _strides.back();
	const int64_t& increment_num = _shape.back();
	const int64_t& add = _strides[0];
	int64_t current_add = 0;
	int64_t current_index = 0;
	indexes[0] = *(begin_a);
	for(int64_t j = 1; j < total_size; ++j){
		if(j % increment_num == 0){
			current_index = current_add * add;
			indexes[j] = *(begin_a + current_index);
			++current_add;
			continue;
		}
		current_index += increment;
		indexes[j] = *(begin_a + current_index);
	}
	return std::move(indexes);	
}

void Perm2D::perm_in_place(int64_t* first, int64_t* last, const int64_t& total) const{
	if(_strides[0] > _strides[1])
		return; //this is basically for only a 2D array an in-place transpose, and will be treated as such
	const int64_t& n = _shape.back(); //the old rows
	const int64_t mn1 = total - 1;
	std::vector<bool> visited(total);
	int64_t* cycle = first;
	 while (++cycle != last) {
		if (visited[cycle - first])
		    continue;
		int a = cycle - first;
		do  {
		    a = a == mn1 ? mn1 : (n * a) % mn1;
		    std::swap(*(first + a), *cycle);
		    visited[a] = true;
		} while ((first + a) != cycle);
	}
}


Perm3D::Perm3D(const std::vector<int64_t> &str, const std::vector<int64_t> &shp)
	:PermND(str, shp)
	{utils::throw_exception(_strides.size() == 3 && _shape.size() == 3,
			"\nRuntime Error: Unable to permute 3D with strides of size $ and shape of size $", _strides.size(), _shape.size());}


inline int64_t Perm3D::get_index(const int64_t i) const {const int64_t& increment = _strides.back();
	const int64_t& increment_num = _shape.back();
	const int64_t increment_num_2 = increment_num * _shape[1];
	const int64_t& add_a = _strides[1];
	const int64_t& add_b = _strides[2];
	int64_t dived = (i % increment_num_2);
	int64_t j = (i / increment_num_2) * add_b;
	if(dived == 0)
		return j;
	j += (dived / (increment_num)) * add_a;
	dived = (i % increment_num);
	if(dived == 0)
		return j;
	return j + (dived * increment);
}

std::shared_ptr<PermIndexItND> Perm3D::begin(int64_t i) const{
	/* std::vector<int64_t> increment_nums = {_strides.back(), _shape.back(), _shape.back() * _shape[_strides.size() - (2)]}; */
	std::vector<int64_t> increment_nums = {_strides[2], _shape[2], _shape[2] * _shape[1]};
	if(is_contiguous())
		return std::make_unique<PermIndexItND_contig>(std::move(increment_nums),
				_strides.cbegin(), i, i);
	return std::make_unique<PermIndexIt3D>(std::move(increment_nums),
				_strides.cbegin(), i, get_index(i));
}

std::vector<int64_t> Perm3D::return_indexes() const{
	int64_t total_size = std::accumulate(_shape.cbegin(), _shape.cend(), 1, std::multiplies<int64_t>());
	std::vector<int64_t> indexes(total_size);
	if(is_contiguous()){
		std::iota(indexes.begin(), indexes.end(), 0);
		return std::move(indexes);
	}
	std::shared_ptr<PermIndexIt3D> start = std::reinterpret_pointer_cast<PermIndexIt3D>(begin());
	for(int64_t i = 0; i < total_size; ++i, ++(*start))
		indexes[i] = *(*start);
	return std::move(indexes);	
}

std::vector<int64_t> Perm3D::return_indexes(std::vector<int64_t>::const_iterator begin_a) const{
	int64_t total_size = std::accumulate(_shape.cbegin(), _shape.cend(), 1, std::multiplies<int64_t>());
	if(is_contiguous())
		return std::vector<int64_t>(begin_a, begin_a + total_size);
	std::vector<int64_t> indexes(total_size);
	std::shared_ptr<PermIndexIt3D> start = std::reinterpret_pointer_cast<PermIndexIt3D>(begin());
	for(int64_t i = 0; i < total_size; ++i, ++(*start))
		indexes[i] = *(begin_a + *(*start));
	return std::move(indexes);
}

std::vector<int64_t> Perm3D::return_indexes(const int64_t* begin_a) const{
	int64_t total_size = std::accumulate(_shape.cbegin(), _shape.cend(), 1, std::multiplies<int64_t>());
	if(is_contiguous())
		return std::vector<int64_t>(begin_a, begin_a + total_size);
	std::vector<int64_t> indexes(total_size);
	std::shared_ptr<PermIndexIt3D> start = std::reinterpret_pointer_cast<PermIndexIt3D>(begin());
	for(int64_t i = 0; i < total_size; ++i, ++(*start))
		indexes[i] = *(begin_a + *(*start));
	return std::move(indexes);	
}

Perm4D::Perm4D(const std::vector<int64_t> &str, const std::vector<int64_t> &shp)
	:PermND(str, shp)
	{utils::throw_exception(_strides.size() == 4 && _shape.size() == 4,
			"\nRuntime Error: Unable to permute 4D with strides of size $ and shape of size $", _strides.size(), _shape.size());}


inline int64_t Perm4D::get_index(const int64_t i) const {
	const int64_t& increment = _strides.back();
	const int64_t& increment_num = _shape.back();
	const int64_t increment_num_2 = increment_num * _shape[_shape.size() - 2];
	const int64_t increment_num_3 = increment_num_2 * _shape[_shape.size() - 3];
	const int64_t& add_a = _strides[_strides.size() - 2];
	const int64_t& add_b = _strides[_strides.size() - 3];
	const int64_t& add_c = _strides[_strides.size() - 4];
	return ((i / increment_num_3) * add_c) + 
		(((i % increment_num_3) / increment_num_2) * add_b) + 
		((((i % increment_num_3) % increment_num_2) / increment_num) * add_a)
		+ (increment * (i % increment_num));
}

std::shared_ptr<PermIndexItND> Perm4D::begin(int64_t i) const{
	/* std::vector<int64_t> increment_nums = {_strides.back(), _shape.back(), _shape.back() * _shape[_strides.size() - (2)]}; */
	std::vector<int64_t> increment_nums = {_strides[3], _shape[3], _shape[3] * _shape[2], _shape[3] * _shape[2] * _shape[1]};
	if(is_contiguous())
		return std::make_unique<PermIndexItND_contig>(std::move(increment_nums),
				_strides.cbegin(), i, i);
	return std::make_unique<PermIndexIt4D>(std::move(increment_nums),
				_strides.cbegin(), i, get_index(i));
}

std::vector<int64_t> Perm4D::return_indexes() const{
	int64_t total_size = std::accumulate(_shape.cbegin(), _shape.cend(), 1, std::multiplies<int64_t>());
	std::vector<int64_t> indexes(total_size);
	if(is_contiguous()){
		std::iota(indexes.begin(), indexes.end(), 0);
		return std::move(indexes);
	}
	std::shared_ptr<PermIndexIt4D> start = std::reinterpret_pointer_cast<PermIndexIt4D>(begin());
	for(int64_t i = 0; i < total_size; ++i, ++(*start))
		indexes[i] = *(*start);
	return std::move(indexes);	
}

std::vector<int64_t> Perm4D::return_indexes(std::vector<int64_t>::const_iterator begin_a) const{
	int64_t total_size = std::accumulate(_shape.cbegin(), _shape.cend(), 1, std::multiplies<int64_t>());
	if(is_contiguous())
		return std::vector<int64_t>(begin_a, begin_a + total_size);
	std::vector<int64_t> indexes(total_size);
	std::shared_ptr<PermIndexIt4D> start = std::reinterpret_pointer_cast<PermIndexIt4D>(begin());
	for(int64_t i = 0; i < total_size; ++i, ++(*start))
		indexes[i] = *(begin_a + *(*start));
	return std::move(indexes);
}

std::vector<int64_t> Perm4D::return_indexes(const int64_t* begin_a) const{
	int64_t total_size = std::accumulate(_shape.cbegin(), _shape.cend(), 1, std::multiplies<int64_t>());
	if(is_contiguous())
		return std::vector<int64_t>(begin_a, begin_a + total_size);
	std::vector<int64_t> indexes(total_size);
	std::shared_ptr<PermIndexIt4D> start = std::reinterpret_pointer_cast<PermIndexIt4D>(begin());
	for(int64_t i = 0; i < total_size; ++i, ++(*start))
		indexes[i] = *(begin_a + *(*start));
	return std::move(indexes);	
}

Perm5D::Perm5D(const std::vector<int64_t> &str, const std::vector<int64_t> &shp)
	:PermND(str, shp)
	{utils::throw_exception(_strides.size() == 5 && _shape.size() == 5,
			"\nRuntime Error: Unable to permute 5D with strides of size $ and shape of size $", _strides.size(), _shape.size());}


inline int64_t Perm5D::get_index(const int64_t i) const {
	const int64_t& increment = _strides.back();
	const int64_t& increment_num = _shape.back();
	const int64_t increment_num_2 = increment_num * _shape[_shape.size() - 2];
	const int64_t increment_num_3 = increment_num_2 * _shape[_shape.size() - 3];
	const int64_t increment_num_4 = increment_num_3 * _shape[_shape.size() - 5];
	const int64_t& add_a = _strides[_strides.size() - 2];
	const int64_t& add_b = _strides[_strides.size() - 3];
	const int64_t& add_c = _strides[_strides.size() - 4];
	const int64_t& add_d = _strides[_strides.size() - 5];
	return ((i / increment_num_4) * add_d) + 
		(((i % increment_num_4) / increment_num_3) * add_c) + 
		((((i % increment_num_4) % increment_num_3) / increment_num_2) * add_b) +
		(((((i % increment_num_4) % increment_num_3) % increment_num_2) / increment_num) * add_a)
		+ (increment * (i % increment_num));
}

std::shared_ptr<PermIndexItND> Perm5D::begin(int64_t i) const{
	/* std::vector<int64_t> increment_nums = {_strides.back(), _shape.back(), _shape.back() * _shape[_strides.size() - (2)]}; */
	std::vector<int64_t> increment_nums = {_strides[4], _shape[4], _shape[4] * _shape[3], _shape[4] * _shape[3] * _shape[2], _shape[4] * _shape[3] * _shape[2] * _shape[1]};
	if(is_contiguous())
		return std::make_unique<PermIndexItND_contig>(std::move(increment_nums),
				_strides.cbegin(), i, i);
	return std::make_unique<PermIndexIt5D>(std::move(increment_nums),
				_strides.cbegin(), i, get_index(i));
}

std::vector<int64_t> Perm5D::return_indexes() const{
	int64_t total_size = std::accumulate(_shape.cbegin(), _shape.cend(), 1, std::multiplies<int64_t>());
	std::vector<int64_t> indexes(total_size);
	if(is_contiguous()){
		std::iota(indexes.begin(), indexes.end(), 0);
		return std::move(indexes);
	}
	std::shared_ptr<PermIndexIt5D> start = std::reinterpret_pointer_cast<PermIndexIt5D>(begin());
	for(int64_t i = 0; i < total_size; ++i, ++(*start))
		indexes[i] = *(*start);
	return std::move(indexes);	
}

std::vector<int64_t> Perm5D::return_indexes(std::vector<int64_t>::const_iterator begin_a) const{
	int64_t total_size = std::accumulate(_shape.cbegin(), _shape.cend(), 1, std::multiplies<int64_t>());
	if(is_contiguous())
		return std::vector<int64_t>(begin_a, begin_a + total_size);
	std::vector<int64_t> indexes(total_size);
	std::shared_ptr<PermIndexIt5D> start = std::reinterpret_pointer_cast<PermIndexIt5D>(begin());
	for(int64_t i = 0; i < total_size; ++i, ++(*start))
		indexes[i] = *(begin_a + *(*start));
	return std::move(indexes);
}

std::vector<int64_t> Perm5D::return_indexes(const int64_t* begin_a) const{
	int64_t total_size = std::accumulate(_shape.cbegin(), _shape.cend(), 1, std::multiplies<int64_t>());
	if(is_contiguous())
		return std::vector<int64_t>(begin_a, begin_a + total_size);
	std::vector<int64_t> indexes(total_size);
	std::shared_ptr<PermIndexIt5D> start = std::reinterpret_pointer_cast<PermIndexIt5D>(begin());
	for(int64_t i = 0; i < total_size; ++i, ++(*start))
		indexes[i] = *(begin_a + *(*start));
	return std::move(indexes);	
}

std::unique_ptr<PermND> create_perm(const std::vector<int64_t>& stride, const std::vector<int64_t>& shape){
	if(stride.size() == 2)
		return std::make_unique<Perm2D>(stride, shape);
	if(stride.size() == 3)
		return std::make_unique<Perm3D>(stride, shape);
	if(stride.size() == 4)
		return std::make_unique<Perm4D>(stride, shape);
	if(stride.size() == 5)
		return std::make_unique<Perm5D>(stride, shape);
	return std::make_unique<PermND>(stride, shape);
}

}
}
