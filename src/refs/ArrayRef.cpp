#include "ArrayRef.h"

#include <iterator>
#include <memory.h>
#include <memory>
#include <sys/_types/_int64_t.h>
#include <vector>
#include <array>
#include <initializer_list>
#include <numeric>
#include <execution>

#include <algorithm>
#include <iostream>
#include <assert.h>

namespace nt{
template<typename T>
ArrayRef<T>::ArrayRef()
	:_vals(std::make_unique<T[]>(1)), _total_size(1), _empty(true)
{}

template<typename T>
ArrayRef<T>::ArrayRef(const ArrayRef<T> &Arr)
	:ArrayRef(Arr.data(), Arr._total_size)
{}

template<typename T>
ArrayRef<T>::ArrayRef(ArrayRef<T>&& Arr)
	:_vals(std::move(Arr._vals)), _total_size(Arr._total_size), _empty(Arr._empty)
{}

template<typename T>
ArrayRef<T>::ArrayRef(const T &OneEle)
	:_vals(std::make_unique<T[]>(1)), _total_size(1), _empty(false)
{_vals[0] = OneEle;}

template<typename T>
ArrayRef<T>::ArrayRef(const T *data, size_t length)
	:_vals(std::make_unique<T[]>(length)), _total_size(length), _empty(false)
{std::copy(data, data + length, _vals.get());}

template<typename T>
ArrayRef<T>::ArrayRef(const std::vector<T> &Vec)
	:_vals(std::make_unique<T[]>(Vec.size() == 0 ? 1 : Vec.size())), _total_size(Vec.size() == 0 ? 1 : Vec.size()), _empty(Vec.size() == 0 ? true : false)
{std::copy(Vec.cbegin(), Vec.cend(), _vals.get());}

template<typename T>
template<size_t N>
ArrayRef<T>::ArrayRef(const std::array<T, N> &Arr)
	:_vals(std::make_unique<T[]>(N == 0 ? 1 : N)), _total_size(N == 0 ? 1 : N), _empty(N == 0 ? true : false)
{std::copy(Arr.cbegin(), Arr.cend(), _vals.get());}

template<typename T>
template<size_t N>
ArrayRef<T>::ArrayRef(const T (&Arr)[N])
	:_vals(std::make_unique<T[]>(N == 0 ? 1 : N)), _total_size(N == 0 ? 1 : N), _empty(N == 0 ? true : false)
{std::copy(&Arr[0], &Arr[N-1], _vals.get());}

template<typename T>
ArrayRef<T>::ArrayRef(const std::initializer_list<T> &Vec)
	:_vals(std::make_unique<T[]>(Vec.size() == 0 ? 1 : Vec.size())), _total_size(Vec.size() == 0 ? 1 : Vec.size()), _empty(Vec.size() == 0 ? true : false)
	{std::copy(Vec.begin(), Vec.end(), _vals.get());}

template<typename T>
ArrayRef<T>::ArrayRef(std::unique_ptr<T[]> vals, size_t ts)
	:_vals(std::move(vals)),
	_total_size(ts),
	_empty(ts == 0)
{}


template<typename T>
ArrayRef<T>& ArrayRef<T>::operator=(const ArrayRef<T>& Arr){
	_vals = std::make_unique<T[]>(Arr._total_size);
	std::copy(Arr.begin(), Arr.end(), _vals.get());
	_total_size = Arr._total_size;
	_empty = Arr._empty;
	return *this;
}

template<typename T>
ArrayRef<T>& ArrayRef<T>::operator=(ArrayRef<T>&& Arr){
	_vals = std::move(Arr._vals);
	_total_size = Arr._total_size;
	_empty = Arr._empty;
	return *this;
}

template<typename T>
const bool ArrayRef<T>::operator==(const ArrayRef<T> &Arr) const {
	if(Arr._total_size != _total_size)
		return false;
	if(Arr._empty != _empty)
		return false;
	return std::equal(begin(), end(), Arr.begin());
}

template<typename T>
const bool ArrayRef<T>::operator!=(const ArrayRef<T> &Arr) const {
	if(Arr._total_size != _total_size)
		return true;
	if(Arr._empty != _empty)
		return true;
	return !(std::equal(begin(), end(), Arr.begin()));
}

template<typename T>
const T* ArrayRef<T>::data() const{return _vals.get();}
template<typename T>
size_t ArrayRef<T>::size() const {return _empty == true ? 0 : _total_size;}
template<typename T>
const T& ArrayRef<T>::front() const {return _vals[0];}
template<typename T>
const T& ArrayRef<T>::back() const {return _vals[_total_size-1];}

template<typename T>
T& ArrayRef<T>::front() {return _vals[0];}
template<typename T>
T& ArrayRef<T>::back() {return _vals[_total_size-1];}


template<typename T>
const T* ArrayRef<T>::begin() const {return &_vals[0];}
template<typename T>
const T* ArrayRef<T>::end() const {return &_vals[_total_size];}
template<typename T>
const T* ArrayRef<T>::cbegin() const {return begin();}
template<typename T>
const T* ArrayRef<T>::cend() const {return end();}
template<typename T>
typename ArrayRef<T>::reverse_iterator ArrayRef<T>::rbegin() const {return reverse_iterator(&_vals[_total_size]);}
template<typename T>
typename ArrayRef<T>::reverse_iterator ArrayRef<T>::rend() const {return reverse_iterator(&_vals[0]);}
template<typename T>
bool ArrayRef<T>::empty() const {return _empty;}
template<typename T>
const T& ArrayRef<T>::operator[](int16_t index) const{
	index = index < 0 ? size() + index : index;
	return _vals[index];
}
template<typename T>
T& ArrayRef<T>::operator[](int16_t index){
	index = index < 0 ? size() + index : index;
	return _vals[index];
}

template<typename T>
const T& ArrayRef<T>::at(uint16_t index) const {assert(index < _total_size);return _vals[index];}

template<typename T>
std::vector<T> ArrayRef<T>::to_vec() const {return std::vector<T>(begin(), end());}

template<typename T>
ArrayRef<T> ArrayRef<T>::permute(const std::vector<uint32_t> &Vec) const {
	assert(Vec.size() == _total_size);
	std::vector<T> output(_total_size);
	for(uint32_t i = 0; i < _total_size; ++i)
		output[i] = at(Vec[i]);
	return ArrayRef<T>(std::move(output));
}



template<typename T>
const T ArrayRef<T>::multiply() const{return std::accumulate(cbegin(), cend(), 1, std::multiplies<T>());}

template<typename T>
ArrayRef<T> ArrayRef<T>::pop_front() const {
	if(_empty || size() == 1)
		return ArrayRef<T>();
	return ArrayRef<T>(data() + 1, _total_size - 1);
}


template<typename T>
T* ArrayRef<T>::d_data(){return _vals.get();}

template<typename T>
std::ostream& operator<<(std::ostream &out, const ArrayRef<T>& data) {
	if(data.empty())
		return out << "{}";
	out<<"{";
	auto begin = data.begin();
	for(uint32_t i = 0; i < data.size()-1; ++i){
		out<< *(begin + i)<<",";
	}
	out << *(begin + (data.size() - 1))<<"}";
	return out;
}


template class ArrayRef<uint32_t>;
template class ArrayRef<uint64_t>;
template class ArrayRef<uint8_t>;
template class ArrayRef<uint16_t>;
template class ArrayRef<int64_t>;
template class ArrayRef<int32_t>;
template class ArrayRef<int16_t>;
template class ArrayRef<int8_t>;

template std::ostream& operator<<(std::ostream &out, const ArrayRef<uint64_t>& data);
template std::ostream& operator<<(std::ostream &out, const ArrayRef<uint32_t>& data);
template std::ostream& operator<<(std::ostream &out, const ArrayRef<uint16_t>& data);
template std::ostream& operator<<(std::ostream &out, const ArrayRef<uint8_t>& data);
template std::ostream& operator<<(std::ostream &out, const ArrayRef<int64_t>& data);
template std::ostream& operator<<(std::ostream &out, const ArrayRef<int32_t>& data);
template std::ostream& operator<<(std::ostream &out, const ArrayRef<int16_t>& data);
template std::ostream& operator<<(std::ostream &out, const ArrayRef<int8_t>& data);
}
