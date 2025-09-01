#include "ArrayRef.h"

#include <iterator>
#include <memory.h>
#include <memory>

#include <vector>
#include <array>
#include <initializer_list>
#include <numeric>
#include <execution>

#include <algorithm>
#include <iostream>
#include <assert.h>
#include <utility>
#include "../utils/type_traits.h"

namespace nt{

template<typename T>
ArrayRef<T>::ArrayRef()
	:_vals(nullptr, ArrayRefDeleteNothing<T>), _total_size(0), _empty(true)
{}

template<typename T>
ArrayRef<T>::ArrayRef(const ArrayRef<T> &Arr)
	:_vals(MetaNewArr(T, Arr._total_size), MetaFreeArr<T>), _total_size(Arr._total_size), _empty(Arr._empty)
{
	if(Arr._empty)
		_vals.reset(nullptr);
	else
		std::copy(Arr._vals.get(), Arr._vals.get() + _total_size, _vals.get());
}

template<typename T>
ArrayRef<T>::ArrayRef(ArrayRef<T>&& Arr)
	:_vals(std::move(Arr._vals)), _total_size(std::exchange(Arr._total_size, 0)), _empty(std::exchange(Arr._empty, true))
{}


template<typename T>
ArrayRef<T>::ArrayRef(const T *data, size_t length)
	:_vals(MetaNewArr(T, length), MetaFreeArr<T>), _total_size(length), _empty(false)
{std::copy(data, data + length, _vals.get());}

template<typename T>
ArrayRef<T>::ArrayRef(const std::vector<T> &Vec)
	:_vals(MetaNewArr(T, Vec.size()), MetaFreeArr<T>), _total_size(Vec.size()), _empty(Vec.size() == 0 ? true : false)
{
	if(!_empty){std::copy(Vec.cbegin(), Vec.cend(), _vals.get());}
	else{_vals.reset(nullptr);}
}

template<typename T>
template<size_t N>
ArrayRef<T>::ArrayRef(const std::array<T, N> &Arr)
	:_vals(MetaNewArr(T, static_cast<int64_t>(N)), MetaFreeArr<T>), _total_size(N), _empty(N == 0 ? true : false)
{
	if(!_empty){std::copy(Arr.cbegin(), Arr.cend(), _vals.get());}
	else{_vals.reset(nullptr);}
}

template<typename T>
template<size_t N>
ArrayRef<T>::ArrayRef(const T (&Arr)[N])
	:_vals(MetaNewArr(T, static_cast<int64_t>(N)), MetaFreeArr<T>), _total_size(N), _empty(N == 0 ? true : false)
{
	if(!_empty){std::copy(&Arr[0], &Arr[N-1], _vals.get());}
	else{_vals.reset(nullptr);}
}

template<typename T>
ArrayRef<T>::ArrayRef(const std::initializer_list<T> &Vec)
	:_vals(MetaNewArr(T, Vec.size()), MetaFreeArr<T>), _total_size(Vec.size()), _empty(Vec.size() == 0 ? true : false)
{std::copy(Vec.begin(), Vec.end(), _vals.get());}

template<typename T>
ArrayRef<T>::ArrayRef(const std::unique_ptr<T[], void(*)(T*)>& vals, size_t ts)
	:_vals(MetaNewArr(T, ts), MetaFreeArr<T>),
	_total_size(ts),
	_empty(ts == 0)
{
	if(!_empty){std::copy(vals.get(), vals.get() + ts, _vals.get());}
	else{_vals.reset(nullptr);}
}

template<typename T>
ArrayRef<T>::ArrayRef(std::unique_ptr<T[], void(*)(T*)>&& vals, size_t ts)
	:_vals(std::move(vals)),
	_total_size(ts),
	_empty(ts == 0)
{}

template<typename T>
ArrayRef<T>& ArrayRef<T>::operator=(const ArrayRef<T>& Arr){
	if(Arr._empty){
		_vals.reset(nullptr);
		_total_size = Arr._total_size;
		_empty = Arr._empty;
		return *this;
	}
	_vals = std::unique_ptr<T[], void(*)(T*)>(MetaNewArr(T, Arr._total_size), MetaFreeArr<T>);
	std::copy(Arr._vals.get(), Arr._vals.get() + Arr._total_size, _vals.get());
	_total_size = Arr._total_size;
	_empty = Arr._empty;
	return *this;
}

template<typename T>
ArrayRef<T>& ArrayRef<T>::operator=(ArrayRef<T>&& Arr){
	_vals = std::move(Arr._vals);
	_total_size = std::exchange(Arr._total_size, 0);
	_empty = std::exchange(Arr._empty, true);
	return *this;
}

template<typename T>
bool ArrayRef<T>::operator==(const ArrayRef<T> &Arr) const {
	if(Arr._vals == nullptr || _vals == nullptr){return false;}
	if(Arr._total_size != _total_size)
		return false;
	if(Arr._empty != _empty)
		return false;
	return std::equal(begin(), end(), Arr.begin());
}

template<typename T>
bool ArrayRef<T>::operator!=(const ArrayRef<T> &Arr) const {
	if(Arr._vals == nullptr || _vals == nullptr){return true;}
	if(Arr._total_size != _total_size)
		return true;
	if(Arr._empty != _empty)
		return true;
	return !(std::equal(begin(), end(), Arr.begin()));
}

template<typename T>
const T* ArrayRef<T>::data() const{return _vals.get();}
template<typename T>
size_t ArrayRef<T>::size() const {return _total_size;}
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
const T& ArrayRef<T>::operator[](size_t index) const{
	index = index < 0 ? size() + index : index;
	return _vals[index];
}
template<typename T>
T& ArrayRef<T>::operator[](size_t index){
	index = index < 0 ? size() + index : index;
	return _vals[index];
}

template<typename T>
const T& ArrayRef<T>::at(size_t index) const {assert(index < _total_size);return _vals[index];}

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
T ArrayRef<T>::multiply() const{return std::accumulate(cbegin(), cend(), T(1), std::multiplies<T>());}

template<typename T>
ArrayRef<T> ArrayRef<T>::pop_front() const {
	if(_empty || size() == 1)
		return ArrayRef<T>();
	return ArrayRef<T>(data() + 1, _total_size - 1);
}


template<typename T>
ArrayRef<T> ArrayRef<T>::clone() const {
	return *this;
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
