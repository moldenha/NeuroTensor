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
#include "../intrusive_ptr/intrusive_ptr.hpp"
#include "../intrusive_ptr/intrusive_tracked_list.hpp"

namespace nt{

template<typename T>
ArrayRef<T>::ArrayRef()
	:_vals(nullptr)
{}

template<typename T>
ArrayRef<T>::ArrayRef(const ArrayRef<T> &Arr)
	:_vals(Arr._vals)
{}

template<typename T>
ArrayRef<T>::ArrayRef(ArrayRef<T>&& Arr)
	:_vals(std::move(Arr._vals))
{}


template<typename T>
ArrayRef<T>::ArrayRef(const std::vector<T> &Vec)
	:_vals(
            Vec.size() == 0 ?
            intrusive_ptr<intrusive_list<T>>(nullptr) :
            make_intrusive<intrusive_list<T>>(Vec.size())
    )
{
	if(!_empty){std::copy(Vec.cbegin(), Vec.cend(), _vals->begin());}
	else{_vals = nullptr;}
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
	:_vals(Vec.size() == 0 
            ? intrusive_ptr<intrusive_list<T>>(nullptr)
            : make_intrusive<intrusive_list<T>>(Vec)
    )
{}

template<typename T>
ArrayRef<T>& ArrayRef<T>::operator=(const ArrayRef<T>& Arr){
    _vals = Arr._vals;
	return *this;
}

template<typename T>
ArrayRef<T>& ArrayRef<T>::operator=(ArrayRef<T>&& Arr){
	_vals = std::move(Arr._vals);
    return *this;
}

template<typename T>
bool ArrayRef<T>::operator==(const ArrayRef<T> &Arr) const {
	if(Arr._vals == nullptr || _vals == nullptr){return false;}
	if(Arr._vals->size() != _vals->size())
		return false;
	return std::equal(begin(), end(), Arr.begin());
}

template<typename T>
bool ArrayRef<T>::operator!=(const ArrayRef<T> &Arr) const {
    return !(*this == Arr);
}

template<typename T>
const T* ArrayRef<T>::data() const{return _vals->ptr();}
template<typename T>
size_t ArrayRef<T>::size() const {return _vals->size();}
template<typename T>
const T& ArrayRef<T>::front() const {return _vals->front();}
template<typename T>
const T& ArrayRef<T>::back() const {return _vals->back();}

template<typename T>
T& ArrayRef<T>::front() {return _vals->front();}
template<typename T>
T& ArrayRef<T>::back() {return _vals->back();}


template<typename T>
const T* ArrayRef<T>::begin() const {return _vals->begin();}
template<typename T>
const T* ArrayRef<T>::end() const {return _vals->end();}
template<typename T>
const T* ArrayRef<T>::cbegin() const {return _vals->cbegin();}
template<typename T>
const T* ArrayRef<T>::cend() const {return _vals->cend();}
template<typename T>
typename ArrayRef<T>::reverse_iterator ArrayRef<T>::rbegin() const {return reverse_iterator(_vals->end());}
template<typename T>
typename ArrayRef<T>::reverse_iterator ArrayRef<T>::rend() const {return reverse_iterator(_vals->begin());}
template<typename T>
bool ArrayRef<T>::empty() const {return _vals == nullptr;}
template<typename T>
const T& ArrayRef<T>::operator[](size_t index) const{
	index = index < 0 ? size() + index : index;
	return _vals->at(index);
}
template<typename T>
T& ArrayRef<T>::operator[](size_t index){
	index = index < 0 ? size() + index : index;
	return _vals->at(index);
}

template<typename T>
const T& ArrayRef<T>::at(size_t index) const {
    assert(index < _total_size);
    return _vals->at(index);
}

template<typename T>
std::vector<T> ArrayRef<T>::to_vec() const {
    std::vector<T> out(this->size());
    std::copy(this->cbegin(), this->cend(), out.begin());
    return std::move(out);
}

template<typename T>
ArrayRef<T> ArrayRef<T>::permute(const std::vector<uint32_t> &Vec) const {
	assert(Vec.size() == this->size());
    ArrayRef out(this->size());
    T* out_ptr_ = out._vals->ptr();
    for(int64_t i = 0; i < this->size(); ++i){
        out_ptr_[i] = this->at(Vec[i]);
    }
    return std::move(out);	
}



template<typename T>
T ArrayRef<T>::multiply() const{
    return std::accumulate(cbegin(), cend(), 
            T(1), std::multiplies<T>());
}

template<typename T>
ArrayRef<T> ArrayRef<T>::pop_front() const {
    if(this->empty() || this->size() == 1)
        return ArrayRef<T>();
    return ArrayRef<T>(this->cbegin() + 1, this->size() - 1);
}


template<typename T>
ArrayRef<T> ArrayRef<T>::clone() const {
    return ArrayRef<T>(this->cbegin(), this->size());
}


template<typename T>
T* ArrayRef<T>::d_data(){return _vals->ptr();}

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
