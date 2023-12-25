#include "Itterator.h"
#include "dtype/DType.h"
#include <_types/_uint32_t.h>
#include <iostream>

namespace nt{
template<typename A>
static_mat_iterator<A>::static_mat_iterator(pointer ptr)
	:m_ptr(ptr)
{}

template<typename A>
static_mat_iterator<A>& static_mat_iterator<A>::operator++(){
	m_ptr++;
	return *this;
}

template<typename A>
typename static_mat_iterator<A>::reference static_mat_iterator<A>::operator*() const {return *m_ptr;}

template<typename A>
typename static_mat_iterator<A>::reference static_mat_iterator<A>::operator[](uint32_t col) const {return *(m_ptr+col);}

template<typename A>
typename static_mat_iterator<A>::pointer static_mat_iterator<A>::operator->() const {return m_ptr;}

template<typename A>
typename static_mat_iterator<A>::pointer static_mat_iterator<A>::m_pt() const {return m_ptr;}

template<typename A>
typename static_mat_iterator<A>::pointer static_mat_iterator<A>::operator+(int i) {return m_ptr + i;}

template<typename A>
static_mat_iterator<A> static_mat_iterator<A>::operator++(int){
	static_mat_iterator<A> tmp = *this;
	++(*this);
	return tmp;
}

template<typename A>
bool static_mat_iterator<A>::operator==(const static_mat_iterator& b){
	return m_ptr == b.m_ptr;
}

template<typename A>
bool static_mat_iterator<A>::operator!=(const static_mat_iterator& b){
	return m_ptr != b.m_ptr;
}

template class static_mat_iterator<uint64_t>;
template class static_mat_iterator<uint32_t>;
template class static_mat_iterator<uint16_t>;
template class static_mat_iterator<uint8_t>;
template class static_mat_iterator<int64_t>;
template class static_mat_iterator<int32_t>;
template class static_mat_iterator<int16_t>;
template class static_mat_iterator<int8_t>;
template class static_mat_iterator<double>;
template class static_mat_iterator<float>;
template class static_mat_iterator<Tensor>;


//Tensor(std::shared_ptr<float[]>, std::shared_ptr<SizeRef>, uint32_t)

d_type_list& static_tensor_iterator::m_pt() {return m_ptr;}

Tensor static_tensor_iterator::operator*(){
	std::shared_ptr<SizeRef> a(const_cast<SizeRef*>(_size.get()), [](SizeRef* _v){});
	return Tensor(ArrayVoid(m_ptr.g_ptr(), val_add, m_ptr.d_type()), a);
}

Tensor static_tensor_iterator::operator[](uint32_t col){
	std::shared_ptr<SizeRef> a(const_cast<SizeRef*>(_size.get()), [](SizeRef* _v){});
	return Tensor(ArrayVoid((m_ptr + (val_add * col)).g_ptr(), val_add, m_ptr.d_type()), a);
}

Tensor static_tensor_iterator::operator->(){
	std::shared_ptr<SizeRef> a(const_cast<SizeRef*>(_size.get()), [](SizeRef* _v){});
	return Tensor(ArrayVoid(m_ptr.g_ptr(), val_add, m_ptr.d_type()), a);
}


Tensor static_tensor_iterator::operator+(int i){
	return Tensor(ArrayVoid((m_ptr + (val_add * i)).g_ptr(), val_add, m_ptr.d_type()), _size);
} 

bool static_tensor_iterator::operator==(const static_tensor_iterator& b) const {return b.m_ptr == m_ptr;}
bool static_tensor_iterator::operator!=(const static_tensor_iterator& b) const {return b.m_ptr != m_ptr;}

static_tensor_iterator& static_tensor_iterator::operator++(){
	m_ptr += val_add;
	return *this;
}

static_tensor_iterator static_tensor_iterator::operator++(int){
	static_tensor_iterator tmp = *this;
	++(*this);
	return tmp;
}

static_tensor_iterator::static_tensor_iterator(d_type_list ptr, SizeRef s, uint32_t va)
	:m_ptr(ptr),
	_size(std::make_shared<SizeRef>(std::move(s))),
	val_add(va)
{}

static_tensor_iterator::static_tensor_iterator(d_type_list ptr, SizeRef s)
	:m_ptr(ptr),
	_size(std::make_shared<SizeRef>(std::move(s)))
{
	val_add = _size->multiply();
}


}
