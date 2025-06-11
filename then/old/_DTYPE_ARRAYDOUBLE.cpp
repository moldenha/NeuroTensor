#include "_DTYPE_ARRAYDOUBLE.h"
#include "DType.h"

namespace nt{
ArrayDoubleTypes::ArrayDoubleTypes(){}

void ArrayDoubleTypes::set(const d_type& val, void* begin){
	value_t _v = val.cast_num<value_t>();
	value_t* _beg = reinterpret_cast<value_t*>(begin);
	value_t* _end = my_end(begin);
	std::fill(_beg, _end, _v);
}

std::shared_ptr<void> ArrayDoubleTypes::make_shared(size_t _size){
	size = _size;
	return std::make_unique<value_t[]>(_size);
}
void ArrayDoubleTypes::make_size(size_t _size) {size = _size;}

void* ArrayDoubleTypes::end_ptr(void* ptr){
	value_t* casted = reinterpret_cast<value_t*>(ptr);
	return casted + size;
}

const void* ArrayDoubleTypes::cend_ptr(const void* ptr) const{
	const value_t* casted = reinterpret_cast<const value_t*>(ptr);
	return casted + size;
}

std::shared_ptr<void> ArrayDoubleTypes::share_from(const std::shared_ptr<void>& ptr, uint32_t x) const{
	const std::shared_ptr<value_t[]> *p = reinterpret_cast<const std::shared_ptr<value_t[]>*>(&ptr);
	return std::shared_ptr<value_t>((*p), &(*p)[x]);
}
ArrayDoubleTypes::value_t* ArrayDoubleTypes::my_end(void* begin){
	value_t* casted = reinterpret_cast<value_t*>(begin);
	return casted + size;
}

const ArrayDoubleTypes::value_t* ArrayDoubleTypes::my_end_c(const void* begin) const{
	const value_t* casted = reinterpret_cast<const value_t*>(begin);
	return casted + size;
}

d_type_list ArrayDoubleTypes::cbegin(const void* inp) const{
	return d_type_list(reinterpret_cast<const value_t*>(inp));
}

d_type_list ArrayDoubleTypes::cend(const void* inp) const{
	return d_type_list(my_end_c(inp));
}
d_type_list ArrayDoubleTypes::begin(void* inp){
	return d_type_list(reinterpret_cast<value_t*>(inp));
}

d_type_list ArrayDoubleTypes::end(void* inp){
	return d_type_list(my_end(inp));
}

}
