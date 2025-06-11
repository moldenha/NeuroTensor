#include "_DTYPE_ARRAYTENSOR.h"
#include "DType.h"


namespace nt{

ArrayTensorTypes::value_t* ArrayTensorTypes::my_end(void *begin){
	value_t* casted = reinterpret_cast<value_t*>(begin);
	return casted + size;
}

const ArrayTensorTypes::value_t* ArrayTensorTypes::my_end_c(const void* begin) const{
	const value_t* casted = reinterpret_cast<const value_t*>(begin);
	return casted + size;
}

ArrayTensorTypes::ArrayTensorTypes(){}

void ArrayTensorTypes::set(const d_type& val, void* begin){
	value_t* _beg = reinterpret_cast<value_t*>(begin);
	value_t* _end = my_end(begin);
	if(val.data.index() == 4){
		const Tensor& t = std::get<4>(val.data).get();
		std::fill(_beg, _end, t);
		return;
	}
	std::fill(_beg, _end, val);
}

std::shared_ptr<void> ArrayTensorTypes::make_shared(size_t _size){
	size = _size;
	return std::make_unique<value_t[]>(_size);
}

void* ArrayTensorTypes::end_ptr(void* ptr){
	value_t* casted = reinterpret_cast<value_t*>(ptr);
	return casted + size;
}

const void* ArrayTensorTypes::cend_ptr(const void* ptr) const{
	const value_t* casted = reinterpret_cast<const value_t*>(ptr);
	return casted + size;
}

std::shared_ptr<void> ArrayTensorTypes::share_from(const std::shared_ptr<void>& ptr, uint32_t x) const{
	const std::shared_ptr<value_t[]> *p = reinterpret_cast<const std::shared_ptr<value_t[]>*>(&ptr);
	return std::shared_ptr<value_t>((*p), &(*p)[x]);
}

void ArrayTensorTypes::make_size(size_t _size) {size = _size;}

d_type_list ArrayTensorTypes::begin(void* inp){
	return d_type_list(reinterpret_cast<value_t*>(inp));
}

d_type_list ArrayTensorTypes::end(void* inp){
	return d_type_list(my_end(inp));
}

d_type_list ArrayTensorTypes::cbegin(const void* inp) const{
	return d_type_list(reinterpret_cast<const value_t*>(inp));
}

d_type_list ArrayTensorTypes::cend(const void* inp) const{
	return d_type_list(my_end_c(inp));
}

}


