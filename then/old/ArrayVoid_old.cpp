
#include <memory.h>
#include <utility>

#include "DType.h"
#include "DType_enum.h"
#include "_DTYPE_ARRAYDOUBLE.h"
#include "_DTYPE_ARRAYFLOAT.h"
#include "_DTYPE_ARRAYLONG.h"
#include "_DTYPE_ARRAYTENSOR.h"
#include "ArrayVoid.h"


namespace nt{

std::unique_ptr<ArrayIntTypes> ArrayVoid::get_typed(DType _type) const{
	switch(_type){
		case DType::Integer:
			return std::make_unique<ArrayIntTypes>();
		case DType::Double:
			return std::make_unique<ArrayDoubleTypes>();
		case DType::Float:
			return std::make_unique<ArrayFloatTypes>();
		case DType::Long:
			return std::make_unique<ArrayLongTypes>();
		case DType::TensorObj:
			return std::make_unique<ArrayTensorTypes>();
		default:
			return std::make_unique<ArrayIntTypes>();
	}

}

ArrayVoid::ArrayVoid(uint32_t _size, DType _t)
	:typed(get_typed(_t)),
	_vals(nullptr),
	dtype(_t),
	size(_size)
{_vals = typed->make_shared(_size);}

ArrayVoid::ArrayVoid(void* ptr, uint32_t _size, DType _t)
	:typed(get_typed(_t)),
	_vals(ptr, [](void *p){}),
	dtype(_t),
	size(_size)
{typed->make_size(_size);}

ArrayVoid& ArrayVoid::operator=(const ArrayVoid &Arr){
	_vals = Arr._vals;
	typed = get_typed(Arr.dtype);
	size = Arr.size;
	dtype = Arr.dtype;
	typed->make_size(size);
	return *this;
}

ArrayVoid& ArrayVoid::operator=(ArrayVoid&& Arr){
	_vals = std::move(Arr._vals);
	typed = std::move(Arr.typed);
	size = Arr.size;
	dtype = Arr.dtype;
	return *this;
}

ArrayVoid::ArrayVoid(const ArrayVoid& Arr)
	:_vals(Arr._vals),
	typed(get_typed(Arr.dtype)),
	size(Arr.size),
	dtype(Arr.dtype)
{typed->make_size(size);}

ArrayVoid::ArrayVoid(ArrayVoid&& Arr)
	:_vals(std::move(Arr._vals)),
	typed(std::move(Arr.typed)),
	size(std::exchange(Arr.size, 0)),
	dtype(Arr.dtype)
{}

ArrayVoid::ArrayVoid(const std::shared_ptr<void>& _v, std::unique_ptr<ArrayIntTypes>&& _t, uint32_t _s, DType _dt)
	:_vals(_v),
	typed(std::move(_t)),
	size(_s),
	dtype(_dt)
{
	typed->make_size(_s);
}

const uint32_t ArrayVoid::Size() const{
	return size;
}

const void* ArrayVoid::data_ptr() const {return _vals.get();}
void* ArrayVoid::data_ptr() {return _vals.get();}
void ArrayVoid::operator=(const d_type& val){
	typed->set(val, data_ptr());
}


std::shared_ptr<void> ArrayVoid::share_part(uint32_t index) const{
	return typed->share_from(_vals, index);
}

ArrayVoid ArrayVoid::share_array(uint32_t index) const{
	assert(index < size);
	std::shared_ptr<void> nv = typed->share_from(_vals, index);
	uint32_t ns = size - index;
	return ArrayVoid(nv, get_typed(dtype), ns, dtype);
}

ArrayVoid ArrayVoid::share_array(uint32_t index, uint32_t ns) const{
	assert(index < size);
	std::shared_ptr<void> nv = typed->share_from(_vals, index);
	return ArrayVoid(nv, get_typed(dtype), ns, dtype);
}

void* ArrayVoid::begin_ptr() {return data_ptr();}
void* ArrayVoid::end_ptr() {return typed->end_ptr(data_ptr());}
const void* ArrayVoid::cbegin_ptr() const {return data_ptr();}
const void* ArrayVoid::cend_ptr() const {return typed->cend_ptr(data_ptr());}
const uint32_t ArrayVoid::use_count() const {return _vals.use_count();}
d_type_list ArrayVoid::begin(){
	return typed->begin(data_ptr());
}
d_type_list ArrayVoid::end(){
	return typed->end(data_ptr());
}

d_type_list ArrayVoid::cbegin() const{
	return typed->cbegin(data_ptr());
}
d_type_list ArrayVoid::cend() const{
	return typed->cend(data_ptr());
}


}
