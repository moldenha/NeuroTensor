



#include <memory.h>
#include <new>


#include <utility>

#include "../Tensor.h"
#include "DType.h"
#include "DType_enum.h"
#include "ArrayVoid.h"
#include "ArrayVoid.hpp"
#include "DType_list.h"
#include <functional>
#include <numeric>
#include "Scalar.h"
#include "../utils/utils.h"
#include "compatible/DType_compatible_all.h"
#include "../types/Types.h"
#include "DType_operators.h"
//%s/"../convert/Convert.h"/"..\/convert\/Convert.h"

namespace nt{

std::shared_ptr<void> ArrayVoid::make_shared(size_t _size, DType _type) const{return DTypeFuncs::make_shared_array(_size, _type);}

const size_t ArrayVoid::dtype_size(DType _type) const{
	return DTypeFuncs::size_of_dtype(_type);
}



ArrayVoid::ArrayVoid(uint32_t _size, DType _t)
	:_vals(this->make_shared(_size, _t)),
	dtype(_t),
	size(_size),
	type_size(this->dtype_size(_t)),
	_last_index(_size),
	_strides(std::shared_ptr<void*>((void**)calloc(_size, DTypeFuncs::size_of_dtype_p(_t)), free)),
	_start(0),
	available_size(_size)
{
	DTypeFuncs::initialize_strides(_strides.get(), _vals.get(), _size, _t);	
}

ArrayVoid& ArrayVoid::operator=(const ArrayVoid &Arr){
	_vals = Arr._vals;
	size = Arr.size;
	dtype = Arr.dtype;
	type_size = Arr.type_size;
	_last_index = Arr._last_index;
	_strides = Arr._strides;
	_start = Arr._start;
	available_size = Arr.available_size;
	return *this;
}

ArrayVoid& ArrayVoid::operator=(ArrayVoid&& Arr){
	_vals = std::move(Arr._vals);
	size = Arr.size;
	dtype = Arr.dtype;
	type_size = Arr.type_size;
	_last_index = Arr._last_index;
	_strides = std::move(Arr._strides);
	_start = Arr._start;
	available_size = Arr.available_size;
	return *this;
}

ArrayVoid::ArrayVoid(const ArrayVoid& Arr)
	:_vals(Arr._vals),
	size(Arr.size),
	dtype(Arr.dtype),
	type_size(Arr.type_size),
	_last_index(Arr._last_index),
	_strides(Arr._strides),
	_start(Arr._start),
	available_size(Arr.available_size)
{}

ArrayVoid::ArrayVoid(ArrayVoid&& Arr)
	:_vals(std::move(Arr._vals)),
	size(std::exchange(Arr.size, 0)),
	dtype(Arr.dtype),
	type_size(std::exchange(Arr.type_size, 0)),
	_last_index(std::exchange(Arr._last_index, 0)),
	_strides(std::move(Arr._strides)),
	_start(std::exchange(Arr._start, 0)),
	available_size(std::exchange(Arr.available_size, 0))
{}

ArrayVoid::ArrayVoid(const std::shared_ptr<void>& _v, const std::shared_ptr<void*>& str, const std::size_t size, const std::size_t start, std::size_t avail, DType _dt)
	:_vals(_v),
	size(size),
	dtype(_dt),
	type_size(this->dtype_size(_dt)),
	_last_index(size + start),
	_strides(str),
	_start(start),
	available_size(avail)
{}

ArrayVoid::ArrayVoid(const std::shared_ptr<void>& _v, std::shared_ptr<void*>&& str, const std::size_t size, const std::size_t start, std::size_t avail, DType _dt)
	:_vals(_v),
	size(size),
	dtype(_dt),
	type_size(this->dtype_size(_dt)),
	_last_index(size + start),
	_strides(std::move(str)),
	_start(start),
	available_size(avail)
{}

const std::size_t ArrayVoid::Size() const{
	return size;
}

const void* ArrayVoid::data_ptr() const {return reinterpret_cast<uint8_t*>(_vals.get()) + (_start * type_size);}
void* ArrayVoid::data_ptr() {return reinterpret_cast<uint8_t*>(_vals.get()) + (_start * type_size) ;}
void* ArrayVoid::data_ptr_end() {return reinterpret_cast<uint8_t*>(_vals.get()) + (_last_index * type_size);}
const void* ArrayVoid::data_ptr_end() const {return reinterpret_cast<uint8_t*>(_vals.get()) + (_last_index * type_size);}
void** ArrayVoid::strides_cbegin() const {return _strides.get() + _start;}
void** ArrayVoid::strides_begin() {return _strides.get() + _start;}
void** ArrayVoid::strides_end() {return _strides.get() + _last_index;}
void** ArrayVoid::strides_cend() const {return _strides.get() + _last_index;}

void ArrayVoid::resize(const size_t inp){
	utils::throw_exception(inp <= size, "Runtime Error: New size cannot be greater than old size $, this function is meant to restrict view not increase it", size);
	size_t dif = size - inp;
	size = inp;
	_last_index -= dif;
}

void ArrayVoid::operator=(Scalar val){
	if(dtype != DType::TensorObj)
		this->execute_function<WRAP_DTYPES<NumberTypesL>>()([&val](auto begin, auto end){
			auto v = val.to<typename std::remove_const<typename decltype(begin)::value_type>::type>();
			std::fill(begin, end, v);
		});
	else{	
		std::fill(tbegin<Tensor>(), tend<Tensor>(), val);
	}
}


std::shared_ptr<void> ArrayVoid::share_part(uint32_t index) const{
	return DTypeFuncs::share_part_ptr(index, dtype, _vals);
}

ArrayVoid ArrayVoid::share_array(uint32_t index) const{
	utils::throw_exception(index <= size, "\nRuntime Error: cannot grab array from index $ with size of only $ ArrayVoid::share_array(index)",index, size);
	return ArrayVoid(_vals, _strides, size - index, index + _start, available_size, dtype);
}

ArrayVoid ArrayVoid::share_array(uint32_t index, uint32_t ns) const{
	utils::throw_exception(ns <= size, "\nRuntime Error: cannot grab array with start of $ with size of only $ ArrayVoid::share_array(index, ns)",ns, size);
	utils::throw_exception(index <= size, "\nRuntime Error: cannot grab array from index $ with size of only $ ArrayVoid::share_array(index, ns)",index, size);	
	return ArrayVoid(_vals, _strides, ns, index + _start, available_size, dtype);
	/* return ArrayVoid(_vals, make_unique_strides(index, index + ns), ns, 0, available_size, dtype); */
}
/* const std::vector<uint64_t>& ArrayVoid::get_strides() const{return _strides;} */
/* std::vector<uint64_t>& ArrayVoid::get_strides(){return _strides;} */

ArrayVoid ArrayVoid::change_stride(const std::vector<std::size_t>& val){
	utils::throw_exception(val.size() <= available_size, "\nRuntime Error: Expected to have permutation index of size at most $ but got $ ArrayVoid::change_stride", available_size, val.size());
	//std::shared_ptr<void*> n_strides = make_unique_strides(); <- this way is honestly a bit memory a performance inefficient
	//the reason shared_ptr's are used in the first place is so that memory isn't coppied when one makes a smaller tensor, or defines a tensor to be the same as another
	//However, those functions don't change the size of the strides
	//So, instead, this will make it so that the size of the strides will actually change memory wise, and then start = 0, and stop at the new size
	std::shared_ptr<void*> n_strides = make_unique_strides(_start, _last_index);
	void** original = n_strides.get();
	void** arr = n_strides.get();
	for(uint64_t i = 0; i < val.size(); ++i){
		arr[i] = original[val[i]];	
	}
	return ArrayVoid(_vals, std::move(n_strides), size, 0, available_size, dtype);
}
ArrayVoid ArrayVoid::range(std::vector<my_range> ranges) const{
	const size_t old_size = size;
	std::for_each(ranges.begin(), ranges.end(), [&old_size](auto& val){val.fix(old_size);});
	uint32_t n_s = 0;
	std::for_each(ranges.cbegin(), ranges.cend(), [&n_s](const auto& val){n_s += val.length();});
	void** original = _strides.get();
	std::shared_ptr<void*> n_strides = make_unique_strides(false);
	void** arr = n_strides.get() + _start;
	std::size_t n_size = 0;
	for(const my_range &x : ranges){
		for(uint32_t j = x.begin + 1; j < x.end; ++j){
			*arr = original[j];
			++arr;
			++n_size;
		}
	}
	return ArrayVoid(_vals, std::move(n_strides), n_size, _start, available_size, dtype);
}

bool ArrayVoid::is_contiguous() const{
	void** arr = _strides.get();
	for(uint64_t i = 1; i < available_size; ++i){
		if(reinterpret_cast<char*>(arr[i]) != reinterpret_cast<char*>(arr[i-1]) + type_size)
			return false;
	}
	return true;
}

ArrayVoid ArrayVoid::contiguous() const{
	ArrayVoid n_vals(size, dtype);
	this->copy(n_vals);
	return std::move(n_vals);
}

const uint32_t ArrayVoid::use_count() const {return _vals.use_count();}
template<typename T>
tdtype_list<T> ArrayVoid::tbegin(){return tdtype_list<T>(strides_begin());}
template<typename T>
tdtype_list<T> ArrayVoid::tend(){return tdtype_list<T>(strides_end());}
template<typename T>
tdtype_list<const T> ArrayVoid::tcbegin() const {return tdtype_list<const T>(strides_cbegin());}
template<typename T>
tdtype_list<const T> ArrayVoid::tcend() const {return tdtype_list<const T>(strides_cend());}

template tdtype_list<float> ArrayVoid::tbegin();
template tdtype_list<double> ArrayVoid::tbegin();
template tdtype_list<complex_64> ArrayVoid::tbegin();
template tdtype_list<complex_128> ArrayVoid::tbegin();
template tdtype_list<uint32_t> ArrayVoid::tbegin();
template tdtype_list<int32_t> ArrayVoid::tbegin();
template tdtype_list<uint16_t> ArrayVoid::tbegin();
template tdtype_list<int16_t> ArrayVoid::tbegin();
template tdtype_list<uint8_t> ArrayVoid::tbegin();
template tdtype_list<int8_t> ArrayVoid::tbegin();
template tdtype_list<int64_t> ArrayVoid::tbegin();
template tdtype_list<Tensor> ArrayVoid::tbegin();
template tdtype_list<uint_bool_t> ArrayVoid::tbegin();
#ifdef _HALF_FLOAT_SUPPORT_
template tdtype_list<float16_t> ArrayVoid::tbegin();
template tdtype_list<complex_32> ArrayVoid::tbegin();
#endif
#ifdef _128_FLOAT_SUPPORT_
template tdtype_list<float128_t> ArrayVoid::tbegin();
#endif
#ifdef __SIZEOF_INT128__
template tdtype_list<int128_t> ArrayVoid::tbegin();
#endif

template tdtype_list<float> ArrayVoid::tend();
template tdtype_list<double> ArrayVoid::tend();
template tdtype_list<complex_64> ArrayVoid::tend();
template tdtype_list<complex_128> ArrayVoid::tend();
template tdtype_list<uint32_t> ArrayVoid::tend();
template tdtype_list<int32_t> ArrayVoid::tend();
template tdtype_list<uint16_t> ArrayVoid::tend();
template tdtype_list<int16_t> ArrayVoid::tend();
template tdtype_list<uint8_t> ArrayVoid::tend();
template tdtype_list<int8_t> ArrayVoid::tend();
template tdtype_list<int64_t> ArrayVoid::tend();
template tdtype_list<Tensor> ArrayVoid::tend();
template tdtype_list<uint_bool_t> ArrayVoid::tend();

#ifdef _HALF_FLOAT_SUPPORT_
template tdtype_list<float16_t> ArrayVoid::tend();
template tdtype_list<complex_32> ArrayVoid::tend();
#endif
#ifdef _128_FLOAT_SUPPORT_
template tdtype_list<float128_t> ArrayVoid::tend();
#endif
#ifdef __SIZEOF_INT128__
template tdtype_list<int128_t> ArrayVoid::tend();
#endif



template tdtype_list<const float> ArrayVoid::tcbegin() const;
template tdtype_list<const double> ArrayVoid::tcbegin() const;
template tdtype_list<const complex_64> ArrayVoid::tcbegin() const;
template tdtype_list<const complex_128> ArrayVoid::tcbegin() const;
template tdtype_list<const uint32_t> ArrayVoid::tcbegin() const;
template tdtype_list<const int32_t> ArrayVoid::tcbegin() const;
template tdtype_list<const uint16_t> ArrayVoid::tcbegin() const;
template tdtype_list<const int16_t> ArrayVoid::tcbegin() const;
template tdtype_list<const uint8_t> ArrayVoid::tcbegin() const;
template tdtype_list<const int8_t> ArrayVoid::tcbegin() const;
template tdtype_list<const int64_t> ArrayVoid::tcbegin() const;
template tdtype_list<const Tensor> ArrayVoid::tcbegin() const;
template tdtype_list<const uint_bool_t> ArrayVoid::tcbegin() const;

#ifdef _HALF_FLOAT_SUPPORT_
template tdtype_list<const float16_t> ArrayVoid::tcbegin() const;
template tdtype_list<const complex_32> ArrayVoid::tcbegin() const;
#endif
#ifdef _128_FLOAT_SUPPORT_
template tdtype_list<const float128_t> ArrayVoid::tcbegin() const;
#endif
#ifdef __SIZEOF_INT128__
template tdtype_list<const int128_t> ArrayVoid::tcbegin() const;
#endif

template tdtype_list<const float> ArrayVoid::tcend() const;
template tdtype_list<const double> ArrayVoid::tcend() const;
template tdtype_list<const complex_64> ArrayVoid::tcend() const;
template tdtype_list<const complex_128> ArrayVoid::tcend() const;
template tdtype_list<const uint32_t> ArrayVoid::tcend() const;
template tdtype_list<const int32_t> ArrayVoid::tcend() const;
template tdtype_list<const uint16_t> ArrayVoid::tcend() const;
template tdtype_list<const int16_t> ArrayVoid::tcend() const;
template tdtype_list<const uint8_t> ArrayVoid::tcend() const;
template tdtype_list<const int8_t> ArrayVoid::tcend() const;
template tdtype_list<const int64_t> ArrayVoid::tcend() const;
template tdtype_list<const Tensor> ArrayVoid::tcend() const;
template tdtype_list<const uint_bool_t> ArrayVoid::tcend() const;

#ifdef _HALF_FLOAT_SUPPORT_
template tdtype_list<const float16_t> ArrayVoid::tcend() const;
template tdtype_list<const complex_32> ArrayVoid::tcend() const;
#endif
#ifdef _128_FLOAT_SUPPORT_
template tdtype_list<const float128_t> ArrayVoid::tcend() const;
#endif
#ifdef __SIZEOF_INT128__
template tdtype_list<const int128_t> ArrayVoid::tcend() const;
#endif

/* dtype_list ArrayVoid::begin(){ */
/* 	return dtype_list(data_ptr(), dtype); */
/* } */
/* dtype_list ArrayVoid::end(){ */
/* 	return dtype_list(data_ptr_end(), dtype); */
/* } */

/* const_dtype_list ArrayVoid::cbegin() const{ */
/* 	return const_dtype_list(data_ptr(), dtype); */
/* } */
/* const_dtype_list ArrayVoid::cend() const{ */
/* 	return const_dtype_list(data_ptr_end(), dtype); */
/* } */

/* const std::size_t* ArrayVoid::stride_cbegin() const{return &(*_strides)[_start];} */
/* const std::size_t* ArrayVoid::stride_cend() const{return &(*_strides)[_last_index];} */
/* std::size_t* ArrayVoid::stride_begin() {return &(*_strides)[_start];} */
/* std::size_t* ArrayVoid::stride_end() {return &(*_strides)[_last_index];} */

/* std::vector<std::size_t>::const_iterator ArrayVoid::stride_it_cbegin() const {return _strides->cbegin() + _start;} */
/* std::vector<std::size_t>::const_iterator ArrayVoid::stride_it_cend() const {return _strides->cbegin() + _last_index;} */
/* std::vector<std::size_t>::iterator ArrayVoid::stride_it_begin(){return _strides->begin() + _start;} */
/* std::vector<std::size_t>::iterator ArrayVoid::stride_it_end(){return _strides->begin() + _last_index;} */

/* template<DType dt> */
ArrayVoid ArrayVoid::copy_strides(bool copy) const{
	/* if(dt != dtype) */
	/* 	return copy_strides<DTypeFuncs::next_dtype_it<dt>>(); */

	std::shared_ptr<void*> cpy_str((void**)calloc(size, DTypeFuncs::size_of_dtype_p(dtype)), free);
	if(copy){
		void** original = strides_cbegin();
		void** destination = cpy_str.get();
		for(uint64_t i = 0; i < size; ++i, ++original, ++destination)
			*destination = *original;
	}
	return ArrayVoid(_vals, cpy_str, size, 0, available_size, dtype);
}


ArrayVoid ArrayVoid::new_stride(uint32_t size) const{
	std::shared_ptr<void*> n_str((void**)calloc(size, DTypeFuncs::size_of_dtype_p(dtype)), free);
	return ArrayVoid(_vals, n_str, size, 0, available_size, dtype);
}

std::shared_ptr<void*> ArrayVoid::make_unique_strides(bool pre_order) const{
	if(!pre_order){
		return std::shared_ptr<void*>((void**)calloc(available_size, DTypeFuncs::size_of_dtype_p(dtype)), free);
	}
	std::shared_ptr<void*> cpy_str((void**)calloc(available_size, DTypeFuncs::size_of_dtype_p(dtype)), free);
	void** original = _strides.get();
	void** destination = cpy_str.get();
	for(uint64_t i = 0; i < available_size; ++i)
		destination[i] = original[i];	
	return std::move(cpy_str);
}

std::shared_ptr<void*> ArrayVoid::make_unique_strides(std::size_t start, std::size_t end) const{
	std::size_t total = end - start;
	std::cout<<"making unique from "<<start<<" to "<<end<<std::endl;
	std::shared_ptr<void*> cpy_str((void**)calloc(total, DTypeFuncs::size_of_dtype_p(dtype)), free);
	void** original = strides_cbegin() + start;
	void** destination = cpy_str.get();
	for(uint64_t i = 0; i < total; ++i, ++destination, ++original){
		*destination = *original;
	}
	return std::move(cpy_str);
}




void ArrayVoid::unique_strides(bool pre_order){
	if(_strides.use_count() == 1)
		return;
	if(!pre_order){
		_strides = std::shared_ptr<void*>((void**)calloc(available_size, DTypeFuncs::size_of_dtype_p(dtype)), free);
		return;
	}
	std::shared_ptr<void*> cpy_str((void**)calloc(available_size, DTypeFuncs::size_of_dtype_p(dtype)), free);
	void** original = _strides.get();
	void** destination = cpy_str.get();
	for(uint64_t i = 0; i < available_size; ++i)
		destination[i] = original[i];
	_strides = std::move(cpy_str);
	
}

ArrayVoid& ArrayVoid::iota(Scalar val){
	if(dtype != DType::TensorObj)
		this->execute_function<WRAP_DTYPES<NumberTypesL>>()([&val](auto begin, auto end){std::iota(begin, end, val.to<typename std::remove_const<typename decltype(begin)::value_type>::type>());});
	else{
		Tensor v({1}, DType::Float);
		v = val;
		std::iota(tbegin<Tensor>(), tend<Tensor>(), v);
	}
	return *this;
}

template<DType dt>
bool _my_sub_copy_(ArrayVoid& Arr, const ArrayVoid& my_arr, unsigned long long i){	
	if(my_arr.dtype != dt) return _my_sub_copy_<DTypeFuncs::next_dtype_it<dt>>(Arr, my_arr, i);
	using value_t = DTypeFuncs::dtype_to_type<dt>;
	std::cout << "copying for "<<dt<< " "<< i<<std::endl;
	std::copy(my_arr.tcbegin<value_t>(), my_arr.tcend<value_t>(), Arr.tbegin<value_t>() + i);
	return true;
}

void ArrayVoid::copy(ArrayVoid& Arr, unsigned long long i) const{
	utils::throw_exception(Arr.dtype == dtype, "\nRuntime Error: Expected to copy ArrayVoid to same type $ but got $", dtype, Arr.dtype);
	Arr.execute_function([](auto begin, auto end, const nt::ArrayVoid& arr){
				using value_type = typename decltype(begin)::value_type;
				std::copy(arr.tcbegin<value_type>(), arr.tcend<value_type>(), begin);
			}, *this);
	/* utils::throw_exception(_my_sub_copy_<DType::Integer>(Arr, *this, i), "\nRuntime Error: Was unable to copy ArrayVoid"); */
}

#include "../convert/Convert.h"

template<DType T, DType F, std::enable_if_t<T != DType::TensorObj && F != DType::TensorObj, bool> = true>
bool _my_sub_turn_dtype_(const ArrayVoid& my_arr, ArrayVoid& out){
	if(F != my_arr.dtype){return _my_sub_turn_dtype_<DTypeFuncs::next_dtype_it<F>, T>(my_arr, out);}
	if(T != out.dtype){return _my_sub_turn_dtype_<F, DTypeFuncs::next_dtype_it<T>>(my_arr, out);}
	using my_value_t = ::nt::DTypeFuncs::dtype_to_type_t<F>;
	using out_value_t = ::nt::DTypeFuncs::dtype_to_type_t<T>;
	std::transform(my_arr.tcbegin<my_value_t>(), my_arr.tcend<my_value_t>(), out.tbegin<out_value_t>(), [](const auto& val){return ::nt::convert::convert<T, my_value_t>(val);});
	return true;
}

template<DType T, DType F, std::enable_if_t<T == DType::TensorObj && F != DType::TensorObj, bool> = true>
bool _my_sub_turn_dtype_(const ArrayVoid& my_arr, ArrayVoid& out){
	if(F != my_arr.dtype){return _my_sub_turn_dtype_<DTypeFuncs::next_dtype_it<F>, T>(my_arr, out);}
	if(T != out.dtype){return _my_sub_turn_dtype_<F, DTypeFuncs::next_dtype_it<T>>(my_arr, out);}
	using my_value_t = ::nt::DTypeFuncs::dtype_to_type_t<F>;
	using out_value_t = ::nt::DTypeFuncs::dtype_to_type_t<T>;
	std::transform(my_arr.tcbegin<my_value_t>(), my_arr.tcend<my_value_t>(), out.tbegin<out_value_t>(), 
			[](const auto& val){
			Tensor outp({1}, F);
			outp = val;
			return std::move(outp);
			}
	);
	return true;
}

template<DType T, DType F, std::enable_if_t<T == DType::TensorObj && F == DType::TensorObj, bool> = true>
bool _my_sub_turn_dtype_(const ArrayVoid& my_arr, ArrayVoid& out){
	if(F != my_arr.dtype){return _my_sub_turn_dtype_<DTypeFuncs::next_dtype_it<F>, T>(my_arr, out);}
	if(T != out.dtype){return _my_sub_turn_dtype_<F, DTypeFuncs::next_dtype_it<T>>(my_arr, out);}
	return true;
}


template<DType T, DType F, std::enable_if_t<T != DType::TensorObj && F == DType::TensorObj, bool> = true>
bool _my_sub_turn_dtype_(const ArrayVoid& my_arr, ArrayVoid& out){
	if(F != my_arr.dtype){return _my_sub_turn_dtype_<DTypeFuncs::next_dtype_it<F>, T>(my_arr, out);}
	if(T != out.dtype){return _my_sub_turn_dtype_<F, DTypeFuncs::next_dtype_it<T>>(my_arr, out);}
	using my_value_t = ::nt::DTypeFuncs::dtype_to_type_t<F>;
	using out_value_t = ::nt::DTypeFuncs::dtype_to_type_t<T>;
	std::transform(my_arr.tcbegin<my_value_t>(), my_arr.tcend<my_value_t>(), out.tbegin<out_value_t>(), 
			[](const auto& val){ 
			Scalar s = val.toScalar();
			return s.to<out_value_t>();
			}
	);
	return true;
}


ArrayVoid& ArrayVoid::fill_ptr_(Scalar c){
		switch(dtype){
			case DType::uint32:{
				using value_t = uint32_t;
				value_t* begin = reinterpret_cast<value_t*>(data_ptr());
				value_t* end = reinterpret_cast<value_t*>(data_ptr_end());
				value_t val = c.to<value_t>();
				for(;begin != end; ++begin)
					*begin = val;
				return *this;
			}
			case DType::int32:{
				using value_t = int32_t;
				value_t* begin = reinterpret_cast<value_t*>(data_ptr());
				value_t* end = reinterpret_cast<value_t*>(data_ptr_end());
				value_t val = c.to<value_t>();
				for(;begin != end; ++begin)
					*begin = val;
				return *this;
			}
			case DType::Double:{
				using value_t = double;
				value_t* begin = reinterpret_cast<value_t*>(data_ptr());
				value_t* end = reinterpret_cast<value_t*>(data_ptr_end());
				value_t val = c.to<value_t>();
				for(;begin != end; ++begin)
					*begin = val;
				return *this;
			}
			case DType::Float:{
				using value_t = float;
				value_t* begin = reinterpret_cast<value_t*>(data_ptr());
				value_t* end = reinterpret_cast<value_t*>(data_ptr_end());
				value_t val = c.to<value_t>();
				for(;begin != end; ++begin)
					*begin = val;
				return *this;
			}
			case DType::cfloat:{
				using value_t = complex_64;
				value_t* begin = reinterpret_cast<value_t*>(data_ptr());
				value_t* end = reinterpret_cast<value_t*>(data_ptr_end());
				value_t val = c.to<value_t>();
				for(;begin != end; ++begin)
					*begin = val;
				return *this;
			}
			case DType::cdouble:{
				using value_t = complex_128;
				value_t* begin = reinterpret_cast<value_t*>(data_ptr());
				value_t* end = reinterpret_cast<value_t*>(data_ptr_end());
				value_t val = c.to<value_t>();
				for(;begin != end; ++begin)
					*begin = val;
				return *this;
			}
			case DType::TensorObj:{
				return *this;
			}
			case DType::uint8:{
				using value_t = uint8_t;
				value_t* begin = reinterpret_cast<value_t*>(data_ptr());
				value_t* end = reinterpret_cast<value_t*>(data_ptr_end());
				value_t val = c.to<value_t>();
				for(;begin != end; ++begin)
					*begin = val;
				return *this;
			}
			case DType::int8:{
				using value_t = int8_t;
				value_t* begin = reinterpret_cast<value_t*>(data_ptr());
				value_t* end = reinterpret_cast<value_t*>(data_ptr_end());
				value_t val = c.to<value_t>();
				for(;begin != end; ++begin)
					*begin = val;
				return *this;			
			}
			case DType::uint16:{
				using value_t = uint16_t;
				value_t* begin = reinterpret_cast<value_t*>(data_ptr());
				value_t* end = reinterpret_cast<value_t*>(data_ptr_end());
				value_t val = c.to<value_t>();
				for(;begin != end; ++begin)
					*begin = val;
				return *this;			
			}
			case DType::int16:{
				using value_t = int16_t;
				value_t* begin = reinterpret_cast<value_t*>(data_ptr());
				value_t* end = reinterpret_cast<value_t*>(data_ptr_end());
				value_t val = c.to<value_t>();
				for(;begin != end; ++begin)
					*begin = val;
				return *this;
			}
			case DType::int64:{
				using value_t = int64_t;
				value_t* begin = reinterpret_cast<value_t*>(data_ptr());
				value_t* end = reinterpret_cast<value_t*>(data_ptr_end());
				value_t val = c.to<value_t>();
				for(;begin != end; ++begin)
					*begin = val;
				return *this;
			}
			case DType::Bool:{
				using value_t = uint_bool_t;
				value_t* begin = reinterpret_cast<value_t*>(data_ptr());
				value_t* end = reinterpret_cast<value_t*>(data_ptr_end());
				value_t val = c.to<value_t>();
				for(;begin != end; ++begin)
					*begin = val;
				return *this;
			}
#ifdef _HALF_FLOAT_SUPPORT_
			case DType::Float16:{
				using value_t = float16_t;
				value_t* begin = reinterpret_cast<value_t*>(data_ptr());
				value_t* end = reinterpret_cast<value_t*>(data_ptr_end());
				value_t val = c.to<value_t>();
				for(;begin != end; ++begin)
					*begin = val;
				return *this;
			}
			case DType::Complex32:{
				using value_t = complex_32;
				value_t* begin = reinterpret_cast<value_t*>(data_ptr());
				value_t* end = reinterpret_cast<value_t*>(data_ptr_end());
				value_t val = c.to<value_t>();
				for(;begin != end; ++begin)
					*begin = val;
				return *this;
			}
#endif
#ifdef _128_FLOAT_SUPPORT_
			case DType::Float128:{
				using value_t = float128_t;
				value_t* begin = reinterpret_cast<value_t*>(data_ptr());
				value_t* end = reinterpret_cast<value_t*>(data_ptr_end());
				value_t val = c.to<value_t>();
				for(;begin != end; ++begin)
					*begin = val;
				return *this;
			}
#endif
#ifdef __SIZEOF_INT128__
			case DType::int128:{
				using value_t = int128_t;
				value_t* begin = reinterpret_cast<value_t*>(data_ptr());
				value_t* end = reinterpret_cast<value_t*>(data_ptr_end());
				value_t val = c.to<value_t>();
				for(;begin != end; ++begin)
					*begin = val;
				return *this;
			}
			case DType::uint128:{
				using value_t = uint128_t;
				value_t* begin = reinterpret_cast<value_t*>(data_ptr());
				value_t* end = reinterpret_cast<value_t*>(data_ptr_end());
				value_t val = c.to<value_t>();
				for(;begin != end; ++begin)
					*begin = val;
				return *this;
			}
#endif
	}
}

ArrayVoid ArrayVoid::uint32() const{
	if(dtype == DType::Long)
		return *this;
	ArrayVoid outp(size, DType::Long);
	::nt::utils::throw_exception(_my_sub_turn_dtype_<DType::Long, DType::Integer>(*this, outp), "\nRuntime Error: Unable to convert ArrayVoid of dtype $ to dtype $", dtype, outp.dtype);
	return std::move(outp);
}

ArrayVoid ArrayVoid::int32() const{
	if(dtype == DType::int32)
		return *this;
	ArrayVoid outp(size, DType::int32);
	::nt::utils::throw_exception(_my_sub_turn_dtype_<DType::int32, DType::Integer>(*this, outp), "\nRuntime Error: Unable to convert ArrayVoid of dtype $ to dtype $", dtype, outp.dtype);
	return std::move(outp);
}


ArrayVoid ArrayVoid::Double() const{
	if(dtype == DType::Double)
		return *this;
	ArrayVoid outp(size, DType::Double);
	::nt::utils::throw_exception(_my_sub_turn_dtype_<DType::Double, DType::Integer>(*this, outp), "\nRuntime Error: Unable to convert ArrayVoid of dtype $ to dtype $", dtype, outp.dtype);
	return std::move(outp);
}

ArrayVoid ArrayVoid::Float() const{
	if(dtype == DType::Float)
		return *this;
	ArrayVoid outp(size, DType::Float);
	::nt::utils::throw_exception(_my_sub_turn_dtype_<DType::Float, DType::Integer>(*this, outp), "\nRuntime Error: Unable to convert ArrayVoid of dtype $ to dtype $", dtype, outp.dtype);
	return std::move(outp);
}

ArrayVoid ArrayVoid::cfloat() const{
	if(dtype == DType::cfloat)
		return *this;
	ArrayVoid outp(size, DType::cfloat);
	::nt::utils::throw_exception(_my_sub_turn_dtype_<DType::cfloat, DType::Integer>(*this, outp), "\nRuntime Error: Unable to convert ArrayVoid of dtype $ to dtype $", dtype, outp.dtype);
	return std::move(outp);
}

ArrayVoid ArrayVoid::cdouble() const{
	if(dtype == DType::cdouble)
		return *this;
	ArrayVoid outp(size, DType::cdouble);
	::nt::utils::throw_exception(_my_sub_turn_dtype_<DType::cdouble, DType::Integer>(*this, outp), "\nRuntime Error: Unable to convert ArrayVoid of dtype $ to dtype $", dtype, outp.dtype);
	return std::move(outp);	
}

//s/std::transform(tcbegin<value_t>(), tcend<value_t>(), data_out, [](const auto &val){return static_cast<const out_value_t>(val);});/std::transform(begin, end, data_out, [\&dtype](const auto \&val){\r\t\t\t\tTensor _z({1}, dtype);\r\t\t\t\t_z = val;\r\t\t\t\treturn std::move(_z)});
ArrayVoid ArrayVoid::tensorobj() const{
	if(dtype == DType::TensorObj)
		return *this;
	ArrayVoid outp(size, DType::TensorObj);
	::nt::utils::throw_exception(_my_sub_turn_dtype_<DType::TensorObj, DType::Integer>(*this, outp), "\nRuntime Error: Unable to convert ArrayVoid of dtype $ to dtype $", dtype, outp.dtype);
	return std::move(outp);
}

ArrayVoid ArrayVoid::uint8() const{
	if(dtype == DType::uint8)
		return *this;
	ArrayVoid outp(size, DType::uint8);
	::nt::utils::throw_exception(_my_sub_turn_dtype_<DType::uint8, DType::Integer>(*this, outp), "\nRuntime Error: Unable to convert ArrayVoid of dtype $ to dtype $", dtype, outp.dtype);
	return std::move(outp);
}

ArrayVoid ArrayVoid::int8() const{
	if(dtype == DType::int8)
		return *this;
	ArrayVoid outp(size, DType::int8);
	::nt::utils::throw_exception(_my_sub_turn_dtype_<DType::int8, DType::Integer>(*this, outp), "\nRuntime Error: Unable to convert ArrayVoid of dtype $ to dtype $", dtype, outp.dtype);
	return std::move(outp);
}

ArrayVoid ArrayVoid::uint16() const{
	if(dtype == DType::uint16)
		return *this;
	ArrayVoid outp(size, DType::uint16);
	::nt::utils::throw_exception(_my_sub_turn_dtype_<DType::uint16, DType::Integer>(*this, outp), "\nRuntime Error: Unable to convert ArrayVoid of dtype $ to dtype $", dtype, outp.dtype);
	return std::move(outp);
}

ArrayVoid ArrayVoid::int16() const{
	if(dtype == DType::int16)
		return *this;
	ArrayVoid outp(size, DType::int16);
	::nt::utils::throw_exception(_my_sub_turn_dtype_<DType::int16, DType::Integer>(*this, outp), "\nRuntime Error: Unable to convert ArrayVoid of dtype $ to dtype $", dtype, outp.dtype);
	return std::move(outp);
}

ArrayVoid ArrayVoid::int64() const{
	if(dtype == DType::int64)
		return *this;
	ArrayVoid outp(size, DType::int64);
	::nt::utils::throw_exception(_my_sub_turn_dtype_<DType::int64, DType::Integer>(*this, outp), "\nRuntime Error: Unable to convert ArrayVoid of dtype $ to dtype $", dtype, outp.dtype);
	return std::move(outp);
}

ArrayVoid ArrayVoid::Bool() const{
	if(dtype == DType::Bool)
		return *this;
	ArrayVoid outp(size, DType::Bool);
	::nt::utils::throw_exception(_my_sub_turn_dtype_<DType::Bool, DType::Integer>(*this, outp), "\nRuntime Error: Unable to convert ArrayVoid of dtype $ to dtype $", dtype, outp.dtype);
	return std::move(outp);
}


#ifdef _HALF_FLOAT_SUPPORT_
ArrayVoid ArrayVoid::Float16() const{
	if(dtype == DType::Float16)
		return *this;
	ArrayVoid outp(size, DType::Float16);
	::nt::utils::throw_exception(_my_sub_turn_dtype_<DType::Float16, DType::Integer>(*this, outp), "\nRuntime Error: Unable to convert ArrayVoid of dtype $ to dtype $", dtype, outp.dtype);
	return std::move(outp);
}
ArrayVoid ArrayVoid::Complex32() const{
	if(dtype == DType::Complex32)
		return *this;
	ArrayVoid outp(size, DType::Complex32);
	::nt::utils::throw_exception(_my_sub_turn_dtype_<DType::Complex32, DType::Integer>(*this, outp), "\nRuntime Error: Unable to convert ArrayVoid of dtype $ to dtype $", dtype, outp.dtype);
	return std::move(outp);
}


#endif
#ifdef _128_FLOAT_SUPPORT_
ArrayVoid ArrayVoid::Float128() const{
	if(dtype == DType::Float128)
		return *this;
	ArrayVoid outp(size, DType::Float128);
	::nt::utils::throw_exception(_my_sub_turn_dtype_<DType::Float128, DType::Integer>(*this, outp), "\nRuntime Error: Unable to convert ArrayVoid of dtype $ to dtype $", dtype, outp.dtype);
	return std::move(outp);
}
#endif
#ifdef __SIZEOF_INT128__
ArrayVoid ArrayVoid::Int128() const{
	if(dtype == DType::int128)
		return *this;
	ArrayVoid outp(size, DType::int128);
	::nt::utils::throw_exception(_my_sub_turn_dtype_<DType::int128, DType::Integer>(*this, outp), "\nRuntime Error: Unable to convert ArrayVoid of dtype $ to dtype $", dtype, outp.dtype);
	return std::move(outp);
}
ArrayVoid ArrayVoid::UInt128() const{
	if(dtype == DType::uint128)
		return *this;
	ArrayVoid outp(size, DType::uint128);
	::nt::utils::throw_exception(_my_sub_turn_dtype_<DType::uint128, DType::Integer>(*this, outp), "\nRuntime Error: Unable to convert ArrayVoid of dtype $ to dtype $", dtype, outp.dtype);
	return std::move(outp);
}
#endif



ArrayVoid ArrayVoid::to(DType _dt) const{
	switch(_dt){
		case DType::uint32:
			return uint32();
		case DType::int32:
			return int32();
		case DType::Double:
			return Double();
		case DType::Float:
			return Float();
		case DType::cfloat:
			return cfloat();
		case DType::cdouble:
			return cdouble();
		case DType::TensorObj:
			return tensorobj();
		case DType::uint8:
			return uint8();
		case DType::int8:
			return int8();
		case DType::uint16:
			return uint16();
		case DType::int16:
			return int16();
		case DType::int64:
			return int64();
		case DType::Bool:
			return Bool();
#ifdef _HALF_FLOAT_SUPPORT_
		case DType::Float16:
			return Float16();
		case DType::Complex32:
			return Complex32();
#endif
#ifdef _128_FLOAT_SUPPORT_
		case DType::Float128:
			return Float128();
#endif
#ifdef __SIZEOF_INT128__
		case DType::int128:
			return Int128();
		case DType::uint128:
			return UInt128();
#endif
	}
}

ArrayVoid& ArrayVoid::operator*=(Scalar c){
	utils::throw_exception(dtype != DType::Bool, "*= operation is invalid for DType::Bool");
	if(dtype == DType::TensorObj){
		this->for_each<DType::TensorObj>([&c](auto& inp){inp *= c;});
		return *this;
	}
	else if(DTypeFuncs::is_floating(dtype)){
		double val = c.to<double>();
		this->for_each<WRAP_DTYPES<FloatingTypesL>>([&val](auto& inp){inp *= val;});
		return *this;
	}
	else if(DTypeFuncs::is_integer(dtype)){
		int64_t val = c.to<int64_t>();
		this->for_each<WRAP_DTYPES<IntegerTypesL>>([&val](auto& inp){inp *= val;});
	}
	else if(DTypeFuncs::is_complex(dtype)){
		complex_128 val = c.to<complex_128>();
		this->for_each<WRAP_DTYPES<ComplexTypesL>>([&val](auto& inp){inp *= val;});
	}
	return *this;
}


ArrayVoid& ArrayVoid::operator/=(Scalar c){
	utils::throw_exception(dtype != DType::Bool, "/= operation is invalid for DType::Bool");
	return *this *= c.inverse();
}

ArrayVoid& ArrayVoid::operator-=(Scalar c){
	utils::throw_exception(dtype != DType::Bool, "-= operation is invalid for DType::Bool");
	if(dtype == DType::TensorObj){
		this->for_each<DType::TensorObj>([&c](auto& inp){inp -= c;});
		return *this;
	}
	else if(DTypeFuncs::is_floating(dtype)){
		double val = c.to<double>();
		this->for_each<WRAP_DTYPES<FloatingTypesL>>([&val](auto& inp){inp -= val;});
		return *this;
	}
	else if(DTypeFuncs::is_integer(dtype)){
		int64_t val = c.to<int64_t>();
		this->for_each<WRAP_DTYPES<IntegerTypesL>>([&val](auto& inp){inp -= val;});
	}
	else if(DTypeFuncs::is_complex(dtype)){
		complex_128 val = c.to<complex_128>();
		this->for_each<WRAP_DTYPES<ComplexTypesL>>([&val](auto& inp){inp -= val;});
	}
	return *this;
}

ArrayVoid& ArrayVoid::operator+=(Scalar c){
	utils::throw_exception(dtype != DType::Bool, "+= operation is invalid for DType::Bool");
	if(dtype == DType::TensorObj){
		this->for_each<DType::TensorObj>([&c](auto& inp){inp += c;});
		return *this;
	}
	else if(DTypeFuncs::is_floating(dtype)){
		double val = c.to<double>();
		this->for_each<WRAP_DTYPES<FloatingTypesL>>([&val](auto& inp){inp += val;});
		return *this;
	}
	else if(DTypeFuncs::is_integer(dtype)){
		int64_t val = c.to<int64_t>();
		this->for_each<WRAP_DTYPES<IntegerTypesL>>([&val](auto& inp){inp += val;});
	}
	else if(DTypeFuncs::is_complex(dtype)){
		complex_128 val = c.to<complex_128>();
		this->for_each<WRAP_DTYPES<ComplexTypesL>>([&val](auto& inp){inp += val;});
	}
	return *this;
}



ArrayVoid ArrayVoid::operator*(Scalar c) const{
	utils::throw_exception(dtype != DType::Bool, "* operation is invalid for DType::Bool");
	ArrayVoid output(size, dtype);
	if(dtype == DType::TensorObj){
		output.transform_function<DType::TensorObj>([&c](auto& outp, auto& inp){return inp * c;}, *this);
		return std::move(output);
	}
	else if(DTypeFuncs::is_floating(dtype)){
		double val = c.to<double>();
		output.transform_function<WRAP_DTYPES<FloatingTypesL>>([&val](auto& outp, auto& inp){return inp * val;}, *this);
		return std::move(output);
	}
	else if(DTypeFuncs::is_integer(dtype)){
		int64_t val = c.to<int64_t>();
		output.transform_function<WRAP_DTYPES<IntegerTypesL>>([&val](auto& outp, auto& inp){return inp * val;}, *this);
		return std::move(output);
	}
	else if(DTypeFuncs::is_complex(dtype)){
		complex_128 val = c.to<complex_128>();
		output.transform_function<WRAP_DTYPES<ComplexTypesL>>([&val](auto& outp, auto& inp){return inp * val;}, *this);
		return std::move(output);
	}
	return std::move(output);
}


ArrayVoid ArrayVoid::operator/(Scalar c) const{
	utils::throw_exception(dtype != DType::Bool, "/ operation is invalid for DType::Bool");
	return *this * c.inverse();
}

ArrayVoid ArrayVoid::operator-(Scalar c) const{
	utils::throw_exception(dtype != DType::Bool, "- operation is invalid for DType::Bool");
	ArrayVoid output(size, dtype);
	if(dtype == DType::TensorObj){
		output.transform_function<DType::TensorObj>([&c](auto& outp, auto& inp){return inp - c;}, *this);
		return std::move(output);
	}
	else if(DTypeFuncs::is_floating(dtype)){
		double val = c.to<double>();
		output.transform_function<WRAP_DTYPES<FloatingTypesL>>([&val](auto& outp, auto& inp){return inp - val;}, *this);
		return std::move(output);
	}
	else if(DTypeFuncs::is_integer(dtype)){
		int64_t val = c.to<int64_t>();
		output.transform_function<WRAP_DTYPES<IntegerTypesL>>([&val](auto& outp, auto& inp){return inp - val;}, *this);
		return std::move(output);
	}
	else if(DTypeFuncs::is_complex(dtype)){
		complex_128 val = c.to<complex_128>();
		output.transform_function<WRAP_DTYPES<ComplexTypesL>>([&val](auto& outp, auto& inp){return inp - val;}, *this);
		return std::move(output);
	}
	return std::move(output);
}

ArrayVoid ArrayVoid::operator+(Scalar c) const{
	utils::throw_exception(dtype != DType::Bool, "+ operation is invalid for DType::Bool");
	ArrayVoid output(size, dtype);
	if(dtype == DType::TensorObj){
		output.transform_function<DType::TensorObj>([&c](auto& outp, auto& inp){return inp + c;}, *this);
		return std::move(output);
	}
	else if(DTypeFuncs::is_floating(dtype)){
		double val = c.to<double>();
		output.transform_function<WRAP_DTYPES<FloatingTypesL>>([&val](auto& outp, auto& inp){return inp + val;}, *this);
		return std::move(output);
	}
	else if(DTypeFuncs::is_integer(dtype)){
		int64_t val = c.to<int64_t>();
		output.transform_function<WRAP_DTYPES<IntegerTypesL>>([&val](auto& outp, auto& inp){return inp + val;}, *this);
		return std::move(output);
	}
	else if(DTypeFuncs::is_complex(dtype)){
		complex_128 val = c.to<complex_128>();
		output.transform_function<WRAP_DTYPES<ComplexTypesL>>([&val](auto& outp, auto& inp){return inp + val;}, *this);
		return std::move(output);
	}
	return std::move(output);
}

ArrayVoid ArrayVoid::operator*(const ArrayVoid& A) const{
	utils::throw_exception(size == A.size, "For operators, sizes must be equal, expected size of $ but got $", size, A.size);
	utils::throw_exception(!(dtype == DType::TensorObj && A.dtype != DType::TensorObj), "$ can not have operator * with $", dtype, A.dtype);
	ArrayVoid output(size, dtype);
	if(dtype == DType::TensorObj && A.dtype != DType::TensorObj){
		auto out_begin = output.tbegin<Tensor>();
		A.transform_function<WRAP_DTYPES<NumberTypesL>>([](const auto& t, const auto& o){return t * o;}, *this, out_begin);
		return std::move(output);
	}
	this->cexecute_function_nbool([](auto begin, auto end, auto begin2, ArrayVoid& output){
				using value_t = typename std::remove_const<typename decltype(begin)::value_type>::type;
				std::transform(begin, end, begin2, output.tbegin<value_t>(), std::multiplies<value_t>());}, A, output);
	return std::move(output);
}

ArrayVoid ArrayVoid::operator/(const ArrayVoid& A) const{
	utils::throw_exception(size == A.size, "For operators, sizes must be equal, expected size of $ but got $", size, A.size);
	utils::throw_exception(!(dtype == DType::TensorObj && A.dtype != DType::TensorObj), "$ can not have operator * with $", dtype, A.dtype);
	ArrayVoid output(size, dtype);
	if(dtype == DType::TensorObj && A.dtype != DType::TensorObj){
		auto out_begin = output.tbegin<Tensor>();
		A.transform_function<WRAP_DTYPES<NumberTypesL>>([](const auto& t, const auto& o){return t / o;},*this, out_begin);
		return std::move(output);
	}
	this->cexecute_function_nbool([](auto begin, auto end, auto begin2, ArrayVoid& output){
				using value_t = typename std::remove_const<typename decltype(begin)::value_type>::type;
				std::transform(begin, end, begin2, output.tbegin<value_t>(), std::divides<value_t>());}, A, output);
	return std::move(output);
}

ArrayVoid ArrayVoid::operator-(const ArrayVoid& A) const{
	utils::throw_exception(size == A.size, "For operators, sizes must be equal, expected size of $ but got $", size, A.size);
	utils::throw_exception(!(dtype == DType::TensorObj && A.dtype != DType::TensorObj), "$ can not have operator * with $", dtype, A.dtype);
	ArrayVoid output(size, dtype);
	if(dtype == DType::TensorObj && A.dtype != DType::TensorObj){
		auto out_begin = output.tbegin<Tensor>();
		A.transform_function<WRAP_DTYPES<NumberTypesL>>([](const auto& t, const auto& o){return t - o;},*this, out_begin);
		return std::move(output);
	}
	this->cexecute_function_nbool([](auto begin, auto end, auto begin2, ArrayVoid& output){
				using value_t = typename std::remove_const<typename decltype(begin)::value_type>::type;
				std::transform(begin, end, begin2, output.tbegin<value_t>(), std::minus<value_t>());}, A, output);
	return std::move(output);
}

ArrayVoid ArrayVoid::operator+(const ArrayVoid& A) const{
	utils::throw_exception(size == A.size, "For operators, sizes must be equal, expected size of $ but got $", size, A.size);
	utils::throw_exception(!(dtype == DType::TensorObj && A.dtype != DType::TensorObj), "$ can not have operator * with $", dtype, A.dtype);
	ArrayVoid output(size, dtype);
	if(dtype == DType::TensorObj && A.dtype != DType::TensorObj){
		auto out_begin = output.tbegin<Tensor>();
		A.transform_function<WRAP_DTYPES<NumberTypesL>>([](const auto& t, const auto& o){return t + o;},*this, out_begin);
		return std::move(output);
	}
	this->cexecute_function_nbool([](auto begin, auto end, auto begin2, ArrayVoid& output){
				using value_t = typename std::remove_const<typename decltype(begin)::value_type>::type;
				std::transform(begin, end, begin2, output.tbegin<value_t>(), std::plus<value_t>());}, A, output);
	return std::move(output);
}

ArrayVoid& ArrayVoid::operator*=(const ArrayVoid& A){
	if(dtype == A.dtype){
		this->transform_function_nbool([](auto& a, auto& b){return a * b;}, A);
		return *this;
	}
	else{
		return *this *= A.to(dtype);
	}
}

ArrayVoid& ArrayVoid::operator+=(const ArrayVoid& A){
	if(dtype == A.dtype){
		this->transform_function_nbool([](auto& a, auto& b){return a + b;}, A);
		return *this;
	}
	else{
		return *this *= A.to(dtype);
	}
}

ArrayVoid& ArrayVoid::operator-=(const ArrayVoid& A){
	if(dtype == A.dtype){
		this->transform_function_nbool([](auto& a, auto& b){return a - b;}, A);
		return *this;
	}
	else{
		return *this *= A.to(dtype);
	}}

ArrayVoid& ArrayVoid::operator/=(const ArrayVoid& A){
	if(dtype == A.dtype){
		this->transform_function_nbool([](auto& a, auto& b){return a / b;}, A);
		return *this;
	}
	else{
		return *this *= A.to(dtype);
	}}

ArrayVoid ArrayVoid::operator*(const Tensor& A) const{
	utils::throw_exception(dtype == DType::TensorObj, "\nRuntime Error: expected DType for * of TensorObj to be TensorObj but got $", dtype);
	ArrayVoid output(size, dtype);
	Tensor* OutputIt = reinterpret_cast<Tensor*>(output.data_ptr());
	this->transform_function<DType::TensorObj>(std::bind(std::multiplies<Tensor>(), A, std::placeholders::_1), OutputIt);
	return std::move(output);
}

ArrayVoid ArrayVoid::operator/(const Tensor& A) const{
	utils::throw_exception(dtype == DType::TensorObj, "\nRuntime Error: expected DType for / of TensorObj to be TensorObj but got $", dtype);
	ArrayVoid output(size, dtype);
	Tensor* OutputIt = reinterpret_cast<Tensor*>(output.data_ptr());
	this->transform_function<DType::TensorObj>(std::bind(std::divides<Tensor>(), A, std::placeholders::_1), OutputIt);
	return std::move(output);
}

ArrayVoid ArrayVoid::operator-(const Tensor& A) const{
	utils::throw_exception(dtype == DType::TensorObj, "\nRuntime Error: expected DType for - of TensorObj to be TensorObj but got $", dtype);
	ArrayVoid output(size, dtype);
	Tensor* OutputIt = reinterpret_cast<Tensor*>(output.data_ptr());
	this->transform_function<DType::TensorObj>(std::bind(std::minus<Tensor>(), A, std::placeholders::_1), OutputIt);
	return std::move(output);

}

ArrayVoid ArrayVoid::operator+(const Tensor& A) const{
	utils::throw_exception(dtype == DType::TensorObj, "\nRuntime Error: expected DType for + of TensorObj to be TensorObj but got $", dtype);
	ArrayVoid output(size, dtype);
	Tensor* OutputIt = reinterpret_cast<Tensor*>(output.data_ptr());
	this->transform_function<DType::TensorObj>(std::bind(std::plus<Tensor>(), A, std::placeholders::_1), OutputIt);
	return std::move(output);

}

ArrayVoid& ArrayVoid::operator*=(const Tensor& A){
	this->for_each<DType::TensorObj>([&A](auto& inp){inp *= A;});
	return *this;
}

ArrayVoid& ArrayVoid::operator+=(const Tensor& A){
	this->for_each<DType::TensorObj>([&A](auto& inp){inp += A;});
	return *this;
}

ArrayVoid& ArrayVoid::operator-=(const Tensor& A){
	this->for_each<DType::TensorObj>([&A](auto& inp){inp -= A;});
	return *this;
}

ArrayVoid& ArrayVoid::operator/=(const Tensor& A){
	this->for_each<DType::TensorObj>([&A](auto& inp){inp /= A;});
	return *this;
}


/* template<DType dt, std::enable_if_t<dt != DType::TensorObj, bool> = true> */
/* bool _my_sub_equal_operator_(const ArrayVoid& mine, uint_bool_t* output, const Scalar& c){ */
/* 	if(dt != mine.dtype) return _my_sub_equal_operator_<DTypeFuncs::next_dtype_it<dt>>(mine, output, c); */
/* 	using value_t = DTypeFuncs::dtype_to_type_t<dt>; */
/* 	value_t s = c.to<value_t>(); */
/* 	mine.transform_function<dt>([&s](const auto& val){return uint_bool_t(val == s);}, output); */
/* 	return true; */
/* } */

/* template<DType dt, std::enable_if_t<dt == DType::TensorObj, bool> = true> */
/* bool _my_sub_equal_operator_(const ArrayVoid& mine, uint_bool_t* output, const Scalar& c){ */
/* 	if(dt != mine.dtype) return _my_sub_equal_operator_<DTypeFuncs::next_dtype_it<dt>>(mine, output, c); */
/* 	return false; */	
/* } */


	
ArrayVoid ArrayVoid::operator==(Scalar c) const{
	if(dtype == DType::TensorObj){
		ArrayVoid output(size, DType::TensorObj);
		Tensor* OutputIt = reinterpret_cast<Tensor*>(output.data_ptr());
		this->transform_function<DType::TensorObj>([&c](const auto& val){return val == c;}, OutputIt);
		return std::move(output);
	}
	ArrayVoid output(size, DType::Bool);
	if(DTypeFuncs::is_unsigned(dtype) && c.to<int64_t>() < 0){
		output = uint_bool_t(false);
		return std::move(output);
	}
	if(c.isComplex() && (!DTypeFuncs::is_complex(dtype))){
		output = uint_bool_t(false);
		return std::move(output);
	}
	this->cexecute_function<WRAP_DTYPES<NumberTypesL, DTypeEnum<DType::Bool>>>([](auto begin, auto end, ArrayVoid& op, Scalar& s){
				auto o_begin = op.tbegin<uint_bool_t>();
				using value_t = typename std::remove_const<typename decltype(begin)::value_type>::type;
				value_t i = s.to<value_t>();
				std::transform(begin, end, o_begin, [&i](const auto& val){
						return uint_bool_t(val == i);});
			}, output, c);
	/* utils::throw_exception(_my_sub_equal_operator_<DType::Integer>(*this, reinterpret_cast<uint_bool_t*>(output.data_ptr()), c), "\nRuntime Error: input dtype $ is invalid for == operator", dtype); */
	return std::move(output);
}

/* template<DType dt, std::enable_if_t<dt == DType::TensorObj, bool> = true> */
/* bool _my_sub_equal_operator_(const ArrayVoid& mine, uint_bool_t* output, const Scalar& c){ */
/* 	if(dt != mine.dtype) return _my_sub_equal_operator_<DTypeFuncs::next_dtype_it<dt>>(mine, output, c); */
/* 	return false; */	
/* } */

ArrayVoid ArrayVoid::operator>=(Scalar c) const{
	if(dtype == DType::TensorObj){
		ArrayVoid output(size, DType::TensorObj);
		Tensor* OutputIt = reinterpret_cast<Tensor*>(output.data_ptr());
		this->transform_function<DType::TensorObj>([&c](const auto& val){return val >= c;}, OutputIt);
		return std::move(output);
	}
	ArrayVoid output(size, DType::Bool);
	if(DTypeFuncs::is_unsigned(dtype) && c.to<int64_t>() < 0){
		output = uint_bool_t(false);
		return std::move(output);
	}
	if(c.isComplex() && (!DTypeFuncs::is_complex(dtype))){
		output = uint_bool_t(false);
		return std::move(output);
	}
	this->cexecute_function<WRAP_DTYPES<NumberTypesL>>([](auto begin, auto end, ArrayVoid& op, Scalar& s){
				auto o_begin = op.tbegin<uint_bool_t>();
				using value_t = typename std::remove_const<typename decltype(begin)::value_type>::type;
				value_t i = s.to<value_t>();
				std::transform(begin, end, o_begin, [&i](const auto& val){
						return uint_bool_t(val >= i);});
			}, output, c);
	/* utils::throw_exception(_my_sub_equal_operator_<DType::Integer>(*this, reinterpret_cast<uint_bool_t*>(output.data_ptr()), c), "\nRuntime Error: input dtype $ is invalid for == operator", dtype); */
	return std::move(output);
}

ArrayVoid ArrayVoid::operator<=(Scalar c) const{
	if(dtype == DType::TensorObj){
		ArrayVoid output(size, DType::TensorObj);
		Tensor* OutputIt = reinterpret_cast<Tensor*>(output.data_ptr());
		this->transform_function<DType::TensorObj>([&c](const auto& val){return val <= c;}, OutputIt);
		return std::move(output);
	}
	ArrayVoid output(size, DType::Bool);
	if(DTypeFuncs::is_unsigned(dtype) && c.to<int64_t>() < 0){
		output = uint_bool_t(false);
		return std::move(output);
	}
	if(c.isComplex() && (!DTypeFuncs::is_complex(dtype))){
		output = uint_bool_t(false);
		return std::move(output);
	}
	this->cexecute_function<WRAP_DTYPES<NumberTypesL>>([](auto begin, auto end, ArrayVoid& op, Scalar& s){
				auto o_begin = op.tbegin<uint_bool_t>();
				using value_t = typename std::remove_const<typename decltype(begin)::value_type>::type;
				value_t i = s.to<value_t>();
				std::transform(begin, end, o_begin, [&i](const auto& val){
						return uint_bool_t(val <= i);});
			}, output, c);
	/* utils::throw_exception(_my_sub_equal_operator_<DType::Integer>(*this, reinterpret_cast<uint_bool_t*>(output.data_ptr()), c), "\nRuntime Error: input dtype $ is invalid for == operator", dtype); */
	return std::move(output);
}

ArrayVoid ArrayVoid::operator>(Scalar c) const{
	if(dtype == DType::TensorObj){
		ArrayVoid output(size, DType::TensorObj);
		Tensor* OutputIt = reinterpret_cast<Tensor*>(output.data_ptr());
		this->transform_function<DType::TensorObj>([&c](const auto& val){return val > c;}, OutputIt);
		return std::move(output);
	}
	ArrayVoid output(size, DType::Bool);
	if(DTypeFuncs::is_unsigned(dtype) && c.to<int64_t>() < 0){
		output = uint_bool_t(false);
		return std::move(output);
	}
	if(c.isComplex() && (!DTypeFuncs::is_complex(dtype))){
		output = uint_bool_t(false);
		return std::move(output);
	}
	this->cexecute_function<WRAP_DTYPES<NumberTypesL>>([](auto begin, auto end, ArrayVoid& op, Scalar& s){
				auto o_begin = op.tbegin<uint_bool_t>();
				using value_t = typename std::remove_const<typename decltype(begin)::value_type>::type;
				value_t i = s.to<value_t>();
				std::transform(begin, end, o_begin, [&i](const auto& val){
						return uint_bool_t(val > i);});
			}, output, c);
	/* utils::throw_exception(_my_sub_equal_operator_<DType::Integer>(*this, reinterpret_cast<uint_bool_t*>(output.data_ptr()), c), "\nRuntime Error: input dtype $ is invalid for == operator", dtype); */
	return std::move(output);
}

ArrayVoid ArrayVoid::operator<(Scalar c) const{
	if(dtype == DType::TensorObj){
		ArrayVoid output(size, DType::TensorObj);
		Tensor* OutputIt = reinterpret_cast<Tensor*>(output.data_ptr());
		this->transform_function<DType::TensorObj>([&c](const auto& val){return val < c;}, OutputIt);
		return std::move(output);
	}
	ArrayVoid output(size, DType::Bool);
	if(DTypeFuncs::is_unsigned(dtype) && c.to<int64_t>() < 0){
		output = uint_bool_t(false);
		return std::move(output);
	}
	if(c.isComplex() && (!DTypeFuncs::is_complex(dtype))){
		output = uint_bool_t(false);
		return std::move(output);
	}
	this->cexecute_function<WRAP_DTYPES<NumberTypesL>>([](auto begin, auto end, ArrayVoid& op, Scalar& s){
				auto o_begin = op.tbegin<uint_bool_t>();
				using value_t = typename std::remove_const<typename decltype(begin)::value_type>::type;
				value_t i = s.to<value_t>();
				std::transform(begin, end, o_begin, [&i](const auto& val){
						return uint_bool_t(val < i);});
			}, output, c);
	/* utils::throw_exception(_my_sub_equal_operator_<DType::Integer>(*this, reinterpret_cast<uint_bool_t*>(output.data_ptr()), c), "\nRuntime Error: input dtype $ is invalid for == operator", dtype); */
	return std::move(output);
}


template<DType dt, std::enable_if_t<dt != DType::Bool && dt != DType::TensorObj && !DTypeFuncs::is_dtype_complex_v<dt>, bool> = true>
bool _my_sub_inverse_(const ArrayVoid& my_arr, ArrayVoid& outp){
	if(dt != my_arr.dtype) return _my_sub_inverse_<DTypeFuncs::next_dtype_it<dt>>(my_arr, outp);
	auto begin  = my_arr.tcbegin<::nt::DTypeFuncs::dtype_to_type_t<dt>>();
	auto end  = my_arr.tcend<::nt::DTypeFuncs::dtype_to_type_t<dt>>();
	auto begin_2  = outp.tbegin<::nt::DTypeFuncs::dtype_to_type_t<dt>>();
	std::transform(begin, end, begin_2, [](const auto& val){return 1.0/val;});
	return true;
}

template<DType dt, std::enable_if_t<dt != DType::Bool && dt != DType::TensorObj && DTypeFuncs::is_dtype_complex_v<dt>, bool> = true>
bool _my_sub_inverse_(const ArrayVoid& my_arr, ArrayVoid& outp){
	if(dt != my_arr.dtype) return _my_sub_inverse_<DTypeFuncs::next_dtype_it<dt>>(my_arr, outp);
	using complex_v = ::nt::DTypeFuncs::dtype_to_type_t<dt>; 
	auto begin  = my_arr.tcbegin<::nt::DTypeFuncs::dtype_to_type_t<dt>>();
	auto end  = my_arr.tcend<::nt::DTypeFuncs::dtype_to_type_t<dt>>();
	auto begin_2  = outp.tbegin<::nt::DTypeFuncs::dtype_to_type_t<dt>>();
	std::transform(begin, end, begin_2, [](const auto& val){return complex_v(1.0/val.real(), 1.0/val.imag());});
	return true;
}

template<DType dt, std::enable_if_t<dt == DType::Bool, bool> = true>
bool _my_sub_inverse_(const ArrayVoid& my_arr, ArrayVoid& outp){
	if(dt != my_arr.dtype) return _my_sub_inverse_<DTypeFuncs::next_dtype_it<dt>>(my_arr, outp);
	return false;
}

template<DType dt, std::enable_if_t<dt == DType::TensorObj, bool> = true>
bool _my_sub_inverse_(const ArrayVoid& my_arr, ArrayVoid& outp){
	if(dt != my_arr.dtype) return _my_sub_inverse_<DTypeFuncs::next_dtype_it<dt>>(my_arr, outp);
	auto begin  = my_arr.tcbegin<::nt::DTypeFuncs::dtype_to_type_t<dt>>();
	auto end  = my_arr.tcend<::nt::DTypeFuncs::dtype_to_type_t<dt>>();
	auto begin_2  = outp.tbegin<::nt::DTypeFuncs::dtype_to_type_t<dt>>();
	std::transform(begin, end, begin_2, [](const auto& val){return val.inverse();});
	return true;
}

ArrayVoid ArrayVoid::inverse() const{
	if(dtype == DType::LongLong){
		ArrayVoid output(size, DType::Double);
		this->transform_function<DType::LongLong>([](const auto& inp) -> double {return 1.0/((double)inp);}, reinterpret_cast<double*>(output.data_ptr()));
		return std::move(output);
	}
#ifdef __SIZEOF_INT128__
	if(dtype == DType::int128 || dtype == DType::uint128){
#ifdef _128_FLOAT_SUPPORT_
		ArrayVoid output(size, DType::Float128);
		this->transform_function<DType::uint128, DType::int128>([](const auto& inp) -> float128_t 
				{return 1.0/(::nt::convert::convert<DType::Float128>(inp));}, reinterpret_cast<float128_t*>(output.data_ptr()));
#else
		ArrayVoid output(size, DType::Double);
		this->transform_function<DType::uint128, DType::int128>([](const auto& inp) -> double
				{return 1.0/(::nt::convert::convert<DType::Double>(inp));}, reinterpret_cast<double*>(output.data_ptr()));
#endif
		DType out_dtype = DType::Float128;
	}
#endif
	if(dtype == DType::Integer || dtype == DType::Long || dtype == DType::uint8 || dtype == DType::int8 || dtype == DType::uint16 || dtype == DType::int16){
		ArrayVoid output(size, DType::Float);
		this->transform_function
			<DType::Integer,DType::Long,DType::uint8,DType::int8,DType::uint16,DType::int16>
			([](const auto& inp) -> float {return 1.0/((float)inp);}, reinterpret_cast<float*>(output.data_ptr()));
		return std::move(output);
	}
	if(dtype == DType::TensorObj){
		ArrayVoid output(size, DType::TensorObj);
		this->transform_function<DType::TensorObj>([](const Tensor& inp) -> Tensor {return inp.inverse();}, reinterpret_cast<Tensor*>(output.data_ptr()));
		return std::move(output);
	}
	ArrayVoid output(size, dtype);
	utils::throw_exception(_my_sub_inverse_<DType::Float>(*this, output), "\nRuntime Error: Could not do inverse() for dtype $", dtype);
	return std::move(output);
}


template<DType dt, std::enable_if_t<DTypeFuncs::is_dtype_floating_v<dt>, bool> = true>
bool _my_sub_inverse_(ArrayVoid& mine){
	if(dt != mine.dtype) return _my_sub_inverse_<DTypeFuncs::next_dtype_it<dt>>(mine);
	mine.for_each<dt>([](auto& val){val = 1.0/val;});
	return true;	
}

template<DType dt, std::enable_if_t<DTypeFuncs::is_dtype_complex_v<dt> || dt == DType::TensorObj, bool> = true>
bool _my_sub_inverse_(ArrayVoid& mine){
	if(dt != mine.dtype) return _my_sub_inverse_<DTypeFuncs::next_dtype_it<dt>>(mine);
	using complex_value_t = DTypeFuncs::dtype_to_type_t<dt>;
	mine.for_each<dt>([](auto& val){val.inverse_();});
	return true;	
}


template<DType dt, std::enable_if_t<!DTypeFuncs::is_dtype_floating_v<dt> && !DTypeFuncs::is_dtype_complex_v<dt> && dt != DType::TensorObj, bool> = true>
bool _my_sub_inverse_(ArrayVoid& mine){
	if(dt != mine.dtype) return _my_sub_inverse_<DTypeFuncs::next_dtype_it<dt>>(mine);
	utils::throw_exception(false, "to inverse() scalars must be complex or floating not $", dt);
	return false;	
}

ArrayVoid& ArrayVoid::inverse_(){
	if(dtype != DType::TensorObj && !DTypeFuncs::is_floating(dtype) && !DTypeFuncs::is_complex(dtype))
		floating_();
	utils::throw_exception(_my_sub_inverse_<DType::Integer>(*this), "\nRuntime Error: Unable to perform inverse on self for DType $", dtype);
	return *this;
}



ArrayVoid ArrayVoid::exp() const{
	ArrayVoid output(size, dtype);
	if(DTypeFuncs::is_integer(dtype) || dtype ==  DType::Double || dtype == DType::Float)
		output.transform_function<WRAP_DTYPES<IntegerTypesL, DTypeEnum<DType::Float, DType::Double>>>([](auto& outp, const auto& a){return std::exp(a);}, *this);
	else if(dtype == DType::Complex64 || dtype == DType::Complex128)
		output.transform_function<DType::Complex64, DType::Complex128>([](auto& outp, const auto& a){
				return typename std::remove_reference<typename std::remove_const<decltype(a)>::type>::type(std::exp(a.real()), std::exp(a.imag()));

				}, *this);
#ifdef _128_FLOAT_SUPPORT_
	else if(dtype == DType::Float128)
		output.transform_function<DType::Float128>([](auto& outp, const auto& a){return std::exp(a);}, *this);
#endif
#ifdef _HALF_FLOAT_SUPPORT_
	else if(dtype == DType::Complex32)
		output.transform_function<DType::Complex32>([](auto& outp, const auto& a){return complex_32(std::exp(static_cast<float>(a.real())), std::exp(static_cast<float>(a.imag())));}, *this); 
	else if(dtype == DType::Float16)
		output.transform_function<DType::Float16>([](auto& outp, const auto& a){return static_cast<float16_t>(std::exp(static_cast<float>(a)));}, *this);
#endif
	else	
		output.transform_function<DType::TensorObj>([](auto& outp, const auto& a){return a.exp();}, *this);
	return std::move(output);
}

ArrayVoid& ArrayVoid::exp_(){
	if(DTypeFuncs::is_integer(dtype) || dtype ==  DType::Double || dtype == DType::Float)
		this->for_each<WRAP_DTYPES<IntegerTypesL, DTypeEnum<DType::Float, DType::Double>>>([](auto& a){a = std::exp(a);});
	else if(dtype == DType::Complex64 || dtype == DType::Complex128)
		this->for_each<DType::Complex64, DType::Complex128>([](auto& a){a = typename std::remove_reference<typename std::remove_const<decltype(a)>::type>::type(std::exp(a.real()), std::exp(a.imag()));});
#ifdef _128_FLOAT_SUPPORT_
	else if(dtype == DType::Float128)
		this->for_each<DType::Float128>([](auto& a){a = std::exp(a);});
#endif
#ifdef _HALF_FLOAT_SUPPORT_
	else if(dtype == DType::Complex32)
		this->for_each<DType::Complex32>([](auto &a){a = complex_32(std::exp(static_cast<float>(a.real())), std::exp(static_cast<float>(a.imag())));});
	else if(dtype == DType::Float16)
		this->for_each<DType::Float16>([](auto& a){a= static_cast<float16_t>(std::exp(static_cast<float>(a)));});
#endif
	else
		this->for_each<DType::TensorObj>([](auto& a){a.exp_();});
	return *this;
}


ArrayVoid& ArrayVoid::complex_(){
	DType complex_to = DTypeFuncs::complex_size(DTypeFuncs::size_of_dtype(dtype));
	utils::throw_exception(DTypeFuncs::is_complex(complex_to), "\nRuntime Error: Unable to implicitly convert $ to complex", dtype);
	DTypeFuncs::convert_this_dtype_array(data_ptr(), dtype, complex_to, available_size);
	return *this;
}

ArrayVoid& ArrayVoid::floating_(){
	DType floating_to = DTypeFuncs::floating_size(DTypeFuncs::size_of_dtype(dtype));
	utils::throw_exception(DTypeFuncs::is_floating(floating_to), "\nRuntime Error: Unable to implicitly convert $ to floating", dtype);
	DTypeFuncs::convert_this_dtype_array(data_ptr(), dtype, floating_to, available_size);
	return *this;
}

ArrayVoid& ArrayVoid::integer_(){
	DType integer_to = DTypeFuncs::integer_size(DTypeFuncs::size_of_dtype(dtype));
	utils::throw_exception(DTypeFuncs::is_integer(integer_to), "\nRuntime Error: Unable to implicitly convert $ to integer", dtype);
	DTypeFuncs::convert_this_dtype_array(data_ptr(), dtype, integer_to, available_size);
	return *this;
}
ArrayVoid& ArrayVoid::unsigned_(){
	DType unsigned_to = DTypeFuncs::unsigned_size(DTypeFuncs::size_of_dtype(dtype));
	utils::throw_exception(DTypeFuncs::is_unsigned(unsigned_to), "\nRuntime Error: Unable to implicitly convert $ to unsigned", dtype);
	DTypeFuncs::convert_this_dtype_array(data_ptr(), dtype, unsigned_to, available_size);
	return *this;
}
	
}

