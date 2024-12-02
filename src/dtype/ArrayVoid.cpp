#include <_types/_uint8_t.h>
#include <memory.h>
#include <new>


#include <utility>

#include "../Tensor.h"
#include "DType.h"
#include "DType_enum.h"
#include "ArrayVoid.h"
#include "ArrayVoid.hpp"
#include <functional>
#include <numeric>
#include "Scalar.h"
#include "../utils/utils.h"
#include "compatible/DType_compatible.h"
#include "../types/Types.h"
#include "DType_operators.h"
#include "../memory/iterator.h"
//%s/"../convert/Convert.h"/"..\/convert\/Convert.h"


#include <immintrin.h>

#if defined(__AVX__) && defined(__AVX2__) && defined(__AVX512F__)
    // Compiler supports AVX, AVX2, and AVX-512F instruction sets
    #define SIMD_SUPPORTED 1
#else
    #define SIMD_SUPPORTED 0
#endif

#ifdef USE_PARALLEL
#include <tbb/parallel_for.h>
#endif

namespace nt{


const size_t ArrayVoid::dtype_size(DType _type) const{
	return DTypeFuncs::size_of_dtype(_type);
}

ArrayVoid::ArrayVoid(int64_t _size, DType _t)
	:bucket(_size, _t),
	dtype(_t),
	size(_size)
{}

ArrayVoid::ArrayVoid(int64_t _size, DType _t, void* ptr, void (*func)(void*))
	:bucket(_size, _t, ptr, func),
	dtype(_t),
	size(_size)
{}


#ifdef USE_PARALLEL
ArrayVoid::ArrayVoid(int64_t _size, DTypeShared _t)
	:bucket(_size, DTypeShared_DType(_t), dCPUShared),
	dtype(DTypeShared_DType(_t)),
	size(_size)
{}
#endif

ArrayVoid& ArrayVoid::operator=(const ArrayVoid &Arr){
	bucket = Arr.bucket;
	size = Arr.size;
	dtype = Arr.dtype;
	return *this;
}

ArrayVoid& ArrayVoid::operator=(ArrayVoid&& Arr){
	bucket = std::move(Arr.bucket);
	size = Arr.size;
	Arr.size = 0;
	dtype = Arr.dtype;
	return *this;
}

ArrayVoid::ArrayVoid(const ArrayVoid& Arr)
	:bucket(Arr.bucket),
	size(Arr.size),
	dtype(Arr.dtype)
{}

ArrayVoid::ArrayVoid(ArrayVoid&& Arr)
	:bucket(std::move(Arr.bucket)),
	size(std::exchange(Arr.size, 0)),
	dtype(Arr.dtype)
{}

/* ArrayVoid::ArrayVoid(intrusive_ptr<void*>&& str, const std::size_t size, const std::size_t start, std::size_t avail, DType _dt) */
/* 	:_strides(std::move(str)), */
/* 	size(size), */
/* 	dtype(_dt), */
/* 	type_size(this->dtype_size(_dt)), */
/* 	_last_index(size + start), */
/* 	_start(start), */
/* 	available_size(avail) */
/* {} */

/* ArrayVoid::ArrayVoid(const intrusive_ptr<void*>& str, const std::size_t size, const std::size_t start, std::size_t avail, DType _dt) */
/* 	:_strides(str), */
/* 	size(size), */
/* 	dtype(_dt), */
/* 	type_size(this->dtype_size(_dt)), */
/* 	_last_index(size + start), */
/* 	_start(start), */
/* 	available_size(avail) */
/* {} */

/* ArrayVoid::ArrayVoid(intrusive_ptr<void*>&& str, const std::size_t size, const std::size_t start, std::size_t avail, DType _dt) */
/* 	:_strides(std::move(str)), */
/* 	size(size), */
/* 	dtype(_dt), */
/* 	type_size(this->dtype_size(_dt)), */
/* 	_last_index(size + start), */
/* 	_start(start), */
/* 	available_size(avail) */
/* {std::cout << "strides move called"<<std::endl;} */
ArrayVoid::ArrayVoid(const Bucket& b, uint64_t size, DType dt)
	:bucket(b),
	size(size),
	dtype(dt)
{}

ArrayVoid::ArrayVoid(Bucket&& b, uint64_t size, DType dt)
	:bucket(std::move(b)),
	size(size),
	dtype(dt)
{}

ArrayVoid::ArrayVoid(const Bucket& b)
	:bucket(b),
	size(b.size()),
	dtype(b.dtype)
{}

ArrayVoid::ArrayVoid(Bucket&& b)
	:bucket(std::move(b)),
	size(0),
	dtype(DType::Integer)
{
	size = bucket.size(); 
	dtype = bucket.dtype;
}

void ArrayVoid::swap(ArrayVoid& other){
	other.bucket.swap(bucket);
	std::swap(other.size, size);
	std::swap(other.dtype, dtype);
}

void* ArrayVoid::data_ptr_end() {utils::throw_exception(is_contiguous(), "Must be contiguous for data_ptr_end()");return reinterpret_cast<uint8_t*>(bucket.data_ptr()) + (size * DTypeFuncs::size_of_dtype(dtype));}
const void* ArrayVoid::data_ptr_end() const {utils::throw_exception(is_contiguous(), "Must be contiguous for data_ptr_end()");return reinterpret_cast<const uint8_t*>(bucket.data_ptr()) + (size * DTypeFuncs::size_of_dtype(dtype));}
/* /1* const void* ArrayVoid::data_ptr_end() const {return reinterpret_cast<uint8_t*>(_vals.get()) + (_last_index * type_size);} *1/ */
/* void** ArrayVoid::strides_cbegin() const {return _strides.get() + _start;} */
/* void** ArrayVoid::strides_begin() {return _strides.get() + _start;} */
/* void** ArrayVoid::strides_end() {return _strides.get() + _last_index;} */
/* void** ArrayVoid::strides_cend() const {return _strides.get() + _last_index;} */





ArrayVoid& ArrayVoid::operator=(Scalar val){
	if(dtype != DType::TensorObj){
		if(is_contiguous()){
			fill_ptr_(val);
			return *this;
		}
		this->execute_function_chunk<WRAP_DTYPES<NumberTypesL, DTypeEnum<DType::Bool> > >( [&val](auto begin, auto end){
			using value_t = utils::IteratorBaseType_t<decltype(begin)>;
			if(val.isZero()){
				std::fill(begin, end, value_t(0));
			}
			else{
				std::fill(begin, end, val.to<value_t>());
			}
		});
	}
	else{
		this->execute_function<WRAP_DTYPES<DTypeEnum<DType::TensorObj> > >([&val](auto begin, auto end){
				/* using value_t = IteratorBaseType_t<decltype(begin)>; */
				/* auto v = val.to<value_t>(); */
				std::fill(begin, end, val);
				});
	}
	return *this;
}


ArrayVoid& ArrayVoid::iota(Scalar s){
	if(dtype != DType::TensorObj){
		this->execute_function<WRAP_DTYPES<NumberTypesL> >([&s](auto begin, auto end){
			using value_t = utils::IteratorBaseType_t<decltype(begin)>;
			auto v = s.to<value_t>();
			std::iota(begin, end, v);
			});
	}else{
		this->execute_function<WRAP_DTYPES<DTypeEnum<DType::TensorObj> > >([&s](auto begin, auto end){
				/* using value_t = IteratorBaseType_t<decltype(begin)>; */
				/* auto v = val.to<value_t>(); */
				std::for_each(begin, end, [&s](auto& val){val.arr_void().iota(s);});
				});
	}

	return *this;
}

/* std::shared_ptr<void> ArrayVoid::share_part(uint32_t index) const{ */
/* 	return DTypeFuncs::share_part_ptr(index, dtype, _vals); */
/* } */

ArrayVoid ArrayVoid::share_array(uint64_t index) const{
	utils::throw_exception(index <= size, "\nRuntime Error: cannot grab array from index $ with size of only $ ArrayVoid::share_array(index)",index, size);
	return ArrayVoid(bucket + index);
}

ArrayVoid ArrayVoid::share_array(uint64_t index, uint64_t ns) const{
	if(ns != (index +1)){
		utils::throw_exception((ns+index) <= size, "\nRuntime Error: cannot grab array with start of $,$ to with size of only $ ArrayVoid::share_array(index, ns)",index,ns, size);
	}
	/* utils::throw_exception(index <= size, "\nRuntime Error: cannot grab array from index $ with size of only $ ArrayVoid::share_array(index, ns)",index, size); */
	Bucket b = bucket.new_bounds(index, index+ns);
	/* std::cout << "share_array() b bucket_amt: "<<b.buckets_amt()<<std::endl; */
	return ArrayVoid(std::move(b));
	/* return ArrayVoid(_vals, make_unique_strides(index, index + ns), ns, 0, available_size, dtype); */
}
/* const std::vector<uint64_t>& ArrayVoid::get_strides() const{return _strides;} */
/* std::vector<uint64_t>& ArrayVoid::get_strides(){return _strides;} */


ArrayVoid ArrayVoid::change_stride(const std::vector<std::pair<uint64_t, uint64_t> >& pairs) const {
	std::vector<Bucket> buckets;
	buckets.reserve(pairs.size());
	for(size_t i = 0; i < pairs.size(); ++i){
		buckets.push_back(bucket.new_bounds(pairs[i].first, pairs[i].second));
	}
	return ArrayVoid(Bucket::cat(buckets));
}

ArrayVoid ArrayVoid::change_stride(const std::vector<uint64_t>& val) const {
	utils::throw_exception(val.size() <= size, "\nRuntime Error: Expected to have permutation index of size at most $ but got $ ArrayVoid::change_stride", size, val.size());
	//std::shared_ptr<void*> n_strides = make_unique_strides(); <- this way is honestly a bit memory a performance inefficient
	//the reason shared_ptr's are used in the first place is so that memory isn't coppied when one makes a smaller tensor, or defines a tensor to be the same as another
	//However, those functions don't change the size of the strides
	//So, instead, this will make it so that the size of the strides will actually change memory wise, and then start = 0, and stop at the new size
	
	std::vector<std::pair<uint64_t, uint64_t> > pairs;
	pairs.reserve(val.size());
	uint64_t start = val[0];
	uint64_t end = start+1;
	for(size_t i = 1; i < val.size(); ++i){
		if(val[i] != end){
			pairs.push_back(std::pair<uint64_t, uint64_t>(start, end));
			start = val[i];
			end = start + 1;
			continue;
		}
		++end;
	}
	if(pairs.back().first != start && pairs.back().second != end){
		pairs.push_back(std::pair<uint64_t, uint64_t>(start, end));
	}
	return change_stride(pairs);
}

ArrayVoid ArrayVoid::range(std::vector<my_range> ranges) const{
	const size_t old_size = size;
	std::for_each(ranges.begin(), ranges.end(), [&old_size](auto& val){val.fix(old_size);});
	std::vector<std::pair<uint64_t, uint64_t> > pairs(ranges.size());
	auto pbegin = pairs.begin();
	for(const my_range &x : ranges){
		*pbegin = std::pair<uint64_t, uint64_t>(static_cast<uint64_t>(x.begin), static_cast<uint64_t>(x.end));
		++pbegin;
	}
	return change_stride(pairs);
}




/* template<typename T> */
/* template<typename T> */
/* tdtype_list<T> ArrayVoid::tend(){return tdtype_list<T>(reinterpret_cast<T**>(strides_end()));} */
/* template<typename T> */
/* 	return tdtype_list<const T>(reinterpret_cast<T**>(strides_cbegin()));} */
/* template<typename T> */
/* tdtype_list<const T> ArrayVoid::tcend() const { */
/* 	return tdtype_list<const T>(reinterpret_cast<T**>(strides_cend()));} */

/* template tdtype_list<float> ArrayVoid::tend(); */
/* template tdtype_list<double> ArrayVoid::tend(); */
/* template tdtype_list<complex_64> ArrayVoid::tend(); */
/* template tdtype_list<complex_128> ArrayVoid::tend(); */
/* template tdtype_list<uint32_t> ArrayVoid::tend(); */
/* template tdtype_list<int32_t> ArrayVoid::tend(); */
/* template tdtype_listArrayVoid(uint16_t> ArrayVoid::tend(); */
/* template tdtype_list<int16_t> ArrayVoid::tend(); */
/* template tdtype_list<uint8_t> ArrayVoid::tend(); */
/* template tdtype_list<int8_t> ArrayVoid::tend(); */
/* template tdtype_list<int64_t> ArrayVoid::tend(); */
/* template tdtype_list<Tensor> ArrayVoid::tend(); */
/* template tdtype_list<uint_bool_t> ArrayVoid::tend(); */

/* #ifdef _HALF_FLOAT_SUPPORT_ */
/* template tdtype_list<float16_t> ArrayVoid::tend(); */
/* template tdtype_list<complex_32> ArrayVoid::tend(); */
/* #endif */
/* #ifdef _128_FLOAT_SUPPORT_ */
/* template tdtype_list<float128_t> ArrayVoid::tend(); */
/* #endif */
/* #ifdef __SIZEOF_INT128__ */
/* template tdtype_list<int128_t> ArrayVoid::tend(); */
/* #endif */





/* template tdtype_list<const float> ArrayVoid::tcend() const; */
/* template tdtype_list<const double> ArrayVoid::tcend() const; */
/* template tdtype_list<const complex_64> ArrayVoid::tcend() const; */
/* template tdtype_list<const complex_128> ArrayVoid::tcend() const; */
/* template tdtype_list<const uint32_t> ArrayVoid::tcend() const; */
/* template tdtype_list<const int32_t> ArrayVoid::tcend() const; */
/* template tdtype_list<const uint16_t> ArrayVoid::tcend() const; */
/* template tdtype_list<const int16_t> ArrayVoid::tcend() const; */
/* template tdtype_list<const uint8_t> ArrayVoid::tcend() const; */
/* template tdtype_list<const int8_t> ArrayVoid::tcend() const; */
/* template tdtype_list<const int64_t> ArrayVoid::tcend() const; */
/* template tdtype_list<const Tensor> ArrayVoid::tcend() const; */
/* template tdtype_list<const uint_bool_t> ArrayVoid::tcend() const; */

/* #ifdef _HALF_FLOAT_SUPPORT_ */
/* template tdtype_list<const float16_t> ArrayVoid::tcend() const; */
/* template tdtype_list<const complex_32> ArrayVoid::tcend() const; */
/* #endif */
/* #ifdef _128_FLOAT_SUPPORT_ */
/* template tdtype_list<const float128_t> ArrayVoid::tcend() const; */
/* #endif */
/* #ifdef __SIZEOF_INT128__ */
/* template tdtype_list<const int128_t> ArrayVoid::tcend() const; */
/* #endif */


ArrayVoid ArrayVoid::copy_strides(bool copy) const{
	if(!bucket.is_strided())
		return bucket_all_indices();
	if(!copy)
		return new_strides(size);
	return ArrayVoid(bucket.copy_strides(), size, dtype);
}

/* template<DType dt> */
//maybe move this to be a private function
/* ArrayVoid ArrayVoid::copy_strides(bool copy) const{ */
/* 	/1* if(dt != dtype) *1/ */
/* 	/1* 	return copy_strides<DTypeFuncs::next_dtype_it<dt>>(); *1/ */

/* 	intrusive_ptr<void*> cpy_str = _strides.copy_strides(available_size, DTypeFuncs::size_of_dtype_p(dtype)); */
/* 	if(copy){ */
/* 		void** original = strides_cbegin(); */
/* 		void** destination = cpy_str.get(); */
/* 		for(uint64_t i = 0; i < size; ++i, ++original, ++destination) */
/* 			*destination = *original; */
/* 	} */
/* 	return ArrayVoid(cpy_str, size, 0, available_size, dtype); */
/* } */


//this would potentially break it if I am not mistakened
/* ArrayVoid ArrayVoid::new_stride(uint32_t size) const{ */
/* 	intrusive_ptr<void*> n_str(_strides.vals_(), size, type_size, DTypeFuncs::size_of_dtype_p(dtype), detail::DontOrderStrides{}); */
/* 	return ArrayVoid(n_str, size, 0, available_size, dtype); */
/* } */

/* intrusive_ptr<void*> ArrayVoid::make_unique_strides(bool pre_order) const{ */
/* 	if(!pre_order){ */
/* 		return intrusive_ptr<void*>(_strides.vals_(), available_size, type_size, DTypeFuncs::size_of_dtype_p(dtype), detail::DontOrderStrides{}); */
/* 	} */
/* 	intrusive_ptr<void*> cpy_str(_strides.vals_(), available_size, type_size, DTypeFuncs::size_of_dtype_p(dtype), detail::DontOrderStrides{}); */
/* 	void** original = _strides.get(); */
/* 	void** destination = cpy_str.get(); */
/* 	for(uint64_t i = 0; i < available_size; ++i) */
/* 		destination[i] = original[i]; */	
/* 	return std::move(cpy_str); */
/* } */

/* intrusive_ptr<void*> ArrayVoid::make_unique_strides(std::size_t start, std::size_t end, bool copy) const{ */
/* 	std::size_t total = end - start; */
/* 	/1* std::cout<<"making unique from "<<start<<" to "<<end<<std::endl; *1/ */
/* 	intrusive_ptr<void*> cpy_str(_strides.vals_(), total, type_size, DTypeFuncs::size_of_dtype_p(dtype), detail::DontOrderStrides{}); */
/* 	if(!copy){return std::move(cpy_str);} */
/* 	void** original = strides_cbegin() + start; */
/* 	void** destination = cpy_str.get(); */
/* 	for(uint64_t i = 0; i < total; ++i, ++destination, ++original){ */
/* 		*destination = *original; */
/* 	} */
/* 	return std::move(cpy_str); */
/* } */




/* void ArrayVoid::unique_strides(bool pre_order){ */
/* 	if(_strides.use_count() == 1) */
/* 		return; */
/* 	if(!pre_order){ */
/* 		_strides = intrusive_ptr<void*>(_strides.vals_(), available_size, type_size, DTypeFuncs::size_of_dtype_p(dtype), detail::DontOrderStrides{}); */
/* 		return; */
/* 	} */
/* 	intrusive_ptr<void*> cpy_str(_strides.vals_(), available_size, type_size, DTypeFuncs::size_of_dtype_p(dtype), detail::DontOrderStrides{}); */
/* 	void** original = _strides.get(); */
/* 	void** destination = cpy_str.get(); */
/* 	for(uint64_t i = 0; i < available_size; ++i) */
/* 		destination[i] = original[i]; */
/* 	_strides = std::move(cpy_str); */
	
/* } */

/* ArrayVoid& ArrayVoid::iota(Scalar val){ */
/* 	if(dtype != DType::TensorObj) */
/* 		this->execute_function<WRAP_DTYPES<NumberTypesL>>()([&val](auto begin, auto end){std::iota(begin, end, val.to<utils::IteratorBaseType_t<decltype(begin)> >());}); */
/* 	else{ */
		
/* 		this->execute_function<WRAP_DTYPES<DTypeEnum<DType::TensorObj> > >([&v](auto begin, auto end){std::for_eacstd::iota(begin, end, v);}); */
/* 	} */
/* 	return *this; */
/* } */

template<DType dt>
bool _my_sub_copy_(ArrayVoid& Arr, const ArrayVoid& my_arr, unsigned long long i){	
	if(my_arr.dtype != dt) return _my_sub_copy_<DTypeFuncs::next_dtype_it<dt>>(Arr, my_arr, i);
	using value_t = DTypeFuncs::dtype_to_type<dt>;
	/* std::cout << "copying for "<<dt<< " "<< i<<std::endl; */
	uint32_t type_a = Arr.get_bucket().iterator_type();
	uint32_t type_b = my_arr.get_bucket().iterator_type();
	if(type_b == 1){
		auto begin = my_arr.get_bucket().cbegin_contiguous<value_t>();
		auto end = my_arr.get_bucket().cend_contiguous<value_t>();
		if(type_a == 1){
			auto begina = Arr.get_bucket().begin_contiguous<value_t>();
			std::copy(begin, end, begina);
		}
		else if(type_a == 2){
			auto begina = Arr.get_bucket().begin_blocked<value_t>();
			std::copy(begin, end, begina);
		}
		if(type_a == 3){
			auto begina = Arr.get_bucket().begin_list<value_t>();
			std::copy(begin, end, begina);
		}

	}
	else if(type_b == 2){
		auto begin = my_arr.get_bucket().cbegin_blocked<value_t>();
		auto end = my_arr.get_bucket().cend_blocked<value_t>();
		if(type_a == 1){
			auto begina = Arr.get_bucket().begin_contiguous<value_t>();
			std::copy(begin, end, begina);
		}
		else if(type_a == 2){
			auto begina = Arr.get_bucket().begin_blocked<value_t>();
			std::copy(begin, end, begina);
		}
		if(type_a == 3){
			auto begina = Arr.get_bucket().begin_list<value_t>();
			std::copy(begin, end, begina);
		}

	}
	if(type_b == 3){
		auto begin = my_arr.get_bucket().cbegin_list<value_t>();
		auto end = my_arr.get_bucket().cend_list<value_t>();
		if(type_a == 1){
			auto begina = Arr.get_bucket().begin_contiguous<value_t>();
			std::copy(begin, end, begina);
		}
		else if(type_a == 2){
			auto begina = Arr.get_bucket().begin_blocked<value_t>();
			std::copy(begin, end, begina);
		}
		if(type_a == 3){
			auto begina = Arr.get_bucket().begin_list<value_t>();
			std::copy(begin, end, begina);
		}

	}

	return true;
}

void ArrayVoid::copy(ArrayVoid& Arr, unsigned long long i) const{
	utils::throw_exception(Arr.dtype == dtype, "\nRuntime Error: Expected to copy ArrayVoid to same type $ but got $", dtype, Arr.dtype);
	Arr.execute_function([](auto begin, auto end, const nt::ArrayVoid& arr){
				using value_type = utils::IteratorBaseType_t<decltype(begin)>;
				uint32_t type_a = arr.get_bucket().iterator_type();
				if(type_a == 1){
					auto m_begin = arr.get_bucket().cbegin_contiguous<value_type>();
					auto m_end = arr.get_bucket().cend_contiguous<value_type>();
					std::copy(m_begin, m_end, begin);
				}
				else if(type_a == 2){
					auto m_begin = arr.get_bucket().cbegin_blocked<value_type>();
					auto m_end = arr.get_bucket().cend_blocked<value_type>();
					std::copy(m_begin, m_end, begin);
				}
				else if(type_a == 3){
					auto m_begin = arr.get_bucket().cbegin_list<value_type>();
					auto m_end = arr.get_bucket().cend_list<value_type>();
					std::copy(m_begin, m_end, begin);
				}

			}, *this);
	/* utils::throw_exception(_my_sub_copy_<DType::Integer>(Arr, *this, i), "\nRuntime Error: Was unable to copy ArrayVoid"); */
}

#include "../convert/Convert.h"

template<DType F, DType T, std::enable_if_t<T != DType::TensorObj && F != DType::TensorObj, bool> = true>
bool _my_sub_turn_dtype_(const ArrayVoid& my_arr, ArrayVoid& out){
	if(F != my_arr.dtype){
		return _my_sub_turn_dtype_<DTypeFuncs::next_dtype_it<F>, T>(my_arr, out);
	}
	if(T != out.dtype){return _my_sub_turn_dtype_<F, DTypeFuncs::next_dtype_it<T>>(my_arr, out);}
	using my_value_t = ::nt::DTypeFuncs::dtype_to_type_t<F>;
	using out_value_t = ::nt::DTypeFuncs::dtype_to_type_t<T>;
	uint32_t type_a = out.get_bucket().iterator_type();
	uint32_t type_b = my_arr.get_bucket().iterator_type();
	utils::throw_exception(type_a == 1, "Expected in turn dtype for output to be contiguous, but got iterator type $, problem with creation", type_a);
	if(type_b == 1){
		auto begin = my_arr.get_bucket().cbegin_contiguous<my_value_t>();
		auto end = my_arr.get_bucket().cend_contiguous<my_value_t>();
		out_value_t* o_begin = reinterpret_cast<out_value_t*>(out.data_ptr());
		std::transform(begin, end, o_begin, [](const my_value_t& val){return ::nt::convert::convert<T, my_value_t>(val);});
	}
	else if(type_b == 2){
		auto begin = my_arr.get_bucket().cbegin_blocked<my_value_t>();
		auto end = my_arr.get_bucket().cend_blocked<my_value_t>();
		out_value_t* o_begin = reinterpret_cast<out_value_t*>(out.data_ptr());
		std::transform(begin, end, o_begin, [](const my_value_t& val){return ::nt::convert::convert<T, my_value_t>(val);});
	}
	else if(type_b == 3){
		auto begin = my_arr.get_bucket().cbegin_list<my_value_t>();
		auto end = my_arr.get_bucket().cend_list<my_value_t>();
		out_value_t* o_begin = reinterpret_cast<out_value_t*>(out.data_ptr());
		std::transform(begin, end, o_begin, [](const my_value_t& val){return ::nt::convert::convert<T, my_value_t>(val);});
	}
	return true;
}

template<DType F, DType T, std::enable_if_t<T == DType::TensorObj && F != DType::TensorObj, bool> = true>
bool _my_sub_turn_dtype_(const ArrayVoid& my_arr, ArrayVoid& out){
	if(F != my_arr.dtype){return _my_sub_turn_dtype_<DTypeFuncs::next_dtype_it<F>, T>(my_arr, out);}
	if(T != out.dtype){return _my_sub_turn_dtype_<F, DTypeFuncs::next_dtype_it<T>>(my_arr, out);}
	using my_value_t = ::nt::DTypeFuncs::dtype_to_type_t<F>;
	using out_value_t = ::nt::DTypeFuncs::dtype_to_type_t<T>;
	uint32_t type_a = out.get_bucket().iterator_type();
	uint32_t type_b = my_arr.get_bucket().iterator_type();
	utils::throw_exception(type_a == 1, "Expected in turn dtype for output to be contiguous, but got iterator type $, problem with creation", type_a);
	if(type_b == 1){
		auto begin = my_arr.get_bucket().cbegin_contiguous<my_value_t>();
		auto end = my_arr.get_bucket().cend_contiguous<my_value_t>();
		out_value_t* o_begin = reinterpret_cast<out_value_t*>(out.data_ptr());
		std::transform(begin, end, o_begin, [](const my_value_t& val){Tensor outp({1}, F); outp.fill_(val); return std::move(outp);});
	}
	else if(type_b == 2){
		auto begin = my_arr.get_bucket().cbegin_blocked<my_value_t>();
		auto end = my_arr.get_bucket().cend_blocked<my_value_t>();
		out_value_t* o_begin = reinterpret_cast<out_value_t*>(out.data_ptr());
		std::transform(begin, end, o_begin, [](const my_value_t& val){Tensor outp({1}, F); outp.fill_(val); return std::move(outp);});
	}
	else if(type_b == 3){
		auto begin = my_arr.get_bucket().cbegin_list<my_value_t>();
		auto end = my_arr.get_bucket().cend_list<my_value_t>();
		out_value_t* o_begin = reinterpret_cast<out_value_t*>(out.data_ptr());
		std::transform(begin, end, o_begin, [](const my_value_t& val){Tensor outp({1}, F); outp.fill_(val); return std::move(outp);});
	}
	return true;
}

template<DType F, DType T, std::enable_if_t<T == DType::TensorObj && F == DType::TensorObj, bool> = true>
bool _my_sub_turn_dtype_(const ArrayVoid& my_arr, ArrayVoid& out){
	if(F != my_arr.dtype){return _my_sub_turn_dtype_<DTypeFuncs::next_dtype_it<F>, T>(my_arr, out);}
	if(T != out.dtype){return _my_sub_turn_dtype_<F, DTypeFuncs::next_dtype_it<T>>(my_arr, out);}
	return true;
}


template<DType F, DType T, std::enable_if_t<T != DType::TensorObj && F == DType::TensorObj, bool> = true>
bool _my_sub_turn_dtype_(const ArrayVoid& my_arr, ArrayVoid& out){
	if(F != my_arr.dtype){return _my_sub_turn_dtype_<DTypeFuncs::next_dtype_it<F>, T>(my_arr, out);}
	if(T != out.dtype){return _my_sub_turn_dtype_<F, DTypeFuncs::next_dtype_it<T>>(my_arr, out);}
	using my_value_t = ::nt::DTypeFuncs::dtype_to_type_t<F>;
	using out_value_t = ::nt::DTypeFuncs::dtype_to_type_t<T>;
	uint32_t type_a = out.get_bucket().iterator_type();
	uint32_t type_b = my_arr.get_bucket().iterator_type();
	utils::throw_exception(type_a == 1, "Expected in turn dtype for output to be contiguous, but got iterator type $, problem with creation", type_a);
	if(type_b == 1){
		auto begin = my_arr.get_bucket().cbegin_contiguous<my_value_t>();
		auto end = my_arr.get_bucket().cend_contiguous<my_value_t>();
		out_value_t* o_begin = reinterpret_cast<out_value_t*>(out.data_ptr());
		std::transform(begin, end, o_begin, [](const Tensor& val){return val.toScalar().to<out_value_t>();});
	}
	else if(type_b == 2){
		auto begin = my_arr.get_bucket().cbegin_blocked<my_value_t>();
		auto end = my_arr.get_bucket().cend_blocked<my_value_t>();
		out_value_t* o_begin = reinterpret_cast<out_value_t*>(out.data_ptr());
		std::transform(begin, end, o_begin, [](const Tensor& val){return val.toScalar().to<out_value_t>();});
	}
	else if(type_b == 3){
		auto begin = my_arr.get_bucket().cbegin_list<my_value_t>();
		auto end = my_arr.get_bucket().cend_list<my_value_t>();
		out_value_t* o_begin = reinterpret_cast<out_value_t*>(out.data_ptr());
		std::transform(begin, end, o_begin, [](const Tensor& val){return val.toScalar().to<out_value_t>();});
	}
	return true;
}


ArrayVoid& ArrayVoid::fill_ptr_(Scalar c){
	utils::throw_exception(is_contiguous(), "needed tensor to be contiguous to fill_ptr_");

#if SIMD_SUPPORTED
	uint32_t increments = DTypeFuncs::is_complex(dtype) ? (32 / DTypeFuncs::size_of_dtype(dtype) / 2) : (32 / DTypeFuncs::size_of_dtype(dtype));
	uint32_t left = (_last_index - _start) % increments;
	uint32_t end1 = (_last_index - _start) - left;
#endif

	switch(dtype){
		case DType::uint32:{
			using value_t = uint32_t;
			value_t* begin = reinterpret_cast<value_t*>(data_ptr());
			value_t val = c.isZero() ? value_t(0) : c.to<value_t>();
#if SIMD_SUPPORTED
			__m256i value =_mm256_set1_epi32(c.to<value_t>());
			value_t* end = begin + end1;
			for(;begin != end; begin += increments)
				_mm256_storeu_si256((__m256i*)begin, value);
			end += left;
			for(;begin != end; ++begin)
				*begin = val;
#else
			value_t* end = reinterpret_cast<value_t*>(data_ptr_end());
			for(;begin != end; ++begin)
				*begin = val;
#endif
			return *this;
		}
		case DType::int32:{
			using value_t = int32_t;
			value_t* begin = reinterpret_cast<value_t*>(data_ptr());
			value_t val = c.isZero() ? value_t(0) : c.to<value_t>();
#if SIMD_SUPPORTED
			__m256i value =_mm256_set1_epi32(c.to<value_t>());
			value_t* end = begin + end1;
			for(;begin != end; begin += increments)
				_mm256_storeu_si256((__m256i*)begin, value);
			end += left;
			for(;begin != end; ++begin)
				*begin = val;
#else
			value_t* end = reinterpret_cast<value_t*>(data_ptr_end());
			for(;begin != end; ++begin)
				*begin = val;
#endif
			return *this;
		}
		case DType::Double:{
			using value_t = double;
			value_t* begin = reinterpret_cast<value_t*>(data_ptr());
			value_t val = c.isZero() ? value_t(0) : c.to<value_t>();
#if SIMD_SUPPORTED
			__m256 value = _mm256_set1_pd(c.to<value_t>());
			value_t* end = begin + end1;
			for(;begin != end; begin += increments)
				_mm256_storeu_pd(begin, value);
			end += left;
			for(;begin != end; ++begin)
				*begin = val;
#else
			value_t* end = reinterpret_cast<value_t*>(data_ptr_end());
			for(;begin != end; ++begin)
				*begin = val;
#endif
			return *this;
		}
		case DType::Float:{
			using value_t = float;
			value_t* begin = reinterpret_cast<value_t*>(data_ptr());
			value_t val = c.isZero() ? value_t(0) : c.to<value_t>();
#if SIMD_SUPPORTED
			__m256 value = _mm256_set1_ps(c.to<value_t>());
			value_t* end = begin + end1;
			for(;begin != end; begin += increments)
				_mm256_storeu_ps(begin, value);
			end += left;
			for(;begin != end; ++begin)
				*begin = val;

#else
			value_t* end = reinterpret_cast<value_t*>(data_ptr_end());
			for(;begin != end; ++begin)
				*begin = val;
#endif
			return *this;
		}
		case DType::cfloat:{
			using value_t = complex_64;
			value_t* begin = reinterpret_cast<value_t*>(data_ptr());
			value_t val = c.isZero() ? value_t(0) : c.to<value_t>();
#if SIMD_SUPPORTED
			__m256 value_real = _mm256_set1_ps(c.to<value_t>().real());
			__m256 value_imag = _mm256_set1_ps(c.to<value_t>().imag());
			value_t* end = begin + end1;
			for(;begin != end; begin += increments){
				_mm256_storeu_ps(&begin->real(), value_real);
				_mm256_storeu_ps(&begin->imag(), value_imag);
			}
			end += left;
			for(;begin != end; ++begin)
				*begin = val;

#else
			value_t* end = reinterpret_cast<value_t*>(data_ptr_end());
			for(;begin != end; ++begin)
				*begin = val;
#endif
			for(;begin != end; ++begin)
				*begin = val;
			return *this;
		}
		case DType::cdouble:{
			using value_t = complex_128;
			value_t* begin = reinterpret_cast<value_t*>(data_ptr());
			value_t val = c.isZero() ? value_t(0) : c.to<value_t>();
#if SIMD_SUPPORTED
			__m256 value_real = _mm256_set1_pd(c.to<value_t>().real());
			__m256 value_imag = _mm256_set1_pd(c.to<value_t>().imag());
			value_t* end = begin + end1;
			for(;begin != end; begin += increments){
				_mm256_storeu_pd(&begin->real(), value_real);
				_mm256_storeu_pd(&begin->imag(), value_imag);
			}
			end += left;
			for(;begin != end; ++begin)
				*begin = val;
#else
			value_t* end = reinterpret_cast<value_t*>(data_ptr_end());
			for(;begin != end; ++begin)
				*begin = val;
#endif
			return *this;
		}
		case DType::TensorObj:{
			return *this;
		}
		case DType::uint8:{
			using value_t = uint8_t;
			value_t* begin = reinterpret_cast<value_t*>(data_ptr());
			value_t val = c.isZero() ? value_t(0) : c.to<value_t>();
#if SIMD_SUPPORTED
			__m256i value =_mm256_set1_epi8(c.to<value_t>());
			value_t* end = begin + end1;
			for(;begin != end; begin += increments)
				_mm256_storeu_si256((__m256i*)begin, value);
			end += left;
			for(;begin != end; ++begin)
				*begin = val;
#else
			value_t* end = reinterpret_cast<value_t*>(data_ptr_end());
			for(;begin != end; ++begin)
				*begin = val;
#endif
			return *this;
		}
		case DType::int8:{
			using value_t = int8_t;
			value_t* begin = reinterpret_cast<value_t*>(data_ptr());
#if SIMD_SUPPORTED
			__m256i value =_mm256_set1_epi8(c.to<value_t>());
			value_t* end = begin + end1;
			for(;begin != end; begin += increments)
				_mm256_storeu_si256((__m256i*)begin, value);
			end += left;
			for(;begin != end; ++begin)
				*begin = val;
#else
			value_t val = c.isZero() ? value_t(0) : c.to<value_t>();
			value_t* end = reinterpret_cast<value_t*>(data_ptr_end());
			for(;begin != end; ++begin)
				*begin = val;
#endif
			return *this;			
		}
		case DType::uint16:{
			using value_t = uint16_t;
			value_t* begin = reinterpret_cast<value_t*>(data_ptr());
#if SIMD_SUPPORTED
			__m256i value =_mm256_set1_epi16(c.to<value_t>());
			value_t* end = begin + end1;
			for(;begin != end; begin += increments)
				_mm256_storeu_si256((__m256i*)begin, value);
			end += left;
			for(;begin != end; ++begin)
				*begin = val;
#else
			value_t val = c.isZero() ? value_t(0) : c.to<value_t>();
			value_t* end = reinterpret_cast<value_t*>(data_ptr_end());
			for(;begin != end; ++begin)
				*begin = val;
#endif
			return *this;			
		}
		case DType::int16:{
			using value_t = int16_t;
			value_t* begin = reinterpret_cast<value_t*>(data_ptr());
#if SIMD_SUPPORTED
			__m256i value =_mm256_set1_epi16(c.to<value_t>());
			value_t* end = begin + end1;
			for(;begin != end; begin += increments)
				_mm256_storeu_si256((__m256i*)begin, value);
			end += left;
			for(;begin != end; ++begin)
				*begin = val;
#else
			value_t val = c.isZero() ? value_t(0) : c.to<value_t>();
			value_t* end = reinterpret_cast<value_t*>(data_ptr_end());
			for(;begin != end; ++begin)
				*begin = val;
#endif			
			return *this;
		}
		case DType::int64:{
			using value_t = int64_t;
			value_t* begin = reinterpret_cast<value_t*>(data_ptr());
#if SIMD_SUPPORTED
			__m256i value =_mm256_set1_epi64x(c.to<value_t>());
			value_t* end = begin + end1;
			for(;begin != end; begin += increments)
				_mm256_storeu_si256((__m256i*)begin, value);
			end += left;
			for(;begin != end; ++begin)
				*begin = val;
#else
			value_t val = c.isZero() ? value_t(0) : c.to<value_t>();
			value_t* end = reinterpret_cast<value_t*>(data_ptr_end());
			for(;begin != end; ++begin)
				*begin = val;
#endif
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
#ifdef _HALF_FLOAT_SUPPORT_ // there is no way to do the following using SIMD instructions, but it can probably be done with a uint16_t array, and a bitset, probably
		case DType::Float16:{
			using value_t = float16_t;
			value_t* begin = reinterpret_cast<value_t*>(data_ptr());
			value_t* end = reinterpret_cast<value_t*>(data_ptr_end());
			value_t val = c.isZero() ? value_t(0) : c.to<value_t>();
			for(;begin != end; ++begin)
				*begin = val;
			return *this;
		}
		case DType::Complex32:{
			using value_t = complex_32;
			value_t* begin = reinterpret_cast<value_t*>(data_ptr());
			value_t* end = reinterpret_cast<value_t*>(data_ptr_end());
			value_t val = c.isZero() ? value_t(0) : c.to<value_t>();
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
			value_t val = c.isZero() ? value_t(0) : c.to<value_t>();
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
			value_t val = c.isZero() ? value_t(0) : c.to<value_t>();
			for(;begin != end; ++begin)
				*begin = val;
			return *this;
		}
		case DType::uint128:{
			using value_t = uint128_t;
			value_t* begin = reinterpret_cast<value_t*>(data_ptr());
			value_t* end = reinterpret_cast<value_t*>(data_ptr_end());
			value_t val = c.isZero() ? value_t(0) : c.to<value_t>();
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

ArrayVoid ArrayVoid::to(DeviceType dt) const{
	return ArrayVoid(this->bucket.to_device(dt), size, dtype);
}


Tensor ArrayVoid::split(const uint64_t sp) const{
	Tensor buckets = bucket.split<Tensor>(sp);
	Tensor* begin = reinterpret_cast<Tensor*>(buckets.data_ptr());
	Tensor* end = begin + buckets.numel();
	typedef typename SizeRef::ArrayRefInt::value_type m_size_t;
	for(;begin != end; ++begin){
		m_size_t size = begin->_vals.bucket.size();
		begin->_size = SizeRef({size});
		begin->dtype = dtype;

	}
	return std::move(buckets);
}

Tensor ArrayVoid::split(const uint64_t sp, SizeRef s_outp) const{
	Tensor buckets = bucket.split<Tensor>(sp);
	Tensor* begin = reinterpret_cast<Tensor*>(buckets.data_ptr());
	Tensor* end = begin + buckets.numel();
	typedef typename SizeRef::ArrayRefInt::value_type m_size_t;
	for(;begin != end; ++begin){
		begin->_size = s_outp;
		begin->dtype = dtype;
	}
	return std::move(buckets);
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
	utils::throw_exception((dtype == DType::TensorObj && A.dtype != DType::TensorObj) 
			|| (dtype == A.dtype), "$ can not have operator * with $", dtype, A.dtype);
	ArrayVoid output(size, dtype); // this is going to just be contiguous
	if(dtype == DType::TensorObj && A.dtype != DType::TensorObj){
		auto out_begin = output.bucket.begin_contiguous<Tensor>();
		A.transform_function<WRAP_DTYPES<NumberTypesL>>([](const auto& t, const auto& o){return t * o;}, *this, out_begin);
		return std::move(output);
	}
	this->cexecute_function_nbool([](auto begin, auto end, auto begin2, ArrayVoid& output){
				using value_t = utils::IteratorBaseType_t<decltype(begin)>;
				std::transform(begin, end, begin2, output.bucket.begin_contiguous<value_t>(), std::multiplies<value_t>());}, A, output);
	return std::move(output);
}

ArrayVoid ArrayVoid::operator/(const ArrayVoid& A) const{
	utils::throw_exception(size == A.size, "For operators, sizes must be equal, expected size of $ but got $", size, A.size);
	utils::throw_exception((dtype == DType::TensorObj && A.dtype != DType::TensorObj) 
			|| (dtype == A.dtype), "$ can not have operator * with $", dtype, A.dtype);
	ArrayVoid output(size, dtype);
	if(dtype == DType::TensorObj && A.dtype != DType::TensorObj){
		auto out_begin = output.bucket.begin_contiguous<Tensor>();
		A.transform_function<WRAP_DTYPES<NumberTypesL>>([](const auto& t, const auto& o){return t / o;},*this, out_begin);
		return std::move(output);
	}
	this->cexecute_function_nbool([](auto begin, auto end, auto begin2, ArrayVoid& output){
				using value_t = utils::IteratorBaseType_t<decltype(begin)>;
				std::transform(begin, end, begin2, output.bucket.begin_contiguous<value_t>(), std::divides<value_t>());}, A, output);
	return std::move(output);
}

ArrayVoid ArrayVoid::operator-(const ArrayVoid& A) const{
	utils::throw_exception(size == A.size, "For operators, sizes must be equal, expected size of $ but got $", size, A.size);
	utils::throw_exception((dtype == DType::TensorObj && A.dtype != DType::TensorObj) 
			|| (dtype == A.dtype), "$ can not have operator * with $", dtype, A.dtype);
	ArrayVoid output(size, dtype);
	if(dtype == DType::TensorObj && A.dtype != DType::TensorObj){
		auto out_begin = output.bucket.begin_contiguous<Tensor>();
		A.transform_function<WRAP_DTYPES<NumberTypesL>>([](const auto& t, const auto& o){return t - o;},*this, out_begin);
		return std::move(output);
	}
	this->cexecute_function_nbool([](auto begin, auto end, auto begin2, ArrayVoid& output){
				using value_t = utils::IteratorBaseType_t<decltype(begin)>;
				std::transform(begin, end, begin2, output.bucket.begin_contiguous<value_t>(), std::minus<value_t>());}, A, output);
	return std::move(output);
}

ArrayVoid ArrayVoid::operator+(const ArrayVoid& A) const{
	utils::throw_exception(size == A.size, "For operators, sizes must be equal, expected size of $ but got $", size, A.size);
	utils::throw_exception((dtype == DType::TensorObj && A.dtype != DType::TensorObj) 
			|| (dtype == A.dtype), "$ can not have operator * with $", dtype, A.dtype);
	ArrayVoid output(size, dtype);
	if(dtype == DType::TensorObj && A.dtype != DType::TensorObj){
		auto out_begin = output.bucket.begin_contiguous<Tensor>();
		A.transform_function<WRAP_DTYPES<NumberTypesL>>([](const auto& t, const auto& o){return t + o;},*this, out_begin);
		return std::move(output);
	}
	this->cexecute_function_nbool([](auto begin, auto end, auto begin2, ArrayVoid& output){
				using value_t = utils::IteratorBaseType_t<decltype(begin)>;
				std::transform(begin, end, begin2, output.bucket.begin_contiguous<value_t>(), std::plus<value_t>());}, A, output);
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
	}
}

ArrayVoid& ArrayVoid::operator/=(const ArrayVoid& A){
	if(dtype == A.dtype){
		this->transform_function_nbool([](auto& a, auto& b){return a / b;}, A);
		return *this;
	}
	else{
		return *this *= A.to(dtype);
	}
}

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
	uint_bool_t* o_begin = reinterpret_cast<uint_bool_t*>(output.data_ptr());
	const uint64_t& m_size = size;
	this->cexecute_function<WRAP_DTYPES<NumberTypesL, DTypeEnum<DType::Bool>>>([&o_begin, &m_size](auto begin, auto end, Scalar& s){
				using value_t = utils::IteratorBaseType_t<decltype(begin)>;
				value_t i = s.to<value_t>();
#ifdef USE_PARALLEL
				tbb::parallel_for(tbb::blocked_range<uint64_t>(0, m_size),
					[&o_begin, &begin, &i](tbb::blocked_range<uint64_t> r){
					auto cur_b = begin + r.begin();
					auto cur_e = begin + r.end();
					uint_bool_t* cur_o = o_begin + r.begin();
					for(;cur_b != cur_e; ++cur_b, ++cur_o){
						*cur_o = uint_bool_t(*cur_b == i);
					}});
#else
				std::transform(begin, end, o_begin, [&i](const auto& val){
						return uint_bool_t(val == i);});
#endif
			}, c);
	/* utils::throw_exception(_my_sub_equal_operator_<DType::Integer>(*this, reinterpret_cast<uint_bool_t*>(output.data_ptr()), c), "\nRuntime Error: input dtype $ is invalid for == operator", dtype); */
	return std::move(output);
}

ArrayVoid ArrayVoid::operator!=(Scalar c) const{
	if(dtype == DType::TensorObj){
		ArrayVoid output(size, DType::TensorObj);
		Tensor* OutputIt = reinterpret_cast<Tensor*>(output.data_ptr());
		this->transform_function<DType::TensorObj>([&c](const auto& val){return val != c;}, OutputIt);
		return std::move(output);
	}
	const size_t n_size = size;
	ArrayVoid output(n_size, DType::Bool);
	if(DTypeFuncs::is_unsigned(dtype) && c.to<int64_t>() < 0){
		output = uint_bool_t(true);
		return std::move(output);
	}
	if(c.isComplex() && (!DTypeFuncs::is_complex(dtype))){
		output = uint_bool_t(true);
		return std::move(output);
	}
	this->cexecute_function<WRAP_DTYPES<NumberTypesL, DTypeEnum<DType::Bool>>>([](auto begin, auto end, ArrayVoid& op, Scalar& s){
				auto o_begin = op.get_bucket().begin_contiguous<uint_bool_t>();
				using value_t = utils::IteratorBaseType_t<decltype(begin)>;
				value_t i = s.to<value_t>();
				std::transform(begin, end, o_begin, [&i](const auto& val){
						return uint_bool_t(!(val == i));});
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
				auto o_begin = op.get_bucket().begin_contiguous<uint_bool_t>();
				using value_t = utils::IteratorBaseType_t<decltype(begin)>;
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
				auto o_begin = op.get_bucket().begin_contiguous<uint_bool_t>();
				using value_t = utils::IteratorBaseType_t<decltype(begin)>;
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
				auto o_begin = op.get_bucket().begin_contiguous<uint_bool_t>();
				using value_t = utils::IteratorBaseType_t<decltype(begin)>;
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
				auto o_begin = op.get_bucket().begin_contiguous<uint_bool_t>();
				using value_t = utils::IteratorBaseType_t<decltype(begin)>;
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
	using my_value_t = ::nt::DTypeFuncs::dtype_to_type_t<dt>;
	using out_value_t = ::nt::DTypeFuncs::dtype_to_type_t<dt>;
	uint32_t type_a = outp.get_bucket().iterator_type();
	uint32_t type_b = my_arr.get_bucket().iterator_type();
	utils::throw_exception(type_a == 1, "Expected in sub inverse for output to be contiguous, but got iterator type $, problem with creation", type_a);
	if(type_b == 1){
		auto begin = my_arr.get_bucket().cbegin_contiguous<my_value_t>();
		auto end = my_arr.get_bucket().cend_contiguous<my_value_t>();
		out_value_t* o_begin = reinterpret_cast<out_value_t*>(outp.data_ptr());
		std::transform(begin, end, o_begin, [](const my_value_t& val){return 1.0/val;});
	}
	else if(type_b == 2){
		auto begin = my_arr.get_bucket().cbegin_blocked<my_value_t>();
		auto end = my_arr.get_bucket().cend_blocked<my_value_t>();
		out_value_t* o_begin = reinterpret_cast<out_value_t*>(outp.data_ptr());
		std::transform(begin, end, o_begin, [](const my_value_t& val){return 1.0/val;});
	}
	else if(type_b == 3){
		auto begin = my_arr.get_bucket().cbegin_list<my_value_t>();
		auto end = my_arr.get_bucket().cend_list<my_value_t>();
		out_value_t* o_begin = reinterpret_cast<out_value_t*>(outp.data_ptr());
		std::transform(begin, end, o_begin, [](const my_value_t& val){return 1.0/val;});
	}

	return true;
}

template<DType dt, std::enable_if_t<dt != DType::Bool && dt != DType::TensorObj && DTypeFuncs::is_dtype_complex_v<dt>, bool> = true>
bool _my_sub_inverse_(const ArrayVoid& my_arr, ArrayVoid& outp){
	if(dt != my_arr.dtype) return _my_sub_inverse_<DTypeFuncs::next_dtype_it<dt>>(my_arr, outp);
	using my_value_t = ::nt::DTypeFuncs::dtype_to_type_t<dt>;
	using out_value_t = ::nt::DTypeFuncs::dtype_to_type_t<dt>;
	uint32_t type_a = outp.get_bucket().iterator_type();
	uint32_t type_b = my_arr.get_bucket().iterator_type();
	utils::throw_exception(type_a == 1, "Expected in sub inverse for output to be contiguous, but got iterator type $, problem with creation", type_a);
	if(type_b == 1){
		auto begin = my_arr.get_bucket().cbegin_contiguous<my_value_t>();
		auto end = my_arr.get_bucket().cend_contiguous<my_value_t>();
		out_value_t* o_begin = reinterpret_cast<out_value_t*>(outp.data_ptr());
		std::transform(begin, end, o_begin, [](const my_value_t& val){return out_value_t(1.0/val.real(), 1.0/val.imag());});
	}
	else if(type_b == 2){
		auto begin = my_arr.get_bucket().cbegin_blocked<my_value_t>();
		auto end = my_arr.get_bucket().cend_blocked<my_value_t>();
		out_value_t* o_begin = reinterpret_cast<out_value_t*>(outp.data_ptr());
		std::transform(begin, end, o_begin, [](const my_value_t& val){return out_value_t(1.0/val.real(), 1.0/val.imag());});
	}
	else if(type_b == 3){
		auto begin = my_arr.get_bucket().cbegin_list<my_value_t>();
		auto end = my_arr.get_bucket().cend_list<my_value_t>();
		out_value_t* o_begin = reinterpret_cast<out_value_t*>(outp.data_ptr());
		std::transform(begin, end, o_begin, [](const my_value_t& val){return out_value_t(1.0/val.real(), 1.0/val.imag());});
	}

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
	using my_value_t = ::nt::DTypeFuncs::dtype_to_type_t<dt>;
	using out_value_t = ::nt::DTypeFuncs::dtype_to_type_t<dt>;
	uint32_t type_a = outp.get_bucket().iterator_type();
	uint32_t type_b = my_arr.get_bucket().iterator_type();
	utils::throw_exception(type_a == 1, "Expected in sub inverse for output to be contiguous, but got iterator type $, problem with creation", type_a);
	if(type_b == 1){
		auto begin = my_arr.get_bucket().cbegin_contiguous<my_value_t>();
		auto end = my_arr.get_bucket().cend_contiguous<my_value_t>();
		out_value_t* o_begin = reinterpret_cast<out_value_t*>(outp.data_ptr());
		std::transform(begin, end, o_begin, [](const my_value_t& val){return val.inverse();});
	}
	else if(type_b == 2){
		auto begin = my_arr.get_bucket().cbegin_blocked<my_value_t>();
		auto end = my_arr.get_bucket().cend_blocked<my_value_t>();
		out_value_t* o_begin = reinterpret_cast<out_value_t*>(outp.data_ptr());
		std::transform(begin, end, o_begin, [](const my_value_t& val){return val.inverse();});
	}
	else if(type_b == 3){
		auto begin = my_arr.get_bucket().cbegin_list<my_value_t>();
		auto end = my_arr.get_bucket().cend_list<my_value_t>();
		out_value_t* o_begin = reinterpret_cast<out_value_t*>(outp.data_ptr());
		std::transform(begin, end, o_begin, [](const my_value_t& val){return val.inverse();});
	}
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

ArrayVoid ArrayVoid::pow(int64_t p) const {
	if(p < 0){return inverse().pow(-p);}
	ArrayVoid output(size, dtype);
	if(p == 0){output.fill_ptr_(1);}
	this->cexecute_function<WRAP_DTYPES<RealNumberTypesL> >(
			[&p](auto a_begin, auto a_end, void* b_begin){
			using value_t = utils::IteratorBaseType_t<decltype(a_begin)>;
			value_t* begin = reinterpret_cast<value_t*>(b_begin);
			for(;a_begin != a_end; ++a_begin, ++begin){
				if constexpr (std::is_same_v<value_t, float16_t>){
					*begin = ::nt::convert::convert<DType::Float16>(std::pow(::nt::convert::convert<DType::Float32>(*a_begin), p));
				}
				else{
					*begin = std::pow(*a_begin, p);
				}
			}
			}, output.data_ptr());
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
#if defined(__SIZEOF_LONG_DOUBLE__) && __SIZEOF_LONG_DOUBLE__ == 16
	else if(dtype == DType::Float128)
		output.transform_function<DType::Float128>([](auto& outp, const auto& a){return std::exp(a);}, *this);
#else
	else if(dtype == DType::Float128)
		output.transform_function<DType::Float128>([](auto& outp, const auto& a){return std::exp(static_cast<long double>(a));}, *this);
#endif
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
#ifdef _128_FLOAT_SUPPORT
#if defined(__SIZEOF_LONG_DOUBLE__) && __SIZEOF_LONG_DOUBLE__ == 16
	else if(dtype == DType::Float128)
		this->for_each<DType::Float128>([](auto& a){a = std::exp(a);});
#else
	else if(dtype == DType::Float128)
		this->for_each<DType::Float128>([](auto& a){a = std::exp(static_cast<long double>(a));});
#endif
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
	utils::throw_exception(is_contiguous(), "In order to implicitly convert to a type, must be contiguous");
	DType complex_to = DTypeFuncs::complex_size(DTypeFuncs::size_of_dtype(dtype));
	utils::throw_exception(DTypeFuncs::is_complex(complex_to), "\nRuntime Error: Unable to implicitly convert $ to complex", dtype);
	DTypeFuncs::convert_this_dtype_array(data_ptr(), dtype, complex_to, size);
	return *this;
}

ArrayVoid& ArrayVoid::floating_(){
	DType floating_to = DTypeFuncs::floating_size(DTypeFuncs::size_of_dtype(dtype));
	utils::throw_exception(DTypeFuncs::is_floating(floating_to), "\nRuntime Error: Unable to implicitly convert $ to floating", dtype);
	DTypeFuncs::convert_this_dtype_array(data_ptr(), dtype, floating_to, size);
	return *this;
}

ArrayVoid& ArrayVoid::integer_(){
	utils::throw_exception(is_contiguous(), "In order to implicitly convert to a type, must be contiguous");
	DType integer_to = DTypeFuncs::integer_size(DTypeFuncs::size_of_dtype(dtype));
	utils::throw_exception(DTypeFuncs::is_integer(integer_to), "\nRuntime Error: Unable to implicitly convert $ to integer", dtype);
	DTypeFuncs::convert_this_dtype_array(data_ptr(), dtype, integer_to, size);
	return *this;
}
ArrayVoid& ArrayVoid::unsigned_(){
	utils::throw_exception(is_contiguous(), "In order to implicitly convert to a type, must be contiguous");
	DType unsigned_to = DTypeFuncs::unsigned_size(DTypeFuncs::size_of_dtype(dtype));
	utils::throw_exception(DTypeFuncs::is_unsigned(unsigned_to), "\nRuntime Error: Unable to implicitly convert $ to unsigned", dtype);
	DTypeFuncs::convert_this_dtype_array(data_ptr(), dtype, unsigned_to, size);
	return *this;
}

/*
ArrayVoid::ArrayVoid(const std::shared_ptr<void>& _v, std::shared_ptr<void*>&& str, const std::size_t size, const std::size_t start, std::size_t avail, DType _dt, bool shared)
	:_vals(_v),
	size(size),
	dtype(_dt),
	type_size(this->dtype_size(_dt)),
	_last_index(size + start),
	_strides(std::move(str)),
	_start(start),
	available_size(avail),
	is_shared(shared)
{}
*/

#ifdef USE_PARALLEL
ArrayVoid ArrayVoid::shared_memory() const{
	utils::throw_exception(is_shared() == false, "Expected to make shared from non-shared memory");
	if(!is_contiguous()){
		return this->contiguous().shared_memory();
	}
	return ArrayVoid(bucket.to_shared());
}

ArrayVoid ArrayVoid::from_shared_memory() const{
	utils::throw_exception(is_shared() == true, "Expected to make non-shared from shared memory");
	if(!is_contiguous()){
		return this->contiguous().from_shared_memory();
	}
	return ArrayVoid(bucket.to_cpu());
}
#endif

/* intrusive_ptr<void> ArrayVoid::MakeContiguousMemory(uint32_t _size, DType _type){ */
/* 	return intrusive_ptr<void>(_size, DTypeFuncs::size_of_dtype(_type)); */
/* } */


/* intrusive_ptr<void> ArrayVoid::MakeContiguousMemory(uint32_t _size, DType _type, Scalar s){ */
/* 	intrusive_ptr<void> ptr(_size, DTypeFuncs::size_of_dtype(_type)); */
/* 	switch(_type){ */
/* 		case DType::Integer:{ */
/* 			using value_t = int32_t; */
/* 			std::fill(reinterpret_cast<value_t*>(ptr.get()), reinterpret_cast<value_t*>(ptr.get()) + _size, s.to<value_t>()); */
/* 			return ptr; */
/* 		} */
/* 		case DType::Float:{ */
/* 			using value_t = float; */
/* 			std::fill(reinterpret_cast<value_t*>(ptr.get()), reinterpret_cast<value_t*>(ptr.get()) + _size, s.to<value_t>()); */
/* 			return ptr; */
/* 		} */
/* 		case DType::Double:{ */
/* 			using value_t = double; */
/* 			std::fill(reinterpret_cast<value_t*>(ptr.get()), reinterpret_cast<value_t*>(ptr.get()) + _size, s.to<value_t>()); */
/* 			return ptr; */
/* 		} */
/* 		case DType::Long:{ */
/* 			using value_t = uint32_t; */
/* 			std::fill(reinterpret_cast<value_t*>(ptr.get()), reinterpret_cast<value_t*>(ptr.get()) + _size, s.to<value_t>()); */
/* 			return ptr; */
/* 		} */
/* 		case DType::Complex64:{ */
/* 			using value_t = complex_64; */
/* 			std::fill(reinterpret_cast<value_t*>(ptr.get()), reinterpret_cast<value_t*>(ptr.get()) + _size, s.to<value_t>()); */
/* 			return ptr; */
/* 		} */
/* 		case DType::Complex128:{ */
/* 			using value_t = complex_128; */
/* 			std::fill(reinterpret_cast<value_t*>(ptr.get()), reinterpret_cast<value_t*>(ptr.get()) + _size, s.to<value_t>()); */
/* 			return ptr; */
/* 		} */
/* 		case DType::uint8:{ */
/* 			using value_t = uint8_t; */
/* 			std::fill(reinterpret_cast<value_t*>(ptr.get()), reinterpret_cast<value_t*>(ptr.get()) + _size, s.to<value_t>()); */
/* 			return ptr; */
/* 		} */
/* 		case DType::int8:{ */
/* 			using value_t = int8_t; */
/* 			std::fill(reinterpret_cast<value_t*>(ptr.get()), reinterpret_cast<value_t*>(ptr.get()) + _size, s.to<value_t>()); */
/* 			return ptr; */
/* 		} */
/* 		case DType::int16:{ */
/* 			using value_t = int16_t; */
/* 			std::fill(reinterpret_cast<value_t*>(ptr.get()), reinterpret_cast<value_t*>(ptr.get()) + _size, s.to<value_t>()); */
/* 			return ptr; */
/* 		} */
/* 		case DType::uint16:{ */
/* 			using value_t = uint16_t; */
/* 			std::fill(reinterpret_cast<value_t*>(ptr.get()), reinterpret_cast<value_t*>(ptr.get()) + _size, s.to<value_t>()); */
/* 			return ptr; */
/* 		} */
/* 		case DType::LongLong:{ */
/* 			using value_t = int64_t; */
/* 			std::fill(reinterpret_cast<value_t*>(ptr.get()), reinterpret_cast<value_t*>(ptr.get()) + _size, s.to<value_t>()); */
/* 			return ptr; */
/* 		} */
/* 		case DType::Bool:{ */
/* 			using value_t = uint_bool_t; */
/* 			std::fill(reinterpret_cast<value_t*>(ptr.get()), reinterpret_cast<value_t*>(ptr.get()) + _size, s.to<value_t>()); */
/* 			return ptr; */
/* 		} */
/* 		case DType::TensorObj:{ */
/* 			return ptr; */
/* 		} */
/* #ifdef _128_FLOAT_SUPPORT_ */
/* 		case DType::Float128:{ */
/* 			using value_t = float128_t; */
/* 			std::fill(reinterpret_cast<value_t*>(ptr.get()), reinterpret_cast<value_t*>(ptr.get()) + _size, s.to<value_t>()); */
/* 			return ptr; */
/* 		} */
/* #endif */
/* #ifdef _HALF_FLOAT_SUPPORT_ */
/* 		case DType::Float16:{ */
/* 			using value_t = float16_t; */
/* 			std::fill(reinterpret_cast<value_t*>(ptr.get()), reinterpret_cast<value_t*>(ptr.get()) + _size, s.to<value_t>()); */
/* 			return ptr; */
/* 		} */
/* 		case DType::Complex32:{ */
/* 			using value_t = complex_32; */
/* 			std::fill(reinterpret_cast<value_t*>(ptr.get()), reinterpret_cast<value_t*>(ptr.get()) + _size, s.to<value_t>()); */
/* 			return ptr; */
/* 		} */
/* #endif */
/* #ifdef __SIZEOF_INT128__ */
/* 		case DType::int128:{ */
/* 			using value_t = int128_t; */
/* 			std::fill(reinterpret_cast<value_t*>(ptr.get()), reinterpret_cast<value_t*>(ptr.get()) + _size, s.to<value_t>()); */
/* 			return ptr; */
/* 		} */
/* 		case DType::uint128:{ */
/* 			using value_t = uint128_t; */
/* 			std::fill(reinterpret_cast<value_t*>(ptr.get()), reinterpret_cast<value_t*>(ptr.get()) + _size, s.to<value_t>()); */
/* 			return ptr; */
/* 		} */
/* #endif */
/* 	} */
/* 	return ptr; */
/* } */

/* #ifdef USE_PARALLEL */
/* intrusive_ptr<void> ArrayVoid::MakeContiguousMemory(uint32_t _size, DTypeShared _type, Scalar s){ */
/* 	intrusive_ptr<void> ptr = intrusive_ptr<void>::make_shared(_size, DTypeFuncs::size_of_dtype(DTypeShared_DType(_type))); */
/* 	switch(DTypeShared_DType(_type)){ */
/* 		case DType::Integer:{ */
/* 			using value_t = int32_t; */
/* 			std::fill(reinterpret_cast<value_t*>(ptr.get()), reinterpret_cast<value_t*>(ptr.get()) + _size, s.to<value_t>()); */
/* 			return ptr; */
/* 		} */
/* 		case DType::Float:{ */
/* 			using value_t = float; */
/* 			std::fill(reinterpret_cast<value_t*>(ptr.get()), reinterpret_cast<value_t*>(ptr.get()) + _size, s.to<value_t>()); */
/* 			return ptr; */
/* 		} */
/* 		case DType::Double:{ */
/* 			using value_t = double; */
/* 			std::fill(reinterpret_cast<value_t*>(ptr.get()), reinterpret_cast<value_t*>(ptr.get()) + _size, s.to<value_t>()); */
/* 			return ptr; */
/* 		} */
/* 		case DType::Long:{ */
/* 			using value_t = uint32_t; */
/* 			std::fill(reinterpret_cast<value_t*>(ptr.get()), reinterpret_cast<value_t*>(ptr.get()) + _size, s.to<value_t>()); */
/* 			return ptr; */
/* 		} */
/* 		case DType::Complex64:{ */
/* 			using value_t = complex_64; */
/* 			std::fill(reinterpret_cast<value_t*>(ptr.get()), reinterpret_cast<value_t*>(ptr.get()) + _size, s.to<value_t>()); */
/* 			return ptr; */
/* 		} */
/* 		case DType::Complex128:{ */
/* 			using value_t = complex_128; */
/* 			std::fill(reinterpret_cast<value_t*>(ptr.get()), reinterpret_cast<value_t*>(ptr.get()) + _size, s.to<value_t>()); */
/* 			return ptr; */
/* 		} */
/* 		case DType::uint8:{ */
/* 			using value_t = uint8_t; */
/* 			std::fill(reinterpret_cast<value_t*>(ptr.get()), reinterpret_cast<value_t*>(ptr.get()) + _size, s.to<value_t>()); */
/* 			return ptr; */
/* 		} */
/* 		case DType::int8:{ */
/* 			using value_t = int8_t; */
/* 			std::fill(reinterpret_cast<value_t*>(ptr.get()), reinterpret_cast<value_t*>(ptr.get()) + _size, s.to<value_t>()); */
/* 			return ptr; */
/* 		} */
/* 		case DType::int16:{ */
/* 			using value_t = int16_t; */
/* 			std::fill(reinterpret_cast<value_t*>(ptr.get()), reinterpret_cast<value_t*>(ptr.get()) + _size, s.to<value_t>()); */
/* 			return ptr; */
/* 		} */
/* 		case DType::uint16:{ */
/* 			using value_t = uint16_t; */
/* 			std::fill(reinterpret_cast<value_t*>(ptr.get()), reinterpret_cast<value_t*>(ptr.get()) + _size, s.to<value_t>()); */
/* 			return ptr; */
/* 		} */
/* 		case DType::LongLong:{ */
/* 			using value_t = int64_t; */
/* 			std::fill(reinterpret_cast<value_t*>(ptr.get()), reinterpret_cast<value_t*>(ptr.get()) + _size, s.to<value_t>()); */
/* 			return ptr; */
/* 		} */
/* 		case DType::Bool:{ */
/* 			using value_t = uint_bool_t; */
/* 			std::fill(reinterpret_cast<value_t*>(ptr.get()), reinterpret_cast<value_t*>(ptr.get()) + _size, s.to<value_t>()); */
/* 			return ptr; */
/* 		} */
/* 		case DType::TensorObj:{ */
/* 			return ptr; */
/* 		} */
/* #ifdef _128_FLOAT_SUPPORT_ */
/* 		case DType::Float128:{ */
/* 			using value_t = float128_t; */
/* 			std::fill(reinterpret_cast<value_t*>(ptr.get()), reinterpret_cast<value_t*>(ptr.get()) + _size, s.to<value_t>()); */
/* 			return ptr; */
/* 		} */
/* #endif */
/* #ifdef _HALF_FLOAT_SUPPORT_ */
/* 		case DType::Float16:{ */
/* 			using value_t = float16_t; */
/* 			std::fill(reinterpret_cast<value_t*>(ptr.get()), reinterpret_cast<value_t*>(ptr.get()) + _size, s.to<value_t>()); */
/* 			return ptr; */
/* 		} */
/* 		case DType::Complex32:{ */
/* 			using value_t = complex_32; */
/* 			std::fill(reinterpret_cast<value_t*>(ptr.get()), reinterpret_cast<value_t*>(ptr.get()) + _size, s.to<value_t>()); */
/* 			return ptr; */
/* 		} */
/* #endif */
/* #ifdef __SIZEOF_INT128__ */
/* 		case DType::int128:{ */
/* 			using value_t = int128_t; */
/* 			std::fill(reinterpret_cast<value_t*>(ptr.get()), reinterpret_cast<value_t*>(ptr.get()) + _size, s.to<value_t>()); */
/* 			return ptr; */
/* 		} */
/* 		case DType::uint128:{ */
/* 			using value_t = uint128_t; */
/* 			std::fill(reinterpret_cast<value_t*>(ptr.get()), reinterpret_cast<value_t*>(ptr.get()) + _size, s.to<value_t>()); */
/* 			return ptr; */
/* 		} */
/* #endif */
/* 	} */
/* 	return ptr; */
/* } */


/* intrusive_ptr<void> ArrayVoid::MakeContiguousMemory(uint32_t _size, DTypeShared _type){ */
/* 	return intrusive_ptr<void>::make_shared(_size, DTypeFuncs::size_of_dtype(DTypeShared_DType(_type))); */
/* } */
/* #endif */


}
