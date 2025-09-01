#include <memory.h>
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
#include "../types/Types.h"
#include "../memory/iterator.h"
#include "../functional/cpu/sum_exp_log.h"
#include "../functional/cpu/activation_functions.h"
#include "../functional/cpu/operators.h"
#include "../functional/cpu/fill.h"
#include "../functional/cpu/compare.h"
#include "../functional/cpu/convert.h"
#include <cstdint>



namespace nt{


const size_t ArrayVoid::dtype_size(DType _type) const{
	return DTypeFuncs::size_of_dtype(_type);
}

ArrayVoid::ArrayVoid(int64_t _size, DType _t)
	:bucket(_size, _t),
	size(_size)
{}

ArrayVoid::ArrayVoid(int64_t _size, DType _t, void* ptr, void (*func)(void*))
	:bucket(_size, _t, ptr, func),
	size(_size)
{}


#ifdef USE_PARALLEL
ArrayVoid::ArrayVoid(int64_t _size, DTypeShared _t)
	:bucket(_size, DTypeShared_DType(_t), dCPUShared),
	size(_size)
{}
#endif

ArrayVoid& ArrayVoid::operator=(const ArrayVoid &Arr){
	bucket = Arr.bucket;
	size = Arr.size;
	return *this;
}

ArrayVoid& ArrayVoid::operator=(ArrayVoid&& Arr){
	bucket = std::move(Arr.bucket);
	size = Arr.size;
	Arr.size = 0;
	return *this;
}

ArrayVoid::ArrayVoid(const ArrayVoid& Arr)
	:bucket(Arr.bucket),
	size(Arr.size)
{}

ArrayVoid::ArrayVoid(ArrayVoid&& Arr)
	:bucket(std::move(Arr.bucket)),
	size(std::exchange(Arr.size, 0))
{}


ArrayVoid::ArrayVoid(std::nullptr_t)
	:bucket(nullptr),
	size(0)
{}




ArrayVoid::ArrayVoid(const Bucket& b, uint64_t size, DType dt)
	:bucket(b),
	size(size) {}

ArrayVoid::ArrayVoid(Bucket&& b, uint64_t size, DType dt)
	:bucket(std::move(b)),
	size(size) {}

ArrayVoid::ArrayVoid(const Bucket& b)
	:bucket(b),
	size(b.size()) {}

ArrayVoid::ArrayVoid(Bucket&& b)
	:bucket(std::move(b)),
	size(0)
{
	size = bucket.size(); 
}

void ArrayVoid::swap(ArrayVoid& other){
	other.bucket.swap(bucket);
	std::swap(other.size, size);
}

void* ArrayVoid::data_ptr_end() {utils::throw_exception(is_contiguous(), "Must be contiguous for data_ptr_end()");return reinterpret_cast<uint8_t*>(bucket.data_ptr()) + (size * DTypeFuncs::size_of_dtype(dtype()));}
const void* ArrayVoid::data_ptr_end() const {utils::throw_exception(is_contiguous(), "Must be contiguous for data_ptr_end()");return reinterpret_cast<const uint8_t*>(bucket.data_ptr()) + (size * DTypeFuncs::size_of_dtype(dtype()));}
/* /1* const void* ArrayVoid::data_ptr_end() const {return reinterpret_cast<uint8_t*>(_vals.get()) + (_last_index * type_size);} *1/ */
/* void** ArrayVoid::strides_cbegin() const {return _strides.get() + _start;} */
/* void** ArrayVoid::strides_begin() {return _strides.get() + _start;} */
/* void** ArrayVoid::strides_end() {return _strides.get() + _last_index;} */
/* void** ArrayVoid::strides_cend() const {return _strides.get() + _last_index;} */





ArrayVoid& ArrayVoid::operator=(Scalar val){
	if(dtype() != DType::TensorObj){
        functional::cpu::_fill_scalar_(*this, val);
        return *this;	
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
    utils::throw_exception(dtype() != DType::Bool, "Cannot get iota or arange from boolean dtype object");
	if(dtype() != DType::TensorObj){
        functional::cpu::_iota_(*this, s);
        return *this;
	}else{
		this->execute_function<WRAP_DTYPES<DTypeEnum<DType::TensorObj> > >([&s](auto begin, auto end){
				/* using value_t = IteratorBaseType_t<decltype(begin)>; */
				/* auto v = val.to<value_t>(); */
				std::for_each(begin, end, [&s](auto& val){val.arr_void().iota(s);});
				});
	}
	return *this;
}


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
	/* return ArrayVoid(_vals, make_unique_strides(index, index + ns), ns, 0, available_size, dtype()); */
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

ArrayVoid ArrayVoid::range(std::vector<range_> ranges) const{
	const size_t old_size = size;
	std::for_each(ranges.begin(), ranges.end(), [&old_size](auto& val){val.fix(old_size);});
	std::vector<std::pair<uint64_t, uint64_t> > pairs(ranges.size());
	auto pbegin = pairs.begin();
	for(const range_ &x : ranges){
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

/* template tdtype_list<float16_t> ArrayVoid::tend(); */
/* template tdtype_list<complex_32> ArrayVoid::tend(); */
/* template tdtype_list<float128_t> ArrayVoid::tend(); */
/* template tdtype_list<int128_t> ArrayVoid::tend(); */





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

/* template tdtype_list<const float16_t> ArrayVoid::tcend() const; */
/* template tdtype_list<const complex_32> ArrayVoid::tcend() const; */
/* template tdtype_list<const float128_t> ArrayVoid::tcend() const; */
/* template tdtype_list<const int128_t> ArrayVoid::tcend() const; */


ArrayVoid ArrayVoid::copy_strides(bool copy) const{
	if(!bucket.is_strided())
		return bucket_all_indices();
	if(!copy)
		return new_strides(size);
	return ArrayVoid(bucket.copy_strides(), size, dtype());
}

/* template<DType dt> */
//maybe move this to be a private function
/* ArrayVoid ArrayVoid::copy_strides(bool copy) const{ */
/* 	/1* if(dt != dtype()) *1/ */
/* 	/1* 	return copy_strides<DTypeFuncs::next_dtype_it<dt>>(); *1/ */

/* 	intrusive_ptr<void*> cpy_str = _strides.copy_strides(available_size, DTypeFuncs::size_of_dtype_p(dtype())); */
/* 	if(copy){ */
/* 		void** original = strides_cbegin(); */
/* 		void** destination = cpy_str.get(); */
/* 		for(uint64_t i = 0; i < size; ++i, ++original, ++destination) */
/* 			*destination = *original; */
/* 	} */
/* 	return ArrayVoid(cpy_str, size, 0, available_size, dtype()); */
/* } */


//this would potentially break it if I am not mistakened
/* ArrayVoid ArrayVoid::new_stride(uint32_t size) const{ */
/* 	intrusive_ptr<void*> n_str(_strides.vals_(), size, type_size, DTypeFuncs::size_of_dtype_p(dtype()), detail::DontOrderStrides{}); */
/* 	return ArrayVoid(n_str, size, 0, available_size, dtype()); */
/* } */

/* intrusive_ptr<void*> ArrayVoid::make_unique_strides(bool pre_order) const{ */
/* 	if(!pre_order){ */
/* 		return intrusive_ptr<void*>(_strides.vals_(), available_size, type_size, DTypeFuncs::size_of_dtype_p(dtype()), detail::DontOrderStrides{}); */
/* 	} */
/* 	intrusive_ptr<void*> cpy_str(_strides.vals_(), available_size, type_size, DTypeFuncs::size_of_dtype_p(dtype()), detail::DontOrderStrides{}); */
/* 	void** original = _strides.get(); */
/* 	void** destination = cpy_str.get(); */
/* 	for(uint64_t i = 0; i < available_size; ++i) */
/* 		destination[i] = original[i]; */	
/* 	return std::move(cpy_str); */
/* } */

/* intrusive_ptr<void*> ArrayVoid::make_unique_strides(std::size_t start, std::size_t end, bool copy) const{ */
/* 	std::size_t total = end - start; */
/* 	/1* std::cout<<"making unique from "<<start<<" to "<<end<<std::endl; *1/ */
/* 	intrusive_ptr<void*> cpy_str(_strides.vals_(), total, type_size, DTypeFuncs::size_of_dtype_p(dtype()), detail::DontOrderStrides{}); */
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
/* 		_strides = intrusive_ptr<void*>(_strides.vals_(), available_size, type_size, DTypeFuncs::size_of_dtype_p(dtype()), detail::DontOrderStrides{}); */
/* 		return; */
/* 	} */
/* 	intrusive_ptr<void*> cpy_str(_strides.vals_(), available_size, type_size, DTypeFuncs::size_of_dtype_p(dtype()), detail::DontOrderStrides{}); */
/* 	void** original = _strides.get(); */
/* 	void** destination = cpy_str.get(); */
/* 	for(uint64_t i = 0; i < available_size; ++i) */
/* 		destination[i] = original[i]; */
/* 	_strides = std::move(cpy_str); */
	
/* } */

/* ArrayVoid& ArrayVoid::iota(Scalar val){ */
/* 	if(dtype() != DType::TensorObj) */
/* 		this->execute_function<WRAP_DTYPES<NumberTypesL>>()([&val](auto begin, auto end){std::iota(begin, end, val.to<utils::IteratorBaseType_t<decltype(begin)> >());}); */
/* 	else{ */
		
/* 		this->execute_function<WRAP_DTYPES<DTypeEnum<DType::TensorObj> > >([&v](auto begin, auto end){std::for_eacstd::iota(begin, end, v);}); */
/* 	} */
/* 	return *this; */
/* } */

template<DType dt>
bool _my_sub_copy_(ArrayVoid& Arr, const ArrayVoid& my_arr, unsigned long long i){	
	if(my_arr.dtype() != dt) return _my_sub_copy_<DTypeFuncs::next_dtype_it<dt>>(Arr, my_arr, i);
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
	utils::throw_exception(Arr.dtype() == dtype(), "\nRuntime Error: Expected to copy ArrayVoid to same type $ but got $", dtype(), Arr.dtype());
    functional::cpu::_set_(Arr, *this);
}

ArrayVoid& ArrayVoid::fill_(Scalar c){
	if(dtype() != DType::TensorObj){
        functional::cpu::_fill_scalar_(*this, c);
	}
	else{
		this->execute_function<WRAP_DTYPES<DTypeEnum<DType::TensorObj> > >([&c](auto begin, auto end){
				/* using value_t = IteratorBaseType_t<decltype(begin)>; */
				/* auto v = val.to<value_t>(); */
				std::fill(begin, end, c);
				});
	}
	return *this;
}




ArrayVoid ArrayVoid::to(DType _dt) const{
    if(_dt == this->dtype()) return *this;
    ArrayVoid out(size, _dt);
    functional::cpu::_convert(*this, out);
    return std::move(out);
}

#define X(type, dtype_enum_a, dtype_enum_b)\
ArrayVoid ArrayVoid::dtype_enum_a() const {return to(DType::dtype_enum_a);}

NT_GET_X_FLOATING_DTYPES_ 
NT_GET_X_COMPLEX_DTYPES_
NT_GET_X_SIGNED_INTEGER_DTYPES_
NT_GET_X_UNSIGNED_INTEGER_DTYPES_
NT_GET_X_OTHER_DTYPES_
#undef X

ArrayVoid ArrayVoid::to(DeviceType dt) const{
	return ArrayVoid(this->bucket.to_device(dt), size, dtype());
}

/*
Tensor ArrayVoid::split(const uint64_t sp) const{
	Tensor buckets = bucket.split<Tensor>(sp);
	Tensor* begin = reinterpret_cast<Tensor*>(buckets.data_ptr());
	Tensor* end = begin + buckets.numel();
	typedef typename SizeRef::ArrayRefInt::value_type m_size_t;
	for(;begin != end; ++begin){
		m_size_t size = begin->_vals.bucket.size();
		begin->_size = SizeRef({size});
		begin->dtype() = dtype;

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
		begin->dtype() = dtype;
	}
	return std::move(buckets);
}

*/

ArrayVoid& ArrayVoid::operator*=(Scalar c){
	utils::throw_exception(dtype() != DType::Bool, "*= operation is invalid for DType::Bool");
	if(dtype() != DType::TensorObj){
        functional::cpu::_operator_mdsa_scalar_(*this, c, 0);
    }
	else{
		this->for_each<DType::TensorObj>([&c](auto& inp){inp *= c;});
	}
	return *this;
}


ArrayVoid& ArrayVoid::operator/=(Scalar c){
	utils::throw_exception(dtype() != DType::Bool, "/= operation is invalid for DType::Bool");
    //if floating or complex, a multiplication (times the inverse) is the same thing and faster
    if(DTypeFuncs::is_complex(dtype()) || DTypeFuncs::is_floating(dtype()))
        return *this *= c.inverse();
	if(dtype() != DType::TensorObj){
        functional::cpu::_operator_mdsa_scalar_(*this, c, 1);
    }
	else{
		this->for_each<DType::TensorObj>([&c](auto& inp){inp /= c;});
	}
    return *this;

}

ArrayVoid& ArrayVoid::operator-=(Scalar c){
	utils::throw_exception(dtype() != DType::Bool, "-= operation is invalid for DType::Bool");
	if(dtype() != DType::TensorObj){
        functional::cpu::_operator_mdsa_scalar_(*this, c, 2);
    }
	else{
		this->for_each<DType::TensorObj>([&c](auto& inp){inp -= c;});
	}
	return *this;
}

ArrayVoid& ArrayVoid::operator+=(Scalar c){
	utils::throw_exception(dtype() != DType::Bool, "+= operation is invalid for DType::Bool");
	if(dtype() != DType::TensorObj){
        functional::cpu::_operator_mdsa_scalar_(*this, c, 3);
    }
	else{
		this->for_each<DType::TensorObj>([&c](auto& inp){inp += c;});
	}
	return *this;
}



ArrayVoid ArrayVoid::operator*(Scalar c) const{
	return this->clone() *= c;
}


ArrayVoid ArrayVoid::operator/(Scalar c) const{
    return this->clone() /= c;
}

ArrayVoid ArrayVoid::operator-(Scalar c) const{
	return this->clone() -= c;
}

ArrayVoid ArrayVoid::operator+(Scalar c) const{
	return this->clone() += c;
}

//make a double chunk operator
//maybe for chunks it would be most efficient to break all the chunks into seperate tensors
//and then multiply them individually

ArrayVoid ArrayVoid::operator*(const ArrayVoid& A) const{
	utils::throw_exception(size == A.size, "For operators, sizes must be equal, expected size of $ but got $", size, A.size);
	utils::throw_exception((dtype() == DType::TensorObj && A.dtype() != DType::TensorObj) 
			|| (dtype() == A.dtype()), "$ can not have operator * with $", dtype(), A.dtype());
	ArrayVoid output(size, dtype()); // this is going to just be contiguous
	if(dtype() == DType::TensorObj && A.dtype() != DType::TensorObj){
		auto out_begin = output.bucket.begin_contiguous<Tensor>();
		A.transform_function<WRAP_DTYPES<NumberTypesL>>([](const auto& t, const auto& o){return t * o;}, *this, out_begin);
		return std::move(output);
	}
    functional::cpu::_operator_mdsa(*this, A, output, 0);
	return std::move(output);
}

ArrayVoid ArrayVoid::operator/(const ArrayVoid& A) const{
	utils::throw_exception(size == A.size, "For operators, sizes must be equal, expected size of $ but got $", size, A.size);
	utils::throw_exception((dtype() == DType::TensorObj && A.dtype() != DType::TensorObj) 
			|| (dtype() == A.dtype()), "$ can not have operator * with $", dtype(), A.dtype());
	ArrayVoid output(size, dtype());
	if(dtype() == DType::TensorObj && A.dtype() != DType::TensorObj){
		auto out_begin = output.bucket.begin_contiguous<Tensor>();
		A.transform_function<WRAP_DTYPES<NumberTypesL>>([](const auto& t, const auto& o){return t / o;},*this, out_begin);
		return std::move(output);
	}
    functional::cpu::_operator_mdsa(*this, A, output, 1);
	return std::move(output);
}

ArrayVoid ArrayVoid::operator-(const ArrayVoid& A) const{
	utils::throw_exception(size == A.size, "For operators, sizes must be equal, expected size of $ but got $", size, A.size);
	utils::throw_exception((dtype() == DType::TensorObj && A.dtype() != DType::TensorObj) 
			|| (dtype() == A.dtype()), "$ can not have operator * with $", dtype(), A.dtype());
	ArrayVoid output(size, dtype());
	if(dtype() == DType::TensorObj && A.dtype() != DType::TensorObj){
		auto out_begin = output.bucket.begin_contiguous<Tensor>();
		A.transform_function<WRAP_DTYPES<NumberTypesL>>([](const auto& t, const auto& o){return t - o;},*this, out_begin);
		return std::move(output);
	}
    functional::cpu::_operator_mdsa(*this, A, output, 2);
	return std::move(output);
}

ArrayVoid ArrayVoid::operator+(const ArrayVoid& A) const{
	utils::throw_exception(size == A.size, "For operators, sizes must be equal, expected size of $ but got $", size, A.size);
	utils::throw_exception((dtype() == DType::TensorObj && A.dtype() != DType::TensorObj) 
			|| (dtype() == A.dtype()), "$ can not have operator * with $", dtype(), A.dtype());
	ArrayVoid output(size, dtype());
	if(dtype() == DType::TensorObj && A.dtype() != DType::TensorObj){
		auto out_begin = output.bucket.begin_contiguous<Tensor>();
		A.transform_function<WRAP_DTYPES<NumberTypesL>>([](const auto& t, const auto& o){return t + o;},*this, out_begin);
		return std::move(output);
	}
    functional::cpu::_operator_mdsa(*this, A, output, 3);
	return std::move(output);
}

ArrayVoid& ArrayVoid::operator*=(const ArrayVoid& A){
	if(dtype() == DType::TensorObj && A.dtype() != DType::TensorObj){
		this->execute_function<WRAP_DTYPES<DTypeEnum<DType::TensorObj> > >([&A](auto begin, auto end){
			for(;begin != end; ++begin){
				begin->arr_void() *= A;
			}
		});
	}
	if(dtype() != A.dtype()){
		return *this *= A.to(dtype());
	}
	utils::throw_exception(size == A.size, "For operators, sizes must be equal, expected size of $ but got $", size, A.size);
    functional::cpu::_operator_mdsa_(*this, A, 0);
	return *this;

}

ArrayVoid& ArrayVoid::operator+=(const ArrayVoid& A){
	if(dtype() == DType::TensorObj && A.dtype() != DType::TensorObj){
		this->execute_function<WRAP_DTYPES<DTypeEnum<DType::TensorObj> > >([&A](auto begin, auto end){
			for(;begin != end; ++begin){
				begin->arr_void() += A;
			}
		});
	}
	if(dtype() != A.dtype()){
		return *this += A.to(dtype());
	}
	utils::throw_exception(size == A.size, "For operators, sizes must be equal, expected size of $ but got $", size, A.size);
    functional::cpu::_operator_mdsa_(*this, A, 3);
	return *this;

}


ArrayVoid& ArrayVoid::operator-=(const ArrayVoid& A){
	if(dtype() == DType::TensorObj && A.dtype() != DType::TensorObj){
		this->execute_function<WRAP_DTYPES<DTypeEnum<DType::TensorObj> > >([&A](auto begin, auto end){
			for(;begin != end; ++begin){
				begin->arr_void() -= A;
			}
		});
	}
	if(dtype() != A.dtype()){
		return *this -= A.to(dtype());
	}
	utils::throw_exception(size == A.size, "For operators, sizes must be equal, expected size of $ but got $", size, A.size);
    functional::cpu::_operator_mdsa_(*this, A, 2);
	return *this;
}


ArrayVoid& ArrayVoid::operator/=(const ArrayVoid& A){
	if(dtype() == DType::TensorObj && A.dtype() != DType::TensorObj){
		this->execute_function<WRAP_DTYPES<DTypeEnum<DType::TensorObj> > >([&A](auto begin, auto end){
			for(;begin != end; ++begin){
				begin->arr_void() /= A;
			}
		});
	}
	if(dtype() != A.dtype()){
		return *this /= A.to(dtype());
	}
	utils::throw_exception(size == A.size, "For operators, sizes must be equal, expected size of $ but got $", size, A.size);
    functional::cpu::_operator_mdsa_(*this, A, 1);
	return *this;
}


ArrayVoid ArrayVoid::operator*(const Tensor& A) const{
	utils::throw_exception(dtype() == DType::TensorObj, "\nRuntime Error: expected DType for * of TensorObj to be TensorObj but got $", dtype());
	ArrayVoid output(size, dtype());
	Tensor* OutputIt = reinterpret_cast<Tensor*>(output.data_ptr());
	this->transform_function<DType::TensorObj>(std::bind(std::multiplies<Tensor>(), A, std::placeholders::_1), OutputIt);
	return std::move(output);
}

ArrayVoid ArrayVoid::operator/(const Tensor& A) const{
	utils::throw_exception(dtype() == DType::TensorObj, "\nRuntime Error: expected DType for / of TensorObj to be TensorObj but got $", dtype());
	ArrayVoid output(size, dtype());
	Tensor* OutputIt = reinterpret_cast<Tensor*>(output.data_ptr());
	this->transform_function<DType::TensorObj>(std::bind(std::divides<Tensor>(), A, std::placeholders::_1), OutputIt);
	return std::move(output);
}

ArrayVoid ArrayVoid::operator-(const Tensor& A) const{
	utils::throw_exception(dtype() == DType::TensorObj, "\nRuntime Error: expected DType for - of TensorObj to be TensorObj but got $", dtype());
	ArrayVoid output(size, dtype());
	Tensor* OutputIt = reinterpret_cast<Tensor*>(output.data_ptr());
	this->transform_function<DType::TensorObj>(std::bind(std::minus<Tensor>(), A, std::placeholders::_1), OutputIt);
	return std::move(output);

}

ArrayVoid ArrayVoid::operator+(const Tensor& A) const{
	utils::throw_exception(dtype() == DType::TensorObj, "\nRuntime Error: expected DType for + of TensorObj to be TensorObj but got $", dtype());
	ArrayVoid output(size, dtype());
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
/* 	if(dt != mine.dtype()) return _my_sub_equal_operator_<DTypeFuncs::next_dtype_it<dt>>(mine, output, c); */
/* 	using value_t = DTypeFuncs::dtype_to_type_t<dt>; */
/* 	value_t s = c.to<value_t>(); */
/* 	mine.transform_function<dt>([&s](const auto& val){return uint_bool_t(val == s);}, output); */
/* 	return true; */
/* } */

/* template<DType dt, std::enable_if_t<dt == DType::TensorObj, bool> = true> */
/* bool _my_sub_equal_operator_(const ArrayVoid& mine, uint_bool_t* output, const Scalar& c){ */
/* 	if(dt != mine.dtype()) return _my_sub_equal_operator_<DTypeFuncs::next_dtype_it<dt>>(mine, output, c); */
/* 	return false; */	
/* } */


	
ArrayVoid ArrayVoid::operator==(Scalar c) const{
	if(dtype() == DType::TensorObj){
		ArrayVoid output(size, DType::TensorObj);
		Tensor* OutputIt = reinterpret_cast<Tensor*>(output.data_ptr());
		this->transform_function<DType::TensorObj>([&c](const auto& val){return val == c;}, OutputIt);
		return std::move(output);
	}
	ArrayVoid output(size, DType::Bool);
	if(DTypeFuncs::is_unsigned(dtype()) && c.to<int64_t>() < 0){
		output = uint_bool_t(false);
		return std::move(output);
	}
	if(c.isComplex() && (!DTypeFuncs::is_complex(dtype()))){
		output = uint_bool_t(false);
		return std::move(output);
	}
    functional::cpu::_equal(output, *this, c);
	return std::move(output);
}

ArrayVoid ArrayVoid::operator!=(Scalar c) const{
	if(dtype() == DType::TensorObj){
		ArrayVoid output(size, DType::TensorObj);
		Tensor* OutputIt = reinterpret_cast<Tensor*>(output.data_ptr());
		this->transform_function<DType::TensorObj>([&c](const auto& val){return val != c;}, OutputIt);
		return std::move(output);
	}
	const size_t n_size = size;
	ArrayVoid output(n_size, DType::Bool);
	if(DTypeFuncs::is_unsigned(dtype()) && c.to<int64_t>() < 0){
		output = uint_bool_t(true);
		return std::move(output);
	}
	if(c.isComplex() && (!DTypeFuncs::is_complex(dtype()))){
		output = uint_bool_t(true);
		return std::move(output);
	}
    functional::cpu::_not_equal(output, *this, c);
	return std::move(output);
}

/* template<DType dt, std::enable_if_t<dt == DType::TensorObj, bool> = true> */
/* bool _my_sub_equal_operator_(const ArrayVoid& mine, uint_bool_t* output, const Scalar& c){ */
/* 	if(dt != mine.dtype()) return _my_sub_equal_operator_<DTypeFuncs::next_dtype_it<dt>>(mine, output, c); */
/* 	return false; */	
/* } */

ArrayVoid ArrayVoid::operator>=(Scalar c) const{
	if(dtype() == DType::TensorObj){
		ArrayVoid output(size, DType::TensorObj);
		Tensor* OutputIt = reinterpret_cast<Tensor*>(output.data_ptr());
		this->transform_function<DType::TensorObj>([&c](const auto& val){return val >= c;}, OutputIt);
		return std::move(output);
	}
	ArrayVoid output(size, DType::Bool);
	if(DTypeFuncs::is_unsigned(dtype()) && c.to<int64_t>() < 0){
		output = uint_bool_t(false);
		return std::move(output);
	}
	if(c.isComplex() && (!DTypeFuncs::is_complex(dtype()))){
		output = uint_bool_t(false);
		return std::move(output);
	}
    functional::cpu::_greater_than_equal(output, *this, c);
	return std::move(output);
}

ArrayVoid ArrayVoid::operator<=(Scalar c) const{
	if(dtype() == DType::TensorObj){
		ArrayVoid output(size, DType::TensorObj);
		Tensor* OutputIt = reinterpret_cast<Tensor*>(output.data_ptr());
		this->transform_function<DType::TensorObj>([&c](const auto& val){return val <= c;}, OutputIt);
		return std::move(output);
	}
	ArrayVoid output(size, DType::Bool);
	if(DTypeFuncs::is_unsigned(dtype()) && c.to<int64_t>() < 0){
		output = uint_bool_t(false);
		return std::move(output);
	}
	if(c.isComplex() && (!DTypeFuncs::is_complex(dtype()))){
		output = uint_bool_t(false);
		return std::move(output);
	}
    functional::cpu::_less_than_equal(output, *this, c);
	return std::move(output);
}

ArrayVoid ArrayVoid::operator>(Scalar c) const{
	if(dtype() == DType::TensorObj){
		ArrayVoid output(size, DType::TensorObj);
		Tensor* OutputIt = reinterpret_cast<Tensor*>(output.data_ptr());
		this->transform_function<DType::TensorObj>([&c](const auto& val){return val > c;}, OutputIt);
		return std::move(output);
	}
	ArrayVoid output(size, DType::Bool);
	if(DTypeFuncs::is_unsigned(dtype()) && c.to<int64_t>() < 0){
		output = uint_bool_t(false);
		return std::move(output);
	}
	if(c.isComplex() && (!DTypeFuncs::is_complex(dtype()))){
		output = uint_bool_t(false);
		return std::move(output);
	}
    functional::cpu::_greater_than(output, *this, c);
	return std::move(output);
}

ArrayVoid ArrayVoid::operator<(Scalar c) const{
	if(dtype() == DType::TensorObj){
		ArrayVoid output(size, DType::TensorObj);
		Tensor* OutputIt = reinterpret_cast<Tensor*>(output.data_ptr());
		this->transform_function<DType::TensorObj>([&c](const auto& val){return val < c;}, OutputIt);
		return std::move(output);
	}
	ArrayVoid output(size, DType::Bool);
	if(DTypeFuncs::is_unsigned(dtype()) && c.to<int64_t>() < 0){
		output = uint_bool_t(false);
		return std::move(output);
	}
	if(c.isComplex() && (!DTypeFuncs::is_complex(dtype()))){
		output = uint_bool_t(false);
		return std::move(output);
	}
    functional::cpu::_less_than(output, *this, c);
	return std::move(output);
}


/* template<DType dt, std::enable_if_t<dt != DType::Bool && dt != DType::TensorObj && !DTypeFuncs::is_dtype_complex_v<dt>, bool> = true> */
/* bool _my_sub_inverse_(const ArrayVoid& my_arr, ArrayVoid& outp){ */
/* 	if(dt != my_arr.dtype()) return _my_sub_inverse_<DTypeFuncs::next_dtype_it<dt>>(my_arr, outp); */
/* 	using my_value_t = ::nt::DTypeFuncs::dtype_to_type_t<dt>; */
/* 	using out_value_t = ::nt::DTypeFuncs::dtype_to_type_t<dt>; */
/* 	uint32_t type_a = outp.get_bucket().iterator_type(); */
/* 	uint32_t type_b = my_arr.get_bucket().iterator_type(); */
/* 	utils::throw_exception(type_a == 1, "Expected in sub inverse for output to be contiguous, but got iterator type $, problem with creation", type_a); */
/* 	if(type_b == 1){ */
/* 		auto begin = my_arr.get_bucket().cbegin_contiguous<my_value_t>(); */
/* 		auto end = my_arr.get_bucket().cend_contiguous<my_value_t>(); */
/* 		out_value_t* o_begin = reinterpret_cast<out_value_t*>(outp.data_ptr()); */
/* 		std::transform(begin, end, o_begin, [](const my_value_t& val){return 1.0/val;}); */
/* 	} */
/* 	else if(type_b == 2){ */
/* 		auto begin = my_arr.get_bucket().cbegin_blocked<my_value_t>(); */
/* 		auto end = my_arr.get_bucket().cend_blocked<my_value_t>(); */
/* 		out_value_t* o_begin = reinterpret_cast<out_value_t*>(outp.data_ptr()); */
/* 		std::transform(begin, end, o_begin, [](const my_value_t& val){return 1.0/val;}); */
/* 	} */
/* 	else if(type_b == 3){ */
/* 		auto begin = my_arr.get_bucket().cbegin_list<my_value_t>(); */
/* 		auto end = my_arr.get_bucket().cend_list<my_value_t>(); */
/* 		out_value_t* o_begin = reinterpret_cast<out_value_t*>(outp.data_ptr()); */
/* 		std::transform(begin, end, o_begin, [](const my_value_t& val){return 1.0/val;}); */
/* 	} */

/* 	return true; */
/* } */

/* template<DType dt, std::enable_if_t<dt != DType::Bool && dt != DType::TensorObj && DTypeFuncs::is_dtype_complex_v<dt>, bool> = true> */
/* bool _my_sub_inverse_(const ArrayVoid& my_arr, ArrayVoid& outp){ */
/* 	if(dt != my_arr.dtype()) return _my_sub_inverse_<DTypeFuncs::next_dtype_it<dt>>(my_arr, outp); */
/* 	using my_value_t = ::nt::DTypeFuncs::dtype_to_type_t<dt>; */
/* 	using out_value_t = ::nt::DTypeFuncs::dtype_to_type_t<dt>; */
/* 	uint32_t type_a = outp.get_bucket().iterator_type(); */
/* 	uint32_t type_b = my_arr.get_bucket().iterator_type(); */
/* 	utils::throw_exception(type_a == 1, "Expected in sub inverse for output to be contiguous, but got iterator type $, problem with creation", type_a); */
/* 	if(type_b == 1){ */
/* 		auto begin = my_arr.get_bucket().cbegin_contiguous<my_value_t>(); */
/* 		auto end = my_arr.get_bucket().cend_contiguous<my_value_t>(); */
/* 		out_value_t* o_begin = reinterpret_cast<out_value_t*>(outp.data_ptr()); */
/* 		std::transform(begin, end, o_begin, [](const my_value_t& val){return out_value_t(1.0/val.real(), 1.0/val.imag());}); */
/* 	} */
/* 	else if(type_b == 2){ */
/* 		auto begin = my_arr.get_bucket().cbegin_blocked<my_value_t>(); */
/* 		auto end = my_arr.get_bucket().cend_blocked<my_value_t>(); */
/* 		out_value_t* o_begin = reinterpret_cast<out_value_t*>(outp.data_ptr()); */
/* 		std::transform(begin, end, o_begin, [](const my_value_t& val){return out_value_t(1.0/val.real(), 1.0/val.imag());}); */
/* 	} */
/* 	else if(type_b == 3){ */
/* 		auto begin = my_arr.get_bucket().cbegin_list<my_value_t>(); */
/* 		auto end = my_arr.get_bucket().cend_list<my_value_t>(); */
/* 		out_value_t* o_begin = reinterpret_cast<out_value_t*>(outp.data_ptr()); */
/* 		std::transform(begin, end, o_begin, [](const my_value_t& val){return out_value_t(1.0/val.real(), 1.0/val.imag());}); */
/* 	} */

/* 	return true; */
/* } */

/* template<DType dt, std::enable_if_t<dt == DType::Bool, bool> = true> */
/* bool _my_sub_inverse_(const ArrayVoid& my_arr, ArrayVoid& outp){ */
/* 	if(dt != my_arr.dtype()) return _my_sub_inverse_<DTypeFuncs::next_dtype_it<dt>>(my_arr, outp); */
/* 	return false; */
/* } */

/* template<DType dt, std::enable_if_t<dt == DType::TensorObj, bool> = true> */
/* bool _my_sub_inverse_(const ArrayVoid& my_arr, ArrayVoid& outp){ */
/* 	if(dt != my_arr.dtype()) return _my_sub_inverse_<DTypeFuncs::next_dtype_it<dt>>(my_arr, outp); */
/* 	using my_value_t = ::nt::DTypeFuncs::dtype_to_type_t<dt>; */
/* 	using out_value_t = ::nt::DTypeFuncs::dtype_to_type_t<dt>; */
/* 	uint32_t type_a = outp.get_bucket().iterator_type(); */
/* 	uint32_t type_b = my_arr.get_bucket().iterator_type(); */
/* 	utils::throw_exception(type_a == 1, "Expected in sub inverse for output to be contiguous, but got iterator type $, problem with creation", type_a); */
/* 	if(type_b == 1){ */
/* 		auto begin = my_arr.get_bucket().cbegin_contiguous<my_value_t>(); */
/* 		auto end = my_arr.get_bucket().cend_contiguous<my_value_t>(); */
/* 		out_value_t* o_begin = reinterpret_cast<out_value_t*>(outp.data_ptr()); */
/* 		std::transform(begin, end, o_begin, [](const my_value_t& val){return val.inverse();}); */
/* 	} */
/* 	else if(type_b == 2){ */
/* 		auto begin = my_arr.get_bucket().cbegin_blocked<my_value_t>(); */
/* 		auto end = my_arr.get_bucket().cend_blocked<my_value_t>(); */
/* 		out_value_t* o_begin = reinterpret_cast<out_value_t*>(outp.data_ptr()); */
/* 		std::transform(begin, end, o_begin, [](const my_value_t& val){return val.inverse();}); */
/* 	} */
/* 	else if(type_b == 3){ */
/* 		auto begin = my_arr.get_bucket().cbegin_list<my_value_t>(); */
/* 		auto end = my_arr.get_bucket().cend_list<my_value_t>(); */
/* 		out_value_t* o_begin = reinterpret_cast<out_value_t*>(outp.data_ptr()); */
/* 		std::transform(begin, end, o_begin, [](const my_value_t& val){return val.inverse();}); */
/* 	} */
/* 	return true; */
/* } */

ArrayVoid ArrayVoid::inverse() const{
    if(dtype() == DType::TensorObj){
        ArrayVoid output(size, DType::TensorObj);
		this->transform_function<DType::TensorObj>([](const Tensor& inp) -> Tensor {return inp.inverse();}, reinterpret_cast<Tensor*>(output.data_ptr()));
		return std::move(output);
    }
    return functional::cpu::_inverse(*this);
}

ArrayVoid ArrayVoid::pow(Scalar p) const {
	return this->clone().pow_(p);
}


ArrayVoid& ArrayVoid::pow_(Scalar p) {
	if(dtype() == DType::TensorObj){
		this->execute_function<WRAP_DTYPES<DTypeEnum<DType::TensorObj> > >([&p](auto a_begin, auto a_end){
			for(;a_begin != a_end; ++a_begin)
				a_begin->pow_(p);
		});
        return *this;
		
	}
    functional::cpu::_pow_(*this, p);
    return *this;
}

/* template<DType dt, std::enable_if_t<DTypeFuncs::is_dtype_floating_v<dt>, bool> = true> */
/* bool _my_sub_inverse_(ArrayVoid& mine){ */
/* 	if(dt != mine.dtype()) return _my_sub_inverse_<DTypeFuncs::next_dtype_it<dt>>(mine); */
/* 	mine.for_each<dt>([](auto& val){val = 1.0/val;}); */
/* 	return true; */	
/* } */

/* template<DType dt, std::enable_if_t<DTypeFuncs::is_dtype_complex_v<dt> || dt == DType::TensorObj, bool> = true> */
/* bool _my_sub_inverse_(ArrayVoid& mine){ */
/* 	if(dt != mine.dtype()) return _my_sub_inverse_<DTypeFuncs::next_dtype_it<dt>>(mine); */
/* 	using complex_value_t = DTypeFuncs::dtype_to_type_t<dt>; */
/* 	mine.for_each<dt>([](auto& val){val.inverse_();}); */
/* 	return true; */	
/* } */


/* template<DType dt, std::enable_if_t<!DTypeFuncs::is_dtype_floating_v<dt> && !DTypeFuncs::is_dtype_complex_v<dt> && dt != DType::TensorObj, bool> = true> */
/* bool _my_sub_inverse_(ArrayVoid& mine){ */
/* 	if(dt != mine.dtype()) return _my_sub_inverse_<DTypeFuncs::next_dtype_it<dt>>(mine); */
/* 	utils::throw_exception(false, "to inverse() scalars must be complex or floating not $", dt); */
/* 	return false; */	
/* } */

ArrayVoid& ArrayVoid::inverse_(){
	if(dtype() == DType::TensorObj){
		this->execute_function<WRAP_DTYPES<DTypeEnum<DType::TensorObj> > >([](auto begin, auto end){ std::for_each(begin, end, [](auto& tensor){tensor.inverse_();});});
	    return *this;
    }
    functional::cpu::_inverse_(*this);
	return *this;
}



ArrayVoid ArrayVoid::exp() const{
	return this->clone().exp_();
	/* ArrayVoid output(size, dtype()); */
	/* if(DTypeFuncs::is_complex(dtype()) || DTypeFuncs::is_floating(dtype()) || DTypeFuncs::is_integer(dtype())){ */
	/* 	this->cexecute_function_chunk<WRAP_DTYPES<FloatingTypesL, ComplexTypesL, IntegerTypesL> >([](auto begin, auto end, void* begin_ptr){ */
	/* 		using value_t = utils::IteratorBaseType_t<decltype(begin)>; */
	/* 		mp::exp(begin, end, reinterpret_cast<value_t*>(begin)); */
	/* 	}, output.data_ptr()) */
	/* } */
	/* else if(dtype() == DType::TensorObj){ */

	/* 	this->cexecute_function<WRAP_DTYPES<DTypeEnum<DType::TensorObj> > >([](auto begin, auto end, void* begin_ptr){ */
	/* 		Tensor* t_ptr = reinterpret_cast<Tensor*>(begin_ptr); */
	/* 		for(;begin != end; ++begin, ++t_ptr){ */
	/* 			*t_ptr = begin->exp(); */
	/* 		} */
	/* 	}, output.data_ptr()); */
	/* } */
	/* return std::move(output); */
}

ArrayVoid& ArrayVoid::exp_(){
	if(DTypeFuncs::is_complex(dtype()) || DTypeFuncs::is_floating(dtype()) || DTypeFuncs::is_integer(dtype())){
        functional::cpu::_exp_(*this);
        return *this;
	}
	else if(dtype() == DType::TensorObj){
		this->execute_function<WRAP_DTYPES<DTypeEnum<DType::TensorObj> > >([](auto begin, auto end){
			for(;begin != end; ++begin){
				begin->exp_();
			}
		});
	}
	return *this;
}


ArrayVoid& ArrayVoid::complex_(){
    functional::cpu::_complex_(*this);
	return *this;
}

ArrayVoid& ArrayVoid::floating_(){
    functional::cpu::_floating_(*this);
	return *this;
}

ArrayVoid& ArrayVoid::integer_(){
    functional::cpu::_integer_(*this);
	return *this;
}
ArrayVoid& ArrayVoid::unsigned_(){
    functional::cpu::_unsigned_(*this);
	return *this;
}


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


}
