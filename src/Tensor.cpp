#include "Tensor.h"
#include "dtype/DType_list.h"
#include "dtype/ranges.h"
#include "functional/functional.h"
#include "refs/SizeRef.h"
#include "dtype/ArrayVoid.h"
#include "dtype/DType.h"
#include "dtype/DType_enum.h"




#include <functional>
#include <i386/types.h>
#include <memory.h>
#include <algorithm>
#include <numeric>
#include <ratio>
#include <sys/_types/_int32_t.h>
#include <sys/_types/_int8_t.h>
#include <cassert>
#include <format>
#include <vector>
#include "types/Types.h"
#include "utils/utils.h"
#include <type_traits>
#include "permute/permute.h"
#include "dtype/Scalar.h"
#include <cmath>
#include "dtype/ArrayVoid.hpp"
#include <sys/types.h>

#ifdef USE_PARALLEL
	#include <unistd.h>
	#include <tbb/parallel_for_each.h>
	#include <tbb/parallel_for.h>
#endif

#define assertm(exp, msg) assert(((void)msg, exp))

namespace nt{
Tensor::Tensor(DType dt)
	:_vals(1, dt), _total_size(0), _size({1}), dtype(dt), sub_tensor(false)
	{}

Tensor::Tensor(SizeRef v, DType dt)
	:_vals(v.multiply(), dt), _size(std::move(v)), dtype(dt), sub_tensor(false)
	{
		_total_size = _vals.Size();
	}

Tensor::Tensor(ArrayVoid ptr, SizeRef v)
	:_vals(ptr), _size(std::move(v)), sub_tensor(true), dtype(ptr.dtype), _total_size(ptr.Size())
	{
		dtype = _vals.dtype;
	}
Tensor::Tensor(std::string_view _sv)
	:_vals(_sv.size(), DType::uint8), _size({static_cast<SizeRef::value_type>(_sv.size())}), sub_tensor(false), dtype(DType::uint8), _total_size(_sv.size())
{
	char* begin = reinterpret_cast<char*>(data_ptr());
	std::transform(_sv.cbegin(), _sv.cend(), begin, [](const char& v){return v;}); 
}

/* Tensor::Tensor(ArrayVoid ptr, std::shared_ptr<SizeRef> v) */
/* 	:_vals(ptr), _size(v), sub_tensor(true), dtype(ptr.dtype) */
/* { */
/* 	_total_size = _vals.Size(); */
/* } */

Tensor::Tensor(const Tensor& t)
	:_vals(t._vals), _total_size(t._total_size), _size(t._size), sub_tensor(false), dtype(t.dtype)
{}

Tensor::Tensor(Tensor&& t)
	:_vals(std::move(t._vals)), _total_size(t._total_size), _size(std::move(t._size)), sub_tensor(false), dtype(t.dtype)
{}

/* template<std::size_t N> */
/* Tensor::Tensor(typename utils::NestedInitializerLists_type<Scalar, N>::type v, DType dt) */
/* 	:_vals(SizeRef(utils::aquire_shape<Scalar, N>(v)).multiply(), dt), */
/* 	_size(std::make_unique<SizeRef>(utils::aquire_shape<Scalar, N>(v))), */
/* 	sub_tensor(false), */
/* 	dtype(dt) */
/* { */
/* 	_total_size = _vals.Size(); */
/* 	_vals.execute_function([&v](auto& begin, auto& end){ */
/* 				using value_type = typename std::remove_const<typename decltype(begin)::value_type>::type; */
/* 				utils::flatten_func<N, Scalar>(v, [&begin](const Scalar& a){*begin = a.to<value_type>();}); */
/* 			}); */
/* } */

/* template<> */
/* Tensor::Tensor(typename utils::NestedInitializerLists_type<Scalar, 1>::type v, DType dt) */
/* 	:_vals(SizeRef(utils::aquire_shape<Scalar, 1>(v)).multiply(), dt), */
/* 	_size(std::make_unique<SizeRef>(utils::aquire_shape<Scalar, 1>(v))), */
/* 	sub_tensor(false), */
/* 	dtype(dt) */
/* { */
/* 	_total_size = _vals.Size(); */
/* 	_vals.execute_function<WRAP_DTYPES<NumberTypesL, DTypeEnum<DType::Bool>>>([&v](auto begin, auto end){ */
/* 				using value_type = typename std::remove_const<typename decltype(begin)::value_type>::type; */
/* 				utils::flatten_func<Scalar, 1>(v, [&begin](const Scalar& a){*begin = a.to<value_type>();}); */
/* 			}); */
/* } */


/* template Tensor<1>::Tensor(typename utils::NestedInitializerLists_type<Scalar, 1>::type, DType); */
/* Tensor::Tensor<2>(typename utils::NestedInitializerLists_type<Scalar, 2>::type, DType); */
/* Tensor::Tensor<3>(typename utils::NestedInitializerLists_type<Scalar, 3>::type, DType); */
/* Tensor::Tensor<4>(typename utils::NestedInitializerLists_type<Scalar, 4>::type, DType); */
/* Tensor::Tensor<5>(typename utils::NestedInitializerLists_type<Scalar, 5>::type, DType); */
/* Tensor::Tensor<6>(typename utils::NestedInitializerLists_type<Scalar, 6>::type, DType); */
/* Tensor::Tensor<7>(typename utils::NestedInitializerLists_type<Scalar, 7>::type, DType); */
/* Tensor::Tensor<8>(typename utils::NestedInitializerLists_type<Scalar, 8>::type, DType); */
/* Tensor::Tensor<9>(typename utils::NestedInitializerLists_type<Scalar, 9>::type, DType); */
/* Tensor::Tensor<10>(typename utils::NestedInitializerLists_type<Scalar, 10>::type, DType); */
/* Tensor::Tensor<11>(typename utils::NestedInitializerLists_type<Scalar, 11>::type, DType); */


Tensor& Tensor::operator=(const Tensor &t){
	if(shape() == t.shape() && dtype == t.dtype){
		_vals.transform_function([](auto& a, auto& b){return b;}, t._vals);
		return *this;
	}
	if(dtype == DType::TensorObj && _total_size == 1){
		this->item<Tensor>() = t;
	}
	_vals = t._vals;
	_size = t._size;
	_total_size = t._total_size;
	return *this;
}

Tensor& Tensor::set_(const Tensor& t){
	utils::throw_exception(t.dtype == dtype, "Expected dtype of $ but got $", dtype, t.dtype);
	utils::throw_exception(t.shape() == shape(), "Expected shape to be $ but got shape $", shape(), t.shape());
	_vals.transform_function([](auto& a, auto& b){return b;}, t._vals);
	return *this;
}

Tensor& Tensor::operator=(Tensor&& t){
	if(dtype == DType::TensorObj && sub_tensor && _total_size == 1){
		this->item<Tensor>() = std::move(t);
		return *this;
	}
	_vals = std::move(t._vals);
	_size = std::move(t._size);
	dtype = t.dtype;
	_total_size = t._total_size;
	return *this;
}

Tensor& Tensor::operator++(){return _add(1);}

Tensor& Tensor::operator=(Scalar val){
	_vals = (val);
	return *this;
}

Tensor& Tensor::operator+=(Scalar val){
	_vals += val;
	return *this;
}

Tensor& Tensor::operator+=(const Tensor& val){
	return functional::add_(*this, val);
}

Tensor& Tensor::operator-=(Scalar val){
	_vals -= val;
	return *this;
}

Tensor& Tensor::operator-=(const Tensor &val){
	return functional::subtract_(*this, val);
}


Tensor Tensor::operator+(Scalar val) const{
	Tensor outp(_vals + val, shape());
	return std::move(outp);
}

Tensor Tensor::operator+(const Tensor& val) const{
	return functional::add(*this, val);	
}

//s/;/{\r\tTensor outp = this->contiguous();\r\toutp._multiply(val);\r\treturn std::move(outp);\r}


Tensor Tensor::operator*(Scalar val) const{
	Tensor output(_vals * val, shape());
	return std::move(output);
}

Tensor Tensor::operator*(const Tensor& a) const{
	return functional::hadamard_multiply(*this, a);
}

//s/;/{\r\t_multiply(val);\r\treturn *this;\r}
Tensor& Tensor::operator*=(Scalar val){
	_vals *= val;
	return *this;
}

Tensor& Tensor::operator*=(const Tensor& a){
	functional::hadamard_multiply_this(*this, a);
	return *this;
}

//s/;/{\r\t_subtract(val);\r\treturn *this;\r}
//s/;/{\r\tTensor outp = this->contiguous();\r\toutp._subtract(val);\r\treturn std::move(outp);\r}
Tensor Tensor::operator-(Scalar val) const{
	return Tensor(_vals - val, shape());
}

Tensor Tensor::operator-(const Tensor &val) const{
	return functional::subtract(*this, val);
}

Tensor Tensor::operator/(Scalar val) const{
	return Tensor(_vals / val, shape());
}

Tensor Tensor::operator/(const Tensor &val) const{
	return functional::divide(*this, val);
}

Tensor& Tensor::operator/=(Scalar val){
	_vals /= val;
	return *this;
}

Tensor& Tensor::operator/=(const Tensor &val){
	return functional::divide_(*this, val);
}


Tensor Tensor::operator-() const {return *this * -1;}
Tensor operator+(Scalar s, const Tensor &t){return t + s;}
Tensor operator-(Scalar s, const Tensor &t){return -t + s;}
Tensor operator*(Scalar s, const Tensor &t){return t * s;}
Tensor operator/(Scalar s, const Tensor &t){
	Tensor a = t.inverse();
	a *= s;
	return a;
}


Tensor Tensor::operator==(const Tensor &val) const{
	utils::throw_exception(val.shape() == val.shape(), "\nRuntimeError: Expected input tensor to have a shape of $ but got $", shape(), val.shape());
	utils::throw_exception(val.dtype == dtype, "\nRuntimeError: Expected input tensor to have a dtype of $ but got $", dtype, val.dtype);
	if(dtype == DType::TensorObj){
		Tensor output(shape(), DType::TensorObj);
		_vals.transform_function<DType::TensorObj>([](const Tensor& a, const Tensor& b){return a == b;}, val._vals, reinterpret_cast<Tensor*>(output.data_ptr()));
		return std::move(output);
	}
	Tensor output(shape(), DType::Bool);
	_vals.transform_function<WRAP_DTYPES<NumberTypesL>>([](auto& a, auto& b) -> uint_bool_t {return uint_bool_t(a == b);}, val._vals, reinterpret_cast<uint_bool_t*>(output.data_ptr()));
	return std::move(output);
}

Tensor Tensor::operator>=(const Tensor &val) const{
	utils::throw_exception(val.shape() == val.shape(), "\nRuntimeError: Expected input tensor to have a shape of $ but got $", shape(), val.shape());
	utils::throw_exception(val.dtype == dtype, "\nRuntimeError: Expected input tensor to have a dtype of $ but got $", dtype, val.dtype);
	if(dtype == DType::TensorObj){
		Tensor output(shape(), DType::TensorObj);
		_vals.transform_function<DType::TensorObj>([](const Tensor& a, const Tensor& b){return a >= b;}, val._vals, reinterpret_cast<Tensor*>(output.data_ptr()));
		return std::move(output);
	}
	Tensor output(shape(), DType::Bool);
	_vals.transform_function<WRAP_DTYPES<NumberTypesL>>([](auto& a, auto& b) -> uint_bool_t {return uint_bool_t(a >= b);}, val._vals, reinterpret_cast<uint_bool_t*>(output.data_ptr()));
	return std::move(output);
}

Tensor Tensor::operator<=(const Tensor &val) const{
	utils::throw_exception(val.shape() == val.shape(), "\nRuntimeError: Expected input tensor to have a shape of $ but got $", shape(), val.shape());
	utils::throw_exception(val.dtype == dtype, "\nRuntimeError: Expected input tensor to have a dtype of $ but got $", dtype, val.dtype);
	if(dtype == DType::TensorObj){
		Tensor output(shape(), DType::TensorObj);
		_vals.transform_function<DType::TensorObj>([](const Tensor& a, const Tensor& b){return a <= b;}, val._vals, reinterpret_cast<Tensor*>(output.data_ptr()));
		return std::move(output);
	}
	Tensor output(shape(), DType::Bool);
	_vals.transform_function<WRAP_DTYPES<NumberTypesL>>([](auto& a, auto& b) -> uint_bool_t {return uint_bool_t(a <= b);}, val._vals, reinterpret_cast<uint_bool_t*>(output.data_ptr()));
	return std::move(output);
}

Tensor Tensor::operator>(const Tensor &val) const{
	utils::throw_exception(val.shape() == val.shape(), "\nRuntimeError: Expected input tensor to have a shape of $ but got $", shape(), val.shape());
	utils::throw_exception(val.dtype == dtype, "\nRuntimeError: Expected input tensor to have a dtype of $ but got $", dtype, val.dtype);
	if(dtype == DType::TensorObj){
		Tensor output(shape(), DType::TensorObj);
		_vals.transform_function<DType::TensorObj>([](const Tensor& a, const Tensor& b){return a > b;}, val._vals, reinterpret_cast<Tensor*>(output.data_ptr()));
		return std::move(output);
	}
	Tensor output(shape(), DType::Bool);
	_vals.transform_function<WRAP_DTYPES<NumberTypesL>>([](auto& a, auto& b) -> uint_bool_t {return uint_bool_t(a > b);}, val._vals, reinterpret_cast<uint_bool_t*>(output.data_ptr()));
	return std::move(output);
}

Tensor Tensor::operator<(const Tensor &val) const{
	utils::throw_exception(val.shape() == val.shape(), "\nRuntimeError: Expected input tensor to have a shape of $ but got $", shape(), val.shape());
	utils::throw_exception(val.dtype == dtype, "\nRuntimeError: Expected input tensor to have a dtype of $ but got $", dtype, val.dtype);
	if(dtype == DType::TensorObj){
		Tensor output(shape(), DType::TensorObj);
		_vals.transform_function<DType::TensorObj>([](const Tensor& a, const Tensor& b){return a < b;}, val._vals, reinterpret_cast<Tensor*>(output.data_ptr()));
		return std::move(output);
	}
	Tensor output(shape(), DType::Bool);
	_vals.transform_function<WRAP_DTYPES<NumberTypesL>>([](auto& a, auto& b) -> uint_bool_t {return uint_bool_t(a < b);}, val._vals, reinterpret_cast<uint_bool_t*>(output.data_ptr()));
	return std::move(output);
}


Tensor Tensor::contiguous() const{
	/* std::copy((const float*)_vals.get(), (const float*)(_vals.get() + _total_size), copy.get()); */
	return Tensor(_vals.contiguous(), _size);
}

const uint32_t Tensor::contig_count() const {return _vals.use_count();}
const bool Tensor::is_contiguous() const {return _vals.is_contiguous();}
const size_t Tensor::dims() const {return  shape().size();}

template<typename T>
T& Tensor::item(){
	assert(_total_size == 1);
	T* casted = reinterpret_cast<T*>(_vals.data_ptr());
	return *(casted);
}

template float& Tensor::item<float>();
template double& Tensor::item<double>();
template int64_t& Tensor::item<int64_t>();
template int32_t& Tensor::item<int32_t>();
template uint32_t& Tensor::item<uint32_t>();
template int16_t& Tensor::item<int16_t>();
template uint16_t& Tensor::item<uint16_t>();
template int8_t& Tensor::item<int8_t>();
template uint8_t& Tensor::item<uint8_t>();
template Tensor& Tensor::item<Tensor>();
template uint_bool_t& Tensor::item<uint_bool_t>();
template complex_64& Tensor::item<complex_64>();
template complex_128& Tensor::item<complex_128>();
#ifdef __SIZEOF_INT128__
template uint128_t& Tensor::item<uint128_t>();
template int128_t& Tensor::item<int128_t>();
#endif
#ifdef _HALF_FLOAT_SUPPORT_
template float16_t& Tensor::item<float16_t>();
template complex_32& Tensor::item<complex_32>();
#endif
#ifdef _128_FLOAT_SUPPORT_
template float128_t& Tensor::item<float128_t>();
#endif


template<typename T>
const T& Tensor::item() const{
	assert(_total_size == 1);
	const T* casted = reinterpret_cast<const T*>(_vals.data_ptr());
	return *(casted);
}

template const float& Tensor::item<float>() const;
template const double& Tensor::item<double>() const;
template const int64_t& Tensor::item<int64_t>() const;
template const int32_t& Tensor::item<int32_t>() const;
template const uint32_t& Tensor::item<uint32_t>() const;
template const int16_t& Tensor::item<int16_t>() const;
template const uint16_t& Tensor::item<uint16_t>() const;
template const int8_t& Tensor::item<int8_t>() const;
template const uint8_t& Tensor::item<uint8_t>() const;
template const Tensor& Tensor::item<Tensor>() const;
template const uint_bool_t& Tensor::item<uint_bool_t>() const;
template const complex_64& Tensor::item<complex_64>() const;
template const complex_128& Tensor::item<complex_128>() const;
#ifdef __SIZEOF_INT128__
template const uint128_t& Tensor::item<uint128_t>() const;
template const int128_t& Tensor::item<int128_t>() const;
#endif
#ifdef _HALF_FLOAT_SUPPORT_
template const float16_t& Tensor::item<float16_t>() const;
template const complex_32& Tensor::item<complex_32 >() const;
#endif
#ifdef _128_FLOAT_SUPPORT_
template const float128_t& Tensor::item<float128_t>() const;
#endif

Scalar Tensor::toScalar() const{
	switch(dtype){
		case DType::Integer:
			return Scalar(*reinterpret_cast<int32_t*>(_vals.strides_cbegin()[0]));
		case DType::Float:
			return Scalar(*reinterpret_cast<float*>(_vals.strides_cbegin()[0]));
		case DType::Double:
			return Scalar(*reinterpret_cast<double*>(_vals.strides_cbegin()[0]));
		case DType::Long:
			return Scalar(*reinterpret_cast<uint32_t*>(_vals.strides_cbegin()[0]));
		case DType::Complex64:
			return Scalar(*reinterpret_cast<complex_64*>(_vals.strides_cbegin()[0]));
		case DType::Complex128:
			return Scalar(*reinterpret_cast<complex_128*>(_vals.strides_cbegin()[0]));
		case DType::uint8:
			return Scalar(*reinterpret_cast<uint8_t*>(_vals.strides_cbegin()[0]));
		case DType::int8:
			return Scalar(*reinterpret_cast<int8_t*>(_vals.strides_cbegin()[0]));
		case DType::int16:
			return Scalar(*reinterpret_cast<int16_t*>(_vals.strides_cbegin()[0]));
		case DType::uint16:
			return Scalar(*reinterpret_cast<uint16_t*>(_vals.strides_cbegin()[0]));
		case DType::LongLong:
			return Scalar(*reinterpret_cast<int64_t*>(_vals.strides_cbegin()[0]));
		case DType::Bool:
			return Scalar(*reinterpret_cast<uint_bool_t*>(_vals.strides_cbegin()[0]));
		case DType::TensorObj:
			return Scalar(0);
#ifdef _128_FLOAT_SUPPORT_
		case DType::Float128:
			return Scalar(*reinterpret_cast<float128_t*>(_vals.strides_cbegin()[0]));
#endif
#ifdef _HALF_FLOAT_SUPPORT_
		case DType::Float16:
			return Scalar(*reinterpret_cast<float16_t*>(_vals.strides_cbegin()[0]));
		case DType::Complex32:
			return Scalar(*reinterpret_cast<complex_32*>(_vals.strides_cbegin()[0]));
#endif
#ifdef __SIZEOF_INT128__
		case DType::int128:
			return Scalar(*reinterpret_cast<int128_t*>(_vals.strides_cbegin()[0]));
		case DType::uint128:
			return Scalar(*reinterpret_cast<uint128_t*>(_vals.strides_cbegin()[0]));
#endif
	}
}


const SizeRef& Tensor::shape() const {
	return _size;
}

Tensor Tensor::operator[](int32_t x){
	x = x < 0 ? x + dims() : x;
	if(_total_size == 1) {
		utils::throw_exception(x == 0, "\nRuntimeError: Expected singleton to have indexed of at most $ but instead got $", 0, x);
		return *this;
	}
	
	//I do really like the assertion below
	//assert(x < shape()[0]);
	utils::throw_exception(x < shape()[0], "RuntimeError: Expected x to be less than $ but got $", shape()[0], x);
	if(dims() == 1){
		return Tensor(_vals.share_array(x, 1), SizeRef({1}));
		/* std::cout<<"dims 1"<<std::endl; */
		/* std::cout << n_size.multiply()<<std::endl; */
		/* std::cout << n_size<<std::endl; */
		/* return Tensor(_vals.share_array( */
	}
	/* if(dims() == 1 && dtype == DType::TensorObj) */
	/* 	return reinterpret_cast<Tensor*>(data_ptr())[x]; */
	SizeRef n_size = shape().pop_front();
	return Tensor(_vals.share_array(x * n_size.multiply(), n_size.multiply()), std::move(n_size));
} 

const Tensor Tensor::operator[](int32_t x) const{
	x = x < 0 ? x + dims() : x;
	if(_total_size == 1) {
		utils::throw_exception(x == 0, "\nRuntimeError: Expected singleton to have indexed of at most $ but instead got $", 0, x);
		return *this;
	}
	utils::throw_exception(x < shape()[0], "RuntimeError: Expected x to be less than $ but got $", shape()[0], x);
	if(dims() == 1){
		return Tensor(_vals.share_array(x, 1), SizeRef({1}));
	}

	SizeRef n_size = shape().pop_front();
	return Tensor(_vals.share_array(x * n_size.multiply(), n_size.multiply()), std::move(n_size));
}


Tensor Tensor::operator[](const Tensor& t){
	utils::throw_exception(t.dtype == DType::Bool, "RuntimeError: expected DType Bool but got $", t.dtype);
	ArrayVoid cpy = _vals.copy_strides(false);
	void** my_ptr = _vals.strides_begin();
	void** outp_ptr = cpy.strides_begin();
	auto begin  = t.arr_void().tcbegin<uint_bool_t>();
	auto end = t.arr_void().tcend<uint_bool_t>();
	uint32_t n_size = 0;
	for(;begin != end; ++begin, ++my_ptr){
		if(*begin == true){
			*outp_ptr = *my_ptr;
			++outp_ptr;
			++n_size;
		}
	}
	cpy.resize(n_size);
	return Tensor(std::move(cpy), {n_size});
}

Tensor Tensor::operator[](std::vector<my_range> r){
	for(uint32_t i = 0; i < r.size(); ++i){
		r[i].fix(shape()[i]);
	}
	std::vector<uint32_t> stri = strides();
	std::vector<uint32_t> n(numel());
	std::iota(n.begin(), n.end(), 0);
	n.erase(n.cbegin(), n.cbegin() + (r[0].begin * stri[1]));
	n.erase(n.cbegin() + (r[0].length() * stri[1]), n.cend());
	uint32_t r_length = r[0].length();
	for(uint32_t i = 1; i < r.size(); ++i){
		uint32_t start = r[i].begin * stri[i+1];
		uint32_t size = r[i].length() * stri[i+1];
		uint32_t left = (stri[i]) - (size+start);
		uint32_t q;
		for(q  = 0; q < r_length-1; ++q){
			n.erase(n.cbegin() + (size * q), n.cbegin() + (size * q) + start);
			n.erase(n.cbegin() + (size * (q+1)), n.cbegin() + (size * (q+1)) + left);
		}
		n.erase(n.cbegin() + (size * (q)), n.cbegin() + (size * (q)) + start);
		n.erase(n.cbegin() + (size * (r_length)), n.end());
		r_length *= r[i].length();
	}
	ArrayVoid cpy = _vals.copy_strides(false);
	auto begin  = n.cbegin();
	auto end = n.cend();
	void** outp_ptr = cpy.strides_begin();
	void** inp_ptr = _vals.strides_cbegin();
	for(;begin != end; ++begin, ++outp_ptr){
		*outp_ptr = inp_ptr[*begin];
	}
	cpy.resize(n.size());
	std::vector<uint32_t> n_shape = shape().Vec();
	for(uint32_t i = 0; i < r.size(); ++i)
		n_shape[i] = r[i].length();
	return Tensor(cpy, SizeRef(std::move(n_shape)));
}


//

Tensor Tensor::operator[](const my_range& x){
	uint32_t a = x.begin < 0 ? x.begin + dims() : x.begin;
	uint32_t b = x.end < 0 ? x.end + dims() : x.end;
	assert(a < shape()[0] && b < shape()[0]);
	std::vector<typename SizeRef::ArrayRefInt::value_type> vec = shape().Vec();
	vec[0] = b - a;
	SizeRef n_size(std::move(vec));
	return Tensor(_vals.share_array(a * n_size.multiply(1), (b-a) * n_size.multiply(1)), std::move(n_size));
}

const Tensor Tensor::operator[](const my_range& x) const{
	uint32_t a = x.begin < 0 ? x.begin + dims() : x.begin;
	uint32_t b = x.end < 0 ? x.end + dims() : x.end;
	assert(a < shape()[0] && b < shape()[0]);
	std::vector<typename SizeRef::ArrayRefInt::value_type> vec = shape().Vec();
	vec[0] = b - a;
	SizeRef n_size(std::move(vec));
	return Tensor(_vals.share_array(a * n_size.multiply(1), (b-a) * n_size.multiply(1)), std::move(n_size));
}

void Tensor::print() const{
	std::cout<<*this<<std::endl;
}


template<typename T>
void print_tensor_template_func(tdtype_list<const T>& data, tdtype_list<const T>& end, std::ostream& out, const SizeRef& t_s, bool sub_matrix, uint32_t print_space){
	if(!sub_matrix){
		out << t_s<<std::endl;
		out << "Tensor(";
		print_space += 7;
	}
	if(t_s.empty()){
		out << "[]"<<std::endl;
		return;
	}
	if(t_s.size() == 1){
		out << "[";
		for(;(data + 1) != end; ++data)
			out << *data<<",";
		out << *data<<"]";
		if(!sub_matrix){out << ')'<< std::endl;}
		return;
	}
	if(t_s.size() == 2){
		const uint32_t _rows = t_s.front();
		const uint32_t _cols = t_s.back();
		out <<"[";
		++print_space;
		for(uint32_t x = 0; x < _rows; ++x){
			if(x != 0 || !sub_matrix){out << "[";}
			for(uint32_t y = 0; y < _cols-1; ++y){
				out<<*data<<",";
				++data;
			}
			if(x == _rows - 1){
				out << *data << "]]"<<std::endl;
			}
			else{
				out<< *data<<"],";
				out << std::endl;
				for(uint32_t i = 0; i < print_space; ++i)
					out << " ";
			}
			++data;
		}
		if(!sub_matrix){out << "])" << std::endl;}
		else { 
			out << std::endl;
			for(uint32_t i = 0; i < print_space - 1; ++i)
				out << " ";
		}
		return;
	}
	for(uint32_t i = 0; i < t_s.front(); ++i){
		if(!sub_matrix && i != 0){
			for(uint32_t j = 0; j < 7; ++j)
				out << " ";
		}
		out<<"[";
		print_tensor_template_func(data, end, out, t_s.pop_front(), true, print_space + 1);
		if(!sub_matrix){
			out << "\033[F";
			for(uint32_t j = 0; j < t_s.size() - 3; ++j)
				out << "]";
			if(i != t_s.front() - 1){
				out << ',';
				out << std::endl<<std::endl;
			}
			else{
				out << ")";
				out << std::endl << std::endl;
			}
		}
	}	
}

inline static constexpr auto print_tensor_func = [](auto data, auto end, std::ostream& out, const SizeRef& t_s, bool sub_matrix, uint32_t print_space){
	if(!sub_matrix){
		out << "Tensor([";
		print_space += 8;
	}
	if(t_s.empty() || t_s.front() == 0){
		out << "[], {0})";
		return;
	}
	if(t_s.size() == 1){
		out << "[";
		for(;(data + 1) != end; ++data)
			out << *data<<",";
		out << *data<<"]";
		if(!sub_matrix){out <<", "<< t_s<<')'<< std::endl;}
		return;
	}
	if(t_s.size() == 2){
		const uint32_t _rows = t_s.front();
		const uint32_t _cols = t_s.back();
		out <<"[";
		++print_space;
		for(uint32_t x = 0; x < _rows; ++x){
			if(x != 0 || !sub_matrix){out << "[";}
			for(uint32_t y = 0; y < _cols-1; ++y){
				out<<*data<<",";
				++data;
			}
			if(x == _rows - 1){
				out << *data << "]]"<<std::endl;
			}
			else{
				out<< *data<<"],";
				out << std::endl;
				for(uint32_t i = 0; i < print_space; ++i)
					out << " ";
			}
			++data;
		}
		if(!sub_matrix){out << "], "<<t_s<<")" << std::endl;}
		else { 
			out << std::endl;
			for(uint32_t i = 0; i < print_space - 1; ++i)
				out << " ";
		}
		return;
	}
	for(uint32_t i = 0; i < t_s.front(); ++i){
		if(!sub_matrix && i != 0){
			for(uint32_t j = 0; j < 9; ++j)
				out << " ";
		}
		out<<"[";
		print_tensor_template_func(data, end, out, t_s.pop_front(), true, print_space + 1);
		if(!sub_matrix){
			out << "\033[F";
			for(uint32_t j = 0; j < t_s.size() - 3; ++j)
				out << "]";
			if(i != t_s.front() - 1){
				/* out << ','; */
				out << std::endl<<std::endl;
			}
			else{
				out <<"], "<<t_s<<")";
				out << std::endl;
			}
		}
	}

};


std::ostream& operator << (std::ostream &out, const Tensor &_t){
	if(_t.dtype == DType::TensorObj && _t.numel() == 1){
		return out << *_t.arr_void().tcbegin<Tensor>() <<std::endl;
	}
	if(_t.dtype == DType::Bool)
		out << std::boolalpha;
	_t.arr_void().cexecute_function<WRAP_DTYPES<AllTypesL>>(print_tensor_func, out, _t.shape(), false, 0);
	if(_t.dtype == DType::Bool)
		out << std::noboolalpha;
	out << std::endl<< _t.dtype;
	return out;	
}


Tensor Tensor::view(SizeRef nv) const{
	uint32_t total = nv.multiply();
	assert(total == _total_size);
	return Tensor(_vals, std::move(nv));
}

Tensor Tensor::flatten(int8_t _a, int8_t _b) const{
	_a = _a < 0 ? _a + dims() : _a;
	_b = _b < 0 ? _b + dims() : _b;
	uint8_t begin = _a < _b ? _a : _b;
	uint8_t end = _a < _b ? _b : _a;
	++end;
	typedef typename SizeRef::ArrayRefInt::value_type value_t;
	size_t n_dims = dims() - (end - begin) + 1;
	std::vector<value_t> n_vals(n_dims);
	std::copy(shape().cbegin(), shape().cbegin() + begin, n_vals.begin());
	n_vals[begin] = std::accumulate(shape().begin() + begin, shape().begin() + end, 1.0, std::multiplies<value_t>());
	std::copy(shape().cbegin() + end, shape().cend(), n_vals.begin() + begin + 1);
	return Tensor(_vals, std::move(n_vals));
}


Tensor Tensor::permute(std::vector<uint32_t> Perm) const{
	std::vector<uint32_t> _strides = this->strides();
	_strides.erase(_strides.begin());
	assert(Perm.size() == _strides.size());
	std::vector<uint32_t> _shape = shape().Vec();
	for(uint32_t i = 0; i < Perm.size(); ++i){
		if(Perm[i] == i) continue;
		std::swap(_strides[i], _strides[Perm[i]]);
		std::swap(_shape[i], _shape[Perm[i]]);
		std::swap(Perm[i], Perm[Perm[i]]);
	}
	ArrayVoid cpy = _vals.copy_strides(false);
	permute::Permute(_vals.strides_cbegin(), cpy.strides_begin(), _vals.Size(), _shape, _strides);
	return Tensor(cpy, SizeRef(std::move(_shape)));
}

Tensor Tensor::transpose(int8_t _a, int8_t _b) const{
	_a = _a < 0 ? dims() + _a : _a;
	_b = _b < 0 ? dims() + _b : _b;
	/* if((_a == dims() - 1 && _b == dims() - 2) || (_a == dims() - 2 && _b == dims() - 1)){ */
	/* 	Tensor cpy = this->contiguous(); */
	/* 	std::cout<<"doing row col swap"<<std::endl; */
	/* 	cpy.RowColSwap(); */
	/* 	return std::move(cpy); */
	/* } */
	std::vector<uint32_t> Vec(dims());
	std::iota(Vec.begin(), Vec.end(), 0);
	std::swap(Vec[_a], Vec[_b]);
	return permute(std::move(Vec));
}


//(row, col) = (row * _cols) + col


void transpose_RowColSwap(void** first, void** last, const uint32_t& n, const uint32_t& mn1, const uint32_t& total)
{
    std::vector<bool> visited(total);
    void** cycle = first;
    while (++cycle != last) {
        if (visited[cycle - first])
            continue;
        int a = cycle - first;
        do  {
            a = a == mn1 ? mn1 : (n * a) % mn1;
            std::swap(*(first + a), *cycle);
            visited[a] = true;
        } while ((first + a) != cycle);
    }
}

//swaps rows and collumns (causes segmentation fault)
Tensor& Tensor::RowColSwap(){
	utils::throw_exception(dims() >= 2, "RuntimeError: Expected dims to be greater than or equal to 2 but got $", dims());
	if(dims() > 2){
		/* Tensor parts = this->split_axis(-3); */
		/* Tensor* ts = reinterpret_cast<Tensor*>(parts.data_ptr()); */
		/* for(uint32_t i = 0; i < parts.numel(); ++i, ++ts) */
		/* 	ts->RowColSwap(); */
		/* _size = shape().transpose(-1,-2); */
		/* return *this; */
		uint32_t i_total = shape().flatten(0,-3)[0];
		uint32_t total = shape()[-1] * shape()[-2];
		const uint32_t& rows = shape()[-2];
		const uint32_t mn1 = total - 1;
		for(uint32_t i = 0; i < i_total; ++i){
			void** first = _vals.strides_begin() + (i * total);
			void** last = first + total;
			transpose_RowColSwap(first, last, rows, mn1, total);
		}
		_size = shape().transpose(-1,-2);
		return *this;

	}
	const uint32_t& rows = shape()[-2];
	const uint32_t mn1 = numel() - 1;
	void** first = _vals.strides_begin();
	void** last = _vals.strides_end();
	transpose_RowColSwap(first, last, rows, mn1, numel());
	_size = shape().transpose(-1,-2);
	return *this;

}


//the difference between the const and non-const version is that the const version does not change the view
//both can be parallized and used for faster opperations though like matrix multiplication and splitting axis view
const Tensor& Tensor::RowColSwap() const{
	utils::throw_exception(dims() >= 2, "RuntimeError: Expected dims to be greater than or equal to 2 but got $", dims());
	if(dims() > 2){
		uint32_t i_total = shape().flatten(0,-3)[0];
		uint32_t total = shape()[-1] * shape()[-2];
		const uint32_t& rows = shape()[-2];
		const uint32_t mn1 = total - 1;
		for(uint32_t i = 0; i < i_total; ++i){
			void** first = _vals.strides_cbegin() + (i * total);
			void** last = first + total;
			transpose_RowColSwap(first, last, rows, mn1, total);
		}
		const_cast<SizeRef&>(_size) = shape().transpose(-1,-2);
		return *this;

	}
	const uint32_t& rows = shape()[-2];
	const uint32_t mn1 = numel() - 1;
	void** first = _vals.strides_cbegin();
	void** last = _vals.strides_cend();
	transpose_RowColSwap(first, last, rows, mn1, numel());
	const_cast<SizeRef&>(_size) = shape().transpose(-1,-2);
	return *this;
}


template<typename T>
void transpose_RowColSwap_contiguous(T* first, T* last, const uint32_t& n, const uint32_t& mn1, const uint32_t& total)
{
    std::vector<bool> visited(total);
    T* cycle = first;
    while (++cycle != last) {
        if (visited[cycle - first])
            continue;
        int a = cycle - first;
        do  {
            a = a == mn1 ? mn1 : (n * a) % mn1;
            std::swap(*(first + a), *cycle);
            visited[a] = true;
        } while ((first + a) != cycle);
    }
}



const Tensor& Tensor::RowColSwap_contiguous() const {
	utils::throw_exception(dims() >= 2, "RuntimeError: Expected dims to be greater than or equal to 2 but got $", dims());
	if(dims() > 2){
		uint32_t i_total = shape().flatten(0,-3)[0];
		uint32_t total = shape()[-1] * shape()[-2];
		const uint32_t& rows = shape()[-2];
		const uint32_t mn1 = total - 1;
		switch(dtype){
			case DType::Float:{
				using value_type = float;
				for(uint32_t i = 0; i < i_total; ++i){
					value_type* first = const_cast<value_type*>(reinterpret_cast<const value_type*>(_vals.data_ptr())) + (i * total);
					value_type* last = first + total;
					transpose_RowColSwap_contiguous(first, last, rows, mn1, total);
				}
				return *this;
			}
			case DType::Double:{
				using value_type = double;
				for(uint32_t i = 0; i < i_total; ++i){
					value_type* first = const_cast<value_type*>(reinterpret_cast<const value_type*>(_vals.data_ptr())) + (i * total);
					value_type* last = first + total;
					transpose_RowColSwap_contiguous(first, last, rows, mn1, total);
				}
				return *this;
			}
#ifdef _HALF_FLOAT_SUPPORT_
			case DType::Float16:{
				using value_type = float16_t;
				for(uint32_t i = 0; i < i_total; ++i){
					value_type* first = const_cast<value_type*>(reinterpret_cast<const value_type*>(_vals.data_ptr())) + (i * total);
					value_type* last = first + total;
					transpose_RowColSwap_contiguous(first, last, rows, mn1, total);
				}
				return *this;
			}
			case DType::Complex32:{
				using value_type = complex_32;
				for(uint32_t i = 0; i < i_total; ++i){
					value_type* first = const_cast<value_type*>(reinterpret_cast<const value_type*>(_vals.data_ptr())) + (i * total);
					value_type* last = first + total;
					transpose_RowColSwap_contiguous(first, last, rows, mn1, total);
				}
				return *this;
			}
#endif
#ifdef _128_FLOAT_SUPPORT_
			case DType::Float128:{
				using value_type = float128_t;
				for(uint32_t i = 0; i < i_total; ++i){
					value_type* first = const_cast<value_type*>(reinterpret_cast<const value_type*>(_vals.data_ptr())) + (i * total);
					value_type* last = first + total;
					transpose_RowColSwap_contiguous(first, last, rows, mn1, total);
				}
				return *this;
			}
#endif
#ifdef __SIZEOF_INT128__
			case DType::int128:{
				using value_type = int128_t;
				for(uint32_t i = 0; i < i_total; ++i){
					value_type* first = const_cast<value_type*>(reinterpret_cast<const value_type*>(_vals.data_ptr())) + (i * total);
					value_type* last = first + total;
					transpose_RowColSwap_contiguous(first, last, rows, mn1, total);
				}
				return *this;
			}
			case DType::uint128:{
				using value_type = uint128_t;
				for(uint32_t i = 0; i < i_total; ++i){
					value_type* first = const_cast<value_type*>(reinterpret_cast<const value_type*>(_vals.data_ptr())) + (i * total);
					value_type* last = first + total;
					transpose_RowColSwap_contiguous(first, last, rows, mn1, total);
				}
				return *this;
			}
#endif
			case DType::Complex64:{
				using value_type = complex_64;
				for(uint32_t i = 0; i < i_total; ++i){
					value_type* first = const_cast<value_type*>(reinterpret_cast<const value_type*>(_vals.data_ptr())) + (i * total);
					value_type* last = first + total;
					transpose_RowColSwap_contiguous(first, last, rows, mn1, total);
				}
				return *this;
			}
			case DType::Complex128:{
				using value_type = complex_128;
				for(uint32_t i = 0; i < i_total; ++i){
					value_type* first = const_cast<value_type*>(reinterpret_cast<const value_type*>(_vals.data_ptr())) + (i * total);
					value_type* last = first + total;
					transpose_RowColSwap_contiguous(first, last, rows, mn1, total);
				}
				return *this;
			}
			case DType::int8:{
				using value_type = int8_t;
				for(uint32_t i = 0; i < i_total; ++i){
					value_type* first = const_cast<value_type*>(reinterpret_cast<const value_type*>(_vals.data_ptr())) + (i * total);
					value_type* last = first + total;
					transpose_RowColSwap_contiguous(first, last, rows, mn1, total);
				}
				return *this;
			}
			case DType::uint8:{
				using value_type = uint8_t;
				for(uint32_t i = 0; i < i_total; ++i){
					value_type* first = const_cast<value_type*>(reinterpret_cast<const value_type*>(_vals.data_ptr())) + (i * total);
					value_type* last = first + total;
					transpose_RowColSwap_contiguous(first, last, rows, mn1, total);
				}
				return *this;
			}
			case DType::int16:{
				using value_type = int16_t;
				for(uint32_t i = 0; i < i_total; ++i){
					value_type* first = const_cast<value_type*>(reinterpret_cast<const value_type*>(_vals.data_ptr())) + (i * total);
					value_type* last = first + total;
					transpose_RowColSwap_contiguous(first, last, rows, mn1, total);
				}
				return *this;
			}
			case DType::uint16:{
				using value_type = uint16_t;
				for(uint32_t i = 0; i < i_total; ++i){
					value_type* first = const_cast<value_type*>(reinterpret_cast<const value_type*>(_vals.data_ptr())) + (i * total);
					value_type* last = first + total;
					transpose_RowColSwap_contiguous(first, last, rows, mn1, total);
				}
				return *this;
			}
			case DType::int32:{
				using value_type = int32_t;
				for(uint32_t i = 0; i < i_total; ++i){
					value_type* first = const_cast<value_type*>(reinterpret_cast<const value_type*>(_vals.data_ptr())) + (i * total);
					value_type* last = first + total;
					transpose_RowColSwap_contiguous(first, last, rows, mn1, total);
				}
				return *this;
			}
			case DType::uint32:{
				using value_type = uint32_t;
				for(uint32_t i = 0; i < i_total; ++i){
					value_type* first = const_cast<value_type*>(reinterpret_cast<const value_type*>(_vals.data_ptr())) + (i * total);
					value_type* last = first + total;
					transpose_RowColSwap_contiguous(first, last, rows, mn1, total);
				}
				return *this;
			}
			case DType::int64:{
				using value_type = int64_t;
				for(uint32_t i = 0; i < i_total; ++i){
					value_type* first = const_cast<value_type*>(reinterpret_cast<const value_type*>(_vals.data_ptr())) + (i * total);
					value_type* last = first + total;
					transpose_RowColSwap_contiguous(first, last, rows, mn1, total);
				}
				return *this;
			}
			case DType::Bool:{
				using value_type = uint_bool_t;
				for(uint32_t i = 0; i < i_total; ++i){
					value_type* first = const_cast<value_type*>(reinterpret_cast<const value_type*>(_vals.data_ptr())) + (i * total);
					value_type* last = first + total;
					transpose_RowColSwap_contiguous(first, last, rows, mn1, total);
				}
				return *this;
			}
			case DType::TensorObj:{
				using value_type = Tensor;
				for(uint32_t i = 0; i < i_total; ++i){
					value_type* first = const_cast<value_type*>(reinterpret_cast<const value_type*>(_vals.data_ptr())) + (i * total);
					value_type* last = first + total;
					transpose_RowColSwap_contiguous(first, last, rows, mn1, total);
				}
				return *this;
			}

		}
		return *this;

	}
	const uint32_t& rows = shape()[-2];
	const uint32_t mn1 = numel() - 1;

	switch(dtype){
		case DType::Float:{
			using value_type = float;
			value_type* first = const_cast<value_type*>(reinterpret_cast<const value_type*>(_vals.data_ptr()));
			value_type* last = const_cast<value_type*>(reinterpret_cast<const value_type*>(_vals.data_ptr_end()));
			transpose_RowColSwap_contiguous(first, last, rows, mn1, numel());
			return *this;
		}
		case DType::Double:{
			using value_type = double;
			value_type* first = const_cast<value_type*>(reinterpret_cast<const value_type*>(_vals.data_ptr()));
			value_type* last = const_cast<value_type*>(reinterpret_cast<const value_type*>(_vals.data_ptr_end()));
			transpose_RowColSwap_contiguous(first, last, rows, mn1, numel());
			return *this;
		}
#ifdef _HALF_FLOAT_SUPPORT_
		case DType::Float16:{
			using value_type = float16_t;
			value_type* first = const_cast<value_type*>(reinterpret_cast<const value_type*>(_vals.data_ptr()));
			value_type* last = const_cast<value_type*>(reinterpret_cast<const value_type*>(_vals.data_ptr_end()));
			transpose_RowColSwap_contiguous(first, last, rows, mn1, numel());
			return *this;
		}
		case DType::Complex32:{
			using value_type = complex_32;
			value_type* first = const_cast<value_type*>(reinterpret_cast<const value_type*>(_vals.data_ptr()));
			value_type* last = const_cast<value_type*>(reinterpret_cast<const value_type*>(_vals.data_ptr_end()));
			transpose_RowColSwap_contiguous(first, last, rows, mn1, numel());
			return *this;
		}
#endif
#ifdef _128_FLOAT_SUPPORT_
		case DType::Float128:{
			using value_type = float128_t;
			value_type* first = const_cast<value_type*>(reinterpret_cast<const value_type*>(_vals.data_ptr()));
			value_type* last = const_cast<value_type*>(reinterpret_cast<const value_type*>(_vals.data_ptr_end()));
			transpose_RowColSwap_contiguous(first, last, rows, mn1, numel());
			return *this;
		}
#endif
#ifdef __SIZEOF_INT128__
		case DType::int128:{
			using value_type = int128_t;
			value_type* first = const_cast<value_type*>(reinterpret_cast<const value_type*>(_vals.data_ptr()));
			value_type* last = const_cast<value_type*>(reinterpret_cast<const value_type*>(_vals.data_ptr_end()));
			transpose_RowColSwap_contiguous(first, last, rows, mn1, numel());
			return *this;
		}
		case DType::uint128:{
			using value_type = uint128_t;
			value_type* first = const_cast<value_type*>(reinterpret_cast<const value_type*>(_vals.data_ptr()));
			value_type* last = const_cast<value_type*>(reinterpret_cast<const value_type*>(_vals.data_ptr_end()));
			transpose_RowColSwap_contiguous(first, last, rows, mn1, numel());
			return *this;
		}
#endif
		case DType::Complex64:{
			using value_type = complex_64;
			value_type* first = const_cast<value_type*>(reinterpret_cast<const value_type*>(_vals.data_ptr()));
			value_type* last = const_cast<value_type*>(reinterpret_cast<const value_type*>(_vals.data_ptr_end()));
			transpose_RowColSwap_contiguous(first, last, rows, mn1, numel());
			return *this;
		}
		case DType::Complex128:{
			using value_type = complex_128;
			value_type* first = const_cast<value_type*>(reinterpret_cast<const value_type*>(_vals.data_ptr()));
			value_type* last = const_cast<value_type*>(reinterpret_cast<const value_type*>(_vals.data_ptr_end()));
			transpose_RowColSwap_contiguous(first, last, rows, mn1, numel());
			return *this;
		}
		case DType::int8:{
			using value_type = int8_t;
			value_type* first = const_cast<value_type*>(reinterpret_cast<const value_type*>(_vals.data_ptr()));
			value_type* last = const_cast<value_type*>(reinterpret_cast<const value_type*>(_vals.data_ptr_end()));
			transpose_RowColSwap_contiguous(first, last, rows, mn1, numel());
			return *this;
		}
		case DType::uint8:{
			using value_type = uint8_t;
			value_type* first = const_cast<value_type*>(reinterpret_cast<const value_type*>(_vals.data_ptr()));
			value_type* last = const_cast<value_type*>(reinterpret_cast<const value_type*>(_vals.data_ptr_end()));
			transpose_RowColSwap_contiguous(first, last, rows, mn1, numel());
			return *this;
		}
		case DType::int16:{
			using value_type = int16_t;
			value_type* first = const_cast<value_type*>(reinterpret_cast<const value_type*>(_vals.data_ptr()));
			value_type* last = const_cast<value_type*>(reinterpret_cast<const value_type*>(_vals.data_ptr_end()));
			transpose_RowColSwap_contiguous(first, last, rows, mn1, numel());
			return *this;
		}
		case DType::uint16:{
			using value_type = uint16_t;
			value_type* first = const_cast<value_type*>(reinterpret_cast<const value_type*>(_vals.data_ptr()));
			value_type* last = const_cast<value_type*>(reinterpret_cast<const value_type*>(_vals.data_ptr_end()));
			transpose_RowColSwap_contiguous(first, last, rows, mn1, numel());
			return *this;
		}
		case DType::int32:{
			using value_type = int32_t;
			value_type* first = const_cast<value_type*>(reinterpret_cast<const value_type*>(_vals.data_ptr()));
			value_type* last = const_cast<value_type*>(reinterpret_cast<const value_type*>(_vals.data_ptr_end()));
			transpose_RowColSwap_contiguous(first, last, rows, mn1, numel());
			return *this;
		}
		case DType::uint32:{
			using value_type = uint32_t;
			value_type* first = const_cast<value_type*>(reinterpret_cast<const value_type*>(_vals.data_ptr()));
			value_type* last = const_cast<value_type*>(reinterpret_cast<const value_type*>(_vals.data_ptr_end()));
			transpose_RowColSwap_contiguous(first, last, rows, mn1, numel());
			return *this;
		}
		case DType::int64:{
			using value_type = int64_t;
			value_type* first = const_cast<value_type*>(reinterpret_cast<const value_type*>(_vals.data_ptr()));
			value_type* last = const_cast<value_type*>(reinterpret_cast<const value_type*>(_vals.data_ptr_end()));
			transpose_RowColSwap_contiguous(first, last, rows, mn1, numel());
			return *this;
		}
		case DType::Bool:{
			using value_type = uint_bool_t;
			value_type* first = const_cast<value_type*>(reinterpret_cast<const value_type*>(_vals.data_ptr()));
			value_type* last = const_cast<value_type*>(reinterpret_cast<const value_type*>(_vals.data_ptr_end()));
			transpose_RowColSwap_contiguous(first, last, rows, mn1, numel());
			return *this;
		}
		case DType::TensorObj:{
			using value_type = Tensor;
			value_type* first = const_cast<value_type*>(reinterpret_cast<const value_type*>(_vals.data_ptr()));
			value_type* last = const_cast<value_type*>(reinterpret_cast<const value_type*>(_vals.data_ptr_end()));
			transpose_RowColSwap_contiguous(first, last, rows, mn1, numel());
			return *this;
		}
	}
	return *this;
}


Tensor& Tensor::RowColSwap_contiguous() {
	utils::throw_exception(dims() >= 2, "RuntimeError: Expected dims to be greater than or equal to 2 but got $", dims());
	if(dims() > 2){
		uint32_t i_total = shape().flatten(0,-3)[0];
		uint32_t total = shape()[-1] * shape()[-2];
		const uint32_t& rows = shape()[-2];
		const uint32_t mn1 = total - 1;
		switch(dtype){
			case DType::Float:{
				using value_type = float;
				for(uint32_t i = 0; i < i_total; ++i){
					value_type* first = reinterpret_cast<value_type*>(_vals.data_ptr()) + (i * total);
					value_type* last = first + total;
					transpose_RowColSwap_contiguous(first, last, rows, mn1, total);
				}
				return *this;
			}
			case DType::Double:{
				using value_type = double;
				for(uint32_t i = 0; i < i_total; ++i){
					value_type* first = reinterpret_cast<value_type*>(_vals.data_ptr()) + (i * total);
					value_type* last = first + total;
					transpose_RowColSwap_contiguous(first, last, rows, mn1, total);
				}
				return *this;
			}
#ifdef _HALF_FLOAT_SUPPORT_
			case DType::Float16:{
				using value_type = float16_t;
				for(uint32_t i = 0; i < i_total; ++i){
					value_type* first = reinterpret_cast<value_type*>(_vals.data_ptr()) + (i * total);
					value_type* last = first + total;
					transpose_RowColSwap_contiguous(first, last, rows, mn1, total);
				}
				return *this;
			}
			case DType::Complex32:{
				using value_type = complex_32;
				for(uint32_t i = 0; i < i_total; ++i){
					value_type* first = reinterpret_cast<value_type*>(_vals.data_ptr()) + (i * total);
					value_type* last = first + total;
					transpose_RowColSwap_contiguous(first, last, rows, mn1, total);
				}
				return *this;
			}
#endif
#ifdef _128_FLOAT_SUPPORT_
			case DType::Float128:{
				using value_type = float128_t;
				for(uint32_t i = 0; i < i_total; ++i){
					value_type* first = reinterpret_cast<value_type*>(_vals.data_ptr()) + (i * total);
					value_type* last = first + total;
					transpose_RowColSwap_contiguous(first, last, rows, mn1, total);
				}
				return *this;
			}
#endif
#ifdef __SIZEOF_INT128__
			case DType::int128:{
				using value_type = int128_t;
				for(uint32_t i = 0; i < i_total; ++i){
					value_type* first = reinterpret_cast<value_type*>(_vals.data_ptr()) + (i * total);
					value_type* last = first + total;
					transpose_RowColSwap_contiguous(first, last, rows, mn1, total);
				}
				return *this;
			}
			case DType::uint128:{
				using value_type = uint128_t;
				for(uint32_t i = 0; i < i_total; ++i){
					value_type* first = reinterpret_cast<value_type*>(_vals.data_ptr()) + (i * total);
					value_type* last = first + total;
					transpose_RowColSwap_contiguous(first, last, rows, mn1, total);
				}
				return *this;
			}
#endif
			case DType::Complex64:{
				using value_type = complex_64;
				for(uint32_t i = 0; i < i_total; ++i){
					value_type* first = reinterpret_cast<value_type*>(_vals.data_ptr()) + (i * total);
					value_type* last = first + total;
					transpose_RowColSwap_contiguous(first, last, rows, mn1, total);
				}
				return *this;
			}
			case DType::Complex128:{
				using value_type = complex_128;
				for(uint32_t i = 0; i < i_total; ++i){
					value_type* first = reinterpret_cast<value_type*>(_vals.data_ptr()) + (i * total);
					value_type* last = first + total;
					transpose_RowColSwap_contiguous(first, last, rows, mn1, total);
				}
				return *this;
			}
			case DType::int8:{
				using value_type = int8_t;
				for(uint32_t i = 0; i < i_total; ++i){
					value_type* first = reinterpret_cast<value_type*>(_vals.data_ptr()) + (i * total);
					value_type* last = first + total;
					transpose_RowColSwap_contiguous(first, last, rows, mn1, total);
				}
				return *this;
			}
			case DType::uint8:{
				using value_type = uint8_t;
				for(uint32_t i = 0; i < i_total; ++i){
					value_type* first = reinterpret_cast<value_type*>(_vals.data_ptr()) + (i * total);
					value_type* last = first + total;
					transpose_RowColSwap_contiguous(first, last, rows, mn1, total);
				}
				return *this;
			}
			case DType::int16:{
				using value_type = int16_t;
				for(uint32_t i = 0; i < i_total; ++i){
					value_type* first = reinterpret_cast<value_type*>(_vals.data_ptr()) + (i * total);
					value_type* last = first + total;
					transpose_RowColSwap_contiguous(first, last, rows, mn1, total);
				}
				return *this;
			}
			case DType::uint16:{
				using value_type = uint16_t;
				for(uint32_t i = 0; i < i_total; ++i){
					value_type* first = reinterpret_cast<value_type*>(_vals.data_ptr()) + (i * total);
					value_type* last = first + total;
					transpose_RowColSwap_contiguous(first, last, rows, mn1, total);
				}
				return *this;
			}
			case DType::int32:{
				using value_type = int32_t;
				for(uint32_t i = 0; i < i_total; ++i){
					value_type* first = reinterpret_cast<value_type*>(_vals.data_ptr()) + (i * total);
					value_type* last = first + total;
					transpose_RowColSwap_contiguous(first, last, rows, mn1, total);
				}
				return *this;
			}
			case DType::uint32:{
				using value_type = uint32_t;
				for(uint32_t i = 0; i < i_total; ++i){
					value_type* first = reinterpret_cast<value_type*>(_vals.data_ptr()) + (i * total);
					value_type* last = first + total;
					transpose_RowColSwap_contiguous(first, last, rows, mn1, total);
				}
				return *this;
			}
			case DType::int64:{
				using value_type = int64_t;
				for(uint32_t i = 0; i < i_total; ++i){
					value_type* first = reinterpret_cast<value_type*>(_vals.data_ptr()) + (i * total);
					value_type* last = first + total;
					transpose_RowColSwap_contiguous(first, last, rows, mn1, total);
				}
				return *this;
			}
			case DType::Bool:{
				using value_type = uint_bool_t;
				for(uint32_t i = 0; i < i_total; ++i){
					value_type* first = reinterpret_cast<value_type*>(_vals.data_ptr()) + (i * total);
					value_type* last = first + total;
					transpose_RowColSwap_contiguous(first, last, rows, mn1, total);
				}
				return *this;
			}
			case DType::TensorObj:{
				using value_type = Tensor;
				for(uint32_t i = 0; i < i_total; ++i){
					value_type* first = reinterpret_cast<value_type*>(_vals.data_ptr()) + (i * total);
					value_type* last = first + total;
					transpose_RowColSwap_contiguous(first, last, rows, mn1, total);
				}
				return *this;
			}

		}
		return *this;

	}
	const uint32_t& rows = shape()[-2];
	const uint32_t mn1 = numel() - 1;

	switch(dtype){
		case DType::Float:{
			using value_type = float;
			value_type* first = reinterpret_cast<value_type*>(_vals.data_ptr());
			value_type* last = reinterpret_cast<value_type*>(_vals.data_ptr_end());
			transpose_RowColSwap_contiguous(first, last, rows, mn1, numel());
			return *this;
		}
		case DType::Double:{
			using value_type = double;
			value_type* first = reinterpret_cast<value_type*>(_vals.data_ptr());
			value_type* last = reinterpret_cast<value_type*>(_vals.data_ptr_end());
			transpose_RowColSwap_contiguous(first, last, rows, mn1, numel());
			return *this;
		}
#ifdef _HALF_FLOAT_SUPPORT_
		case DType::Float16:{
			using value_type = float16_t;
			value_type* first = reinterpret_cast<value_type*>(_vals.data_ptr());
			value_type* last = reinterpret_cast<value_type*>(_vals.data_ptr_end());
			transpose_RowColSwap_contiguous(first, last, rows, mn1, numel());
			return *this;
		}
		case DType::Complex32:{
			using value_type = complex_32;
			value_type* first = reinterpret_cast<value_type*>(_vals.data_ptr());
			value_type* last = reinterpret_cast<value_type*>(_vals.data_ptr_end());
			transpose_RowColSwap_contiguous(first, last, rows, mn1, numel());
			return *this;
		}
#endif
#ifdef _128_FLOAT_SUPPORT_
		case DType::Float128:{
			using value_type = float128_t;
			value_type* first = reinterpret_cast<value_type*>(_vals.data_ptr());
			value_type* last = reinterpret_cast<value_type*>(_vals.data_ptr_end());
			transpose_RowColSwap_contiguous(first, last, rows, mn1, numel());
			return *this;
		}
#endif
#ifdef __SIZEOF_INT128__
		case DType::int128:{
			using value_type = int128_t;
			value_type* first = reinterpret_cast<value_type*>(_vals.data_ptr());
			value_type* last = reinterpret_cast<value_type*>(_vals.data_ptr_end());
			transpose_RowColSwap_contiguous(first, last, rows, mn1, numel());
			return *this;
		}
		case DType::uint128:{
			using value_type = uint128_t;
			value_type* first = reinterpret_cast<value_type*>(_vals.data_ptr());
			value_type* last = reinterpret_cast<value_type*>(_vals.data_ptr_end());
			transpose_RowColSwap_contiguous(first, last, rows, mn1, numel());
			return *this;
		}
#endif
		case DType::Complex64:{
			using value_type = complex_64;
			value_type* first = reinterpret_cast<value_type*>(_vals.data_ptr());
			value_type* last = reinterpret_cast<value_type*>(_vals.data_ptr_end());
			transpose_RowColSwap_contiguous(first, last, rows, mn1, numel());
			return *this;
		}
		case DType::Complex128:{
			using value_type = complex_128;
			value_type* first = reinterpret_cast<value_type*>(_vals.data_ptr());
			value_type* last = reinterpret_cast<value_type*>(_vals.data_ptr_end());
			transpose_RowColSwap_contiguous(first, last, rows, mn1, numel());
			return *this;
		}
		case DType::int8:{
			using value_type = int8_t;
			value_type* first = reinterpret_cast<value_type*>(_vals.data_ptr());
			value_type* last = reinterpret_cast<value_type*>(_vals.data_ptr_end());
			transpose_RowColSwap_contiguous(first, last, rows, mn1, numel());
			return *this;
		}
		case DType::uint8:{
			using value_type = uint8_t;
			value_type* first = reinterpret_cast<value_type*>(_vals.data_ptr());
			value_type* last = reinterpret_cast<value_type*>(_vals.data_ptr_end());
			transpose_RowColSwap_contiguous(first, last, rows, mn1, numel());
			return *this;
		}
		case DType::int16:{
			using value_type = int16_t;
			value_type* first = reinterpret_cast<value_type*>(_vals.data_ptr());
			value_type* last = reinterpret_cast<value_type*>(_vals.data_ptr_end());
			transpose_RowColSwap_contiguous(first, last, rows, mn1, numel());
			return *this;
		}
		case DType::uint16:{
			using value_type = uint16_t;
			value_type* first = reinterpret_cast<value_type*>(_vals.data_ptr());
			value_type* last = reinterpret_cast<value_type*>(_vals.data_ptr_end());
			transpose_RowColSwap_contiguous(first, last, rows, mn1, numel());
			return *this;
		}
		case DType::int32:{
			using value_type = int32_t;
			value_type* first = reinterpret_cast<value_type*>(_vals.data_ptr());
			value_type* last = reinterpret_cast<value_type*>(_vals.data_ptr_end());
			transpose_RowColSwap_contiguous(first, last, rows, mn1, numel());
			return *this;
		}
		case DType::uint32:{
			using value_type = uint32_t;
			value_type* first = reinterpret_cast<value_type*>(_vals.data_ptr());
			value_type* last = reinterpret_cast<value_type*>(_vals.data_ptr_end());
			transpose_RowColSwap_contiguous(first, last, rows, mn1, numel());
			return *this;
		}
		case DType::int64:{
			using value_type = int64_t;
			value_type* first = reinterpret_cast<value_type*>(_vals.data_ptr());
			value_type* last = reinterpret_cast<value_type*>(_vals.data_ptr_end());
			transpose_RowColSwap_contiguous(first, last, rows, mn1, numel());
			return *this;
		}
		case DType::Bool:{
			using value_type = uint_bool_t;
			value_type* first = reinterpret_cast<value_type*>(_vals.data_ptr());
			value_type* last = reinterpret_cast<value_type*>(_vals.data_ptr_end());
			transpose_RowColSwap_contiguous(first, last, rows, mn1, numel());
			return *this;
		}
		case DType::TensorObj:{
			using value_type = Tensor;
			value_type* first = reinterpret_cast<value_type*>(_vals.data_ptr());
			value_type* last = reinterpret_cast<value_type*>(_vals.data_ptr_end());
			transpose_RowColSwap_contiguous(first, last, rows, mn1, numel());
			return *this;
		}
	}
	return *this;
}


Tensor Tensor::real() const{
	utils::throw_exception(DTypeFuncs::is_complex(dtype), "RuntimeError: Expected dtype to be a complex number but got $", dtype);
#ifdef _HALF_FLOAT_SUPPORT_
	Tensor output(shape(), (dtype == DType::Complex128 ? DType::Double : dtype == DType::Complex64 ? DType::Float : DType::Float16));
#else
	Tensor output(shape(), (dtype == DType::Complex128 ? DType::Double : DType::Float));
#endif
	if(dtype == DType::Complex128){
		const complex_128* begin = reinterpret_cast<const complex_128*>(_vals.data_ptr());
		const complex_128* end = reinterpret_cast<const complex_128*>(_vals.data_ptr_end());
		double* start = reinterpret_cast<double*>(output._vals.data_ptr());
		for(;begin != end; ++begin, ++start){
			*start = begin->real();
		}
		return std::move(output);
	}
	else if(dtype == DType::Complex64){
		const complex_64* begin = reinterpret_cast<const complex_64*>(_vals.data_ptr());
		const complex_64* end = reinterpret_cast<const complex_64*>(_vals.data_ptr_end());
		float* start = reinterpret_cast<float*>(output._vals.data_ptr());
		for(;begin != end; ++begin, ++start){
			*start = begin->real();
		}
		return std::move(output);
	}
#ifdef _HALF_FLOAT_SUPPORT_
	else if(dtype == DType::Complex32){
		const complex_32* begin = reinterpret_cast<const complex_32*>(_vals.data_ptr());
		const complex_32* end = reinterpret_cast<const complex_32*>(_vals.data_ptr_end());
		float16_t* start = reinterpret_cast<float16_t*>(output._vals.data_ptr());
		for(;begin != end; ++begin, ++start){
			*start = begin->real();
		}
		return std::move(output);
	}
#endif
	return std::move(output);

}


Tensor Tensor::imag() const{
	utils::throw_exception(DTypeFuncs::is_complex(dtype), "RuntimeError: Expected dtype to be a complex number but got $", dtype);
#ifdef _HALF_FLOAT_SUPPORT_
	Tensor output(shape(), (dtype == DType::Complex128 ? DType::Double : dtype == DType::Complex64 ? DType::Float : DType::Float16));
#else
	Tensor output(shape(), (dtype == DType::Complex128 ? DType::Double : DType::Float));
#endif	
	if(dtype == DType::Complex128){
		const complex_128* begin = reinterpret_cast<const complex_128*>(_vals.data_ptr());
		const complex_128* end = reinterpret_cast<const complex_128*>(_vals.data_ptr_end());
		double* start = reinterpret_cast<double*>(output._vals.data_ptr());
		for(;begin != end; ++begin, ++start){
			*start = begin->imag();
		}
		return std::move(output);
	}
	if(dtype == DType::Complex64){
		const complex_64* begin = reinterpret_cast<const complex_64*>(_vals.data_ptr());
		const complex_64* end = reinterpret_cast<const complex_64*>(_vals.data_ptr_end());
		float* start = reinterpret_cast<float*>(output._vals.data_ptr());
		for(;begin != end; ++begin, ++start){
			*start = begin->imag();
		}
		return std::move(output);
	}
#ifdef _HALF_FLOAT_SUPPORT_
	if(dtype == DType::Complex32){
		const complex_32* begin = reinterpret_cast<const complex_32*>(_vals.data_ptr());
		const complex_32* end = reinterpret_cast<const complex_32*>(_vals.data_ptr_end());
		float16_t* start = reinterpret_cast<float16_t*>(output._vals.data_ptr());
		for(;begin != end; ++begin, ++start){
			*start = begin->imag();
		}
		return std::move(output);
	}
#endif
	return std::move(output);

}

//I am going to remake this
//now that I can have a Tensor with a dtype of Tensor
//There may as well just be a Tensor with all the Tensor objects inside of it
//that way there is very little overhead, except for at the beggining

Tensor::Tensor(uint32_t i, const ArrayVoid& Arr, SizeRef&& _s)
	:_size({i}),
	_total_size(i),
	sub_tensor(false),
	_vals(i, DType::TensorObj),
	dtype(DType::TensorObj)
{
	uint32_t count = 0;
	uint32_t inner = _s.multiply();
	Tensor* it = reinterpret_cast<Tensor*>(this->data_ptr());
	Tensor* end = it + i;
	/* for(auto it = val_begin(); it != val_end(); ++it){ */ 
	for(;it != end; ++it, count += inner){
		*it = Tensor(Arr.share_array(count, inner), _s);
	}
}


//returns a tensor of tensors split along a specific axis and accumulated along that axis
Tensor Tensor::split_axis(int8_t dim){
	typedef typename SizeRef::ArrayRefInt::value_type value_t;
	dim = dim < 0 ? dim + dims() : dim;
	if(dim == (dims() - 1)){
		return transpose(-1, -2).split_axis(-2);
	}
	dim += 1;
	std::vector<value_t> n_vals(dims() - dim);
	for(uint32_t i = 0; i < dims() - dim; ++i){
		n_vals[i] = shape()[i+dim];
	}
	SizeRef n2_size(n_vals);
	uint32_t i = _total_size / n2_size.multiply();
	return Tensor(i, _vals, std::move(n2_size));
}


// 0 == cols
// 1 == rows
// 2 == z
// 3 == q
// ....
const Tensor Tensor::split_axis(int8_t dim) const{
	typedef typename SizeRef::ArrayRefInt::value_type value_t;
	dim = dim < 0 ? dim + dims() : dim;
	if(dim == (dims() - 1)){
		return transpose(-1, -2).split_axis(-2);
	}
	dim += 1;
	std::vector<value_t> n_vals(dims() - dim);
	for(uint32_t i = 0; i < dims() - dim; ++i){
		n_vals[i] = shape()[i+dim];
	}
	SizeRef n2_size(n_vals);
	uint32_t i = _total_size / n2_size.multiply();
	return Tensor(i, _vals, std::move(n2_size));
}


const Tensor Tensor::split_axis_1_() const{
	typedef typename SizeRef::ArrayRefInt::value_type value_t;
	RowColSwap();
	SizeRef n_shape = shape().transpose(-1,-2);
	uint32_t dim = (dims() - 2) + 1;
	std::vector<value_t> n_vals(dims() - dim);
	for(uint32_t i = 0; i < dims() - dim; ++i){
		n_vals[i] = n_shape[i+dim];
	}
	SizeRef n2_size(n_vals);
	uint32_t i = _total_size / n2_size.multiply();
	return Tensor(i, _vals, std::move(n2_size));
}

dtype_list Tensor::val_begin(){
	return dtype_list(_vals.strides_begin(), dtype);
}

dtype_list Tensor::val_end(){
	return dtype_list(_vals.strides_begin() + _vals.Size(), dtype);
}

const_dtype_list Tensor::val_cbegin() const{
	return const_dtype_list(_vals.strides_cbegin(), dtype);
}

const_dtype_list Tensor::val_cend() const{
	return const_dtype_list(_vals.strides_cbegin() + _vals.Size(), dtype);
}

#ifndef USE_PARALLEL
//TODO: this function needs to be completed and actually done
//TODO: come up with a dimension swapper (this is going to be similar to RowColSwap except it swaps any 2 dimensions)
Tensor Tensor::unfold(int32_t dim, uint32_t size, uint32_t step) const{
	dim = dim < 0 ? dims() + dim : dim;
	utils::throw_exception(dim < dims(), "Expected to get an appropriate dimension less than $ but got $\n", dims(), dim);
	utils::throw_exception(size <= shape()[dim], "maximum size for Tensor at dimension $ is $ but got size of $", dim, shape()[dim], size);

	std::vector<uint32_t> vec = shape().Vec();
	uint32_t unfolds = uint32_t((shape()[dim] - size)/step + 1);
	vec[dim] = unfolds;
	vec.push_back(size);
	SizeRef outp_size(std::move(vec)); 
	uint32_t n_vals_size = outp_size.multiply();
	ArrayVoid n_vals = _vals.new_stride(n_vals_size);
	std::cout << "getting proc"<<std::endl;
	Tensor proc = dim == dims()-1 ? *this : this->transpose(-1, dim);
	std::cout << "got proc"<<std::endl;
	std::vector<uint32_t> _strides = proc.strides();
	_strides.erase(_strides.cbegin());

	std::vector<uint32_t> outp_strides = outp_size.strides();
	outp_strides.erase(outp_strides.cbegin());

	
	int i_dim = dim;
	while(i_dim != dims()-2 && i_dim != dims()-1){
		std::swap(_strides[i_dim], _strides[i_dim+1]);
		++i_dim;
	}


	//this is going to give the correct strides at the right one,
	//and then the variable unfolds needs to be taken into account
	//this will be used to add the number of _strides[-1] naturally based on where it is in the dim compared to the _strides
	//this kinda fills in the last piece of the puzzle
	
	void** n_ptr = n_vals.strides_begin();
	void** m_ptr = proc._vals.strides_begin();

	for(uint32_t i = 0; i < n_vals_size; ++i, ++n_ptr){
		uint32_t current_add = 0;
		uint32_t i_s = i;
		for(uint32_t j = 0; j < outp_strides.size(); ++j){
			if(j == dim && i_s >= outp_strides[j]){
				uint32_t i_n = i_s / outp_strides[j];
				i_s = i_s % outp_strides[j];
				current_add += i_n * _strides.back();
				continue;
			}
			if(i_s >= outp_strides[j]){
				uint32_t j_i = j > dim ? j - 1 : j;
				uint32_t i_n = i_s / outp_strides[j];
				i_s = i_s % outp_strides[j];
				current_add += i_n * _strides[j_i];

			}
		}
		*n_ptr = m_ptr[current_add];
	}
	return Tensor(std::move(n_vals), std::move(outp_size));
}


#else
Tensor Tensor::unfold(int32_t dim, uint32_t size, uint32_t step) const{
	dim = dim < 0 ? dims() + dim : dim;
	utils::throw_exception(dim < dims(), "Expected to get an appropriate dimension less than $ but got $\n", dims(), dim);
	utils::throw_exception(size <= shape()[dim], "maximum size for Tensor at dimension $ is $ but got size of $", dim, shape()[dim], size);

	std::vector<uint32_t> vec = shape().Vec();
	uint32_t unfolds = uint32_t((shape()[dim] - size)/step + 1);
	vec[dim] = unfolds;
	vec.push_back(size);
	SizeRef outp_size(std::move(vec)); 
	uint32_t n_vals_size = outp_size.multiply();
	int pipe_fd[2];
	Tensor proc;
	ArrayVoid n_vals = _vals.new_stride(n_vals_size);
	std::vector<uint32_t> _strides = proc.shape().transpose(-1, dim).strides();
	_strides.erase(_strides.cbegin());

	std::vector<uint32_t> outp_strides = outp_size.strides();
	outp_strides.erase(outp_strides.cbegin());

	int i_dim = dim;
	while(i_dim != dims()-2 && i_dim != dims()-1){
		std::swap(_strides[i_dim], _strides[i_dim+1]);
		++i_dim;
	}
	
	std::vector<uint32_t> indexes(n_vals_size);
	if (pipe(pipe_fd) == -1) {
		perror("pipe");
		return *this;
	}
	pid_t pid = fork();
	if (pid < 0) {
		// Error occurred
		std::cerr << "Failed to fork.\n";
		return *this;
	}
	else if(pid == 0){
		close(pipe_fd[0]);
		std::vector<uint32_t> indexes(n_vals_size);
		tbb::parallel_for(tbb::blocked_range<uint32_t>(0, n_vals_size),
			[&](tbb::blocked_range<uint32_t> b){
				for(uint32_t i = b.begin(); i < b.end(); ++i){
					uint32_t current_add = 0;
					uint32_t i_s = i;
					for(uint32_t j = 0; j < outp_strides.size(); ++j){
						if(j == dim && i_s >= outp_strides[j]){
							uint32_t i_n = i_s / outp_strides[j];
							i_s = i_s % outp_strides[j];
							current_add += i_n * _strides.back();
							continue;
						}
						if(i_s >= outp_strides[j]){
							uint32_t j_i = j > dim ? j - 1 : j;
							uint32_t i_n = i_s / outp_strides[j];
							i_s = i_s % outp_strides[j];
							current_add += i_n * _strides[j_i];

						}
					}
					indexes[i] = current_add;
				}
			});
		write(pipe_fd[1], indexes.data(), indexes.size() * sizeof(uint32_t));
		close(pipe_fd[1]);
	}
	else{
	close(pipe_fd[1]);
	proc = dim == dims()-1 ? *this : this->transpose(-1, dim);
	read(pipe_fd[0], indexes.data(), indexes.size() * sizeof(uint32_t));
	close(pipe_fd[0]);	
	wait(nullptr);
	}



	//this is going to give the correct strides at the right one,
	//and then the variable unfolds needs to be taken into account
	//this will be used to add the number of _strides[-1] naturally based on where it is in the dim compared to the _strides
	//this kinda fills in the last piece of the puzzle
	
	void** n_ptr = n_vals.strides_begin();
	void** m_ptr = proc._vals.strides_begin();
	tbb::parallel_for(tbb::blocked_range<uint32_t>(0, n_vals_size, 10),
			[&](tbb::blocked_range<uint32_t> b){
				void** r_ptr = n_ptr + b.begin();
				for(uint32_t i = b.begin(); i < b.end(); ++i, ++r_ptr){
					*r_ptr = m_ptr[indexes[i]];
	
				}
			});
	return Tensor(std::move(n_vals), std::move(outp_size));
}
#endif


void* Tensor::data_ptr(){
	return _vals.data_ptr();
}
const void* Tensor::data_ptr() const{
	return _vals.data_ptr();
}

//share from a specific point in memory
Tensor Tensor::div(uint32_t i){
	return Tensor(_vals.share_array(i, i), {i});
}

ArrayVoid& Tensor::arr_void() { return _vals;}
const ArrayVoid& Tensor::arr_void() const { return _vals;}



//fill, subtract, add, multiply
Tensor& Tensor::_subtract(Scalar val){
	_vals -= val;
	return *this;
}
Tensor& Tensor::_subtract(const Tensor& val){return functional::subtract_(*this, val);}
Tensor& Tensor::_multiply(Scalar val){
	_vals *= val;
	return *this;
}
Tensor& Tensor::_multiply(const Tensor& val){return functional::hadamard_multiply_this(*this, val);}
Tensor& Tensor::_divide(Scalar val){
	_vals /= val;
	return *this;
}
Tensor& Tensor::_divide(const Tensor& val){return functional::divide_(*this, val);}
Tensor& Tensor::_fill(Scalar val){*this = val; return *this;}
Tensor& Tensor::_fill(const Tensor& val){_vals.for_each<DType::TensorObj>([&val](auto& inp){inp = val;});return *this;}
Tensor& Tensor::_add(Scalar val){
	_vals += val;
	return *this;
}
Tensor& Tensor::_add(const Tensor& val){return functional::add_(*this, val);}



Tensor Tensor::operator==(Scalar c) const{
	return Tensor(_vals == c, shape());
}

Tensor Tensor::operator<=(Scalar c) const{
	return Tensor(_vals <= c, shape());	
}

Tensor Tensor::operator>=(Scalar c) const{
	return Tensor(_vals >= c, shape());	
}

Tensor Tensor::operator<(Scalar c) const{
	return Tensor(_vals <= c, shape());	
}

Tensor Tensor::operator>(Scalar c) const{
	return Tensor(_vals >= c, shape());	
}



std::string_view Tensor::sv() const{
	utils::throw_exception(dtype == DType::uint8, "\nRuntimeError: Expected DType for string_view to be uint8 but got $", dtype);
	return std::string_view(reinterpret_cast<const char*>(_vals.data_ptr()), numel());
}

Tensor Tensor::to_dtype(DType _dt) const{
	return Tensor(_vals.to(_dt), shape());
}

Tensor Tensor::Int() const{
	return Tensor(_vals.int32(), shape());
}

Tensor Tensor::Long() const{
	return Tensor(_vals.uint32(), shape());
}

Tensor Tensor::unsqueeze() const{
	std::vector<SizeRef::ArrayRefInt::value_type> Vec = shape().Vec();
	Vec.insert(Vec.begin(), 1);
	return view(SizeRef(std::move(Vec)));
}

Tensor Tensor::squeeze() const{
	utils::throw_exception(shape()[0] == 1, "Expected shape[0] to be 1 but got $", shape()[0]);
	std::vector<SizeRef::ArrayRefInt::value_type> Vec(shape().cbegin() + 1, shape().cend());
	return view(SizeRef(std::move(Vec)));
}

inline static constexpr auto sumation_func = [](auto begin, auto end, ArrayVoid& outp) -> Scalar{
	return std::reduce(begin, end);
};

Tensor Tensor::sum() const{
	if(dtype == DType::TensorObj){

		Tensor outp(shape(), DType::TensorObj);
		_vals.transform_function<DType::TensorObj>([](const Tensor& output) -> Tensor {return output.sum();}, reinterpret_cast<Tensor*>(outp.data_ptr()));
		return std::move(outp);
	}
	Tensor outp(1, dtype);
	outp = _vals.cexecute_function<WRAP_DTYPES<RealNumberTypesL>>()([](auto begin, auto end) -> Scalar {return std::reduce(begin, end);});
	return std::move(outp);
}


Tensor Tensor::max() const{
	if(dtype == DType::TensorObj){
		Tensor outp(shape(), DType::TensorObj);
		_vals.transform_function<DType::TensorObj>([](const Tensor& output) -> Tensor {return output.max();}, reinterpret_cast<Tensor*>(outp.data_ptr()));
		return std::move(outp);
	}
	Tensor outp(1, dtype);
	outp = _vals.cexecute_function<WRAP_DTYPES<RealNumberTypesL>>()([](auto begin, auto end) -> Scalar {return *std::max_element(begin, end);});
	return std::move(outp);
}


Tensor Tensor::sum(int32_t dim) const{
	dim = dim < 0 ? dim + dims() : dim;
	if(dtype == DType::TensorObj){
		Tensor outp(shape(), DType::TensorObj);
		_vals.transform_function<DType::TensorObj>([&dim](const Tensor& output) -> Tensor {return output.sum(dim);}, reinterpret_cast<Tensor*>(outp.data_ptr()));
		return std::move(outp);
	}
	uint32_t total_size = shape().flatten(0,dim)[0];
	Tensor outp(shape()[my_range(0, dim)], dtype);
	const Tensor split = this->split_axis(dim);
	outp._vals.execute_function<WRAP_DTYPES<RealNumberTypesL>>()([](auto begin, auto end, const Tensor* vals){
				using value_t = typename std::remove_const<typename decltype(begin)::value_type>::type;
				for(;begin != end; ++begin, ++vals){
					*begin = vals->sum().toScalar().to<value_t>();
				}
			}, reinterpret_cast<const Tensor*>(split.data_ptr()));
	return std::move(outp);
}

Tensor Tensor::max(int32_t dim) const{
	dim = dim < 0 ? dim + dims() : dim;
	if(dtype == DType::TensorObj){
		Tensor outp(shape(), DType::TensorObj);
		_vals.transform_function<DType::TensorObj>([&dim](const Tensor& output) -> Tensor {return output.max(dim);}, reinterpret_cast<Tensor*>(outp.data_ptr()));
		return std::move(outp);
	}
	uint32_t total_size = shape().flatten(0,dim)[0];
	Tensor outp(shape()[my_range(0, dim)], dtype);
	const Tensor split = this->split_axis(dim);
	outp._vals.execute_function<WRAP_DTYPES<RealNumberTypesL>>()([](auto begin, auto end, const Tensor* vals){
				using value_t = typename std::remove_const<typename decltype(begin)::value_type>::type;
				for(;begin != end; ++begin, ++vals){
					*begin = vals->max().toScalar().to<value_t>();
				}
			}, reinterpret_cast<const Tensor*>(split.data_ptr()));
	return std::move(outp);	
}

Tensor Tensor::exp() const{
	utils::throw_exception(dtype != DType::Bool, "\nRuntimeError: Tried running unsupported DType Bool with function exp()");
	return Tensor(_vals.exp(), shape());
}

Tensor& Tensor::exp_(){
	utils::throw_exception(dtype != DType::Bool, "\nRuntimeError: Tried running unsupported DType Bool with function exp_()");
	_vals.exp_();
	return *this;
}

//this was the function that made me implement the ability to choose a specific dtype for a for_each, execute, and transform_function
Tensor& Tensor::inverse_(){
	_vals.inverse_();
	dtype = _vals.dtype;
	return *this;
}

Tensor Tensor::inverse() const{
	return Tensor(_vals.inverse(), shape());
}



Tensor& Tensor::clip_(Scalar a, Scalar b){
	/* std::cout << "clip min: "<<a << */ 
	_vals.execute_function<WRAP_DTYPES<NumberTypesL>>()([&a, &b](auto begin, auto end){
				using value_t = typename std::remove_const<typename decltype(begin)::value_type>::type;
				value_t lower = a.to<value_t>();
				value_t upper = b.to<value_t>();
				std::transform(begin, end, begin, [&lower, &upper](auto val){return std::clamp(val, lower, upper);});
			});
	return *this;
}


template<std::size_t N>
Tensor Tensor::FromInitializer(typename utils::NestedInitializerLists_type<Scalar, N>::type v, DType dt){
	SizeRef sz(utils::aquire_shape<Scalar, N>(v));
	Tensor output(sz, dt);
	switch(dt){
		case DType::Float:{
			using value_type = float;
			value_type* begin = reinterpret_cast<value_type*>(output.data_ptr());
			utils::flatten_func<Scalar, N>(v, [&begin](const Scalar& a){
					*begin = a.to<value_type>();
					++begin;});
			break;
		}
		case DType::Double:{
			using value_type = double;
			value_type* begin = reinterpret_cast<value_type*>(output.data_ptr());
			utils::flatten_func<Scalar, N>(v, [&begin](const Scalar& a){
					*begin = a.to<value_type>();
					++begin;});
			break;
		}
#ifdef _HALF_FLOAT_SUPPORT_
		case DType::Float16:{
			using value_type = float16_t;
			value_type* begin = reinterpret_cast<value_type*>(output.data_ptr());
			utils::flatten_func<Scalar, N>(v, [&begin](const Scalar& a){
					*begin = a.to<value_type>();
					++begin;});
			break;
		}
		case DType::Complex32:{
			using value_type = complex_32;
			value_type* begin = reinterpret_cast<value_type*>(output.data_ptr());
			utils::flatten_func<Scalar, N>(v, [&begin](const Scalar& a){
					*begin = a.to<value_type>();
					++begin;});
			break;
		}
#endif
#ifdef _128_FLOAT_SUPPORT_
		case DType::Float128:{
			using value_type = float128_t;
			value_type* begin = reinterpret_cast<value_type*>(output.data_ptr());
			utils::flatten_func<Scalar, N>(v, [&begin](const Scalar& a){
					*begin = a.to<value_type>();
					++begin;});
			break;
		}
#endif
#ifdef __SIZEOF_INT128__
		case DType::int128:{
			using value_type = int128_t;
			value_type* begin = reinterpret_cast<value_type*>(output.data_ptr());
			utils::flatten_func<Scalar, N>(v, [&begin](const Scalar& a){
					*begin = a.to<value_type>();
					++begin;});
			break;
		}
		case DType::uint128:{
			using value_type = uint128_t;
			value_type* begin = reinterpret_cast<value_type*>(output.data_ptr());
			utils::flatten_func<Scalar, N>(v, [&begin](const Scalar& a){
					*begin = a.to<value_type>();
					++begin;});
			break;
		}
#endif
		case DType::Complex64:{
			using value_type = complex_64;
			value_type* begin = reinterpret_cast<value_type*>(output.data_ptr());
			utils::flatten_func<Scalar, N>(v, [&begin](const Scalar& a){
					*begin = a.to<value_type>();
					++begin;});
			break;
		}
		case DType::Complex128:{
			using value_type = complex_128;
			value_type* begin = reinterpret_cast<value_type*>(output.data_ptr());
			utils::flatten_func<Scalar, N>(v, [&begin](const Scalar& a){
					*begin = a.to<value_type>();
					++begin;});
			break;
		}
		case DType::int8:{
			using value_type = int8_t;
			value_type* begin = reinterpret_cast<value_type*>(output.data_ptr());
			utils::flatten_func<Scalar, N>(v, [&begin](const Scalar& a){
					*begin = a.to<value_type>();
					++begin;});
			break;
		}
		case DType::uint8:{
			using value_type = uint8_t;
			value_type* begin = reinterpret_cast<value_type*>(output.data_ptr());
			utils::flatten_func<Scalar, N>(v, [&begin](const Scalar& a){
					*begin = a.to<value_type>();
					++begin;});
			break;
		}
		case DType::int16:{
			using value_type = int16_t;
			value_type* begin = reinterpret_cast<value_type*>(output.data_ptr());
			utils::flatten_func<Scalar, N>(v, [&begin](const Scalar& a){
					*begin = a.to<value_type>();
					++begin;});
			break;
		}
		case DType::uint16:{
			using value_type = uint16_t;
			value_type* begin = reinterpret_cast<value_type*>(output.data_ptr());
			utils::flatten_func<Scalar, N>(v, [&begin](const Scalar& a){
					*begin = a.to<value_type>();
					++begin;});
			break;
		}
		case DType::int32:{
			using value_type = int32_t;
			value_type* begin = reinterpret_cast<value_type*>(output.data_ptr());
			utils::flatten_func<Scalar, N>(v, [&begin](const Scalar& a){
					*begin = a.to<value_type>();
					++begin;});
			break;
		}
		case DType::uint32:{
			using value_type = uint32_t;
			value_type* begin = reinterpret_cast<value_type*>(output.data_ptr());
			utils::flatten_func<Scalar, N>(v, [&begin](const Scalar& a){
					*begin = a.to<value_type>();
					++begin;});
			break;
		}
		case DType::int64:{
			using value_type = int64_t;
			value_type* begin = reinterpret_cast<value_type*>(output.data_ptr());
			utils::flatten_func<Scalar, N>(v, [&begin](const Scalar& a){
					*begin = a.to<value_type>();
					++begin;});
			break;
		}
		case DType::Bool:{
			using value_type = uint_bool_t;
			value_type* begin = reinterpret_cast<value_type*>(output.data_ptr());
			utils::flatten_func<Scalar, N>(v, [&begin](const Scalar& a){
					*begin = a.to<value_type>();
					++begin;});
			break;
		}
		case DType::TensorObj:
			break;
	}
	/* output._vals.execute_function<WRAP_DTYPES<NumberTypesL, DTypeEnum<DType::Bool>>>([&v](auto begin, auto end){ */
	/* 	using value_type = typename std::remove_const<typename decltype(begin)::value_type>::type; */
	/* 	utils::flatten_func<Scalar, N>(v, [&begin](const Scalar& a){*begin = a.to<value_type>();}); */
	/* }); */
	return output;
}

template Tensor Tensor::FromInitializer<1ul>(utils::NestedInitializerLists_type<Scalar, 1ul>::type, DType);
template Tensor Tensor::FromInitializer<2ul>(utils::NestedInitializerLists_type<Scalar, 2ul>::type, DType);
template Tensor Tensor::FromInitializer<3ul>(utils::NestedInitializerLists_type<Scalar, 3ul>::type, DType);
template Tensor Tensor::FromInitializer<4ul>(utils::NestedInitializerLists_type<Scalar, 4ul>::type, DType);
template Tensor Tensor::FromInitializer<5ul>(utils::NestedInitializerLists_type<Scalar, 5ul>::type, DType);
template Tensor Tensor::FromInitializer<6ul>(utils::NestedInitializerLists_type<Scalar, 6ul>::type, DType);

Tensor Tensor::clip(Scalar a, Scalar b) const{
	Tensor outp = this->contiguous();
	outp.clip_(a, b);
	return std::move(outp);
}

Tensor Tensor::pad(std::vector<uint32_t> p, const char* mode, double value) const{
	utils::throw_exception(p.size() % 2 == 0, "RuntimeError: The size of the pad must have 2 per dimension");
	utils::throw_exception((p.size() / 2) <= dims(), "RuntimeError: expected padding for at most $ dims but instead got $", dims(), int(p.size() / 2));

	std::vector<uint32_t> n_shape = shape().Vec();
	uint32_t i = 0;
	uint32_t last_dims = uint32_t(p.size() / 2);
	for(; i < (p.size() / 2); ++i){
		n_shape[n_shape.size() - (i+1)] += p[i*2];
		n_shape[n_shape.size() - (i+1)] += p[i*2+1];
	}
	Tensor output(SizeRef(std::move(n_shape)), dtype);
	output = value;
	std::vector<nt::my_range> ranges(dims());
	auto begin = p.cbegin();
	uint32_t start = dims() - int32_t(p.size() / 2);
	for(i = 0; i < dims(); ++i){
		if(i < (start)){
			ranges[i] = nt::my_range(0, shape()[i]);
			continue;
		}
		ranges[i] = nt::my_range(*begin, (-1)*int32_t(*(begin + 1)));
		++begin;
		++begin;
	}
	output[ranges] = *this;
	return std::move(output);
}


Tensor Tensor::flip(int32_t dim) const{
	dim = dim < 0 ? dim + dims() : dim;
	utils::throw_exception(dim < dims() && dim > 0, "RuntimeError: Expected input dim for flip to be less than $ but got $", dims(), dim);
	Tensor output(shape(), dtype);
	const Tensor my_tensors = this->split_axis(dim);
	Tensor out_tensors = output.split_axis(dim);
	tdtype_list<const Tensor> begin = my_tensors.arr_void().tcbegin<Tensor>();
	Tensor* outp = reinterpret_cast<Tensor*>(out_tensors.data_ptr());
	if(dim > 0){
		uint32_t a = dim == (dims() - 1) ? shape().transpose(-1,-2).flatten(0,-3)[0] : shape().flatten(0,dim-1)[0];
		uint32_t b = uint32_t(out_tensors.shape()[0] / a);
		std::cout << a << "," << b << std::endl;
		for(uint32_t i = 0; i < a; ++i, outp += b){
			for(int32_t j = b-1; j >= 0; --j, ++begin){
				*(outp + j) = *begin;
			}
		}
		return output;
	}
	for(int32_t i = out_tensors.numel()-1; i >= 0; --i, ++begin){
		outp[i] = *begin;
	}
	return output;
}


Tensor Tensor::flip() const {
	Tensor output(shape(), dtype);
	_vals.cexecute_function([](auto begin, auto end, ArrayVoid& v, const uint32_t& numel){
				using value_t = typename std::remove_const<typename decltype(begin)::value_type>::type;
				auto v_begin = v.tbegin<value_t>();
				for(int32_t i = numel-1; i >= 0; --i, ++begin){
					*(v_begin + i) = *begin;
				}
			}, output._vals, output.numel());
	return output;
}

Tensor Tensor::flip_(){
	ArrayVoid cpy = _vals.copy_strides(false);
	void** my_strides = _vals.strides_begin();
	void** out_strides = cpy.strides_begin();
	for(int32_t i = numel()-1; i >= 0; --i, ++my_strides){
		out_strides[i] = *my_strides;
	}
	return Tensor(cpy, shape());
}


//there is an issue with dim = -1
//it seems as though maybe the transpose in split axis isn't it letting it access the exact pointer values?
//Which does not make sense
//Wait, no, that does make sense
//this is because the permute function automatically makes a new stride in memory if the use_count > 1
//therefore, the solution would be to get the RowColSwap function working again (which needs to happen anyways)
Tensor Tensor::flip_(int32_t dim){
	dim = dim < 0 ? dim + dims() : dim;
	utils::throw_exception(dim < dims() && dim > 0, "RuntimeError: Expected input dim for flip to be less than $ but got $", dims(), dim);
	Tensor output(_vals.copy_strides(false), shape());
	Tensor my_tensors = (dim == (dims()-1)) ? this->RowColSwap().split_axis(-2) : this->split_axis(dim);
	Tensor out_tensors = (dim == (dims()-1)) ? output.RowColSwap().split_axis(-2) : output.split_axis(dim);
	tdtype_list<Tensor> begin = my_tensors.arr_void().tbegin<Tensor>();
	Tensor* outp = reinterpret_cast<Tensor*>(out_tensors.data_ptr());
	if(dim > 0){
		uint32_t a = dim == (dims() - 1) ? shape().flatten(0,-3)[0] : shape().flatten(0,dim-1)[0];
		uint32_t b = uint32_t(out_tensors.shape()[0] / a);
		for(uint32_t i = 0; i < a; ++i, outp += b){
			for(int32_t j = b-1; j >= 0; --j, ++begin){
				Tensor& from = *begin;
				Tensor& to = (*(outp + j));
				void** to_strides = to._vals.strides_begin();
				void** to_strides_e = to._vals.strides_end();
				void** from_strides = from._vals.strides_begin();
				void** from_strides_e = from._vals.strides_end();
				for(;from_strides != from_strides_e & to_strides != to_strides_e; ++from_strides, ++to_strides)
					*to_strides = *from_strides;
			}
		}
		if(dim == (dims()-1)){
			this->RowColSwap();
			output.RowColSwap();
		}
		return output;
	}
	for(int32_t i = out_tensors.numel()-1; i >= 0; --i, ++begin){
		Tensor& to = outp[i];
		Tensor& from = (*begin);
		void** to_strides = to._vals.strides_begin();
		void** from_strides = from._vals.strides_begin();
		void** from_strides_e = from._vals.strides_end();
		for(;from_strides != from_strides_e; ++from_strides, ++to_strides)
			*to_strides = *from_strides;
	}
	if(dim == (dims()-1)){
		this->RowColSwap();
		output.RowColSwap();
	}
	return output;
	
}


Tensor Tensor::dilate_(uint32_t dil) const{
	std::vector<uint32_t> vec = shape().Vec();
	/* dil -= 1; */
	vec.back() *= dil;
	vec.back() -= (dil-1);
	vec[vec.size()-2] *= dil;
	vec[vec.size()-2] -= (dil-1);
	Tensor outp = functional::zeros(SizeRef(vec), dtype);
	void** outp_strides = outp._vals.strides_begin();
	void** my_strides = _vals.strides_cbegin();
	uint32_t cols = shape()[-1];
	uint32_t i_total = shape().multiply(-2);
	for(uint32_t i = 0; i < numel(); ++i, ++my_strides){
		*outp_strides = *my_strides;
		if((i+1) % cols == 0){
			if((i+1) % i_total == 0){outp_strides += 1; continue;}
			outp_strides += outp.shape().back()+(dil-1);continue;
		}
		outp_strides += dil;


	}
	return outp;
}

//this version only makes 1 extra value in the amount of a certain element
//then it only expands the size of the actual void** ptr
Tensor Tensor::dilate_mem_(uint32_t dil) const{
	std::vector<uint32_t> vec = shape().Vec();
	/* dil -= 1; */
	vec.back() *= dil;
	vec.back() -= (dil-1);
	vec[vec.size()-2] *= dil;
	vec[vec.size()-2] -= (dil-1);
	ArrayVoid outp_arr_b(1, dtype);
	outp_arr_b = 1;
	SizeRef outp_shape(std::move(vec));
	ArrayVoid outp_arr = outp_arr_b.new_stride(outp_shape.multiply());
	void** outp_strides = outp_arr.strides_begin();
	void** my_strides = _vals.strides_cbegin();
	uint32_t cols = shape()[-1];
	uint32_t i_total = shape().multiply(-2);
	void* ptr = outp_arr.data_ptr();
	for(uint32_t i = 0; i < numel(); ++i, ++my_strides){
		*outp_strides = *my_strides;
		if((i+1) % cols == 0){
			if((i+1) % i_total == 0){
				*outp_strides = ptr;
				outp_strides += 1; 
				continue;
			}
			for(uint32_t j = 0; j < outp_shape.back() + (dil - 1); ++j, ++outp_strides){
				*outp_strides = ptr;
			}
			continue;
		}
		outp_strides += dil;
	}
	return Tensor(outp_arr, std::move(outp_shape));

	
}

Tensor Tensor::dilate(uint32_t dil) const {return dilate_(dil).contiguous();}

};


