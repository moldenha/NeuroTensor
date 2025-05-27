#include "Tensor.h"
#include "dtype/ArrayVoid.h"
#include "dtype/DType.h"
#include "dtype/DType_enum.h"
#include "dtype/ranges.h"
#include "functional/functional.h"
#include "memory/iterator.h"
#include "refs/SizeRef.h"

#include <functional>
#include <algorithm>
#include <set>
#include <memory>
#include <numeric>
#include <ratio>
#include <sys/wait.h>

#include <cassert>
#include "dtype/ArrayVoid.hpp"
#include "dtype/Scalar.h"
#include "types/Types.h"
#include "utils/utils.h"
#include <cmath>
#include <set>
#include <type_traits>
#include <vector>
#include "utils/macros.h"



#define _NT_HANDLE_NULL_TENSORS_NON_EMPTY_4_(T1, T2, T3, T4) \
    utils::THROW_EXCEPTION(!is_null() && !T4.is_null() && !T3.is_null() && !T2.is_null() && !T1.is_null(), \
                           "Cannot perform operation $"\
                           " on a null tensor", __NT_FUNCTION_NAME__);

#define _NT_HANDLE_NULL_TENSORS_NON_EMPTY_3_(T1, T2, T3) \
    utils::THROW_EXCEPTION(!is_null() && !T3.is_null() && !T2.is_null() && !T1.is_null(), \
                           "Cannot perform operation $"\
                           " on a null tensor", __NT_FUNCTION_NAME__);

#define _NT_HANDLE_NULL_TENSORS_NON_EMPTY_2_(T1, T2) \
    utils::THROW_EXCEPTION(!is_null() && !T2.is_null() && !T1.is_null(), \
                           "Cannot perform operation $"\
                           " on a null tensor", __NT_FUNCTION_NAME__);


#define _NT_HANDLE_NULL_TENSORS_NON_EMPTY_1_(T1) \
    utils::THROW_EXCEPTION(!is_null() && !T1.is_null(), \
                           "Cannot perform operation $"\
                           " on a null tensor", __NT_FUNCTION_NAME__);

#define _NT_HANDLE_NULL_TENSORS_NON_EMPTY_0_() \
    utils::THROW_EXCEPTION(!is_null(), \
                           "Cannot perform operation $"\
                           " on a null tensor", __NT_FUNCTION_NAME__);



#define _NT_HANDLE_NULL_TENSORS_EMPTY_1(...) utils::THROW_EXCEPTION(!is_null(), "Cannot perform operation $" \
                                                                        " on a null tensor", __NT_FUNCTION_NAME__);
#define _NT_HANDLE_NULL_TENSORS_NON_EMPTY_SELECT_(_1, _2, _3, _4, NAME, ...) NAME
#define _NT_HANDLE_NULL_TENSORS_EMPTY_0(...) _NT_HANDLE_NULL_TENSORS_NON_EMPTY_SELECT_(__VA_ARGS__, _NT_HANDLE_NULL_TENSORS_NON_EMPTY_4_, _NT_HANDLE_NULL_TENSORS_NON_EMPTY_3_, _NT_HANDLE_NULL_TENSORS_NON_EMPTY_2_, _NT_HANDLE_NULL_TENSORS_NON_EMPTY_1_, _NT_HANDLE_NULL_TENSORS_NON_EMPTY_0_)(__VA_ARGS__)

#define __NT_HANDLE_NULL_TENSORS__(...) _NT_GLUE_(_NT_HANDLE_NULL_TENSORS_EMPTY_, _NT_IS_EMPTY_(__VA_ARGS__))(__VA_ARGS__);

#define __NT_HANDLE_MUTABILITY__() \
    utils::THROW_EXCEPTION(_is_mutable, "Cannot perform operation $" \
                                        " on an immutable tensor", __NT_FUNCTION_NAME__);

namespace nt {

DType specify_dtype_from_scalar(const Scalar &s) {
    if (s.isComplex())
        return DType::Complex64;
    if (s.isFloatingPoint())
        return DType::Float32;
    return s.type();
}

Tensor::Tensor(DType dt)
    : _vals(1, dt), _total_size(1), _size({1}), dtype(dt),
      stored_strides(nullptr), _is_mutable(true) {}

Tensor::Tensor(SizeRef v, DType dt)
    : _vals(v.unsigned_multiply(), dt), _size(std::move(v)), dtype(dt),
      stored_strides(nullptr), _is_mutable(true) {
    _total_size = _vals.Size();
}

Tensor::Tensor(ArrayVoid ptr, SizeRef v)
    : _vals(ptr), _size(std::move(v)), dtype(ptr.dtype),
      _total_size(ptr.Size()), stored_strides(nullptr), _is_mutable(true) {
    /* std::cout << "setting dtype"<<std::endl; */
    _total_size = _vals.Size();
    dtype = _vals.dtype;
    /* std::cout << "dtype set"<<std::endl; */
}
Tensor::Tensor(std::string_view _sv)
    : _vals(_sv.size(), DType::uint8),
      _size({static_cast<SizeRef::value_type>(_sv.size())}),
      dtype(DType::uint8), _total_size(_sv.size()), stored_strides(nullptr), _is_mutable(true) {
    char *begin = reinterpret_cast<char *>(data_ptr());
    std::transform(_sv.cbegin(), _sv.cend(), begin,
                   [](const char &v) { return v; });
}

Tensor::Tensor(std::nullptr_t)
    : dtype(nt::DType::Float32), _vals(nullptr), _size(nullptr), _total_size(0),
      stored_strides(nullptr), _is_mutable(false) {}


Tensor::Tensor(const Tensor &t)
    : _vals(t._vals), _total_size(t._total_size), _size(t._size),
      dtype(t.dtype), stored_strides(t.stored_strides), _is_mutable(t._is_mutable) {}

Tensor::Tensor(Tensor &&t)
    : _vals(std::move(t._vals)), _total_size(t._total_size),
      _size(std::move(t._size)), dtype(t.dtype),
      stored_strides(std::move(t.stored_strides)), _is_mutable(t._is_mutable) {}

Tensor::Tensor(Scalar s)
    : _vals(1, specify_dtype_from_scalar(s)), _total_size(1), _size({1}),
      dtype(specify_dtype_from_scalar(s)),
      stored_strides(nullptr), _is_mutable(true){
    if (s.isZero()) {
        _vals.fill_(0);
    } else {
        *this = s;
    }
}



/* template<> */
/* Tensor::Tensor(typename utils::NestedInitializerLists_type<Scalar, 1>::type
 * v, DType dt) */
/* 	:_vals(SizeRef(utils::aquire_shape<Scalar, 1>(v)).multiply(), dt), */
/* 	_size(std::make_unique<SizeRef>(utils::aquire_shape<Scalar, 1>(v))), */
/* 	dtype(dt) */
/* { */
/* 	_total_size = _vals.Size(); */
/* 	_vals.execute_function<WRAP_DTYPES<NumberTypesL,
 * DTypeEnum<DType::Bool>>>([&v](auto begin, auto end){ */
/* 				using value_type = typename
 * std::remove_const<typename decltype(begin)::value_type>::type; */
/* 				utils::flatten_func<Scalar, 1>(v, [&begin](const
 * Scalar& a){*begin = a.to<value_type>();}); */
/* 			}); */
/* } */

/* template Tensor<1>::Tensor(typename
 * utils::NestedInitializerLists_type<Scalar, 1>::type, DType); */
/* Tensor::Tensor<2>(typename utils::NestedInitializerLists_type<Scalar,
 * 2>::type, DType); */
/* Tensor::Tensor<3>(typename utils::NestedInitializerLists_type<Scalar,
 * 3>::type, DType); */
/* Tensor::Tensor<4>(typename utils::NestedInitializerLists_type<Scalar,
 * 4>::type, DType); */
/* Tensor::Tensor<5>(typename utils::NestedInitializerLists_type<Scalar,
 * 5>::type, DType); */
/* Tensor::Tensor<6>(typename utils::NestedInitializerLists_type<Scalar,
 * 6>::type, DType); */
/* Tensor::Tensor<7>(typename utils::NestedInitializerLists_type<Scalar,
 * 7>::type, DType); */
/* Tensor::Tensor<8>(typename utils::NestedInitializerLists_type<Scalar,
 * 8>::type, DType); */
/* Tensor::Tensor<9>(typename utils::NestedInitializerLists_type<Scalar,
 * 9>::type, DType); */
/* Tensor::Tensor<10>(typename utils::NestedInitializerLists_type<Scalar,
 * 10>::type, DType); */
/* Tensor::Tensor<11>(typename utils::NestedInitializerLists_type<Scalar,
 * 11>::type, DType); */

void Tensor::swap(Tensor &other) {
    // utils::throw_exception(_is_mutable && other._is_mutable,
    //                     "Cannot swap immutable tensors");
    _vals.swap(other._vals);
    std::swap(_total_size, other._total_size);
    _size.swap(other._size);
    std::swap(dtype, other.dtype);
    std::swap(stored_strides, other.stored_strides);
    std::swap(_is_mutable, other._is_mutable);
}

Tensor &Tensor::operator=(const Tensor &t) {
    if (is_null()) {
        _vals = t._vals;
        _size = t._size;
        _total_size = t._total_size;
        dtype = t.dtype;
        stored_strides = t.stored_strides;
        _is_mutable = t._is_mutable;
        return *this;
    }
    __NT_HANDLE_MUTABILITY__();
    if (shape() == t.shape() && dtype == t.dtype) {
        return functional::set_(*this, t);
    }
    if (dtype == DType::TensorObj && _total_size == 1) {
        *reinterpret_cast<Tensor *>(data_ptr()) = t;
        return *this;
    }
    _vals = t._vals;
    _size = t._size;
    _total_size = t._total_size;
    dtype = t.dtype;
    stored_strides = t.stored_strides;
    return *this;
}

Tensor &Tensor::set_(const Tensor &t) {return functional::set_(*this, t);}


Tensor &Tensor::operator=(Tensor &&t) {
    if(is_null()){
        _vals = std::move(t._vals);
        _size = std::move(t._size);
        dtype = t.dtype;
        _total_size = t._total_size;
        stored_strides = std::move(t.stored_strides);
        _is_mutable = t._is_mutable;
        return *this;
    }
    if (dtype == DType::TensorObj && this->is_sub_tensor() && _total_size == 1) {
        *reinterpret_cast<Tensor *>(data_ptr()) = std::move(t);
        return *this;
    }
    // __NT_HANDLE_NULL_TENSORS__(t);
    __NT_HANDLE_MUTABILITY__();
    _vals = std::move(t._vals);
    _size = std::move(t._size);
    dtype = t.dtype;
    _total_size = t._total_size;
    stored_strides = std::move(t.stored_strides);
    return *this;
}

Tensor &Tensor::operator++() { return functional::add_(*this, 1); }

Tensor &Tensor::operator=(Scalar val) { return functional::fill_(*this, val); }

Tensor &Tensor::operator+=(Scalar val) { return functional::add_(*this, val); }
Tensor &Tensor::operator+=(const Tensor &val) {
    __NT_HANDLE_NULL_TENSORS__(val);
    __NT_HANDLE_MUTABILITY__();
    if (val.numel() == 1 && val.dtype != DType::TensorObj) {
        return (*this) += val.toScalar();
    }
    return functional::add_(*this, val);
}


Tensor Tensor::operator+(Scalar val) const { return functional::add(*this, val); }
Tensor Tensor::operator+(const Tensor &val) const {
    __NT_HANDLE_NULL_TENSORS__(val);
    if (val.numel() == 1 && val.dtype != DType::TensorObj) {
        return (*this) + val.toScalar();
    }
    return functional::add(*this, val);
}


Tensor &Tensor::operator-=(Scalar val) { return functional::subtract_(*this, val); }
Tensor &Tensor::operator-=(const Tensor &val) {
    __NT_HANDLE_NULL_TENSORS__(val);
    __NT_HANDLE_MUTABILITY__();
    if (val.numel() == 1 && val.dtype != DType::TensorObj) {
        return (*this) -= val.toScalar();
    }
    return functional::subtract_(*this, val);
}
Tensor Tensor::operator-(Scalar val) const {return functional::subtract(*this, val);}
Tensor Tensor::operator-(const Tensor &val) const {
    __NT_HANDLE_NULL_TENSORS__(val);
    return functional::subtract(*this, val);
}

// s/;/{\r\tTensor outp = this->contiguous();\r\toutp._multiply(val);\r\treturn
// std::move(outp);\r}

Tensor Tensor::operator*(Scalar val) const {return functional::multiply(*this, val);}

Tensor Tensor::operator*(const Tensor &a) const {
    __NT_HANDLE_NULL_TENSORS__(a);
    if (a.numel() == 1 && a.dtype != DType::TensorObj) {
        return (*this) * a.toScalar();
    }
    return functional::hadamard_multiply(*this, a);
}

// s/;/{\r\t_multiply(val);\r\treturn *this;\r}
Tensor &Tensor::operator*=(Scalar val) { return functional::multiply_(*this, val); }

Tensor &Tensor::operator*=(const Tensor &a) {
    __NT_HANDLE_NULL_TENSORS__(a);
    __NT_HANDLE_MUTABILITY__();
    if (a.numel() == 1 && a.dtype != DType::TensorObj) {
        return (*this) *= a.toScalar();
    }
    functional::hadamard_multiply_this(*this, a);
    return *this;
}



Tensor Tensor::operator/(Scalar val) const {return functional::divide(*this, val); }

Tensor Tensor::operator/(const Tensor &val) const {
    __NT_HANDLE_NULL_TENSORS__(val);
    if (val.numel() == 1 && val.dtype != DType::TensorObj) {
        return (*this) / val.toScalar();
    }
    return functional::divide(*this, val);
}

Tensor &Tensor::operator/=(Scalar val) {return functional::divide_(*this, val);}
Tensor &Tensor::operator/=(const Tensor &val) {
    __NT_HANDLE_NULL_TENSORS__(val);
    __NT_HANDLE_MUTABILITY__();
    if (val.numel() == 1 && val.dtype != DType::TensorObj) {
        return (*this) /= val.toScalar();
    }
    return functional::divide_(*this, val);
}

Tensor Tensor::operator-() const { return *this * -1; }
Tensor operator+(Scalar s, const Tensor &t) { return t + s; }
Tensor operator-(Scalar s, const Tensor &t) { return -t + s; }
Tensor operator*(Scalar s, const Tensor &t) { return t * s; }
Tensor operator/(Scalar s, const Tensor &t) {
    Tensor a = t.inverse();
    a *= s;
    return a;
}

Tensor Tensor::operator==(const Tensor &val) const {return functional::equal(*this, val);}
Tensor Tensor::operator!=(const Tensor &val) const {return functional::not_equal(*this, val);}
Tensor Tensor::operator>=(const Tensor &val) const {return functional::greater_than_equal(*this, val);}
Tensor Tensor::operator<=(const Tensor &val) const {return functional::less_than_equal(*this, val);}
Tensor Tensor::operator&&(Tensor val) const {return functional::and_op(*this, val);}
Tensor Tensor::operator||(Tensor val) const {return functional::or_op(*this, val);}
Tensor Tensor::operator>(const Tensor &val) const {return functional::greater_than(*this, val);}
Tensor Tensor::operator<(const Tensor &val) const {return functional::less_than(*this, val);}

Tensor Tensor::contiguous() const {
    __NT_HANDLE_NULL_TENSORS__();
    if (is_contiguous())
        return *this;
    /* std::copy((const float*)_vals.get(), (const float*)(_vals.get() +
     * _total_size), copy.get()); */
    return Tensor(_vals.contiguous(), _size);
}

Tensor Tensor::clone() const { __NT_HANDLE_NULL_TENSORS__(); return Tensor(_vals.clone(), _size); }
Tensor Tensor::conditional_mutate_clone() const {return this->_is_mutable ? clone() : *this;}

//this function is really in place to make the user fully aware they are potentially altering a tensor
//that has been marked as immutable
Tensor& Tensor::force_mutable_function(std::function<void(Tensor&)> func){
    if(this->_is_mutable){func(*this); return *this;}
    this->_is_mutable = true;
    func(*this);
    this->_is_mutable = false;
    return *this;
}

const size_t Tensor::dims() const { return shape().size(); }

template <typename T> T &Tensor::item() {
    __NT_HANDLE_NULL_TENSORS__();
    assert(_total_size == 1);
    T *casted = reinterpret_cast<T *>(data_ptr());
    return *(casted);
}

template float &Tensor::item<float>();
template double &Tensor::item<double>();
template int64_t &Tensor::item<int64_t>();
template int32_t &Tensor::item<int32_t>();
template uint32_t &Tensor::item<uint32_t>();
template int16_t &Tensor::item<int16_t>();
template uint16_t &Tensor::item<uint16_t>();
template int8_t &Tensor::item<int8_t>();
template uint8_t &Tensor::item<uint8_t>();
template Tensor &Tensor::item<Tensor>();
template uint_bool_t &Tensor::item<uint_bool_t>();
template bool &Tensor::item<bool>();
template complex_64 &Tensor::item<complex_64>();
template complex_128 &Tensor::item<complex_128>();
#ifdef __SIZEOF_INT128__
template uint128_t &Tensor::item<uint128_t>();
template int128_t &Tensor::item<int128_t>();
#endif
#ifdef _HALF_FLOAT_SUPPORT_
template float16_t &Tensor::item<float16_t>();
template complex_32 &Tensor::item<complex_32>();
#endif
#ifdef _128_FLOAT_SUPPORT_
template float128_t &Tensor::item<float128_t>();
#endif

template <typename T> const T &Tensor::item() const {
    assert(_total_size == 1);
    const T *casted = reinterpret_cast<const T *>(data_ptr());
    return *(casted);
}

template const float &Tensor::item<float>() const;
template const double &Tensor::item<double>() const;
template const int64_t &Tensor::item<int64_t>() const;
template const int32_t &Tensor::item<int32_t>() const;
template const uint32_t &Tensor::item<uint32_t>() const;
template const int16_t &Tensor::item<int16_t>() const;
template const uint16_t &Tensor::item<uint16_t>() const;
template const int8_t &Tensor::item<int8_t>() const;
template const uint8_t &Tensor::item<uint8_t>() const;
template const Tensor &Tensor::item<Tensor>() const;
template const uint_bool_t &Tensor::item<uint_bool_t>() const;
template const bool &Tensor::item<bool>() const;
template const complex_64 &Tensor::item<complex_64>() const;
template const complex_128 &Tensor::item<complex_128>() const;
#ifdef __SIZEOF_INT128__
template const uint128_t &Tensor::item<uint128_t>() const;
template const int128_t &Tensor::item<int128_t>() const;
#endif
#ifdef _HALF_FLOAT_SUPPORT_
template const float16_t &Tensor::item<float16_t>() const;
template const complex_32 &Tensor::item<complex_32>() const;
#endif
#ifdef _128_FLOAT_SUPPORT_
template const float128_t &Tensor::item<float128_t>() const;
#endif




template<typename T>
T& Tensor::item(const std::vector<size_value_t>& vec){
    __NT_HANDLE_NULL_TENSORS__();
    if(vec.size() == 0){return item<T>();}
    utils::throw_exception(vec.size() <= dims(), "Expected to get at most $ indices for item but got $", dims(), vec.size());

    uint64_t cur_mult = 1;
    int64_t counter = 0;
    const auto& sh = shape();
    for(const size_value_t& val : vec){
        cur_mult *= val;
        utils::throw_exception(sh[counter] > val && val >= 0, "Expected indices for item to be positive and within the range of $ but got $", sh[counter], val);
        ++counter;

    }
    uint64_t mult = vec.size() == dims() ? 1 : static_cast<uint64_t>(sh.multiply(counter));
    mult *= cur_mult;
    return *(_vals.execute_function<WRAP_DTYPES<AllTypesL>>(
        [&mult](auto begin, auto end) -> T* {
            return reinterpret_cast<T*>(&(*(begin + mult)));
        }));
}

template float &Tensor::item<float>(const std::vector<Tensor::size_value_t> &);
template double &Tensor::item<double>(const std::vector<Tensor::size_value_t> &);
template int64_t &Tensor::item<int64_t>(const std::vector<Tensor::size_value_t> &);
template int32_t &Tensor::item<int32_t>(const std::vector<Tensor::size_value_t> &);
template uint32_t &Tensor::item<uint32_t>(const std::vector<Tensor::size_value_t> &);
template int16_t &Tensor::item<int16_t>(const std::vector<Tensor::size_value_t> &);
template uint16_t &Tensor::item<uint16_t>(const std::vector<Tensor::size_value_t> &);
template int8_t &Tensor::item<int8_t>(const std::vector<Tensor::size_value_t> &);
template uint8_t &Tensor::item<uint8_t>(const std::vector<Tensor::size_value_t> &);
template Tensor &Tensor::item<Tensor>(const std::vector<Tensor::size_value_t> &);
template uint_bool_t &Tensor::item<uint_bool_t>(const std::vector<Tensor::size_value_t> &);
template bool &Tensor::item<bool>(const std::vector<Tensor::size_value_t> &);
template complex_64 &Tensor::item<complex_64>(const std::vector<Tensor::size_value_t> &);
template complex_128 &Tensor::item<complex_128>(const std::vector<Tensor::size_value_t> &);
#ifdef __SIZEOF_INT128__
template uint128_t &Tensor::item<uint128_t>(const std::vector<Tensor::size_value_t> &);
template int128_t &Tensor::item<int128_t>(const std::vector<Tensor::size_value_t> &);
#endif
#ifdef _HALF_FLOAT_SUPPORT_
template float16_t &Tensor::item<float16_t>(const std::vector<Tensor::size_value_t> &);
template complex_32 &Tensor::item<complex_32>(const std::vector<Tensor::size_value_t> &);
#endif
#ifdef _128_FLOAT_SUPPORT_
template float128_t &Tensor::item<float128_t>(const std::vector<Tensor::size_value_t> &);
#endif

template<typename T>
const T& Tensor::item(const std::vector<size_value_t>& vec) const{
    __NT_HANDLE_NULL_TENSORS__();
    if(vec.size() == 0){return item<T>();}
    utils::throw_exception(vec.size() <= dims(), "Expected to get at most $ indices for item but got $", dims(), vec.size());

    uint64_t cur_mult = 1;
    int64_t counter = 0;
    const auto& sh = shape();
    for(const size_value_t& val : vec){
        cur_mult *= val;
        utils::throw_exception(sh[counter] > val && val >= 0, "Expected indices for item to be positive and within the range of $ but got $", sh[counter], val);
        ++counter;

    }
    uint64_t mult = vec.size() == dims() ? 1 : static_cast<uint64_t>(sh.multiply(counter));
    mult *= cur_mult;
    return *(_vals.cexecute_function<WRAP_DTYPES<AllTypesL>>(
        [&mult](auto begin, auto end) -> const T* {
            return reinterpret_cast<const T*>(&(*(begin + mult)));
        }));
}

template const float &Tensor::item<float>(const std::vector<Tensor::size_value_t> &) const;
template const double &Tensor::item<double>(const std::vector<Tensor::size_value_t> &) const;
template const int64_t &Tensor::item<int64_t>(const std::vector<Tensor::size_value_t> &) const;
template const int32_t &Tensor::item<int32_t>(const std::vector<Tensor::size_value_t> &) const;
template const uint32_t &Tensor::item<uint32_t>(const std::vector<Tensor::size_value_t> &) const;
template const int16_t &Tensor::item<int16_t>(const std::vector<Tensor::size_value_t> &) const;
template const uint16_t &Tensor::item<uint16_t>(const std::vector<Tensor::size_value_t> &) const;
template const int8_t &Tensor::item<int8_t>(const std::vector<Tensor::size_value_t> &) const;
template const uint8_t &Tensor::item<uint8_t>(const std::vector<Tensor::size_value_t> &) const;
template const Tensor &Tensor::item<Tensor>(const std::vector<Tensor::size_value_t> &) const;
template const uint_bool_t &Tensor::item<uint_bool_t>(const std::vector<Tensor::size_value_t> &) const;
template const bool &Tensor::item<bool>(const std::vector<Tensor::size_value_t> &) const;
template const complex_64 &Tensor::item<complex_64>(const std::vector<Tensor::size_value_t> &) const;
template const complex_128 &Tensor::item<complex_128>(const std::vector<Tensor::size_value_t> &) const;
#ifdef __SIZEOF_INT128__
template const uint128_t &Tensor::item<uint128_t>(const std::vector<Tensor::size_value_t> &) const;
template const int128_t &Tensor::item<int128_t>(const std::vector<Tensor::size_value_t> &) const;
#endif
#ifdef _HALF_FLOAT_SUPPORT_
template const float16_t &Tensor::item<float16_t>(const std::vector<Tensor::size_value_t> &) const;
template const complex_32 &Tensor::item<complex_32>(const std::vector<Tensor::size_value_t> &) const;
#endif
#ifdef _128_FLOAT_SUPPORT_
template const float128_t &Tensor::item<float128_t>(const std::vector<Tensor::size_value_t> &) const;
#endif

Scalar Tensor::toScalar() const {
    __NT_HANDLE_NULL_TENSORS__();
    switch (dtype) {
    case DType::Integer:
        return Scalar(reinterpret_cast<const int32_t *>(_vals.data_ptr())[0]);
    case DType::Float:
        return Scalar(reinterpret_cast<const float *>(_vals.data_ptr())[0]);
    case DType::Double:
        return Scalar(reinterpret_cast<const double *>(_vals.data_ptr())[0]);
    case DType::Long:
        return Scalar(reinterpret_cast<const uint32_t *>(_vals.data_ptr())[0]);
    case DType::Complex64:
        return Scalar(
            reinterpret_cast<const complex_64 *>(_vals.data_ptr())[0]);
    case DType::Complex128:
        return Scalar(
            reinterpret_cast<const complex_128 *>(_vals.data_ptr())[0]);
    case DType::uint8:
        return Scalar(reinterpret_cast<const uint8_t *>(_vals.data_ptr())[0]);
    case DType::int8:
        return Scalar(reinterpret_cast<const int8_t *>(_vals.data_ptr())[0]);
    case DType::int16:
        return Scalar(reinterpret_cast<const int16_t *>(_vals.data_ptr())[0]);
    case DType::uint16:
        return Scalar(reinterpret_cast<const uint16_t *>(_vals.data_ptr())[0]);
    case DType::LongLong:
        return Scalar(reinterpret_cast<const int64_t *>(_vals.data_ptr())[0]);
    case DType::Bool:
        return Scalar(
            reinterpret_cast<const uint_bool_t *>(_vals.data_ptr())[0]);
    case DType::TensorObj:
        return Scalar(0);
#ifdef _128_FLOAT_SUPPORT_
    case DType::Float128:
        return Scalar(
            reinterpret_cast<const float128_t *>(_vals.data_ptr())[0]);
#endif
#ifdef _HALF_FLOAT_SUPPORT_
    case DType::Float16:
        return Scalar(reinterpret_cast<const float16_t *>(_vals.data_ptr())[0]);
    case DType::Complex32:
        return Scalar(
            reinterpret_cast<const complex_32 *>(_vals.data_ptr())[0]);
#endif
#ifdef __SIZEOF_INT128__
    case DType::int128:
        return Scalar(reinterpret_cast<const int128_t *>(_vals.data_ptr())[0]);
    case DType::uint128:
        return Scalar(reinterpret_cast<const uint128_t *>(_vals.data_ptr())[0]);
#endif
    }
    return Scalar();
}

const SizeRef &Tensor::shape() const { return _size; }

Tensor Tensor::operator[](size_value_t x) {return functional::at(*this, x);}
const Tensor Tensor::operator[](size_value_t x) const { return functional::at(*this, x);}
Tensor Tensor::operator[](const Tensor &t) const { return functional::at(*this, t);}

Tensor Tensor::index_except(int64_t dim, int64_t excluding_index) const { return functional::index_except(*this, dim, excluding_index);}

// going to re-think the following,
// just going to go along the guise of pop_back
// while (range.back().end == shape[range.size()-1]-1 && range.back().begin = 0
// && range.size() > 0){range.pop_back();} then going to split axis (2 different
// modes per if range.size() == dims()): if(range.size() != dims()){
//      Tensor splitting = this->split_axis(range.size());
//      Tensor outp = Tensor::makeNullArray( *multiply ranges* );
//      then place the correct splitting into the output
// }
//

inline bool idx_is_total(const my_range &range, const SizeRef &shape,
                         const size_t idx) noexcept {
    return range.begin == 0 && range.end == shape[idx];
}

Tensor Tensor::operator[](std::vector<my_range> r) {return functional::op_range(*this, r); }
const Tensor Tensor::operator[](std::vector<my_range> r) const {return functional::op_range(*this, r); }
const Tensor Tensor::operator[](std::vector<size_value_t> xs) const {return functional::at(*this, std::move(xs));}
Tensor Tensor::operator[](std::vector<size_value_t> xs) {return functional::at(*this, std::move(xs));}


// this was the old way which was not nearly as efficient, but outlines how it works:
/*
Tensor Tensor::operator[](std::vector<my_range> r){
        for(uint32_t i = 0; i < r.size(); ++i){
                r[i].fix(shape()[i]);
        }
        std::vector<uint32_t> stri = strides();
        std::vector<uint64_t> n(numel());
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
                        n.erase(n.cbegin() + (size * q), n.cbegin() + (size * q)
+ start); n.erase(n.cbegin() + (size * (q+1)), n.cbegin() + (size * (q+1)) +
left);
                }
                n.erase(n.cbegin() + (size * (q)), n.cbegin() + (size * (q)) +
start); n.erase(n.cbegin() + (size * (r_length)), n.end()); r_length *=
r[i].length();
        }
        std::vector<uint32_t> n_shape = shape().Vec();
        for(uint32_t i = 0; i < r.size(); ++i)
                n_shape[i] = r[i].length();

        return Tensor(_vals.change_stride(n), SizeRef(std::move(n_shape)));
}
*/

//

Tensor Tensor::operator[](const my_range &x) { return nt::functional::op_range(*this, x);}
const Tensor Tensor::operator[](const my_range &x) const { return nt::functional::op_range(*this, x); }

void Tensor::print() const { std::cout << *this << std::endl; }

std::ostream &operator<<(std::ostream &out, const Tensor &_t) {
    return functional::print(out, _t);
}

Tensor Tensor::view(SizeRef nv) const {
    __NT_HANDLE_NULL_TENSORS__();
    size_value_t total = nv.multiply();
    utils::THROW_EXCEPTION(
        total == _total_size,
        "problem with converting shape $ to shape $ my numel is $ and shape numel is $", shape(), nv,
        _total_size, total);
    /* assert(total == _total_size); */
    return Tensor(_vals, std::move(nv)).set_mutability(_is_mutable);
}

// this is a view to happen to every single tensor
Tensor Tensor::view_Tensors(SizeRef nv) const {
    __NT_HANDLE_NULL_TENSORS__();
    utils::THROW_EXCEPTION(
        dtype == DType::TensorObj,
        "Expected view_Tensors to be used on a tensor of tensors");
    Tensor outp = Tensor::makeNullTensorArray(numel());
    Tensor *outputIt = reinterpret_cast<Tensor *>(outp.data_ptr());
    _vals.transform_function<DType::TensorObj>(
        [&nv](auto &inp) { return inp.view(nv); }, outputIt);
    outp.set_mutability(_is_mutable);
    return std::move(outp);
}


Tensor Tensor::view_Tensor_vector(std::vector<size_value_t> nv) const {
    __NT_HANDLE_NULL_TENSORS__();
    utils::THROW_EXCEPTION(
        dtype == DType::TensorObj,
        "Expected view_Tensors to be used on a tensor of tensors");
    Tensor outp = Tensor::makeNullTensorArray(numel());
    size_value_t n = 1;
    bool is_neg = false;
    size_value_t neg_index = 0;
    for (size_value_t i = 0; i < nv.size(); ++i) {
        if (nv[i] < 0) {
            utils::THROW_EXCEPTION(
                is_neg == false,
                "already had negative value in shape at index $", neg_index);
            is_neg = true;
            neg_index = i;
            continue;
        }
        n *= nv[i];
    }
    const int64_t &t_size = numel();
    _vals.transform_function<DType::TensorObj>(
        [&nv, &is_neg, &neg_index, &n](auto &inp) {
            std::vector<size_value_t> nv_cpy = nv;
            if (is_neg) {
                utils::THROW_EXCEPTION(
                    inp.numel() % n == 0,
                    "shape must be divisible by what has been "
                    "given, $ is not divisible by $",
                    inp.numel(), n);
                nv_cpy[neg_index] = inp.numel() / n;
            }
            return inp.view(SizeRef(std::move(nv_cpy)));
        },
        reinterpret_cast<Tensor *>(outp.data_ptr()));
    outp.set_mutability(_is_mutable);
    return std::move(outp);
}

// this is a transpose to happen to every single tensor
Tensor Tensor::transpose_Tensors(size_value_t a, size_value_t b) const {
    __NT_HANDLE_NULL_TENSORS__();
    utils::THROW_EXCEPTION(
        dtype == DType::TensorObj,
        "Expected transpose_Tensors to be used on a tensor of tensors");
    Tensor outp = Tensor::makeNullTensorArray(numel());
    Tensor *outputIt = reinterpret_cast<Tensor *>(outp.data_ptr());
    _vals.transform_function<DType::TensorObj>(
        [&a, &b](auto &inp) { return inp.transpose(a, b); }, outputIt);
    outp.set_mutability(_is_mutable);
    return std::move(outp);
}


Tensor Tensor::flatten(size_value_t _a, size_value_t _b) const {
    __NT_HANDLE_NULL_TENSORS__();
    _a = _a < 0 ? _a + dims() : _a;
    _b = _b < 0 ? _b + dims() : _b;
    size_value_t begin = _a < _b ? _a : _b;
    size_value_t end = _a < _b ? _b : _a;
    ++end;
    typedef typename SizeRef::ArrayRefInt::value_type value_t;
    size_value_t n_dims = dims() - (end - begin) + 1;
    std::vector<value_t> n_vals(n_dims);
    std::copy(shape().cbegin(), shape().cbegin() + begin, n_vals.begin());
    n_vals[begin] =
        std::accumulate(shape().begin() + begin, shape().begin() + end, 1.0,
                        std::multiplies<value_t>());
    std::copy(shape().cbegin() + end, shape().cend(),
              n_vals.begin() + begin + 1);
    return Tensor(_vals, std::move(n_vals)).set_mutability(_is_mutable);
}

void insert_ones(std::vector<Tensor::size_value_t> &vec, Tensor::size_value_t a,
                 Tensor::size_value_t b) {
    // Check for valid position
    if (a < 0 || a > vec.size()) {
        throw std::out_of_range("Invalid position to insert 1's");
    }

    for (Tensor::size_value_t i = 0; i < b; ++i) {
        vec.insert(vec.begin() + a, 1);
    }
}

Tensor Tensor::unflatten(size_value_t _a, size_value_t _b) const {
    __NT_HANDLE_NULL_TENSORS__();
    _a = _a < 0 ? _a + dims() : _a;
    _b = _b < 0 ? _b + dims() : _b;
    size_value_t begin = _a < _b ? _a : _b;
    size_value_t end = _a < _b ? _b : _a;
    size_value_t amt = end - begin;

    typedef typename SizeRef::ArrayRefInt::value_type value_t;
    auto vec = shape().Vec();
    insert_ones(vec, begin, amt);
    return this->view(SizeRef(std::move(vec)));
}



Tensor Tensor::permute(std::vector<size_value_t> Perm)
    const { return functional::permute(*this, std::move(Perm)); }

// when dims are 1 away from each other this is a great and wonderfully
// optimized design when they are not, there is probably much to be desired
Tensor Tensor::transpose(size_value_t _a, size_value_t _b) const { return functional::transpose(*this, _a, _b); }



Tensor &Tensor::RowColSwap_Tensors() {
    __NT_HANDLE_NULL_TENSORS__();
    __NT_HANDLE_MUTABILITY__()
    utils::THROW_EXCEPTION(dtype == DType::TensorObj,
                           "RowColSwap_Tensors is meant to be used on a tensor "
                           "that holds tensors");
    _vals.for_each<DType::TensorObj>([](auto &inp) { inp.RowColSwap(); });
    return *this;
}

Tensor &Tensor::RowColSwap() {return functional::row_col_swap_(*this);}

Tensor Tensor::swap_axis(size_value_t dim, size_value_t a, size_value_t b) const{
    __NT_HANDLE_NULL_TENSORS__();
    dim = (dim < 0) ? dims() + dim : dim;
    utils::throw_exception(b < dims(), "cannot swap axis ($) that are greater than dims ($)", b, dims());
    if(a > b) std::swap(a, b);
    utils::throw_exception(b < shape()[dim], "Expected to swap indices lower than the axis they are swapping got $ and $ for shape $", a, b, shape());
    Tensor split = this->split_axis(dim);
    Tensor* begin = reinterpret_cast<Tensor*>(split.data_ptr());
    Tensor* end = reinterpret_cast<Tensor*>(split.data_ptr_end());
    if(dim == 0){
        begin[a].swap(begin[b]);
        return functional::cat_unordered(split).view(shape());
    }
    int64_t batches = shape().flatten(0, dim)[0];
    int64_t sizes = shape().flatten(dim, -1)[-1];
    for(int64_t i = 0; i < batches; ++i, begin += sizes){
        begin[a].swap(begin[b]);
    }
    return functional::cat_unordered(split).view(shape()).set_mutability(is_mutable());

}

Tensor& Tensor::swap_axis_(size_value_t dim, size_value_t a, size_value_t b){
    __NT_HANDLE_NULL_TENSORS__();
    __NT_HANDLE_MUTABILITY__();
    dim = (dim < 0) ? dims() + dim : dim;
    utils::throw_exception(b < dims(), "cannot swap axis ($) that are greater than dims ($)", b, dims());
    if(a > b) std::swap(a, b);
    utils::throw_exception(b < shape()[dim], "Expected to swap indices lower than the axis they are swapping got $ and $ for shape $", a, b, shape());
    Tensor split = this->split_axis(dim);
    Tensor* begin = reinterpret_cast<Tensor*>(split.data_ptr());
    Tensor* end = reinterpret_cast<Tensor*>(split.data_ptr_end());
    if(dim == 0){
        begin[a].swap(begin[b]);
        Tensor n_tensor = functional::cat_unordered(split).view(shape());
        this->swap(n_tensor);
        return *this;
    }
    int64_t batches = shape().flatten(0, dim)[0];
    int64_t sizes = shape().flatten(dim, -1)[-1];
    for(int64_t i = 0; i < batches; ++i, begin += sizes){
        begin[a].swap(begin[b]);
    }
    Tensor n_tensor = functional::cat_unordered(split).view(shape());
    this->swap(n_tensor);
    return *this;
}

Tensor Tensor::real() const { return functional::real(*this); }
Tensor Tensor::to_complex_from_real() const { return functional::to_complex_from_real(*this); }
Tensor Tensor::imag() const { return functional::imag(*this); }
Tensor Tensor::to_complex_from_imag() const { return functional::to_complex_from_imag(*this); }


// I am going to remake this
// now that I can have a Tensor with a dtype of Tensor
// There may as well just be a Tensor with all the Tensor objects inside of it
// that way there is very little overhead, except for at the beggining

Tensor::Tensor(size_value_t i, const ArrayVoid &Arr, SizeRef &&_s)
    : _size({i}), _total_size(i), _vals(i, DType::TensorObj),
      dtype(DType::TensorObj), stored_strides(nullptr), _is_mutable(true) {
    size_value_t count = 0;
    size_value_t inner = _s.multiply();
    Tensor *it = reinterpret_cast<Tensor *>(this->data_ptr());
    Tensor *end = it + i;
    /* for(auto it = val_begin(); it != val_end(); ++it){ */
    for (; it != end; ++it, count += inner) {
        ArrayVoid a = Arr.share_array(static_cast<uint64_t>(count),
                                      static_cast<uint64_t>(inner));
        *it = Tensor(std::move(a), _s);
    }
}

Tensor::Tensor(ArrayVoid Arr, SizeRef _s, intrusive_ptr<size_value_t[]> strides)
    : _size(std::move(_s)), _total_size(0), 
      _vals(std::move(Arr)), dtype(DType::Float),
      stored_strides(std::move(strides)), _is_mutable(true) {
    _total_size = _vals.Size();
    dtype = _vals.dtype;
}

Tensor::Tensor(ArrayVoid Arr, SizeRef _s,
               const std::vector<size_value_t> &strides)
    : _size(std::move(_s)), _total_size(0), 
      _vals(std::move(Arr)), dtype(DType::Float),
      stored_strides(strides.size()), _is_mutable(true) {
    _total_size = _vals.Size();
    dtype = _vals.dtype;
    for (size_t i = 0; i < strides.size(); ++i)
        stored_strides[i] = strides[i];
}

/*Tensor Tensor::operator[](std::vector<my_range> r){
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
                        n.erase(n.cbegin() + (size * q), n.cbegin() + (size * q)
+ start); n.erase(n.cbegin() + (size * (q+1)), n.cbegin() + (size * (q+1)) +
left);
                }
                n.erase(n.cbegin() + (size * (q)), n.cbegin() + (size * (q)) +
start); n.erase(n.cbegin() + (size * (r_length)), n.end()); r_length *=
r[i].length();
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
}*/

// this will be finished at a later point in time
/*Tensor Tensor::split_axis(std::vector<my_range> ranges){*/
/*	utils::THROW_EXCEPTION(ranges.size() <= dims(), "expeted to have at most
 * $ ranges but got $ ranges for split_axis on tensor shape $", dims(),
 * ranges.size(), shape());*/

/*	std::vector<uint32_t> divs(ranges.size());*/
/*	std::vector<std::vector<std::pair<uint32_t, uint32_t>>>
 * lengths(dims());*/
/*	for(uint32_t i = 0; i < dims(); ++i){*/
/*		if(i < ranges.size())*/
/*			continue;*/
/*		lengths[i] = std::vector<std::pair<uint32_t, uint32_t>>({{0,
 * shape()[i]}});*/
/*	}*/
/*	for(uint32_t i = 0; i < ranges.size(); ++i){*/
/*		ranges[i].fix(shape()[i]); // corrects the ranges so that if one
 * of the input ranges is a negative number, then it is corrected according to
 * the shape of the current tensor*/
/*		utils::THROW_EXCEPTION(shape()[i] >= ranges[i].length(),
 * "Expected range to be less than or equal to length ($) at dim ($) but got
 * ($)", shape()[i], i, ranges[i].length());*/
/*		uint32_t cur_length = ranges[i].length();*/
/*		lengths[i].push_back({0, cur_length});*/
/*		if(cur_length == shape()[i])*/
/*			continue;*/
/*		while(cur_length < shape()[i]){*/
/*			if(cur_length + ranges[i].length() > shape()[i]){*/
/*				lengths[i].push_back({cur_length,
 * shape()[i]});*/
/*			}*/
/*			lengths[i].push_back({cur_length, cur_length +
 * ranges[i].length()});*/
/*			cur_length += ranges[i].length();*/
/*		}*/
/*	}*/
/*	uint32_t total_tensors = 1;*/
/*	for(uint32_t i = 0; i < lengths.size(); ++i)*/
/*		total_tensors *= lengths[i].size();*/
/*	if(total_tensors == 1)*/
/*		return *this;*/
/*	Tensor outp({total_tensors}, DType::TensorObj);*/
/*	std::vector<uint32_t> n(numel());*/
/*	std::vector<uint32_t> stri = strides();*/
/*	uint32_t current_tensor = 0;*/
/*	std::vector<uint32_t> indices(dims(), 0);*/

/*	uint32_t r_length = r[0].length();*/
/*	for(uint32_t i = 1; i < r.size(); ++i){*/
/*		uint32_t start = r[i].begin * stri[i+1];*/
/*		uint32_t size = r[i].length() * stri[i+1];*/
/*		uint32_t left = (stri[i]) - (size+start);*/
/*		uint32_t q;*/
/*		for(q  = 0; q < r_length-1; ++q){*/
/*			n.erase(n.cbegin() + (size * q), n.cbegin() + (size * q)
 * + start);*/
/*			n.erase(n.cbegin() + (size * (q+1)), n.cbegin() + (size
 * * (q+1)) + left);*/
/*		}*/
/*		n.erase(n.cbegin() + (size * (q)), n.cbegin() + (size * (q)) +
 * start);*/
/*		n.erase(n.cbegin() + (size * (r_length)), n.end());*/
/*		r_length *= r[i].length();*/
/*	}*/
/*	for(uint32_t i = 0; i < total_tensors; ++i){*/
/*		std::vector<uint32_t> sub_indices;*/
/*		uint32_t r_length = (lengths[0][indices[0]].second -
 * lengths[0][indices[0]].first);*/

/*		uint32_t stride = 1;*/
/*		for(uint32_t j = dims() - 1; j >= 0; --j){*/
/*			uint32_t index = indices[j];*/
/*			uint32_t begin = lengths[j][index].first;*/
/*			uint32_t end = lengths[j][index].second;*/
/*			sub_indices.push_back(begin);*/
/*			stride *= (end - begin);*/
/*		}*/
/*		auto begin = n.begin();*/
/*		for(uint32_t i = 0; i < dims(); ++i, ++r_cur_index,
 * ++r_lengths_Vec){*/
/*			uint32_t start = r[i].begin * stri[i+1];*/
/*			uint32_t size = r[i].length() * stri[i+1];*/
/*			uint32_t left = (stri[i]) - (size+start);*/

/*		}*/
/*	}*/

/*}*/

// the point of this function is to generate all ranges based on a singular
// range

void print_vec_ranges(std::vector<my_range>& current_ranges){
    std::cout << '(';
    for(int i = 0; i < current_ranges.size()-1; ++i)
            std::cout << current_ranges[i] << ',';
    std::cout << current_ranges.back() << ')' << std::endl;
}

//set instead of unordered set because the order does matter
void generate_ranges(const std::vector<my_range> &ranges,
                     std::vector<my_range> current_ranges, size_t idx,
                     std::set<std::vector<my_range>> &result,
                     const SizeRef &shape) noexcept {
    if (idx >= ranges.size()) return;
    if (idx == (ranges.size() - 1)) {
        if (idx_is_total(ranges[idx], shape, idx)) {
            result.insert(current_ranges);
            return;
        }
        bool at_end = current_ranges[idx].end + ranges[idx].length() >= shape[idx];
        current_ranges[idx] =
            at_end ? my_range(current_ranges[idx].end, shape[idx])
                   : my_range(current_ranges[idx].end,
                              current_ranges[idx].end + ranges[idx].length());
        // std::cout << "inserting "<<current_ranges<<std::endl;
        result.insert(current_ranges);
        if (at_end) {
            return;
        }
        generate_ranges(ranges, current_ranges, idx, result, shape);
    }
    if (idx_is_total(ranges[idx], shape, idx)) {
        generate_ranges(ranges, current_ranges, idx + 1, result, shape);
    } else {
        bool at_end = current_ranges[idx].end + ranges[idx].length() >= shape[idx];
        // std::cout << "current_ranges before add: ";
        // print_vec_ranges(current_ranges);
        current_ranges[idx] =
            at_end ? my_range(current_ranges[idx].end, shape[idx])
                   : my_range(current_ranges[idx].end,
                              current_ranges[idx].end + ranges[idx].length());
        
        // std::cout << "at end and generating range idx: "<<idx << " ranges size: "<<ranges.size()
        //         <<" shape: "<<shape<<std::endl;
        // std::cout << "current_ranges: ";
        // print_vec_ranges(current_ranges);
        // std::cout << "inserting "<<current_ranges<<std::endl;
        result.insert(current_ranges);
        if(!at_end){
            generate_ranges(ranges, current_ranges, idx, result, shape);
        }
        generate_ranges(ranges, current_ranges, idx + 1, result, shape);
    }
}

Tensor Tensor::split_axis(std::vector<my_range> ranges) const {
    __NT_HANDLE_NULL_TENSORS__();
    utils::THROW_EXCEPTION(ranges.size() <= dims(),
                           "expected to have at most $ ranges but got $ ranges "
                           "for split_axis on tensor shape $",
                           dims(), ranges.size(), shape());
    while (ranges.size() < dims()) {
        // Add a my_range(0, -1) to ranges
        ranges.push_back(my_range(0, shape()[ranges.size()]));
    }
    for (uint32_t i = 0; i < ranges.size(); ++i){
        ranges[i].fix(shape()[i]);
        utils::throw_exception(ranges[i].length() > 0, 
                "Cannot increment range at dimension $ with length of 0 got range of $", 
                i, ranges[i]);
    }

    std::set<std::vector<my_range>> result_ranges;
    std::vector<my_range> current_ranges = ranges;
    result_ranges.insert(ranges);

    generate_ranges(ranges, current_ranges, 0, result_ranges, shape());
    if (result_ranges.size() == 1 || result_ranges.size() == 0)
        return *this;
    Tensor output = Tensor::makeNullTensorArray(result_ranges.size());
    output._is_mutable = this->_is_mutable;
    output._size = SizeRef({static_cast<size_value_t>(result_ranges.size())});
    Tensor *begin = reinterpret_cast<Tensor *>(output.data_ptr());
    Tensor *end = begin + result_ranges.size();
    auto ra_begin = result_ranges.cbegin();
    for (; begin != end; ++begin, ++ra_begin){
        *begin = (*this)[*ra_begin];
        begin->_is_mutable = this->_is_mutable;
    }
    return output;
}

/* Tensor Tensor::split_elements(){ */
/* 	return Tensor(_total_size, _vals, shape()); */
/* } */

// returns a tensor of tensors split along a specific axis and accumulated along
// that axis this was the old way of doing it:
/* Tensor Tensor::split_axis(size_value_t dim){ */
/* 	typedef typename SizeRef::ArrayRefInt::value_type value_t; */
/* 	dim = dim < 0 ? dim + dims() : dim; */
/* 	if(dim == (dims() - 1)){ */
/* 		return transpose(-1, -2).split_axis(-2); */
/* 	} */
/* 	dim += 1; */
/* 	std::vector<value_t> n_vals(dims() - dim); */

/* 	for(size_value_t i = 0; i < dims() - dim; ++i){ */
/* 		n_vals[i] = shape()[i+dim]; */
/* 	} */
/* 	SizeRef n2_size(n_vals); */
/* 	size_value_t i = _total_size / n2_size.multiply(); */
/* 	Tensor to_return(i, _vals, std::move(n2_size)); */
/* 	return std::move(to_return); */
/* } */

// this reduced time from 106 seconds to 0.016 seconds in some cases, (much
// faster)
Tensor Tensor::split_axis(size_value_t dim) const {
    __NT_HANDLE_NULL_TENSORS__();
    typedef typename SizeRef::ArrayRefInt::value_type value_t;
    dim = dim < 0 ? dim + dims() : dim;
    if (dim == (dims() - 1) && dims() >= 2) {
        return transpose(-1, -2).split_axis(-2);
    }
    dim += 1;
    std::vector<value_t> n_vals(dims() - dim);

    for (size_value_t i = 0; i < dims() - dim; ++i) {
        n_vals[i] = shape()[i + dim];
    }
    SizeRef n2_size(n_vals);
    const uint64_t splitting = n2_size.multiply();
    size_value_t i = _total_size / splitting;
    Tensor buckets = _vals.get_bucket().split<Tensor>(splitting);
	Tensor* begin = reinterpret_cast<Tensor*>(buckets.data_ptr());
	Tensor* end = begin + buckets.numel();
	typedef typename SizeRef::ArrayRefInt::value_type m_size_t;
	for(;begin != end; ++begin){
		begin->_size = n2_size;
		begin->dtype = dtype;
        begin->_is_mutable = this->_is_mutable;
	}
	return buckets.set_mutability(this->_is_mutable); 
}

// 0 == cols
// 1 == rows
// 2 == z
// 3 == q
// ....
/* const Tensor Tensor::split_axis(size_value_t dim) const{ */
/* 	typedef typename SizeRef::ArrayRefInt::value_type value_t; */
/* 	dim = dim < 0 ? dim + dims() : dim; */
/* 	if(dim == (dims() - 1)){ */
/* 		return transpose(-1, -2).split_axis(-2); */
/* 	} */
/* 	dim += 1; */
/* 	std::vector<value_t> n_vals(dims() - dim); */

/* 	for(size_value_t i = 0; i < dims() - dim; ++i){ */
/* 		n_vals[i] = shape()[i+dim]; */
/* 	} */
/* 	SizeRef n2_size(n_vals); */
/* 	const uint64_t splitting = n2_size.multiply(); */
/* 	size_value_t i = _total_size / splitting; */
/* 	return _vals.split(splitting, n2_size); */
/* } */

// this is for a column like view
Tensor Tensor::split_axis_1() const {
    __NT_HANDLE_NULL_TENSORS__();
    typedef typename SizeRef::ArrayRefInt::value_type value_t;
    Tensor outp = transpose(-1, -2);
    /* RowColSwap(); */
    SizeRef n_shape = outp.shape();
    size_value_t dim = (dims() - 2) + 1;
    std::vector<value_t> n_vals(dims() - dim);
    for (size_value_t i = 0; i < dims() - dim; ++i) {
        n_vals[i] = n_shape[i + dim];
    }
    SizeRef n2_size(n_vals);
    size_value_t i = _total_size / n2_size.multiply();
    const uint64_t splitting = n2_size.multiply();
    // return outp._vals.split(splitting, n2_size);

    Tensor buckets = outp._vals.get_bucket().split<Tensor>(splitting);
	Tensor* begin = reinterpret_cast<Tensor*>(buckets.data_ptr());
	Tensor* end = begin + buckets.numel();
	typedef typename SizeRef::ArrayRefInt::value_type m_size_t;
	for(;begin != end; ++begin){
		begin->_size = n2_size;
		begin->dtype = dtype;
        begin->_is_mutable = this->_is_mutable;
	}
	return buckets.set_mutability(this->_is_mutable); 
}

Tensor Tensor::unfold(size_value_t dim, size_value_t size,
                      size_value_t step) const {
    __NT_HANDLE_NULL_TENSORS__();
    dim = dim < 0 ? dims() + dim : dim;
    utils::THROW_EXCEPTION(
        dim < dims(),
        "Expected to get an appropriate dimension less than $ but got $\n",
        dims(), dim);
    utils::THROW_EXCEPTION(
        size <= shape()[dim],
        "maximum size for Tensor at dimension $ is $ but got size of $", dim,
        shape()[dim], size);

    std::vector<size_value_t> n_strides = this->strides();
    std::vector<size_value_t> get_strides = this->getChangedStrides();
    bool change_get = get_strides != n_strides;

    n_strides.erase(n_strides.begin());
    const size_value_t inserting = n_strides[dim];
    n_strides[dim] *= step;
    // aftetr the dim insert the inserting
    n_strides.push_back(inserting);

    // this is done correctly
    std::vector<size_value_t> vec = this->shape().Vec();
    size_value_t unfolds = size_value_t((this->shape()[dim] - size) / step + 1);
    vec[dim] = unfolds;
    vec.push_back(size);
    SizeRef outp_size(std::move(vec));

    // Use as_strided to create the new tensor
    if (!change_get) {
        return functional::as_strided(*this, outp_size, SizeRef(n_strides), 0,
                                      false);
    }
    Tensor output =
        functional::as_strided(*this, outp_size, SizeRef(n_strides), 0, false);

    get_strides.erase(get_strides.begin());
    const size_value_t inserting_g = get_strides[dim];
    get_strides[dim] *= step;
    get_strides.push_back(inserting_g);
    get_strides.insert(get_strides.begin(), outp_size.multiply());
    output.set_stored_strides(get_strides);
    output._is_mutable = this->_is_mutable;
    return std::move(output);
}

// Helper function to calculate the indices in the output tensor
std::vector<Tensor::size_value_t>
calculate_indices_fold_function(size_t index, const SizeRef &shape) {
    std::vector<Tensor::size_value_t> indices(shape.size());
    for (size_t i = 0; i < shape.size(); ++i) {
        indices[i] = index % shape[i];
        index /= shape[i];
    }
    return indices;
}

// Helper function to add a value to the output tensor at the specified indices
template <typename T>
void add_value_to_output_fold_function(
    T *output_tensor, const SizeRef &output_tensor_shape,
    const std::vector<Tensor::size_value_t> &indices, const T &value) {
    size_t index = 0;
    size_t stride = 1;
    for (size_t i = 0; i < indices.size(); ++i) {
        index += indices[i] * stride;
        stride *= output_tensor_shape[i];
    }

    output_tensor[index] += value;
}

/* Tensor Tensor::fold(size_value_t dim, size_value_t size, size_value_t step,
 * const SizeRef& output_shape) const { */
/*     // Validate inputs */
/*     dim = dim < 0 ? t.dims() + dim : dim; */
/*     utils::THROW_EXCEPTION(dim < t.dims(), "Expected to get an appropriate
 * dimension less than " + std::to_string(dims()) + " but got " +
 * std::to_string(dim)); */

/*     // Calculate the shape and stride for the unfolded tensor */
/*     std::vector<size_value_t> n_shape = shape().Vec(); */
/*     std::vector<size_value_t> stride = getChangedStrides(); */

/*     // Allocate the output tensor */
/*     Tensor output_tensor = ::nt::functional::zeros(output_shape,
 * this->dtype); */

/*     // Iterate over each element in the unfolded tensor and add its value to
 * the appropriate location in the output tensor */
/*     this->_vals.cexecute_function<WRAP_DTYPES<NumberTypesL>>()([&output_tensor,
 * &dim, &n_shape, &size, &stride, &output_shape, &step](auto begin, auto end){
 */
/* 		using value_t = utils::IteratorBaseType_t<decltype(begin)>; */
/* 		value_t* o_begin =
 * reinterpret_cast<value_t*>(output_tensor.data_ptr()); */
/* 		for(size_value_t i = 0; i < n_shape[dim]; ++i){ */
/* 			for(size_value_t j = 0; j < size; ++j){ */
/* 				size_value_t unfold_index = i * stride[dim] +j;
 */
/* 				size_value_t output_index = i * step + j; */

/* 				// Calculate the indices for the output tensor
 */
/* 				std::vector<size_value_t> output_indices =
 * calculate_indices_fold_function(output_index, output_shape); */

/* 				// Add the value from the unfolded tensor to the
 * output tensor */
/* 				add_value_to_output_fold_function<value_t>(o_begin,
 * output_shape, output_indices, begin[unfold_index]); */
/* 			} */
/* 		} */

/* 	}); */

/*     return std::move(output_tensor); */
/* } */

/* Tensor Tensor::unfold(int32_t dim, uint32_t size, uint32_t step) const{ */
/* dim = dim < 0 ? dims() + dim : dim; */
/* utils::THROW_EXCEPTION(dim < dims(), "Expected to get an appropriate
 * dimension less than $ but got $\n", dims(), dim); */
/* utils::THROW_EXCEPTION(size <= shape()[dim], "maximum size for Tensor at
 * dimension $ is $ but got size of $", dim, shape()[dim], size); */

/* std::vector<uint32_t> vec = shape().Vec(); */
/* 	uint32_t unfolds = uint32_t((shape()[dim] - size)/step + 1); */
/* 	vec[dim] = unfolds; */
/* 	vec.push_back(size); */
/* 	SizeRef outp_size(std::move(vec)); */
/* 	uint32_t n_vals_size = outp_size.multiply(); */
/* 	ArrayVoid n_vals = _vals.new_stride(n_vals_size); */
/* 	std::cout << "getting proc"<<std::endl; */
/* 	Tensor proc = dim == dims()-1 ? *this : this->transpose(-1, dim); */
/* 	std::cout << "got proc"<<std::endl; */
/* 	std::vector<uint32_t> _strides = proc.strides(); */
/* 	_strides.erase(_strides.cbegin()); */

/* 	std::vector<uint32_t> outp_strides = outp_size.strides(); */
/* 	outp_strides.erase(outp_strides.cbegin()); */

/* 	int i_dim = dim; */
/* 	while(i_dim != dims()-2 && i_dim != dims()-1){ */
/* 		std::swap(_strides[i_dim], _strides[i_dim+1]); */
/* 		++i_dim; */
/* 	} */

/* 	//this is going to give the correct strides at the right one, */
/* 	//and then the variable unfolds needs to be taken into account */
/* 	//this will be used to add the number of _strides[-1] naturally based on
 * where it is in the dim compared to the _strides */
/* 	//this kinda fills in the last piece of the puzzle */

/* 	void** n_ptr = n_vals.strides_begin(); */
/* 	void** m_ptr = proc._vals.strides_begin(); */

/* 	for(uint32_t i = 0; i < n_vals_size; ++i, ++n_ptr){ */
/* 		uint32_t currentadd_ = 0; */
/* 		uint32_t i_s = i; */
/* 		for(uint32_t j = 0; j < outp_strides.size(); ++j){ */
/* 			if(j == dim && i_s >= outp_strides[j]){ */
/* 				uint32_t i_n = i_s / outp_strides[j]; */
/* 				i_s = i_s % outp_strides[j]; */
/* 				current_add += i_n * _strides.back(); */
/* 				continue; */
/* 			} */
/* 			if(i_s >= outp_strides[j]){ */
/* 				uint32_t j_i = j > dim ? j - 1 : j; */
/* 				uint32_t i_n = i_s / outp_strides[j]; */
/* 				i_s = i_s % outp_strides[j]; */
/* 				current_add += i_n * _strides[j_i]; */

/* 			} */
/* 		} */
/* 		*n_ptr = m_ptr[current_add]; */
/* 	} */
/* 	return Tensor(std::move(n_vals), std::move(outp_size)); */
/* } */

/* #else */
/* Tensor Tensor::unfold(int32_t dim, uint32_t size, uint32_t step) const{ */
/* 	dim = dim < 0 ? dims() + dim : dim; */
/* 	utils::THROW_EXCEPTION(dim < dims(), "Expected to get an appropriate
 * dimension less than $ but got $\n", dims(), dim); */
/* 	utils::THROW_EXCEPTION(size <= shape()[dim], "maximum size for Tensor at
 * dimension $ is $ but got size of $", dim, shape()[dim], size); */

/* 	std::vector<uint32_t> vec = shape().Vec(); */
/* 	uint32_t unfolds = uint32_t((shape()[dim] - size)/step + 1); */
/* 	vec[dim] = unfolds; */
/* 	vec.push_back(size); */
/* 	SizeRef outp_size(std::move(vec)); */
/* 	uint32_t n_vals_size = outp_size.multiply(); */
/* 	int pipe_fd[2]; */
/* 	Tensor proc; */
/* 	ArrayVoid n_vals = _vals.new_stride(n_vals_size); */
/* 	std::vector<uint32_t> _strides = proc.shape().transpose(-1,
 * dim).strides(); */
/* 	_strides.erase(_strides.cbegin()); */

/* 	std::vector<uint32_t> outp_strides = outp_size.strides(); */
/* 	outp_strides.erase(outp_strides.cbegin()); */

/* 	int i_dim = dim; */
/* 	while(i_dim != dims()-2 && i_dim != dims()-1){ */
/* 		std::swap(_strides[i_dim], _strides[i_dim+1]); */
/* 		++i_dim; */
/* 	} */

/* 	std::vector<uint32_t> indexes(n_vals_size); */
/* 	if (pipe(pipe_fd) == -1) { */
/* 		perror("pipe"); */
/* 		return *this; */
/* 	} */
/* 	pid_t pid = fork(); */
/* 	if (pid < 0) { */
/* 		// Error occurred */
/* 		std::cerr << "Failed to fork.\n"; */
/* 		return *this; */
/* 	} */
/* 	else if(pid == 0){ */
/* 		close(pipe_fd[0]); */
/* 		std::vector<uint32_t> indexes(n_vals_size); */
/* 		tbb::parallel_for(tbb::blocked_range<uint32_t>(0, n_vals_size),
 */
/* 			[&](tbb::blocked_range<uint32_t> b){ */
/* 				for(uint32_t i = b.begin(); i < b.end(); ++i){
 */
/* 					uint32_t current_add = 0; */
/* 					uint32_t i_s = i; */
/* 					for(uint32_t j = 0; j <
 * outp_strides.size(); ++j){ */
/* 						if(j == dim && i_s >=
 * outp_strides[j]){ */
/* 							uint32_t i_n = i_s /
 * outp_strides[j]; */
/* 							i_s = i_s %
 * outp_strides[j]; */
/* 							current_add += i_n *
 * _strides.back(); */
/* 							continue; */
/* 						} */
/* 						if(i_s >= outp_strides[j]){ */
/* 							uint32_t j_i = j > dim ?
 * j - 1 : j; */
/* 							uint32_t i_n = i_s /
 * outp_strides[j]; */
/* 							i_s = i_s %
 * outp_strides[j]; */
/* 							current_add += i_n *
 * _strides[j_i]; */

/* 						} */
/* 					} */
/* 					indexes[i] = current_add; */
/* 				} */
/* 			}); */
/* 		write(pipe_fd[1], indexes.data(), indexes.size() *
 * sizeof(uint32_t)); */
/* 		close(pipe_fd[1]); */
/* 	} */
/* 	else{ */
/* 	close(pipe_fd[1]); */
/* 	proc = dim == dims()-1 ? *this : this->transpose(-1, dim); */
/* 	read(pipe_fd[0], indexes.data(), indexes.size() * sizeof(uint32_t)); */
/* 	close(pipe_fd[0]); */
/* 	wait(nullptr); */
/* 	} */

/* 	//this is going to give the correct strides at the right one, */
/* 	//and then the variable unfolds needs to be taken into account */
/* 	//this will be used to add the number of _strides[-1] naturally based on
 * where it is in the dim compared to the _strides */
/* 	//this kinda fills in the last piece of the puzzle */

/* 	void** n_ptr = n_vals.strides_begin(); */
/* 	void** m_ptr = proc._vals.strides_begin(); */
/* 	tbb::parallel_for(tbb::blocked_range<uint32_t>(0, n_vals_size, 10), */
/* 			[&](tbb::blocked_range<uint32_t> b){ */
/* 				void** r_ptr = n_ptr + b.begin(); */
/* 				for(uint32_t i = b.begin(); i < b.end(); ++i,
 * ++r_ptr){ */
/* 					*r_ptr = m_ptr[indexes[i]]; */

/* 				} */
/* 			}); */
/* 	return Tensor(std::move(n_vals), std::move(outp_size)); */
/* } */
/* #endif */

void *Tensor::data_ptr() { return _vals.data_ptr(); }
const void *Tensor::data_ptr() const { return _vals.data_ptr(); }

void *Tensor::data_ptr_end() {
    __NT_HANDLE_NULL_TENSORS__();
    utils::THROW_EXCEPTION(
        is_contiguous(),
        "Can only find end of data pointer on contiguous tensor");
    return (void *)(reinterpret_cast<uint8_t *>(data_ptr()) +
                    (_total_size * DTypeFuncs::size_of_dtype(dtype)));
}

const void *Tensor::data_ptr_end() const {
    __NT_HANDLE_NULL_TENSORS__();
    utils::THROW_EXCEPTION(
        is_contiguous(),
        "Can only find end of data pointer on contiguous tensor");
    return (const void *)(reinterpret_cast<const uint8_t *>(data_ptr()) +
                          (_total_size * DTypeFuncs::size_of_dtype(dtype)));
}

// share from a specific point in memory
Tensor Tensor::div(size_value_t i) const {
    __NT_HANDLE_NULL_TENSORS__();
    return Tensor(_vals.share_array(i, i), {i}).set_mutability(this->_is_mutable);
}

ArrayVoid &Tensor::arr_void() { return _vals; }
const ArrayVoid &Tensor::arr_void() const { return _vals; }

Tensor Tensor::repeat_(size_value_t amt) const {return functional::repeat_(*this, amt);}
Tensor Tensor::repeat_(size_value_t dim, size_value_t amt) const {return functional::repeat_(*this, dim, amt);}
Tensor Tensor::expand(SizeRef s) const {return functional::expand(*this, s);} 
Tensor Tensor::expand_as(const Tensor &t) const {return functional::expand_as(*this, t);}


// fill, subtract, add, multiply
Tensor &Tensor::subtract_(Scalar val) {return functional::subtract_(*this, val);}

Tensor &Tensor::subtract_(const Tensor &val) {
    return functional::subtract_(*this, val);
}
Tensor &Tensor::multiply_(Scalar val) { return functional::multiply_(*this, val);}
Tensor &Tensor::multiply_(const Tensor &val) {
    return functional::hadamard_multiply_this(*this, val);
}
Tensor &Tensor::divide_(Scalar val) {return functional::divide_(*this, val);}
Tensor &Tensor::divide_(const Tensor &val) {
    return functional::divide_(*this, val);
}
Tensor &Tensor::fill_(Scalar val) {return functional::fill_(*this, val);}

Tensor &Tensor::fill_(const Tensor &val) {
    __NT_HANDLE_NULL_TENSORS__();
    __NT_HANDLE_MUTABILITY__();
    if (dtype != DType::TensorObj) {
        utils::THROW_EXCEPTION(
            val.dtype == dtype,
            "For filling in a tensor with another tensor, "
            "dtypes are expected to be equal but got $ and $",
            val.dtype, dtype);
        utils::THROW_EXCEPTION(
            shape() == val.shape(),
            "For filling in a tensor with another tensor, "
            "shapes are expected to be equal but got $ and $",
            val.shape(), shape());
    }
    return functional::set_(*this, val);
}

Tensor& Tensor::fill_diagonal_(Scalar c){return functional::fill_diagonal_(*this, c);}

Tensor Tensor::diagonal() const { return functional::diagonal(*this); }

Tensor &Tensor::add_(Scalar val) { return functional::add_(*this, val); }
Tensor &Tensor::add_(const Tensor &val) { return functional::add_(*this, val); }
Tensor Tensor::operator==(Scalar c) const { return functional::equal(*this, c); }
Tensor Tensor::operator!=(Scalar c) const { return functional::not_equal(*this, c); }
Tensor Tensor::operator<=(Scalar c) const { return functional::less_than_equal(*this, c); }
Tensor Tensor::operator>=(Scalar c) const { return functional::greater_than_equal(*this, c); }
Tensor Tensor::operator<(Scalar c) const { return functional::less_than(*this, c); }
Tensor Tensor::operator>(Scalar c) const { return functional::greater_than(*this, c); }

CommaOperator Tensor::operator<<(Scalar s) {
    __NT_HANDLE_NULL_TENSORS__();
    __NT_HANDLE_MUTABILITY__();
    utils::throw_exception(is_contiguous(), "Must be contiguous to use comma operator");
    CommaOperator out(data_ptr(), data_ptr_end(), dtype);
    return out , s;
}


std::string_view Tensor::sv() const {
    __NT_HANDLE_NULL_TENSORS__();
    __NT_HANDLE_MUTABILITY__();
    utils::THROW_EXCEPTION(
        dtype == DType::uint8,
        "\nRuntimeError: Expected DType for string_view to be uint8 but got $",
        dtype);
    utils::THROW_EXCEPTION(is_contiguous(),
                           "Can only convert contiguous tensor to string_view");
    return std::string_view(reinterpret_cast<const char *>(data_ptr()),
                            numel());
}

Tensor Tensor::to_dtype(DType _dt) const {return functional::to(*this, _dt);}

Tensor Tensor::to_device(DeviceType _dt) const {
    __NT_HANDLE_NULL_TENSORS__();
    return Tensor(_vals.to(_dt), shape());
}

Tensor Tensor::Int() const { __NT_HANDLE_NULL_TENSORS__(); return Tensor(_vals.int32(), shape()); }
Tensor Tensor::Long() const { __NT_HANDLE_NULL_TENSORS__(); return Tensor(_vals.uint32(), shape()); }

Tensor Tensor::unsqueeze(size_value_t dim) const {
    __NT_HANDLE_NULL_TENSORS__();
    dim = dim < 0 ? (dim + dims() + 1) : dim;
    std::vector<SizeRef::ArrayRefInt::value_type> Vec = shape().Vec();
    Vec.insert(Vec.begin() + dim, 1);
    return view(SizeRef(std::move(Vec)));
}

Tensor Tensor::unsqueeze_as(const Tensor &t) const {
    __NT_HANDLE_NULL_TENSORS__(t);
    std::vector<SizeRef::ArrayRefInt::value_type> Vec = shape().Vec();
    while (Vec.size() < t.dims())
        Vec.insert(Vec.begin(), 1);
    return view(SizeRef(std::move(Vec)));
}

Tensor Tensor::unsqueeze_as(const SizeRef &s) const {
    __NT_HANDLE_NULL_TENSORS__();
    std::vector<SizeRef::ArrayRefInt::value_type> Vec = shape().Vec();
    while (Vec.size() < s.size())
        Vec.insert(Vec.begin(), 1);
    return view(SizeRef(std::move(Vec)));
}

// now gets rid of all dimensions that are 1
Tensor Tensor::squeeze(utils::optional_list list) const {
    __NT_HANDLE_NULL_TENSORS__();
    std::vector<SizeRef::ArrayRefInt::value_type> Vec;
    Vec.reserve(shape().size());
    if(!list){
        for (const auto &element : shape()) {
            if (element > 1)
                Vec.push_back(element);
        }
        if (Vec.size() == 0) {
            Vec.push_back(1);
        } // make sure there is an element
        return view(SizeRef(std::move(Vec)));

    }
    Vec = shape().Vec();
    for (auto begin = list->cbegin(); begin != list->cend(); ++begin) {
        int64_t dim = *begin < 0 ? *begin + dims() : *begin;
        utils::throw_exception(dim < dims(), "Trying to squeeze at dim $ but only have $ dims", dim, dims());
        if(Vec.at(dim) == 1){
            Vec.erase(Vec.begin() + dim);
        }
    }
    if (Vec.size() == 0) {
        Vec.push_back(1);
    } // make sure there is an element
    return view(SizeRef(std::move(Vec))); 
        

}






Tensor Tensor::sum(utils::optional_list list, bool keepdim) const { return functional::sum(*this, list, keepdim); }

Tensor Tensor::mean(utils::optional_list list, bool keepdim) const {
    __NT_HANDLE_NULL_TENSORS__();
    if (dtype == DType::TensorObj) {
        Tensor outp(shape(), DType::TensorObj);
        _vals.transform_function<DType::TensorObj>(
            [&](const Tensor &output) -> Tensor {
                return output.mean(list, keepdim);
            },
            reinterpret_cast<Tensor *>(outp.data_ptr()));
        return std::move(outp);
    }
    if (!list) {
        size_value_t total_scale = numel();
        double div = 1.0 / (double)total_scale;
        Tensor output = sum(nullptr, keepdim);
        return output * div;
    }
    size_value_t total_scale = 1;
    for (const auto &dim : list) {
        total_scale *= shape()[dim];
    }
    Tensor output = sum(list, keepdim);
    if(output.shape() == shape()) return output;
    if (total_scale == 0) {
        output.arr_void().fill_(0);
        return std::move(output);
    }
    output /= total_scale;
    return std::move(output);
}

Tensor Tensor::sum_as(const SizeRef &s) const {
    __NT_HANDLE_NULL_TENSORS__();
    utils::THROW_EXCEPTION(s.size() == dims(),
                           "Expected to have information about all dimensions "
                           "whether to sum or not but got $ and $",
                           s, shape());
    if (s.multiply() == 1) {
        return sum();
    }
    Tensor output = this->clone();
    const SizeRef &before_shape = shape();
    std::vector<size_value_t> cur_strides(s.size(), 1);
    for (size_value_t i = s.size() - 2; i >= 0; --i) {
        cur_strides[i] = cur_strides[i + 1] * before_shape[i + 1];
    }

    std::vector<size_value_t> axes;
    axes.reserve(s.size());
    for (size_value_t i = 0; i < s.size(); ++i) {
        if (before_shape[i] != s[i]) {
            utils::THROW_EXCEPTION(
                s[i] == 1,
                "Expected for dimension $ where target shape does not equal "
                "current "
                "shape that target would be 1 but got $ and current is $",
                i, s[i], before_shape[i]);
            axes.push_back(i);
        }
    }
    for (const auto &dim : axes) {
        output = output.transpose(0, dim).sum(0, true).transpose(0, dim);
    }
    return std::move(output.view(s));
}

Tensor Tensor::sum_as(const Tensor &t) const {
    __NT_HANDLE_NULL_TENSORS__(t);
    utils::THROW_EXCEPTION(
        t.dims() == dims(),
        "Expected dims to be equal for sum as but got $ and $", t.dims(),
        dims());
    if (t.numel() == 1) {
        return sum();
    }
    return sum_as(t.shape());
}


result_types::max<Tensor, Tensor> Tensor::max(utils::optional_list list) const { return functional::max(*this, list); }
result_types::max<Tensor, Tensor> Tensor::min(utils::optional_list list) const { return functional::max(*this, list); }


Tensor Tensor::exp() const { return functional::exp(*this); }
Tensor &Tensor::exp_() {
    __NT_HANDLE_NULL_TENSORS__();
    __NT_HANDLE_MUTABILITY__();
    utils::THROW_EXCEPTION(
        dtype != DType::Bool,
        "\nRuntimeError: Tried running unsupported DType Bool "
        "with function exp_()");
    _vals.exp_();
    return *this;
}

// this was the function that made me implement the ability to choose a specific
// dtype for a for_each, execute, and transform_function
Tensor &Tensor::inverse_() {return functional::inverse_(*this);}
Tensor Tensor::inverse() const { return functional::inverse(*this); }

Tensor Tensor::pow(Scalar i) const { __NT_HANDLE_NULL_TENSORS__(); return Tensor(_vals.pow(i), shape()); }

Tensor &Tensor::pow_(Scalar i) {
    __NT_HANDLE_NULL_TENSORS__();
    __NT_HANDLE_MUTABILITY__();
    _vals.pow_(i);
    dtype = _vals.dtype;
    return *this;
}

Tensor &Tensor::clip_(Scalar a, Scalar b) { return functional::clamp_(*this, a, b); }

template <std::size_t N>
Tensor Tensor::FromInitializer(
    typename utils::NestedInitializerLists_type<Scalar, N>::type v, DType dt) {
    SizeRef sz(utils::aquire_shape<Scalar, N>(v));
    Tensor output(sz, dt);
    switch (dt) {
    case DType::Float: {
        using value_type = float;
        value_type *begin = reinterpret_cast<value_type *>(output.data_ptr());
        utils::flatten_func<Scalar, N>(v, [&begin](const Scalar &a) {
            *begin = a.to<value_type>();
            ++begin;
        });
        break;
    }
    case DType::Double: {
        using value_type = double;
        value_type *begin = reinterpret_cast<value_type *>(output.data_ptr());
        utils::flatten_func<Scalar, N>(v, [&begin](const Scalar &a) {
            *begin = a.to<value_type>();
            ++begin;
        });
        break;
    }
#ifdef _HALF_FLOAT_SUPPORT_
    case DType::Float16: {
        using value_type = float16_t;
        value_type *begin = reinterpret_cast<value_type *>(output.data_ptr());
        utils::flatten_func<Scalar, N>(v, [&begin](const Scalar &a) {
            *begin = a.to<value_type>();
            ++begin;
        });
        break;
    }
    case DType::Complex32: {
        using value_type = complex_32;
        value_type *begin = reinterpret_cast<value_type *>(output.data_ptr());
        utils::flatten_func<Scalar, N>(v, [&begin](const Scalar &a) {
            *begin = a.to<value_type>();
            ++begin;
        });
        break;
    }
#endif
#ifdef _128_FLOAT_SUPPORT_
    case DType::Float128: {
        using value_type = float128_t;
        value_type *begin = reinterpret_cast<value_type *>(output.data_ptr());
        utils::flatten_func<Scalar, N>(v, [&begin](const Scalar &a) {
            *begin = a.to<value_type>();
            ++begin;
        });
        break;
    }
#endif
#ifdef __SIZEOF_INT128__
    case DType::int128: {
        using value_type = int128_t;
        value_type *begin = reinterpret_cast<value_type *>(output.data_ptr());
        utils::flatten_func<Scalar, N>(v, [&begin](const Scalar &a) {
            *begin = a.to<value_type>();
            ++begin;
        });
        break;
    }
    case DType::uint128: {
        using value_type = uint128_t;
        value_type *begin = reinterpret_cast<value_type *>(output.data_ptr());
        utils::flatten_func<Scalar, N>(v, [&begin](const Scalar &a) {
            *begin = a.to<value_type>();
            ++begin;
        });
        break;
    }
#endif
    case DType::Complex64: {
        using value_type = complex_64;
        value_type *begin = reinterpret_cast<value_type *>(output.data_ptr());
        utils::flatten_func<Scalar, N>(v, [&begin](const Scalar &a) {
            *begin = a.to<value_type>();
            ++begin;
        });
        break;
    }
    case DType::Complex128: {
        using value_type = complex_128;
        value_type *begin = reinterpret_cast<value_type *>(output.data_ptr());
        utils::flatten_func<Scalar, N>(v, [&begin](const Scalar &a) {
            *begin = a.to<value_type>();
            ++begin;
        });
        break;
    }
    case DType::int8: {
        using value_type = int8_t;
        value_type *begin = reinterpret_cast<value_type *>(output.data_ptr());
        utils::flatten_func<Scalar, N>(v, [&begin](const Scalar &a) {
            *begin = a.to<value_type>();
            ++begin;
        });
        break;
    }
    case DType::uint8: {
        using value_type = uint8_t;
        value_type *begin = reinterpret_cast<value_type *>(output.data_ptr());
        utils::flatten_func<Scalar, N>(v, [&begin](const Scalar &a) {
            *begin = a.to<value_type>();
            ++begin;
        });
        break;
    }
    case DType::int16: {
        using value_type = int16_t;
        value_type *begin = reinterpret_cast<value_type *>(output.data_ptr());
        utils::flatten_func<Scalar, N>(v, [&begin](const Scalar &a) {
            *begin = a.to<value_type>();
            ++begin;
        });
        break;
    }
    case DType::uint16: {
        using value_type = uint16_t;
        value_type *begin = reinterpret_cast<value_type *>(output.data_ptr());
        utils::flatten_func<Scalar, N>(v, [&begin](const Scalar &a) {
            *begin = a.to<value_type>();
            ++begin;
        });
        break;
    }
    case DType::int32: {
        using value_type = int32_t;
        value_type *begin = reinterpret_cast<value_type *>(output.data_ptr());
        utils::flatten_func<Scalar, N>(v, [&begin](const Scalar &a) {
            *begin = a.to<value_type>();
            ++begin;
        });
        break;
    }
    case DType::uint32: {
        using value_type = uint32_t;
        value_type *begin = reinterpret_cast<value_type *>(output.data_ptr());
        utils::flatten_func<Scalar, N>(v, [&begin](const Scalar &a) {
            *begin = a.to<value_type>();
            ++begin;
        });
        break;
    }
    case DType::int64: {
        using value_type = int64_t;
        value_type *begin = reinterpret_cast<value_type *>(output.data_ptr());
        utils::flatten_func<Scalar, N>(v, [&begin](const Scalar &a) {
            *begin = a.to<value_type>();
            ++begin;
        });
        break;
    }
    case DType::Bool: {
        using value_type = uint_bool_t;
        value_type *begin = reinterpret_cast<value_type *>(output.data_ptr());
        utils::flatten_func<Scalar, N>(v, [&begin](const Scalar &a) {
            *begin = a.to<value_type>();
            ++begin;
        });
        break;
    }
    case DType::TensorObj:
        break;
    }
    return output;
}

template Tensor Tensor::FromInitializer<1ul>(
    utils::NestedInitializerLists_type<Scalar, 1ul>::type, DType);
template Tensor Tensor::FromInitializer<2ul>(
    utils::NestedInitializerLists_type<Scalar, 2ul>::type, DType);
template Tensor Tensor::FromInitializer<3ul>(
    utils::NestedInitializerLists_type<Scalar, 3ul>::type, DType);
template Tensor Tensor::FromInitializer<4ul>(
    utils::NestedInitializerLists_type<Scalar, 4ul>::type, DType);
template Tensor Tensor::FromInitializer<5ul>(
    utils::NestedInitializerLists_type<Scalar, 5ul>::type, DType);
template Tensor Tensor::FromInitializer<6ul>(
    utils::NestedInitializerLists_type<Scalar, 6ul>::type, DType);

Tensor Tensor::clip(Scalar a, Scalar b) const { return functional::clamp(*this, a, b); }

Tensor Tensor::pad(std::vector<size_value_t> p, const char *mode,
                   Scalar value) const {return functional::pad(*this, std::move(p), mode, value);}
Tensor Tensor::unpad(std::vector<size_value_t> p) const {return functional::unpad(*this, std::move(p), /*return_contiguous = */true);}
Tensor Tensor::flip(utils::optional_list list) const{return functional::flip(*this, list);}
Tensor Tensor::flip_view(utils::optional_list list) const{return functional::flip_view(*this, list);}
Tensor Tensor::dilate(size_value_t dil) const { return functional::dilate(*this, dil); }
Tensor Tensor::dilate(size_value_t row_dil, size_value_t col_dil) const { return functional::dilate(*this, row_dil, col_dil); }
Tensor Tensor::dilate(size_value_t channel_dil, size_value_t row_dil, size_value_t col_dil) const { return functional::dilate(*this, channel_dil, row_dil, col_dil); }
Tensor Tensor::undilate(size_value_t dil) const {return functional::undilate_(*this, dil).clone();}
Tensor Tensor::undilate(size_value_t row_dil, size_value_t col_dil) const {return functional::undilate_(*this, row_dil, col_dil).clone(); }
Tensor Tensor::undilate(size_value_t chan_dil, size_value_t row_dil, size_value_t col_dil) const {return functional::undilate_(*this, chan_dil, row_dil, col_dil).clone(); }
Tensor Tensor::undilate_(size_value_t dil) const {return functional::undilate_(*this, dil);}
Tensor Tensor::undilate_(size_value_t row_dil, size_value_t col_dil) const {return functional::undilate_(*this, row_dil, col_dil);}
Tensor Tensor::undilate_(size_value_t chan_dil, size_value_t row_dil, size_value_t col_dil) const { return functional::undilate_(*this, row_dil, col_dil, chan_dil);}

} // namespace nt

#undef __NT__HANDLE_MUTABILITY__
#undef _NT_HANDLE_NULL_TENSORS_NON_EMPTY_4_
#undef _NT_HANDLE_NULL_TENSORS_NON_EMPTY_3_
#undef _NT_HANDLE_NULL_TENSORS_NON_EMPTY_2_
#undef _NT_HANDLE_NULL_TENSORS_NON_EMPTY_1_
#undef _NT_HANDLE_NULL_TENSORS_NON_EMPTY_0_
#undef _NT_HANDLE_NULL_TENSORS_EMPTY_1
#undef _NT_HANDLE_NULL_TENSORS_EMPTY_0
#undef __NT_HANDLE_NULL_TENSORS__
#undef __NT__HANDLE_MUTABILITY__

