#include "Tensor.h"
#include "dtype/ArrayVoid.h"
#include "dtype/DType.h"
#include "dtype/DType_enum.h"
#include "dtype/ranges.h"
#include "functional/functional.h"
#include "memory/iterator.h"
#include "refs/SizeRef.h"

#include <_types/_uint32_t.h>
#include <functional>
// #include <i386/types.h>
#include <algorithm>
#include <set>
#include <ios>
#include <memory.h>
#include <memory>
#include <numeric>
#include <ratio>
#include <sys/_types/_int64_t.h>
#include <sys/wait.h>

#include <cassert>
// #include <format>
#include "dtype/ArrayVoid.hpp"
#include "dtype/Scalar.h"
#include "mp/Threading.h"
#include "mp/simde_ops.h"
#include "permute/permute.h"
#include "types/Types.h"
#include "utils/utils.h"
#include <cmath>
#include <set>
#include <sys/types.h>
#include <type_traits>
#include <vector>

#ifdef USE_PARALLEL
#include <tbb/parallel_for.h>
#include <tbb/parallel_for_each.h>
#include <unistd.h>
#endif

#define assertm(exp, msg) assert(((void)msg, exp))

namespace nt {

DType specify_dtype_from_scalar(const Scalar &s) {
    if (s.isComplex())
        return DType::Complex64;
    if (s.isFloatingPoint())
        return DType::Float32;
    return s.type();
}

Tensor::Tensor(DType dt)
    : _vals(1, dt), _total_size(1), _size({1}), dtype(dt), sub_tensor(false),
      stored_strides(nullptr) {}

Tensor::Tensor(SizeRef v, DType dt)
    : _vals(v.unsigned_multiply(), dt), _size(std::move(v)), dtype(dt),
      sub_tensor(false), stored_strides(nullptr) {
    _total_size = _vals.Size();
}

Tensor::Tensor(ArrayVoid ptr, SizeRef v)
    : _vals(ptr), _size(std::move(v)), sub_tensor(true), dtype(ptr.dtype),
      _total_size(ptr.Size()), stored_strides(nullptr) {
    /* std::cout << "setting dtype"<<std::endl; */
    _total_size = _vals.Size();
    dtype = _vals.dtype;
    /* std::cout << "dtype set"<<std::endl; */
}
Tensor::Tensor(std::string_view _sv)
    : _vals(_sv.size(), DType::uint8),
      _size({static_cast<SizeRef::value_type>(_sv.size())}), sub_tensor(false),
      dtype(DType::uint8), _total_size(_sv.size()), stored_strides(nullptr) {
    char *begin = reinterpret_cast<char *>(data_ptr());
    std::transform(_sv.cbegin(), _sv.cend(), begin,
                   [](const char &v) { return v; });
}

Tensor::Tensor(std::nullptr_t)
    : dtype(nt::DType::Float32), _vals(nullptr), _size(nullptr), _total_size(0),
      stored_strides(nullptr), sub_tensor(false) {}

/* Tensor::Tensor(ArrayVoid ptr, std::shared_ptr<SizeRef> v) */
/* 	:_vals(ptr), _size(v), sub_tensor(true), dtype(ptr.dtype) */
/* { */
/* 	_total_size = _vals.Size(); */
/* } */

Tensor::Tensor(const Tensor &t)
    : _vals(t._vals), _total_size(t._total_size), _size(t._size),
      sub_tensor(false), dtype(t.dtype), stored_strides(t.stored_strides) {}

Tensor::Tensor(Tensor &&t)
    : _vals(std::move(t._vals)), _total_size(t._total_size),
      _size(std::move(t._size)), sub_tensor(false), dtype(t.dtype),
      stored_strides(std::move(t.stored_strides)) {}

Tensor::Tensor(Scalar s)
    : _vals(1, specify_dtype_from_scalar(s)), _total_size(1), _size({1}),
      sub_tensor(false), dtype(specify_dtype_from_scalar(s)),
      stored_strides(nullptr) {
    if (s.isZero()) {
        _vals.fill_(0);
    } else {
        *this = s;
    }
}

/* template<std::size_t N> */
/* Tensor::Tensor(typename utils::NestedInitializerLists_type<Scalar, N>::type
 * v, DType dt) */
/* 	:_vals(SizeRef(utils::aquire_shape<Scalar, N>(v)).multiply(), dt), */
/* 	_size(std::make_unique<SizeRef>(utils::aquire_shape<Scalar, N>(v))), */
/* 	sub_tensor(false), */
/* 	dtype(dt) */
/* { */
/* 	_total_size = _vals.Size(); */
/* 	_vals.execute_function([&v](auto& begin, auto& end){ */
/* 				using value_type = typename
 * std::remove_const<typename decltype(begin)::value_type>::type; */
/* 				utils::flatten_func<N, Scalar>(v, [&begin](const
 * Scalar& a){*begin = a.to<value_type>();}); */
/* 			}); */
/* } */

/* template<> */
/* Tensor::Tensor(typename utils::NestedInitializerLists_type<Scalar, 1>::type
 * v, DType dt) */
/* 	:_vals(SizeRef(utils::aquire_shape<Scalar, 1>(v)).multiply(), dt), */
/* 	_size(std::make_unique<SizeRef>(utils::aquire_shape<Scalar, 1>(v))), */
/* 	sub_tensor(false), */
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
    _vals.swap(other._vals);
    std::swap(_total_size, other._total_size);
    _size.swap(other._size);
    const bool ob = other.sub_tensor;
    const_cast<bool &>(other.sub_tensor) = sub_tensor;
    const_cast<bool &>(sub_tensor) = ob;
    std::swap(dtype, other.dtype);
    std::swap(stored_strides, other.stored_strides);
}

Tensor &Tensor::operator=(const Tensor &t) {
    if (is_null()) {
        _vals = t._vals;
        _size = t._size;
        _total_size = t._total_size;
        dtype = t.dtype;
        stored_strides = t.stored_strides;
        return *this;
    }
    if (shape() == t.shape() && dtype == t.dtype) {
        _vals.transform_function([](auto &a, auto &b) { return b; }, t._vals);
        return *this;
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

Tensor &Tensor::set_(const Tensor &t) {
    utils::THROW_EXCEPTION(t.dtype == dtype, "Expected dtype of $ but got $",
                           dtype, t.dtype);
    utils::THROW_EXCEPTION(t.shape() == shape(),
                           "Expected shape to be $ but got shape $", shape(),
                           t.shape());
    _vals.transform_function([](auto &a, auto &b) { return b; }, t._vals);
    return *this;
}

Tensor &Tensor::operator=(Tensor &&t) {
    if (dtype == DType::TensorObj && sub_tensor && _total_size == 1) {
        *reinterpret_cast<Tensor *>(data_ptr()) = std::move(t);
        return *this;
    }

    _vals = std::move(t._vals);
    _size = std::move(t._size);
    dtype = t.dtype;
    _total_size = t._total_size;
    stored_strides = std::move(t.stored_strides);
    return *this;
}

Tensor &Tensor::operator++() { return add_(1); }

Tensor &Tensor::operator=(Scalar val) {
    _vals = (val);
    return *this;
}

Tensor &Tensor::operator+=(Scalar val) {
    _vals += val;
    return *this;
}

Tensor &Tensor::operator+=(const Tensor &val) {
    if (val.numel() == 1 && val.dtype != DType::TensorObj) {
        return (*this) += val.toScalar();
    }
    return functional::add_(*this, val);
}

Tensor &Tensor::operator-=(Scalar val) {
    _vals -= val;
    return *this;
}

Tensor &Tensor::operator-=(const Tensor &val) {
    if (val.numel() == 1 && val.dtype != DType::TensorObj) {
        return (*this) -= val.toScalar();
    }
    return functional::subtract_(*this, val);
}

Tensor Tensor::operator+(Scalar val) const {
    Tensor outp(_vals + val, shape());
    return std::move(outp);
}

Tensor Tensor::operator+(const Tensor &val) const {
    if (val.numel() == 1 && val.dtype != DType::TensorObj) {
        return (*this) + val.toScalar();
    }
    return functional::add(*this, val);
}

// s/;/{\r\tTensor outp = this->contiguous();\r\toutp._multiply(val);\r\treturn
// std::move(outp);\r}

Tensor Tensor::operator*(Scalar val) const {
    Tensor output(_vals * val, shape());
    return std::move(output);
}

Tensor Tensor::operator*(const Tensor &a) const {
    if (a.numel() == 1 && a.dtype != DType::TensorObj) {
        return (*this) * a.toScalar();
    }
    return functional::hadamard_multiply(*this, a);
}

// s/;/{\r\t_multiply(val);\r\treturn *this;\r}
Tensor &Tensor::operator*=(Scalar val) {
    _vals *= val;
    return *this;
}

Tensor &Tensor::operator*=(const Tensor &a) {
    if (a.numel() == 1 && a.dtype != DType::TensorObj) {
        return (*this) *= a.toScalar();
    }
    functional::hadamard_multiply_this(*this, a);
    return *this;
}

// s/;/{\r\t_subtract(val);\r\treturn *this;\r}
// s/;/{\r\tTensor outp = this->contiguous();\r\toutp._subtract(val);\r\treturn
// std::move(outp);\r}
Tensor Tensor::operator-(Scalar val) const {
    return Tensor(_vals - val, shape());
}

Tensor Tensor::operator-(const Tensor &val) const {
    return functional::subtract(*this, val);
}

Tensor Tensor::operator/(Scalar val) const {
    return Tensor(_vals / val, shape());
}

Tensor Tensor::operator/(const Tensor &val) const {
    if (val.numel() == 1 && val.dtype != DType::TensorObj) {
        return (*this) / val.toScalar();
    }
    return functional::divide(*this, val);
}

Tensor &Tensor::operator/=(Scalar val) {
    _vals /= val;
    return *this;
}

Tensor &Tensor::operator/=(const Tensor &val) {
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

Tensor Tensor::operator==(const Tensor &val) const {
    utils::THROW_EXCEPTION(
        val.shape() == shape(),
        "\nRuntimeError: Expected input tensor to have a shape of $ but got $",
        shape(), val.shape());
    utils::THROW_EXCEPTION(
        val.dtype == dtype,
        "\nRuntimeError: Expected input tensor to have a dtype of $ but got $",
        dtype, val.dtype);
    if (dtype == DType::TensorObj) {
        Tensor output(shape(), DType::TensorObj);
        _vals.transform_function<DType::TensorObj>(
            [](const Tensor &a, const Tensor &b) { return a == b; }, val._vals,
            reinterpret_cast<Tensor *>(output.data_ptr()));
        return std::move(output);
    }
    Tensor output(shape(), DType::Bool);
    _vals.transform_function<WRAP_DTYPES<NumberTypesL, DTypeEnum<nt::DType::Bool>>>(
        [](auto &a, auto &b) -> uint_bool_t { return uint_bool_t(a == b); },
        val._vals, reinterpret_cast<uint_bool_t *>(output.data_ptr()));
    return std::move(output);
}

Tensor Tensor::operator!=(const Tensor &val) const {
    utils::THROW_EXCEPTION(
        val.shape() == shape(),
        "\nRuntimeError: Expected input tensor to have a shape of $ but got $",
        shape(), val.shape());
    utils::THROW_EXCEPTION(
        val.dtype == dtype,
        "\nRuntimeError: Expected input tensor to have a dtype of $ but got $",
        dtype, val.dtype);
    if (dtype == DType::TensorObj) {
        Tensor output(shape(), DType::TensorObj);
        _vals.transform_function<DType::TensorObj>(
            [](const Tensor &a, const Tensor &b) { return a != b; }, val._vals,
            reinterpret_cast<Tensor *>(output.data_ptr()));
        return std::move(output);
    }
    Tensor output(shape(), DType::Bool);
    _vals.transform_function<WRAP_DTYPES<NumberTypesL, DTypeEnum<nt::DType::Bool>>>(
        [](auto &a, auto &b) -> uint_bool_t { return uint_bool_t(a != b); },
        val._vals, reinterpret_cast<uint_bool_t *>(output.data_ptr()));
    return std::move(output);
}

Tensor Tensor::operator>=(const Tensor &val) const {
    utils::THROW_EXCEPTION(
        val.shape() == shape(),
        "\nRuntimeError: Expected input tensor to have a shape of $ but got $",
        shape(), val.shape());
    utils::THROW_EXCEPTION(
        val.dtype == dtype,
        "\nRuntimeError: Expected input tensor to have a dtype of $ but got $",
        dtype, val.dtype);
    if (dtype == DType::TensorObj) {
        Tensor output(shape(), DType::TensorObj);
        _vals.transform_function<DType::TensorObj>(
            [](const Tensor &a, const Tensor &b) { return a >= b; }, val._vals,
            reinterpret_cast<Tensor *>(output.data_ptr()));
        return std::move(output);
    }
    utils::THROW_EXCEPTION(dtype != nt::DType::Bool, "Cannot compare less or greater than with dtype bool");
    Tensor output(shape(), DType::Bool);
    _vals.transform_function<WRAP_DTYPES<NumberTypesL>>(
        [](auto &a, auto &b) -> uint_bool_t { return uint_bool_t(a >= b); },
        val._vals, reinterpret_cast<uint_bool_t *>(output.data_ptr()));
    return std::move(output);
}

Tensor Tensor::operator<=(const Tensor &val) const {
    utils::THROW_EXCEPTION(
        val.shape() == shape(),
        "\nRuntimeError: Expected input tensor to have a shape of $ but got $",
        shape(), val.shape());
    utils::THROW_EXCEPTION(
        val.dtype == dtype,
        "\nRuntimeError: Expected input tensor to have a dtype of $ but got $",
        dtype, val.dtype);
    if (dtype == DType::TensorObj) {
        Tensor output(shape(), DType::TensorObj);
        _vals.transform_function<DType::TensorObj>(
            [](const Tensor &a, const Tensor &b) { return a <= b; }, val._vals,
            reinterpret_cast<Tensor *>(output.data_ptr()));
        return std::move(output);
    }
    utils::THROW_EXCEPTION(dtype != nt::DType::Bool, "Cannot compare less or greater than with dtype bool");
    Tensor output(shape(), DType::Bool);
    _vals.transform_function<WRAP_DTYPES<NumberTypesL>>(
        [](auto &a, auto &b) -> uint_bool_t { return uint_bool_t(a <= b); },
        val._vals, reinterpret_cast<uint_bool_t *>(output.data_ptr()));
    return std::move(output);
}

Tensor Tensor::operator&&(Tensor val) const {
    utils::THROW_EXCEPTION(
        val.shape() == shape(),
        "\nRuntimeError: Expected input tensor to have a shape of $ but got $",
        shape(), val.shape());

    utils::THROW_EXCEPTION(val.dtype == dtype && dtype == DType::Bool,
                           "\nRuntimeError: Expected both tensors to have a "
                           "dtype of Bool but got $ and $",
                           dtype, val.dtype);
    utils::THROW_EXCEPTION(
        val.is_contiguous() && is_contiguous(),
        "\nRuntimeError: Expected both tensors to be contiguous");
    uint_bool_t looking(true);
    Tensor output(shape(), DType::Bool);
    uint_bool_t *o_begin = reinterpret_cast<uint_bool_t *>(output.data_ptr());
    const uint_bool_t *m_begin =
        reinterpret_cast<const uint_bool_t *>(data_ptr());
    uint_bool_t *v_begin = reinterpret_cast<uint_bool_t *>(val.data_ptr());
    uint_bool_t *o_end = o_begin + numel();
    for (; o_begin != o_end; ++o_begin, ++m_begin, ++v_begin) {
        if (*m_begin == looking && *v_begin == looking) {
            *o_begin = looking;
            continue;
        }
        *o_begin = uint_bool_t(false);
    }
    return std::move(output);
}

Tensor Tensor::operator||(Tensor val) const {
    utils::THROW_EXCEPTION(
        val.shape() == shape(),
        "\nRuntimeError: Expected input tensor to have a shape of $ but got $",
        shape(), val.shape());

    utils::THROW_EXCEPTION(val.dtype == dtype && dtype == DType::Bool,
                           "\nRuntimeError: Expected both tensors to have a "
                           "dtype of Bool but got $ and $",
                           dtype, val.dtype);
    utils::THROW_EXCEPTION(
        val.is_contiguous() && is_contiguous(),
        "\nRuntimeError: Expected both tensors to be contiguous");
    uint_bool_t looking(true);
    Tensor output(shape(), DType::Bool);
    uint_bool_t *o_begin = reinterpret_cast<uint_bool_t *>(output.data_ptr());
    const uint_bool_t *m_begin =
        reinterpret_cast<const uint_bool_t *>(data_ptr());
    uint_bool_t *v_begin = reinterpret_cast<uint_bool_t *>(val.data_ptr());
    uint_bool_t *o_end = o_begin + numel();
    for (; o_begin != o_end; ++o_begin, ++m_begin, ++v_begin) {
        if (*m_begin == looking && *v_begin == looking) {
            *o_begin = looking;
            continue;
        }
        *o_begin = uint_bool_t(false);
    }
    return std::move(output);
}

Tensor Tensor::operator>(const Tensor &val) const {
    utils::THROW_EXCEPTION(
        val.shape() == shape(),
        "\nRuntimeError: Expected input tensor to have a shape of $ but got $",
        shape(), val.shape());
    utils::THROW_EXCEPTION(
        val.dtype == dtype,
        "\nRuntimeError: Expected input tensor to have a dtype of $ but got $",
        dtype, val.dtype);
    if (dtype == DType::TensorObj) {
        Tensor output(shape(), DType::TensorObj);
        _vals.transform_function<DType::TensorObj>(
            [](const Tensor &a, const Tensor &b) { return a > b; }, val._vals,
            reinterpret_cast<Tensor *>(output.data_ptr()));
        return std::move(output);
    }
    Tensor output(shape(), DType::Bool);
    _vals.transform_function<WRAP_DTYPES<NumberTypesL>>(
        [](auto &a, auto &b) -> uint_bool_t { return uint_bool_t(a > b); },
        val._vals, reinterpret_cast<uint_bool_t *>(output.data_ptr()));
    return std::move(output);
}

Tensor Tensor::operator<(const Tensor &val) const {
    utils::THROW_EXCEPTION(
        val.shape() == shape(),
        "\nRuntimeError: Expected input tensor to have a shape of $ but got $",
        shape(), val.shape());
    utils::THROW_EXCEPTION(
        val.dtype == dtype,
        "\nRuntimeError: Expected input tensor to have a dtype of $ but got $",
        dtype, val.dtype);
    if (dtype == DType::TensorObj) {
        Tensor output(shape(), DType::TensorObj);
        _vals.transform_function<DType::TensorObj>(
            [](const Tensor &a, const Tensor &b) { return a < b; }, val._vals,
            reinterpret_cast<Tensor *>(output.data_ptr()));
        return std::move(output);
    }
    Tensor output(shape(), DType::Bool);
    _vals.transform_function<WRAP_DTYPES<NumberTypesL>>(
        [](auto &a, auto &b) -> uint_bool_t { return uint_bool_t(a < b); },
        val._vals, reinterpret_cast<uint_bool_t *>(output.data_ptr()));
    return std::move(output);
}

Tensor Tensor::contiguous() const {
    if (is_contiguous())
        return *this;
    /* std::copy((const float*)_vals.get(), (const float*)(_vals.get() +
     * _total_size), copy.get()); */
    return Tensor(_vals.contiguous(), _size);
}

Tensor Tensor::clone() const { return Tensor(_vals.clone(), _size); }

const size_t Tensor::dims() const { return shape().size(); }

template <typename T> T &Tensor::item() {
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

Tensor Tensor::operator[](size_value_t x) {
    x = x < 0 ? x + dims() : x;
    uint64_t nx = static_cast<uint64_t>(x);
    if (_total_size == 1) {
        utils::THROW_EXCEPTION(
            x == 0,
            "\nRuntimeError: Expected singleton to have indexed "
            "of at most $ but instead got $",
            0, x);
        return *this;
    }

    // I do really like the assertion below
    // assert(x < shape()[0]);
    utils::THROW_EXCEPTION(
        x < shape()[0], "RuntimeError: Expected x to be less than $ but got $",
        shape()[0], x);
    /* if(dims() == 1 && dtype == DType::TensorObj) */
    /* 	return reinterpret_cast<Tensor*>(data_ptr())[x]; */
    if (dims() == 1) {
        return Tensor(_vals.share_array(x, 1), SizeRef({1}));
        /* std::cout<<"dims 1"<<std::endl; */
        /* std::cout << n_size.multiply()<<std::endl; */
        /* std::cout << n_size<<std::endl; */
        /* return Tensor(_vals.share_array( */
    }

    SizeRef n_size = shape().pop_front();
    uint64_t mult = static_cast<uint64_t>(n_size.multiply());
    return Tensor(_vals.share_array(nx * mult, mult), std::move(n_size));
    // return Tensor(_vals.share_array(nx * mult, mult), std::move(n_size));
}

const Tensor Tensor::operator[](size_value_t x) const {
    x = x < 0 ? x + dims() : x;
    uint64_t nx = static_cast<uint64_t>(x);
    if (_total_size == 1) {
        utils::THROW_EXCEPTION(
            x == 0,
            "\nRuntimeError: Expected singleton to have indexed "
            "of at most $ but instead got $",
            0, x);
        return *this;
    }

    utils::THROW_EXCEPTION(
        x < shape()[0], "RuntimeError: Expected x to be less than $ but got $",
        shape()[0], x);
    /* if(dims() == 1 && dtype == DType::TensorObj) */
    /* 	return reinterpret_cast<const Tensor*>(data_ptr())[x]; */
    if (dims() == 1) {
        return Tensor(_vals.share_array(x, 1), SizeRef({1}));
    }

    SizeRef n_size = shape().pop_front();
    uint64_t mult = static_cast<uint64_t>(n_size.multiply());
    return Tensor(_vals.share_array(nx * mult, mult), std::move(n_size));
}


Tensor Tensor::operator[](const Tensor &t) const {
    utils::THROW_EXCEPTION(
        t.dtype == DType::Bool || t.dtype == DType::TensorObj || t.dtype == DType::int64,
        "RuntimeError: expected DType Bool, TensorObj, or int64 but got $", t.dtype);
    if (t.dtype == DType::TensorObj) {
        //if it is operations of tensors of tensors, then jus repeat the operation
        if(dtype == DType::TensorObj){
            Tensor output = Tensor::makeNullTensorArray(numel());
            Tensor* ts_begin = reinterpret_cast<Tensor*>(output.data_ptr());
            Tensor* ts_end = ts_begin + numel();
            const Tensor* begin = reinterpret_cast<const Tensor*>(data_ptr());
            const Tensor* t_begin = reinterpret_cast<const Tensor*>(t.data_ptr());
            for(;ts_begin != ts_end; ++ts_begin, ++begin, ++t_begin)
                *ts_begin = (*begin)[*t_begin];
            return std::move(output);
        }
        utils::THROW_EXCEPTION(
            t.is_contiguous(),
            "RuntimeError: Expected indexing tensor to be contiguous");
        utils::THROW_EXCEPTION(
            t.numel() == dims(),
            "Expected indexing tensor to have $ tensors but had $", dims(),
            t.numel());
        ArrayVoid my_vals = _vals.bucket_all_indices();
        const Tensor *begin = reinterpret_cast<const Tensor *>(t.data_ptr());
        const Tensor *end = begin + t.numel();
        const Tensor *begin_cpy = begin;
        for (; begin != end; ++begin)
            utils::THROW_EXCEPTION(
                begin->dtype == DType::int64 && begin->is_contiguous(),
                "Expected indexing tensor to have dtype int64 but got $ and expected to be contiguous",
                begin->dtype);
        begin = begin_cpy;

        // making the strides for indexing:
        const std::vector<size_value_t> s = strides();
        std::vector<const size_value_t> ns;
        for (size_value_t i = 0; i < s.size(); ++i)
            ns.emplace_back(s[i]);

        // keeping track of each int64_t pointer for the indexing
        const size_value_t *ptrs[dims()];
        size_value_t i = 0;
        for (; begin != end; ++begin, ++i) {
            ptrs[i] = reinterpret_cast<const int64_t *>(begin->data_ptr());
        }
        // making a new ArrayVoid to keep track of all the indices
        const size_value_t &n_size = begin_cpy->numel();
        ArrayVoid new_vals = _vals.new_strides(n_size);
        void **out_begin = new_vals.stride_begin();
        void **my_begin = my_vals.stride_begin();
        // finding each index
        for (size_value_t i = 0; i < n_size; ++i, ++out_begin) {
            // getting the ith index to copy
            size_value_t index = 0;
            for (size_value_t j = 0; j < t.numel() - 1; ++j) {
                index += ptrs[j][i] * ns[j + 1];
            }
            index += ptrs[t.numel() - 1][i];
            *out_begin = my_begin[index];
        }

        return Tensor(new_vals, {static_cast<size_value_t>(n_size)});
    }
    else if (t.dtype == DType::int64){
       utils::THROW_EXCEPTION(
            t.is_contiguous(),
            "RuntimeError: Expected indexing tensor to be contiguous");
        const int64_t* t_begin = reinterpret_cast<const int64_t*>(t.data_ptr());
        const int64_t* t_end = reinterpret_cast<const int64_t*>(t.data_ptr_end());
        if(dims() == 1){
            const size_value_t &n_size = t.numel();
            ArrayVoid my_vals = _vals.bucket_all_indices();
            ArrayVoid new_vals = _vals.new_strides(n_size);
            void **out_begin = new_vals.stride_begin();
            void **my_begin = my_vals.stride_begin();
            const int64_t& max = shape()[0];
            for(;t_begin != t_end; ++t_begin, ++out_begin){
                utils::THROW_EXCEPTION(*t_begin < max,
                                "Trying to get index of tensor at dim 0 of $ with that dimension only holding $", *t_begin, max);
                *out_begin = my_begin[*t_begin];
            }
            return Tensor(new_vals, {static_cast<size_value_t>(n_size)});
        }
        std::vector<SizeRef::value_type> Vec = this->shape().Vec();
        Vec[0] = t.numel();
        Tensor split = this->split_axis(0);
        std::vector<Tensor> catting(t.numel(), Tensor::Null());
        auto out_begin = catting.begin();
        Tensor* s_begin = reinterpret_cast<Tensor*>(split.data_ptr());
        for(;t_begin != t_end; ++t_begin, ++out_begin){
            *out_begin = s_begin[*t_begin];
        }
        return functional::cat(std::move(catting)).view(SizeRef(std::move(Vec)));
    
    }
    utils::THROW_EXCEPTION(
        t.dtype == DType::Bool,
        "RuntimeError (at end, logic error): expected DType Bool, TensorObj, or int64 but got $", t.dtype);
    utils::THROW_EXCEPTION(
        t.is_contiguous(),
        "RuntimeError: Expected indexing tensor to be contiguous");
    if(t.numel() != numel() && t.numel() == shape()[0]){
        const uint_bool_t *begin =
            reinterpret_cast<const uint_bool_t *>(t.data_ptr());
        const uint_bool_t *end = begin + t.numel();

        std::vector<SizeRef::value_type> Vec = this->shape().Vec();
        Vec[0] = functional::count(t);
        Tensor split = this->split_axis(0);
        std::vector<Tensor> catting(functional::count(t), Tensor::Null());
        auto out_begin = catting.begin();
        Tensor* s_begin = reinterpret_cast<Tensor*>(split.data_ptr());
        for(;begin != end; ++begin, ++s_begin){
            if(*begin){
                *out_begin++ = *s_begin;
            }
        }
        return functional::cat(std::move(catting)).view(SizeRef(std::move(Vec)));
 
    }
    utils::THROW_EXCEPTION(
        t.numel() == numel(),
        "Numels must be equal for [] operator on Tensor DType::Bool, or equal to shape()[0] ($)", shape()[0]);

    ArrayVoid my_vals = _vals.bucket_all_indices();
    size_value_t new_size = ::nt::functional::count(t);
    ArrayVoid new_vals = _vals.new_strides(new_size);
    const uint_bool_t *begin =
        reinterpret_cast<const uint_bool_t *>(t.data_ptr());
    const uint_bool_t *end = begin + numel();
    void **my_stride = my_vals.stride_begin();
    void **new_stride = new_vals.stride_begin();
    for (; begin != end; ++begin, ++my_stride) {
        if (*begin == true) {
            *new_stride = *my_stride;
            ++new_stride;
        }
    }
    return Tensor(std::move(new_vals), {static_cast<size_value_t>(new_size)});
}

Tensor Tensor::index_except(int64_t dim, int64_t excluding_index) const {
    dim = dim < 0 ? dim + dims() : dim;
    utils::THROW_EXCEPTION(dim < dims() && dim >= 0, "Got invalid dim $", dim);
    bool end_dim = (dim == dim-1);
    auto sh = shape();
    excluding_index = excluding_index < 0 ? excluding_index + sh[dim] : excluding_index;
    utils::THROW_EXCEPTION(excluding_index < sh[dim] && excluding_index >= 0, "Got invalid index $", excluding_index);
    std::vector<size_value_t> Vec = sh.Vec();
    Vec[dim] -= 1;

    Tensor split = this->split_axis(dim);
    int64_t total = (split.numel() / sh[dim]) * Vec[dim];
    Tensor out = makeNullTensorArray(total);
    Tensor* o_begin = reinterpret_cast<Tensor*>(out.data_ptr());
    Tensor* s_begin = reinterpret_cast<Tensor*>(split.data_ptr());
    Tensor* s_end = reinterpret_cast<Tensor*>(split.data_ptr_end());
    int64_t counter = 0;
    const int64_t max_dim = (sh[dim]-1);
    while(s_begin != s_end){
        if(counter != excluding_index){
            *o_begin = *s_begin;
            ++o_begin;
        }
        ++s_begin;
        counter = counter == max_dim ? 0 : counter + 1;
    }
    if(end_dim){
        std::swap(Vec[dim], Vec[dim-1]);
        return functional::cat_unordered(out).view(SizeRef(std::move(Vec))).transpose(-1, -2);
    }
    return functional::cat_unordered(out).view(SizeRef(std::move(Vec)));
}

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

Tensor get_range(const Tensor &t, const my_range &r, size_t idx) {
    if (idx == 0) {
        if (idx_is_total(r, t.shape(), 0)) {
            return t.split_axis(0);
        }
        return t.split_axis(0)[r];
    }
    utils::THROW_EXCEPTION(t.dtype == DType::TensorObj,
                           "Error with dtype format");
    const Tensor *begin_i = reinterpret_cast<const Tensor *>(t.data_ptr());
    const Tensor *end_i = begin_i + t.numel();
    if (begin_i->dims() == 1) {
        /* std::cout << "doing dims1"<<std::endl; */
        if (idx_is_total(r, begin_i->shape(), 0)) {
            return t;
        }
        Tensor output = Tensor::makeNullTensorArray(t.numel());
        Tensor *begin_o = reinterpret_cast<Tensor *>(output.data_ptr());
        for (; begin_i != end_i; ++begin_i, ++begin_o) {
            *begin_o = (*begin_i)[r];
        }
        return std::move(output);
    }
    Tensor output = Tensor::makeNullTensorArray(r.length() * t.numel());
    Tensor *begin_o = reinterpret_cast<Tensor *>(output.data_ptr());

    for (; begin_i != end_i; ++begin_i) {
        /* std::cout << "before split shape: "<<begin_i->shape()<<std::endl; */
        Tensor o = begin_i->split_axis(0);
        Tensor *o_b = reinterpret_cast<Tensor *>(o.data_ptr()) + r.begin;
        Tensor *o_e = reinterpret_cast<Tensor *>(o.data_ptr()) + r.end;
        for (; o_b != o_e; ++o_b, ++begin_o) {
            /* std::cout << o_b->shape() << std::endl; */
            *begin_o = std::move(*o_b);
        }
    }
    return std::move(output);
}

Tensor Tensor::operator[](std::vector<my_range> r) {
    utils::THROW_EXCEPTION(
        r.size() <= dims(),
        "Expected to get less than or equal to $ ranges but got $ ranges",
        dims(), r.size());
    while (r.size() > 0 && idx_is_total(r.back(), shape(), r.size() - 1)) {
        r.pop_back();
    }
    if (r.size() == 1) {
        return (*this)[r[0]];
    } else if (r.size() == 0) {
        return *this;
    }
    for (size_value_t i = 0; i < r.size(); ++i) {
        r[i].fix(shape()[i]);
    }

    Tensor outs = get_range(*this, r[0], 0);
    for (size_t i = 1; i < r.size(); ++i) {
        outs = get_range(outs, r[i], i);
    }
    std::vector<size_value_t> n_shape = shape().Vec();
    for (size_value_t i = 0; i < r.size(); ++i)
        n_shape[i] = r[i].length();

    return nt::functional::cat_unordered(outs).view(n_shape);
}

const Tensor Tensor::operator[](std::vector<my_range> r) const {
    utils::THROW_EXCEPTION(
        r.size() <= dims(),
        "Expected to get less than or equal to $ ranges but got $ ranges",
        dims(), r.size());
    while (r.size() > 0 && idx_is_total(r.back(), shape(), r.size() - 1)) {
        r.pop_back();
    }
    if (r.size() == 1) {
        return (*this)[r[0]];
    } else if (r.size() == 0) {
        return *this;
    }
    for (size_value_t i = 0; i < r.size(); ++i) {
        r[i].fix(shape()[i]);
    }

    Tensor outs = get_range(*this, r[0], 0);
    for (size_t i = 1; i < r.size(); ++i) {
        outs = get_range(outs, r[i], i);
    }
    std::vector<size_value_t> n_shape = shape().Vec();
    for (size_value_t i = 0; i < r.size(); ++i)
        n_shape[i] = r[i].length();

    return nt::functional::cat_unordered(outs).view(n_shape);
}

const Tensor Tensor::operator[](std::vector<size_value_t> xs) const {
    utils::THROW_EXCEPTION(
            xs.size() <= dims(),
            "Expected to get less than or equal to $ indices but got $ indices",
            dims(), xs.size());
    if(xs.size() == 1){return (*this)[xs[0]];}
    if(xs.size() == 0){return *this;}
    for (size_value_t i = 0; i < xs.size(); ++i) {
        xs[i] = xs[i] < 0 ? xs[i] + dims() : xs[i];
    }

    uint64_t cur_mult = 1;
    auto begin = xs.begin();
    auto end = xs.end();
    cur_mult *= *begin;
    SizeRef n_size = shape().pop_front();
    ++begin;
    for(;begin != end; ++begin){
        cur_mult *= *begin;
        n_size = n_size.pop_front();
    }
    uint64_t mult = n_size.size() == 0 ? 1 : static_cast<uint64_t>(n_size.multiply());
    return Tensor(_vals.share_array(cur_mult * mult, mult), std::move(n_size));

}

Tensor Tensor::operator[](std::vector<size_value_t> xs) {
    utils::THROW_EXCEPTION(
            xs.size() <= dims(),
            "Expected to get less than or equal to $ indices but got $ indices",
            dims(), xs.size());
    if(xs.size() == 1){return (*this)[xs[0]];}
    if(xs.size() == 0){return *this;}
    for (size_value_t i = 0; i < xs.size(); ++i) {
        xs[i] = xs[i] < 0 ? xs[i] + dims() : xs[i];
    }

    uint64_t cur_mult = 1;
    auto begin = xs.begin();
    auto end = xs.end();
    cur_mult *= *begin;
    SizeRef n_size = shape().pop_front();
    ++begin;
    for(;begin != end; ++begin){
        cur_mult *= *begin;
        n_size = shape().pop_front();
    }
    uint64_t mult = static_cast<uint64_t>(n_size.multiply());
    return Tensor(_vals.share_array(cur_mult * mult, mult), std::move(n_size));

}


// this was the old way which was not nearly as efficient:
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

Tensor Tensor::operator[](const my_range &x) {
    size_value_t a = x.begin < 0 ? x.begin + shape()[0] : x.begin;
    size_value_t b = x.end < 0 ? x.end + shape()[0] : x.end;
    // std::cout << "doing range from a:"<<a<<" to b:"<<b<<std::endl;
    utils::THROW_EXCEPTION(
        a < shape()[0] && b <= shape()[0],
        "Expected a,b to be less than $ for dim $ but got (a = $), (b = $)",
        shape()[0], dims(), a, b);
    std::vector<typename SizeRef::ArrayRefInt::value_type> vec = shape().Vec();
    vec[0] = b - a;
    SizeRef n_size(std::move(vec));
    return Tensor(
        _vals.share_array(a * n_size.multiply(1), (b - a) * n_size.multiply(1)),
        std::move(n_size));
}

const Tensor Tensor::operator[](const my_range &x) const {
    size_value_t a = x.begin < 0 ? x.begin + shape()[0] : x.begin;
    size_value_t b = x.end < 0 ? x.end + shape()[0] : x.end;
    utils::THROW_EXCEPTION(
        a < shape()[0] && b <= shape()[0],
        "Expected a,b to be less than $ for dim $ but got (a = $), (b = $)",
        shape()[0], dims(), a, b);
    std::vector<typename SizeRef::ArrayRefInt::value_type> vec = shape().Vec();
    vec[0] = b - a;
    SizeRef n_size(std::move(vec));
    return Tensor(
        _vals.share_array(a * n_size.multiply(1), (b - a) * n_size.multiply(1)),
        std::move(n_size));
}

void Tensor::print() const { std::cout << *this << std::endl; }


//this function ensures that int8_t and uint8_t are printed properly, otherwise it just comes out as random-looking characters
//and it is really annoying
template <typename IteratorOut>
void print_byte_tensor_template_func(IteratorOut &data, IteratorOut &end,
                                std::ostream &out, const SizeRef &t_s,
                                bool sub_matrix, uint32_t print_space) {
    if (!sub_matrix) {
        out << t_s << std::endl;
        out << "Tensor(";
        print_space += 7;
    }
    if (t_s.empty()) {
        out << "[]" << std::endl;
        return;
    }
    if (t_s.size() == 1) {
        out << "[";
        for (; (data + 1) != end; ++data)
            out << +(*data) << ",";
        out << +(*data) << "]";
        if (!sub_matrix) {
            if(utils::g_print_dtype_on_tensor){
                out << ", " << t_s << ')' << std::endl;
            }else{
                out <<", " << t_s << ')';
            }
        }
        return;
    }
    if (t_s.size() == 2) {
        const Tensor::size_value_t _rows = t_s.front();
        const Tensor::size_value_t _cols = t_s.back();
        out << "[";
        ++print_space;
        for (Tensor::size_value_t x = 0; x < _rows; ++x) {
            if (x != 0 || !sub_matrix) {
                out << "[";
            }
            for (Tensor::size_value_t y = 0; y < _cols - 1; ++y) {
                out << +(*data) << ",";
                ++data;
            }
            if (x == _rows - 1) {
                out << +(*data) << "]]" << std::endl;
            } else {
                out << +(*data) << "],";
                out << std::endl;
                for (uint32_t i = 0; i < print_space; ++i)
                    out << " ";
            }
            ++data;
        }
        if (!sub_matrix) {
            out << "])" << std::endl;
        } else {
            out << std::endl;
            for (uint32_t i = 0; i < print_space - 1; ++i)
                out << " ";
        }
        return;
    }
    for (Tensor::size_value_t i = 0; i < t_s.front(); ++i) {
        if (!sub_matrix && i != 0) {
            for (Tensor::size_value_t j = 0; j < 7; ++j)
                out << " ";
        }
        out << "[";
        print_byte_tensor_template_func(data, end, out, t_s.pop_front(), true,
                                   print_space + 1);
        if (!sub_matrix) {
            out << "\033[F";
            for (Tensor::size_value_t j = 0; j < t_s.size() - 3; ++j)
                out << "]";
            if (i != t_s.front() - 1) {
                out << ',';
                out << std::endl << std::endl;
            } else {
                out << ")";
                if(utils::g_print_dtype_on_tensor){
                    out << std::endl;
                }
            }
        }
    }
}

template <typename IteratorOut>
void print_tensor_template_func(IteratorOut &data, IteratorOut &end,
                                std::ostream &out, const SizeRef &t_s,
                                bool sub_matrix, uint32_t print_space) {
    
    if (!sub_matrix) {
        out << t_s << std::endl;
        out << "Tensor(";
        print_space += 7;
    }
    if (t_s.empty()) {
        out << "[]" << std::endl;
        return;
    }
    if (t_s.size() == 1) {
        out << "[";
        for (; (data + 1) != end; ++data)
            out << *data << ",";
        out << *data << "]";
        if (!sub_matrix) {
            if(utils::g_print_dtype_on_tensor){
                out << ')' << std::endl;
            }else{
                out << ')';
            }
        }
        return;
    }
    if (t_s.size() == 2) {
        const Tensor::size_value_t _rows = t_s.front();
        const Tensor::size_value_t _cols = t_s.back();
        out << "[";
        ++print_space;
        for (Tensor::size_value_t x = 0; x < _rows; ++x) {
            if (x != 0 || !sub_matrix) {
                out << "[";
            }
            for (Tensor::size_value_t y = 0; y < _cols - 1; ++y) {
                out << *data << ",";
                ++data;
            }
            if (x == _rows - 1) {
                out << *data << "]]" << std::endl;
            } else {
                out << *data << "],";
                out << std::endl;
                for (uint32_t i = 0; i < print_space; ++i)
                    out << " ";
            }
            ++data;
        }
        if (!sub_matrix) {
            out << "])" << std::endl;
        } else {
            out << std::endl;
            for (uint32_t i = 0; i < print_space - 1; ++i)
                out << " ";
        }
        return;
    }
    for (Tensor::size_value_t i = 0; i < t_s.front(); ++i) {
        if (!sub_matrix && i != 0) {
            for (Tensor::size_value_t j = 0; j < 7; ++j)
                out << " ";
        }
        out << "[";
        print_tensor_template_func(data, end, out, t_s.pop_front(), true,
                                   print_space + 1);
        if (!sub_matrix) {
            out << "\033[F";
            for (Tensor::size_value_t j = 0; j < t_s.size() - 3; ++j)
                out << "]";
            if (i != t_s.front() - 1) {
                out << ',';
                out << std::endl << std::endl;
            } else {
                out << ")";
                if(utils::g_print_dtype_on_tensor){
                    out << std::endl;
                }
            }
        }
    }
}

inline static constexpr auto print_tensor_func = [](auto data, auto end,
                                                    std::ostream &out,
                                                    const SizeRef &t_s,
                                                    bool sub_matrix,
                                                    uint32_t print_space) {
    using value_t = utils::IteratorBaseType_t<decltype(data)>;
    if constexpr (std::is_same_v<value_t, int8_t> || std::is_same_v<value_t, uint8_t>){
        print_byte_tensor_template_func<decltype(data)>(
            data, end, out, t_s, false, print_space);
        return;
    }
    if (!sub_matrix) {
        out << "Tensor([";
        print_space += 8;
    }
    if (t_s.empty() || t_s.front() == 0) {
        out << "], {0})";
        return;
    }
    if (t_s.size() == 1) {
        for (; (data + 1) != end; ++data) { // this is the issue, the iterator
                                            // can't handle + 1 for some reason?
            out << *data << ",";
        }
        out << *data << "]";
        if (!sub_matrix) {
            if(utils::g_print_dtype_on_tensor){
                out << ", " << t_s << ')' << std::endl;
            }else{
                out <<", " << t_s << ')';
            }

        }
        return;
    }
    if (t_s.size() == 2) {
        const Tensor::size_value_t _rows = t_s.front();
        const Tensor::size_value_t _cols = t_s.back();
        /* out <<"["; */
        ++print_space;
        for (Tensor::size_value_t x = 0; x < _rows; ++x) {
            if (x != 0 || !sub_matrix) {
                out << "[";
            }
            for (Tensor::size_value_t y = 0; y < _cols - 1; ++y) {
                out << *data << ",";
                ++data;
            }
            if (x == _rows - 1) {
                if (!sub_matrix) {
                    out << *data << "]], " << t_s << ")" << std::endl;
                } else {
                    out << *data << "]]" << std::endl;
                }
            } else {
                out << *data << "],";
                out << std::endl;
                for (uint32_t i = 0; i < print_space; ++i)
                    out << " ";
            }
            ++data;
        }
        /* if(!sub_matrix){out << "], "<<t_s<<")" << std::endl;} */
        if (sub_matrix) {
            out << std::endl;
            for (uint32_t i = 0; i < print_space - 1; ++i)
                out << " ";
        }
        return;
    }
    for (Tensor::size_value_t i = 0; i < t_s.front(); ++i) {
        if (!sub_matrix && i != 0) {
            for (uint32_t j = 0; j < 9; ++j)
                out << " ";
        }
        out << "[";
        print_tensor_template_func<decltype(data)>(
            data, end, out, t_s.pop_front(), true, print_space + 1);
        if (!sub_matrix) {
            out << "\033[F";
            for (Tensor::size_value_t j = 0; j < t_s.size() - 3; ++j)
                out << "]";
            if (i != t_s.front() - 1) {
                /* out << ','; */
                out << std::endl << std::endl;
            } else {
                out << "], " << t_s << ")";
                if(utils::g_print_dtype_on_tensor){
                    out << std::endl;
                }
            }
        }
    }
};

std::ostream &operator<<(std::ostream &out, const Tensor &_t) {
    if (_t.is_null()) {
        return out << "Tensor (Null)";
    }
    if (_t.is_empty()) {
        if(utils::g_print_dtype_on_tensor){
            return out << "Tensor([], " << _t.dtype << ")";
        }else{
            return out << "Tensor([])";
        }
    }
    if (_t.dtype == DType::TensorObj && _t.numel() == 1) {
        return out << *reinterpret_cast<const Tensor *>(_t.data_ptr())
                   << std::endl;
    }
    if (_t.dtype == DType::Bool)
        out << std::boolalpha;
    _t.arr_void().cexecute_function<WRAP_DTYPES<AllTypesL>>(
        print_tensor_func, out, _t.shape(), false, 0);
    if (_t.dtype == DType::Bool)
        out << std::noboolalpha;
    if(utils::g_print_dtype_on_tensor){
        out << std::endl << _t.dtype;
    }
    return out;
}

Tensor Tensor::view(SizeRef nv) const {
    size_value_t total = nv.multiply();
    utils::THROW_EXCEPTION(
        total == _total_size,
        "problem with converting shape $ to shape $ my numel is $ and shape numel is $", shape(), nv,
        _total_size, total);
    /* assert(total == _total_size); */
    return Tensor(_vals, std::move(nv));
}

// this is a view to happen to every single tensor
Tensor Tensor::view_Tensors(SizeRef nv) const {
    utils::THROW_EXCEPTION(
        dtype == DType::TensorObj,
        "Expected view_Tensors to be used on a tensor of tensors");
    Tensor outp = Tensor::makeNullTensorArray(numel());
    Tensor *outputIt = reinterpret_cast<Tensor *>(outp.data_ptr());
    _vals.transform_function<DType::TensorObj>(
        [&nv](auto &inp) { return inp.view(nv); }, outputIt);
    return std::move(outp);
}


Tensor Tensor::view_Tensor_vector(std::vector<size_value_t> nv) const {
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
    return std::move(outp);
}

// this is a transpose to happen to every single tensor
Tensor Tensor::transpose_Tensors(size_value_t a, size_value_t b) const {
    utils::THROW_EXCEPTION(
        dtype == DType::TensorObj,
        "Expected transpose_Tensors to be used on a tensor of tensors");
    Tensor outp = Tensor::makeNullTensorArray(numel());
    Tensor *outputIt = reinterpret_cast<Tensor *>(outp.data_ptr());
    _vals.transform_function<DType::TensorObj>(
        [&a, &b](auto &inp) { return inp.transpose(a, b); }, outputIt);
    return std::move(outp);
}


Tensor Tensor::flatten(size_value_t _a, size_value_t _b) const {
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
    return Tensor(_vals, std::move(n_vals));
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

void printTransposes(
    const std::vector<std::pair<Tensor::size_value_t, Tensor::size_value_t>>
        &transposes) {
    for (const auto &p : transposes) {
        std::cout << "(" << p.first << "," << p.second << ") ";
    }
    std::cout << std::endl;
}

void reduceTransposesHelper(
    std::vector<std::pair<Tensor::size_value_t, Tensor::size_value_t>> &ts) {}

template <typename Function>
void reduceTransposesHelper(
    std::vector<std::pair<Tensor::size_value_t, Tensor::size_value_t>> &ts,
    Function &&func) {
    Tensor::size_value_t i = 0;
    std::forward<Function &&>(func)(ts, i);
}

template <typename Function, typename... Functions>
void reduceTransposesHelper(
    std::vector<std::pair<Tensor::size_value_t, Tensor::size_value_t>> &ts,
    Function &&func, Functions &&...funcs) {
    size_t i = 0;
    std::forward<Function &&>(func)(ts, i);
    reduceTransposesHelper(ts, std::forward<Functions &&>(funcs)...);
}

template <typename... Functions>
void reduceTransposes(
    std::vector<std::pair<Tensor::size_value_t, Tensor::size_value_t>> &ts,
    Functions &&...funcs) {
    reduceTransposesHelper(ts, std::forward<Functions &&>(funcs)...);
}

void deleteAtIndex(
    std::vector<std::pair<Tensor::size_value_t, Tensor::size_value_t>> &vec,
    size_t index) {
    // Check if index is valid
    if (index >= vec.size()) {
        std::cerr << "Index out of range!" << std::endl;
        return;
    }

    // Erase element at the specified index
    vec.erase(vec.begin() + index);
}

bool are_equal(std::pair<Tensor::size_value_t, Tensor::size_value_t> &a,
               std::pair<Tensor::size_value_t, Tensor::size_value_t> &b) {
    return a.first == b.first && a.second == b.second;
}

void deleteLastTwoIfSameTranspose(
    std::vector<std::pair<Tensor::size_value_t, Tensor::size_value_t>> &vec,
    size_t i) {
    // Check if the vector has at least two elements
    if (i >= vec.size())
        return;
    if (vec.size() - i < 2) {
        i = vec.size();
        return; // Nothing to delete
    }

    // Check if the last two elements are the same as the current two elements
    if (vec.size() - i >= 4 && are_equal(vec[i + 2], vec[i]) &&
        are_equal(vec[i + 3], vec[i + 1])) {
        deleteAtIndex(vec, i + 2);
        deleteAtIndex(vec, i + 2);
        i += 2;
        deleteLastTwoIfSameTranspose(vec, i);
    }
    i += 2;
    deleteLastTwoIfSameTranspose(vec, i);
}

void deleteTransposeRepeat(
    std::vector<std::pair<Tensor::size_value_t, Tensor::size_value_t>> &vec,
    size_t i) {
    if (i >= vec.size())
        return;
    if (vec.size() - i < 2) {
        i = vec.size();
        return;
    }
    if (are_equal(vec[i], vec[i + 1])) {
        deleteAtIndex(vec, i);
        deleteAtIndex(vec, i);
        deleteTransposeRepeat(vec, i);
    }
    ++i;
    deleteTransposeRepeat(vec, i);
}

uint64_t calc_index(const std::vector<uint64_t> &current_shape,
                    const std::vector<Tensor::size_value_t> &strides) {
    uint64_t index = 0;
    for (uint32_t i = 0; i < current_shape.size() - 1; ++i) {
        index += current_shape[i] * strides[i];
    }
    return index + current_shape.back();
}

void increment_shape(std::vector<uint64_t> &current_shape,
                     const std::vector<Tensor::size_value_t> &out_shape) {
    for (int i = current_shape.size() - 1; i >= 0; --i) {
        if (current_shape[i] < out_shape[i] - 1) {
            ++current_shape[i];
            break;
        } else {
            current_shape[i] = 0;
        }
    }
}

bool all_zeros(const std::vector<uint64_t> &current_shape) {
    return std::all_of(current_shape.cbegin(), current_shape.cend(),
                       [](const auto &val) { return val == 0; });
}

bool is_equal(const std::vector<uint64_t> &current_shape,
              const std::vector<uint64_t> &maxing) {
    return current_shape == maxing;
}

template <typename T> void print_vector(const std::vector<T> &vec) {
    std::cout << '{';
    for (uint32_t i = 0; i < vec.size() - 1; ++i)
        std::cout << vec[i] << ',';
    std::cout << vec.back() << '}' << std::endl;
}

// TODO:
//  I am sure there is a more elegant way to do this
//  however, this works, and is fairly fast, and requires really no coppies
//
// this is fairly inefficient, and could definitely be made to be more efficient
// it wouldn't be super hard, just use the permute functions above
Tensor Tensor::permute(std::vector<size_value_t> Perm)
    const { // going to figure out a better one for this
    utils::THROW_EXCEPTION(
        Perm.size() <= dims(),
        "Expected to permute at most $ dims but got $ dims to permute", dims(),
        Perm.size());
    std::vector<std::pair<size_value_t, size_value_t>> transposes;
    std::vector<size_value_t> cur_strides = getChangedStrides();
    size_value_t min_dim = dims();
    size_value_t max_dim = 0;
    for (uint32_t i = 0; i < Perm.size(); ++i) {
        if (Perm[i] != i) {
            for (uint32_t j = 0; j < Perm.size(); ++j) {
                if (Perm[j] == i) {
                    transposes.push_back(
                        std::pair<size_value_t, size_value_t>(i, j));
                    std::swap(Perm[i], Perm[j]);
                    std::swap(cur_strides[i], cur_strides[j]);
                    min_dim = std::min<size_value_t>(min_dim, i);
                    min_dim = std::min<size_value_t>(min_dim, j);
                    max_dim = std::max<size_value_t>(max_dim, i);
                    max_dim = std::max<size_value_t>(max_dim, j);
                }
            }
        }
    }
    if (transposes.size() == 1) {
        return transpose(static_cast<size_value_t>(transposes[0].first),
                         static_cast<size_value_t>(transposes[0].second));
    }

    /* std::vector<std::pair<size_value_t, size_value_t> > nTs; */
    /* for(uint32_t i = 0; i < transposes.size(); ++i){ */
    /* 	if(std::abs(transposes[i].first - transposes[i].second) <= 1){ */
    /* 		nTs.push_back(transposes[i]); */
    /* 		continue; */
    /* 	} */
    /* 	size_value_t _a = transposes[i].first; */
    /* 	size_value_t _b = transposes[i].second; */
    /* 	if(_a > _b){std::swap(_a, _b);} */
    /* 	size_value_t original_a = _a; */
    /* 	nTs.push_back(std::pair<size_value_t, size_value_t>(_a, _a+1)); */
    /* 	++_a; */
    /* 	while(_b - _a != 1){ */
    /* 		nTs.push_back(std::pair<size_value_t, size_value_t>(_a, _a+1));
     */
    /* 		++_a; */
    /* 	} */
    /* 	nTs.push_back(std::pair<size_value_t, size_value_t>(_a, _b)); */
    /* 	while(_a != original_a){ */
    /* 		nTs.push_back(std::pair<size_value_t, size_value_t>(_a-1, _a));
     */
    /* 		--_a; */
    /* 	} */
    /* } */

    /* std::cout << "current transposes:"<<std::endl; */
    /* for(const auto& pair : nTs){ */
    /* 	std::cout << '{'<<pair.first<<','<<pair.second<<'}'<<std::endl; */
    /* } */

    /* reduceTransposes(nTs, deleteTransposeRepeat,
     * deleteLastTwoIfSameTranspose);
     */
    /* std::cout << "after reduction transposes:"<<std::endl; */
    /* for(const auto& pair : nTs){ */
    /* 	std::cout << '{'<<pair.first<<','<<pair.second<<'}'<<std::endl; */
    /* } */
    Tensor x = transpose(static_cast<size_value_t>(transposes[0].first),
                         static_cast<size_value_t>(transposes[0].second));
    /* std::cout << "did transpose"<<std::endl; */
    for (size_value_t i = 1; i < transposes.size(); ++i) {
        x = x.transpose(static_cast<size_value_t>(transposes[i].first),
                        static_cast<size_value_t>(transposes[i].second));
    }
    return x;
}

void transpose_RowColSwap(void **first, void **last,
                          const Tensor::size_value_t &n,
                          const Tensor::size_value_t &mn1,
                          const Tensor::size_value_t &total) {
    std::vector<bool> visited(total);
    void **cycle = first;
    while (++cycle != last) {
        if (visited[cycle - first])
            continue;
        int a = cycle - first;
        do {
            a = a == mn1 ? mn1 : (n * a) % mn1;
            std::swap(*(first + a), *cycle);
            visited[a] = true;
        } while ((first + a) != cycle);
    }
}

SizeRef squeeze_and_adjust_transpose(std::vector<Tensor::size_value_t> size_vec, Tensor::size_value_t& a, Tensor::size_value_t& b){
    //a < b
    for(int i = size_vec.size()-1; i >= 0; --i){
        if(size_vec[i] != 1) continue;
        if(i > b) continue;
        b = (b == 0) ? 0 : b-1;
        if(i > a) continue;
        a = (a == 0) ? 0 : a-1;
    }
    size_vec.erase(std::remove(size_vec.begin(), size_vec.end(), 1), size_vec.end());
    return SizeRef(std::move(size_vec));
}

// when dims are 1 away from each other this is a great and wonderfully
// optimized design when they are not, there is probably much to be desired
Tensor Tensor::transpose(size_value_t _a, size_value_t _b) const {
    _a = _a < 0 ? dims() + _a : _a;
    _b = _b < 0 ? dims() + _b : _b;

    utils::THROW_EXCEPTION(
        (_a < dims() && _b < dims()) && (_a >= 0 && _b >= 0),
        "a and b ($,$) are out of range for tensor with dimensionality $", _a,
        _b, dims());

    if (_a > _b) {
        std::swap(_a, _b);
    }
    if (_a == _b) {
        return *this;
    }
    if(std::any_of(shape().begin(), shape().end(), [](const int64_t& i){return i == 1;})){
        //fill in squeeze and adjust
        SizeRef out_shape = shape().transpose(_a, _b);
        SizeRef n_shape = squeeze_and_adjust_transpose(shape().Vec(), _a, _b);
        return this->view(n_shape).transpose(_a, _b).view(out_shape);
    }
    std::vector<size_value_t> cur_strides = getChangedStrides();
    std::swap(cur_strides[_a + 1], cur_strides[_b + 1]);
    if (std::abs(_b - _a) == 1) {
        if (_b == dims() - 1) {
            // this has been sped up using a way to just stride every index of
            // the tensor this is super slow and needs to be sped up
            ArrayVoid out_vals = _vals.get_bucket().is_strided()
                                     ? _vals.copy_strides(true)
                                     : _vals.bucket_all_indices();
            void **o_begin = out_vals.stride_begin();
            size_value_t total = shape()[-1] * shape()[-2];
            const size_value_t &rows = shape()[-2];
            const size_value_t mn1 = total - 1;
            if (dims() > 2) {
                size_value_t i_total = shape().flatten(0, -3)[0];

#ifdef USE_PARALLEL
                tbb::parallel_for(
                    utils::calculateGrainSize1D(i_total),
                    [&](const tbb::blocked_range<size_value_t> &range) {
                        for (size_value_t i = range.begin(); i != range.end();
                             ++i) {
#else
                for (size_value_t i = 0; i < i_total; ++i) {
#endif
                            void **first = o_begin + (i * total);
                            void **last = first + total;
                            transpose_RowColSwap(first, last, rows, mn1, total);
                        }
#ifdef USE_PARALLEL
                    });
#endif
                return Tensor(std::move(out_vals), shape().transpose(_a, _b),
                              cur_strides);
            }
            void **o_end = out_vals.stride_end();
            transpose_RowColSwap(o_begin, o_end, rows, mn1, total);
            return Tensor(std::move(out_vals), shape().transpose(_a, _b),
                          cur_strides);
        }
        Tensor parts = split_axis(_b).view(shape().range(0, _b + 1));
        // split axis makes a list of all the tensors that have that many
        // dimensions
        // (_b-1) view changes the view such that it has the original shape of
        // this tensor up to where it was split
        parts.RowColSwap();
        // its rows and cols are then transposed
        return functional::cat(parts)
            .view(shape().transpose(_a, _b))
            .set_stored_strides(cur_strides);
    }

    if (_b == dims() - 1) {
        ArrayVoid cur = _vals.bucket_all_indices();
        ArrayVoid output = cur.copy_strides(false);
        void **mine = cur.stride_begin();
        void **outp = output.stride_begin();
        auto m_strides = shape().strides();
        auto m_shape = shape().Vec();
        auto out_shape = shape().transpose(_a, _b);
        auto n_strides = out_shape.strides();
        n_strides.erase(n_strides.begin());
        m_strides.erase(m_strides.begin());

        // this is basically if it is not a transpose(0,-1)
        // then it can be parallelized by doing all the upper dimensions in
        // parallel
        /* if(_a != 0){ */
        /* 	while( */

        /* } */
        std::vector<uint64_t> done(m_shape.size());
        for (uint32_t i = 0; i < done.size(); ++i) {
            done[i] = m_shape[i] - 1;
        }
        std::vector<uint64_t> current_shape(dims(), 0);
        current_shape.back() = 1;
        *outp = *mine;
        ++mine;
        for (; !is_equal(current_shape, done);
             increment_shape(current_shape, m_shape), ++mine) {
            std::swap(current_shape[_a], current_shape[_b]);
            *(outp + calc_index(current_shape, n_strides)) = *mine;
            std::swap(current_shape[_a], current_shape[_b]);
        }
        *(outp + numel() - 1) = *mine;
        return Tensor(std::move(output), std::move(out_shape))
            .set_stored_strides(cur_strides);
    }

    // basically, by splitting, this will do all the lower dimensions at the
    // exact same time without needing to do threading to run it in parallel
    // obviously, threading will be added to further parallelize this though
    if (shape().range(0, _b + 1).multiply() == 1) {
        // this is the case when it is all 1's, and there really isn't a change
        // even in the view
        return *this;
    }
    Tensor split = this->split_axis(_b).view(shape().range(
        0, _b + 1)); // this split_axis function needs to be made faster
    Tensor *mine = reinterpret_cast<Tensor *>(split.data_ptr());
    // std::cout << "there are "<<split.numel()<<" tensors in mine"<<std::endl;
    Tensor out =
        Tensor::makeNullTensorArray(shape().range(0, _b + 1).multiply());
    out._size = shape().range(0, _b + 1);
    Tensor *outp = reinterpret_cast<Tensor *>(out.data_ptr());
    auto m_strides = split.shape().strides();
    m_strides.erase(m_strides.begin());
    const auto m_shape = split.shape().Vec();
    auto out_shape = split.shape().transpose(_a, _b);
    auto n_strides = out_shape.strides();
    n_strides.erase(n_strides.begin());

    std::vector<uint64_t> done(m_shape.size());
    for (uint32_t i = 0; i < done.size(); ++i) {
        done[i] = m_shape[i] - 1;
    }
    std::vector<uint64_t> current_shape(out_shape.size(), 0);
    current_shape.back() = 1;
    /* std::cout << "transposing dims "<<_a<<" and "<<_b<<" with input shape of
     * "<<shape() << std::endl; */
    *outp = *mine;
    ++mine;
    for (; !is_equal(current_shape, done);
         increment_shape(current_shape, m_shape), ++mine) {
        std::swap(current_shape[_a], current_shape[_b]);
        /* std::cout << "setting "<< calc_index(current_shape, n_strides) << "
         * of  "
         * << shape().range(0, _b+1).multiply() << std::endl; */
        int64_t index = calc_index(current_shape, n_strides);
        // std::cout << "index "<<index<<" was greater than out numel "<<out.numel()<<std::endl;
        *(outp + index) = *mine;
        std::swap(current_shape[_a], current_shape[_b]);
    }
    *(outp + out.numel() - 1) = *mine;
    return nt::functional::cat(out)
        .view(shape().transpose(_a, _b))
        .set_stored_strides(cur_strides);

    /* size_value_t original_a = _a; */
    /* size_value_t original_b = _b; */
    /* Tensor x = transpose(_a, _a+1); */
    /* ++_a; */
    /* while(_b - _a != 1){ */
    /* 	x = x.transpose(_a, _a+1); */
    /* 	++_a; */
    /* } */
    /* x = x.transpose(_a, _b); */
    /* while(_a != original_a){ */
    /* 	x = x.transpose(_a-1, _a); */
    /* 	--_a; */
    /* } */

    /* return std::move(x); */
}

//(row, col) = (row * _cols) + col

// the issue when dealing with multiple tensors is the swapping, so I am going
// to have to make my own swapping function for individual tensors, ArrayVoids,
// and Buckets
template <typename T>
void transpose_RowColSwap_contiguous(T *first, T *last,
                                     const Tensor::size_value_t &n,
                                     const Tensor::size_value_t &mn1,
                                     const Tensor::size_value_t &total) {
    std::vector<bool> visited(total);
    T *cycle = first;
    while (++cycle != last) {
        if (visited[cycle - first])
            continue;
        int a = cycle - first;
        do {
            a = a == mn1 ? mn1 : (n * a) % mn1;
            std::swap(*(first + a), *cycle);
            visited[a] = true;
        } while ((first + a) != cycle);
    }
}

/* inline static constexpr auto transposerc_once = [](auto first, auto last,
 * const uint32_t& n, const uint64_t& mn1, const uint64_t& total){ */
/* 	std::vector<bool> visited(total); */
/* 	auto cycle = first; */
/* 	while(++cycle != last){ */
/* 		int a = cycle - first; */
/* 		if(visited[a]) */
/* 			continue; */
/* 		do{ */
/* 			a = a == mn1 ? mn1 : (n * a) % mn1; */
/* 			std::swap(*(first + a), *cycle); */
/* 			visited[a] = true; */
/* 		}while((first + a) != cycle); */
/* 	} */
/* }; */

inline static constexpr auto transposeRC_once =
    [](auto first, auto last, const Tensor::size_value_t &n,
       const uint64_t &mn1, const uint64_t &total) {
        std::vector<bool> visited(total);
        auto cycle = first;
        uint64_t counter = 0;
        uint64_t a;
        while (++cycle != last) {
            ++counter;
            if (visited[counter])
                continue;
            a = counter;
            do {
                a = a == mn1 ? mn1 : (n * a) % mn1;
                std::swap(*(first + a), *cycle);
                visited[a] = true;
            } while ((first + a) != cycle);
        }
    };

// this swaps rows and collumns in memory
// potentially faster than transpose(-1,-2)
const Tensor &Tensor::RowColSwap() const {
    utils::THROW_EXCEPTION(dims() >= 2,
                           "RuntimeError: Expected dims to be greater than or "
                           "equal to 2 but got $",
                           dims());
    if (!is_contiguous()) {
        if (dims() > 2) {
            const Tensor parts = split_axis(-3); // div matricies
            const Tensor *begin =
                reinterpret_cast<const Tensor *>(parts.data_ptr());
            const Tensor *end = begin + parts.numel();
            for (; begin != end; ++begin)
                begin->RowColSwap();
            const_cast<SizeRef &>(_size) = shape().transpose(-1, -2);
            return *this;
        }
        const size_value_t &rows = shape()[-2];
        const uint64_t mn1 = numel() - 1;
        const_cast<ArrayVoid &>(_vals).execute_function(transposeRC_once, rows,
                                                        mn1, _total_size);
        const_cast<SizeRef &>(_size) = shape().transpose(-1, -2);
        return *this;
    }
    if (dims() > 2) {
        size_value_t i_total = shape().flatten(0, -3)[0];
        size_value_t total = shape()[-1] * shape()[-2];
        const size_value_t &rows = shape()[-2];
        const size_value_t mn1 = total - 1;
        switch (dtype) {
        case DType::Float: {
            using value_type = float;
#ifdef USE_PARALLEL
            tbb::parallel_for(
                utils::calculateGrainSize1D(i_total),
                [&](const tbb::blocked_range<size_value_t> &range) {
                    for (size_value_t i = range.begin(); i != range.end();
                         ++i) {
#else
            for (size_value_t i = 0; i < i_total; ++i) {
#endif
                        value_type *first =
                            const_cast<value_type *>(
                                reinterpret_cast<const value_type *>(
                                    data_ptr())) +
                            (i * total);
                        value_type *last = first + total;
                        transpose_RowColSwap_contiguous(first, last, rows, mn1,
                                                        total);
                    }
#ifdef USE_PARALLEL
                });
#endif
            const_cast<SizeRef &>(_size) = shape().transpose(-1, -2);
            return *this;
        }
        case DType::Double: {
            using value_type = double;
#ifdef USE_PARALLEL
            tbb::parallel_for(
                utils::calculateGrainSize1D(i_total),
                [&](const tbb::blocked_range<size_value_t> &range) {
                    for (size_value_t i = range.begin(); i != range.end();
                         ++i) {
#else
            for (size_value_t i = 0; i < i_total; ++i) {
#endif
                        value_type *first =
                            const_cast<value_type *>(
                                reinterpret_cast<const value_type *>(
                                    data_ptr())) +
                            (i * total);
                        value_type *last = first + total;
                        transpose_RowColSwap_contiguous(first, last, rows, mn1,
                                                        total);
                    }
#ifdef USE_PARALLEL
                });
#endif
            const_cast<SizeRef &>(_size) = shape().transpose(-1, -2);
            return *this;
        }
#ifdef _HALF_FLOAT_SUPPORT_
        case DType::Float16: {
            using value_type = float16_t;
#ifdef USE_PARALLEL
            tbb::parallel_for(
                utils::calculateGrainSize1D(i_total),
                [&](const tbb::blocked_range<size_value_t> &range) {
                    for (size_value_t i = range.begin(); i != range.end();
                         ++i) {
#else
            for (size_value_t i = 0; i < i_total; ++i) {
#endif
                        value_type *first =
                            const_cast<value_type *>(
                                reinterpret_cast<const value_type *>(
                                    data_ptr())) +
                            (i * total);
                        value_type *last = first + total;
                        transpose_RowColSwap_contiguous(first, last, rows, mn1,
                                                        total);
                    }
#ifdef USE_PARALLEL
                });
#endif
            const_cast<SizeRef &>(_size) = shape().transpose(-1, -2);
            return *this;
        }
        case DType::Complex32: {
            using value_type = complex_32;
#ifdef USE_PARALLEL
            tbb::parallel_for(
                utils::calculateGrainSize1D(i_total),
                [&](const tbb::blocked_range<size_value_t> &range) {
                    for (size_value_t i = range.begin(); i != range.end();
                         ++i) {
#else
            for (size_value_t i = 0; i < i_total; ++i) {
#endif
                        value_type *first =
                            const_cast<value_type *>(
                                reinterpret_cast<const value_type *>(
                                    data_ptr())) +
                            (i * total);
                        value_type *last = first + total;
                        transpose_RowColSwap_contiguous(first, last, rows, mn1,
                                                        total);
                    }
#ifdef USE_PARALLEL
                });
#endif
            const_cast<SizeRef &>(_size) = shape().transpose(-1, -2);
            return *this;
        }
#endif
#ifdef _128_FLOAT_SUPPORT_
        case DType::Float128: {
            using value_type = float128_t;
#ifdef USE_PARALLEL
            tbb::parallel_for(
                utils::calculateGrainSize1D(i_total),
                [&](const tbb::blocked_range<size_value_t> &range) {
                    for (size_value_t i = range.begin(); i != range.end();
                         ++i) {
#else
            for (size_value_t i = 0; i < i_total; ++i) {
#endif
                        value_type *first =
                            const_cast<value_type *>(
                                reinterpret_cast<const value_type *>(
                                    data_ptr())) +
                            (i * total);
                        value_type *last = first + total;
                        transpose_RowColSwap_contiguous(first, last, rows, mn1,
                                                        total);
                    }
#ifdef USE_PARALLEL
                });
#endif
            const_cast<SizeRef &>(_size) = shape().transpose(-1, -2);
            return *this;
        }
#endif
#ifdef __SIZEOF_INT128__
        case DType::int128: {
            using value_type = int128_t;
#ifdef USE_PARALLEL
            tbb::parallel_for(
                utils::calculateGrainSize1D(i_total),
                [&](const tbb::blocked_range<size_value_t> &range) {
                    for (size_value_t i = range.begin(); i != range.end();
                         ++i) {
#else
            for (size_value_t i = 0; i < i_total; ++i) {
#endif
                        value_type *first =
                            const_cast<value_type *>(
                                reinterpret_cast<const value_type *>(
                                    data_ptr())) +
                            (i * total);
                        value_type *last = first + total;
                        transpose_RowColSwap_contiguous(first, last, rows, mn1,
                                                        total);
                    }
#ifdef USE_PARALLEL
                });
#endif
            const_cast<SizeRef &>(_size) = shape().transpose(-1, -2);
            return *this;
        }
        case DType::uint128: {
            using value_type = uint128_t;
#ifdef USE_PARALLEL
            tbb::parallel_for(
                utils::calculateGrainSize1D(i_total),
                [&](const tbb::blocked_range<size_value_t> &range) {
                    for (size_value_t i = range.begin(); i != range.end();
                         ++i) {
#else
            for (size_value_t i = 0; i < i_total; ++i) {
#endif
                        value_type *first =
                            const_cast<value_type *>(
                                reinterpret_cast<const value_type *>(
                                    data_ptr())) +
                            (i * total);
                        value_type *last = first + total;
                        transpose_RowColSwap_contiguous(first, last, rows, mn1,
                                                        total);
                    }
#ifdef USE_PARALLEL
                });
#endif
            const_cast<SizeRef &>(_size) = shape().transpose(-1, -2);
            return *this;
        }
#endif
        case DType::Complex64: {
            using value_type = complex_64;
#ifdef USE_PARALLEL
            tbb::parallel_for(
                utils::calculateGrainSize1D(i_total),
                [&](const tbb::blocked_range<size_value_t> &range) {
                    for (size_value_t i = range.begin(); i != range.end();
                         ++i) {
#else
            for (size_value_t i = 0; i < i_total; ++i) {
#endif
                        value_type *first =
                            const_cast<value_type *>(
                                reinterpret_cast<const value_type *>(
                                    data_ptr())) +
                            (i * total);
                        value_type *last = first + total;
                        transpose_RowColSwap_contiguous(first, last, rows, mn1,
                                                        total);
                    }
#ifdef USE_PARALLEL
                });
#endif
            const_cast<SizeRef &>(_size) = shape().transpose(-1, -2);
            return *this;
        }
        case DType::Complex128: {
            using value_type = complex_128;
#ifdef USE_PARALLEL
            tbb::parallel_for(
                utils::calculateGrainSize1D(i_total),
                [&](const tbb::blocked_range<size_value_t> &range) {
                    for (size_value_t i = range.begin(); i != range.end();
                         ++i) {
#else
            for (size_value_t i = 0; i < i_total; ++i) {
#endif
                        value_type *first =
                            const_cast<value_type *>(
                                reinterpret_cast<const value_type *>(
                                    data_ptr())) +
                            (i * total);
                        value_type *last = first + total;
                        transpose_RowColSwap_contiguous(first, last, rows, mn1,
                                                        total);
                    }
#ifdef USE_PARALLEL
                });
#endif
            const_cast<SizeRef &>(_size) = shape().transpose(-1, -2);
            return *this;
        }
        case DType::int8: {
            using value_type = int8_t;
#ifdef USE_PARALLEL
            tbb::parallel_for(
                utils::calculateGrainSize1D(i_total),
                [&](const tbb::blocked_range<size_value_t> &range) {
                    for (size_value_t i = range.begin(); i != range.end();
                         ++i) {
#else
            for (size_value_t i = 0; i < i_total; ++i) {
#endif
                        value_type *first =
                            const_cast<value_type *>(
                                reinterpret_cast<const value_type *>(
                                    data_ptr())) +
                            (i * total);
                        value_type *last = first + total;
                        transpose_RowColSwap_contiguous(first, last, rows, mn1,
                                                        total);
                    }
#ifdef USE_PARALLEL
                });
#endif
            const_cast<SizeRef &>(_size) = shape().transpose(-1, -2);
            return *this;
        }
        case DType::uint8: {
            using value_type = uint8_t;
#ifdef USE_PARALLEL
            tbb::parallel_for(
                utils::calculateGrainSize1D(i_total),
                [&](const tbb::blocked_range<size_value_t> &range) {
                    for (size_value_t i = range.begin(); i != range.end();
                         ++i) {
#else
            for (size_value_t i = 0; i < i_total; ++i) {
#endif
                        value_type *first =
                            const_cast<value_type *>(
                                reinterpret_cast<const value_type *>(
                                    data_ptr())) +
                            (i * total);
                        value_type *last = first + total;
                        transpose_RowColSwap_contiguous(first, last, rows, mn1,
                                                        total);
                    }
#ifdef USE_PARALLEL
                });
#endif
            const_cast<SizeRef &>(_size) = shape().transpose(-1, -2);
            return *this;
        }
        case DType::int16: {
            using value_type = int16_t;
#ifdef USE_PARALLEL
            tbb::parallel_for(
                utils::calculateGrainSize1D(i_total),
                [&](const tbb::blocked_range<size_value_t> &range) {
                    for (size_value_t i = range.begin(); i != range.end();
                         ++i) {
#else
            for (size_value_t i = 0; i < i_total; ++i) {
#endif
                        value_type *first =
                            const_cast<value_type *>(
                                reinterpret_cast<const value_type *>(
                                    data_ptr())) +
                            (i * total);
                        value_type *last = first + total;
                        transpose_RowColSwap_contiguous(first, last, rows, mn1,
                                                        total);
                    }
#ifdef USE_PARALLEL
                });
#endif
            const_cast<SizeRef &>(_size) = shape().transpose(-1, -2);
            return *this;
        }
        case DType::uint16: {
            using value_type = uint16_t;
#ifdef USE_PARALLEL
            tbb::parallel_for(
                utils::calculateGrainSize1D(i_total),
                [&](const tbb::blocked_range<size_value_t> &range) {
                    for (size_value_t i = range.begin(); i != range.end();
                         ++i) {
#else
            for (size_value_t i = 0; i < i_total; ++i) {
#endif
                        value_type *first =
                            const_cast<value_type *>(
                                reinterpret_cast<const value_type *>(
                                    data_ptr())) +
                            (i * total);
                        value_type *last = first + total;
                        transpose_RowColSwap_contiguous(first, last, rows, mn1,
                                                        total);
                    }
#ifdef USE_PARALLEL
                });
#endif
            const_cast<SizeRef &>(_size) = shape().transpose(-1, -2);
            return *this;
        }
        case DType::int32: {
            using value_type = int32_t;
#ifdef USE_PARALLEL
            tbb::parallel_for(
                utils::calculateGrainSize1D(i_total),
                [&](const tbb::blocked_range<size_value_t> &range) {
                    for (size_value_t i = range.begin(); i != range.end();
                         ++i) {
#else
            for (size_value_t i = 0; i < i_total; ++i) {
#endif
                        value_type *first =
                            const_cast<value_type *>(
                                reinterpret_cast<const value_type *>(
                                    data_ptr())) +
                            (i * total);
                        value_type *last = first + total;
                        transpose_RowColSwap_contiguous(first, last, rows, mn1,
                                                        total);
                    }
#ifdef USE_PARALLEL
                });
#endif
            const_cast<SizeRef &>(_size) = shape().transpose(-1, -2);
            return *this;
        }
        case DType::uint32: {
            using value_type = uint32_t;
#ifdef USE_PARALLEL
            tbb::parallel_for(
                utils::calculateGrainSize1D(i_total),
                [&](const tbb::blocked_range<size_value_t> &range) {
                    for (size_value_t i = range.begin(); i != range.end();
                         ++i) {
#else
            for (size_value_t i = 0; i < i_total; ++i) {
#endif
                        value_type *first =
                            const_cast<value_type *>(
                                reinterpret_cast<const value_type *>(
                                    data_ptr())) +
                            (i * total);
                        value_type *last = first + total;
                        transpose_RowColSwap_contiguous(first, last, rows, mn1,
                                                        total);
                    }
#ifdef USE_PARALLEL
                });
#endif
            const_cast<SizeRef &>(_size) = shape().transpose(-1, -2);
            return *this;
        }
        case DType::int64: {
            using value_type = int64_t;
            std::cout << "swapping for int64_t" << std::endl;
#ifdef USE_PARALLEL
            tbb::parallel_for(
                utils::calculateGrainSize1D(i_total),
                [&](const tbb::blocked_range<size_value_t> &range) {
                    for (size_value_t i = range.begin(); i != range.end();
                         ++i) {
                        std::cout << "swapping from " << i << " to "
                                  << range.end() << std::endl;
#else
            for (size_value_t i = 0; i < i_total; ++i) {
#endif

                        value_type *first =
                            const_cast<value_type *>(
                                reinterpret_cast<const value_type *>(
                                    data_ptr())) +
                            (i * total);
                        value_type *last = first + total;
                        transpose_RowColSwap_contiguous(first, last, rows, mn1,
                                                        total);
                    }
#ifdef USE_PARALLEL
                });
#endif
            const_cast<SizeRef &>(_size) = shape().transpose(-1, -2);
            return *this;
        }
        case DType::Bool: {
            using value_type = uint_bool_t;
#ifdef USE_PARALLEL
            tbb::parallel_for(
                utils::calculateGrainSize1D(i_total),
                [&](const tbb::blocked_range<size_value_t> &range) {
                    for (size_value_t i = range.begin(); i != range.end();
                         ++i) {
#else
            for (size_value_t i = 0; i < i_total; ++i) {
#endif
                        value_type *first =
                            const_cast<value_type *>(
                                reinterpret_cast<const value_type *>(
                                    data_ptr())) +
                            (i * total);
                        value_type *last = first + total;
                        transpose_RowColSwap_contiguous(first, last, rows, mn1,
                                                        total);
                    }
#ifdef USE_PARALLEL
                });
#endif
            const_cast<SizeRef &>(_size) = shape().transpose(-1, -2);
            return *this;
        }
        case DType::TensorObj: {
            using value_type = Tensor;
#ifdef USE_PARALLEL
            tbb::parallel_for(
                utils::calculateGrainSize1D(i_total),
                [&](const tbb::blocked_range<size_value_t> &range) {
                    for (size_value_t i = range.begin(); i != range.end();
                         ++i) {
#else
            for (size_value_t i = 0; i < i_total; ++i) {
#endif
                        value_type *first =
                            const_cast<value_type *>(
                                reinterpret_cast<const value_type *>(
                                    data_ptr())) +
                            (i * total);
                        value_type *last = first + total;
                        transpose_RowColSwap_contiguous(first, last, rows, mn1,
                                                        total);
                    }
#ifdef USE_PARALLEL
                });
#endif
            const_cast<SizeRef &>(_size) = shape().transpose(-1, -2);
            return *this;
        }
        }
        return *this;
    }
    const size_value_t &rows = shape()[-2];
    const size_value_t mn1 = numel() - 1;

    switch (dtype) {
    case DType::Float: {
        using value_type = float;
        value_type *first = const_cast<value_type *>(
            reinterpret_cast<const value_type *>(data_ptr()));
        value_type *last = const_cast<value_type *>(
            reinterpret_cast<const value_type *>(data_ptr_end()));
        transpose_RowColSwap_contiguous(first, last, rows, mn1, numel());
        const_cast<SizeRef &>(_size) = shape().transpose(-1, -2);
        return *this;
    }
    case DType::Double: {
        using value_type = double;
        value_type *first = const_cast<value_type *>(
            reinterpret_cast<const value_type *>(data_ptr()));
        value_type *last = const_cast<value_type *>(
            reinterpret_cast<const value_type *>(data_ptr_end()));
        transpose_RowColSwap_contiguous(first, last, rows, mn1, numel());
        const_cast<SizeRef &>(_size) = shape().transpose(-1, -2);
        return *this;
    }
#ifdef _HALF_FLOAT_SUPPORT_
    case DType::Float16: {
        using value_type = float16_t;
        value_type *first = const_cast<value_type *>(
            reinterpret_cast<const value_type *>(data_ptr()));
        value_type *last = const_cast<value_type *>(
            reinterpret_cast<const value_type *>(data_ptr_end()));
        transpose_RowColSwap_contiguous(first, last, rows, mn1, numel());
        const_cast<SizeRef &>(_size) = shape().transpose(-1, -2);
        return *this;
    }
    case DType::Complex32: {
        using value_type = complex_32;
        value_type *first = const_cast<value_type *>(
            reinterpret_cast<const value_type *>(data_ptr()));
        value_type *last = const_cast<value_type *>(
            reinterpret_cast<const value_type *>(data_ptr_end()));
        transpose_RowColSwap_contiguous(first, last, rows, mn1, numel());
        const_cast<SizeRef &>(_size) = shape().transpose(-1, -2);
        return *this;
    }
#endif
#ifdef _128_FLOAT_SUPPORT_
    case DType::Float128: {
        using value_type = float128_t;
        value_type *first = const_cast<value_type *>(
            reinterpret_cast<const value_type *>(data_ptr()));
        value_type *last = const_cast<value_type *>(
            reinterpret_cast<const value_type *>(data_ptr_end()));
        transpose_RowColSwap_contiguous(first, last, rows, mn1, numel());
        const_cast<SizeRef &>(_size) = shape().transpose(-1, -2);
        return *this;
    }
#endif
#ifdef __SIZEOF_INT128__
    case DType::int128: {
        using value_type = int128_t;
        value_type *first = const_cast<value_type *>(
            reinterpret_cast<const value_type *>(data_ptr()));
        value_type *last = const_cast<value_type *>(
            reinterpret_cast<const value_type *>(data_ptr_end()));
        transpose_RowColSwap_contiguous(first, last, rows, mn1, numel());
        const_cast<SizeRef &>(_size) = shape().transpose(-1, -2);
        return *this;
    }
    case DType::uint128: {
        using value_type = uint128_t;
        value_type *first = const_cast<value_type *>(
            reinterpret_cast<const value_type *>(data_ptr()));
        value_type *last = const_cast<value_type *>(
            reinterpret_cast<const value_type *>(data_ptr_end()));
        transpose_RowColSwap_contiguous(first, last, rows, mn1, numel());
        const_cast<SizeRef &>(_size) = shape().transpose(-1, -2);
        return *this;
    }
#endif
    case DType::Complex64: {
        using value_type = complex_64;
        value_type *first = const_cast<value_type *>(
            reinterpret_cast<const value_type *>(data_ptr()));
        value_type *last = const_cast<value_type *>(
            reinterpret_cast<const value_type *>(data_ptr_end()));
        transpose_RowColSwap_contiguous(first, last, rows, mn1, numel());
        const_cast<SizeRef &>(_size) = shape().transpose(-1, -2);
        return *this;
    }
    case DType::Complex128: {
        using value_type = complex_128;
        value_type *first = const_cast<value_type *>(
            reinterpret_cast<const value_type *>(data_ptr()));
        value_type *last = const_cast<value_type *>(
            reinterpret_cast<const value_type *>(data_ptr_end()));
        transpose_RowColSwap_contiguous(first, last, rows, mn1, numel());
        const_cast<SizeRef &>(_size) = shape().transpose(-1, -2);
        return *this;
    }
    case DType::int8: {
        using value_type = int8_t;
        value_type *first = const_cast<value_type *>(
            reinterpret_cast<const value_type *>(data_ptr()));
        value_type *last = const_cast<value_type *>(
            reinterpret_cast<const value_type *>(data_ptr_end()));
        transpose_RowColSwap_contiguous(first, last, rows, mn1, numel());
        const_cast<SizeRef &>(_size) = shape().transpose(-1, -2);
        return *this;
    }
    case DType::uint8: {
        using value_type = uint8_t;
        value_type *first = const_cast<value_type *>(
            reinterpret_cast<const value_type *>(data_ptr()));
        value_type *last = const_cast<value_type *>(
            reinterpret_cast<const value_type *>(data_ptr_end()));
        transpose_RowColSwap_contiguous(first, last, rows, mn1, numel());
        const_cast<SizeRef &>(_size) = shape().transpose(-1, -2);
        return *this;
    }
    case DType::int16: {
        using value_type = int16_t;
        value_type *first = const_cast<value_type *>(
            reinterpret_cast<const value_type *>(data_ptr()));
        value_type *last = const_cast<value_type *>(
            reinterpret_cast<const value_type *>(data_ptr_end()));
        transpose_RowColSwap_contiguous(first, last, rows, mn1, numel());
        const_cast<SizeRef &>(_size) = shape().transpose(-1, -2);
        return *this;
    }
    case DType::uint16: {
        using value_type = uint16_t;
        value_type *first = const_cast<value_type *>(
            reinterpret_cast<const value_type *>(data_ptr()));
        value_type *last = const_cast<value_type *>(
            reinterpret_cast<const value_type *>(data_ptr_end()));
        transpose_RowColSwap_contiguous(first, last, rows, mn1, numel());
        const_cast<SizeRef &>(_size) = shape().transpose(-1, -2);
        return *this;
    }
    case DType::int32: {
        using value_type = int32_t;
        value_type *first = const_cast<value_type *>(
            reinterpret_cast<const value_type *>(data_ptr()));
        value_type *last = const_cast<value_type *>(
            reinterpret_cast<const value_type *>(data_ptr_end()));
        transpose_RowColSwap_contiguous(first, last, rows, mn1, numel());
        const_cast<SizeRef &>(_size) = shape().transpose(-1, -2);
        return *this;
    }
    case DType::uint32: {
        using value_type = uint32_t;
        value_type *first = const_cast<value_type *>(
            reinterpret_cast<const value_type *>(data_ptr()));
        value_type *last = const_cast<value_type *>(
            reinterpret_cast<const value_type *>(data_ptr_end()));
        transpose_RowColSwap_contiguous(first, last, rows, mn1, numel());
        const_cast<SizeRef &>(_size) = shape().transpose(-1, -2);
        return *this;
    }
    case DType::int64: {
        std::cout << "int64 swapping..." << std::endl;
        using value_type = int64_t;
        value_type *first = const_cast<value_type *>(
            reinterpret_cast<const value_type *>(data_ptr()));
        value_type *last = first + shape().back() * rows;
        std::cout << *(first + mn1) << std::endl;
        std::cout << "transposing" << std::endl;
        transpose_RowColSwap_contiguous(first, last, rows, mn1, numel());
        std::cout << "transposed" << std::endl;
        const_cast<SizeRef &>(_size) = shape().transpose(-1, -2);
        return *this;
    }
    case DType::Bool: {
        using value_type = uint_bool_t;
        value_type *first = const_cast<value_type *>(
            reinterpret_cast<const value_type *>(data_ptr()));
        value_type *last = const_cast<value_type *>(
            reinterpret_cast<const value_type *>(data_ptr_end()));
        transpose_RowColSwap_contiguous(first, last, rows, mn1, numel());
        const_cast<SizeRef &>(_size) = shape().transpose(-1, -2);
        return *this;
    }
    case DType::TensorObj: {
        using value_type = Tensor;
        value_type *first = const_cast<value_type *>(
            reinterpret_cast<const value_type *>(data_ptr()));
        value_type *last = const_cast<value_type *>(
            reinterpret_cast<const value_type *>(data_ptr_end()));
        transpose_RowColSwap_contiguous(first, last, rows, mn1, numel());
        const_cast<SizeRef &>(_size) = shape().transpose(-1, -2);
        return *this;
    }
    }
    return *this;
}

//:%s/\t\t\t\tfor(uint32_t i = 0; i < i_total; ++i){\n\t\t\t\t\tvalue_type\* first = reinterpret_cast<value_type\*>(_vals.data_ptr() + (i \* total);\n\t\t\
t\t\tvalue_type\* last = first + total;\n\t\t\t\t\ttranspose_RowColSwap_contiguous(first, last, rows, mn1, total);\n\t\t\t\t}/#ifdef USE_PARALLEL\n\t\t\t\ttbb::parallel_for(utils::calculateGrainSize1D(i_total), [\&](const tbb::blocked_range<uint32_t>\& range){\n\t\t\t\tfor(uint32_t i = range.begin(); i != range.end(); ++i){\n#else\n\t\t\t\tfor(uint32_t i = 0; i < i_total; ++i){\n#endif\n\t\t\t\t\tvalue_type\* first = reinterpret_cast<value_type\*>(_vals.data_ptr() + (i \* total);\n\t\t\
t\t\tvalue_type\* last = first + total;\n\t\t\t\t\ttranspose_RowColSwap_contiguous(first, last, rows, mn1, total);\n\t\t\t\t}\n#ifdef USE_PARALLEL\n\t\t\t\t});\n#endif

Tensor &Tensor::RowColSwap_Tensors() {
    utils::THROW_EXCEPTION(dtype == DType::TensorObj,
                           "RowColSwap_Tensors is meant to be used on a tensor "
                           "that holds tensors");
    _vals.for_each<DType::TensorObj>([](auto &inp) { inp.RowColSwap(); });
    return *this;
}

Tensor &Tensor::RowColSwap() {
    utils::THROW_EXCEPTION(dims() >= 2,
                           "RuntimeError: Expected dims to be greater than or "
                           "equal to 2 but got $",
                           dims());
    if (!is_contiguous()) {
        if (dims() > 2) {
            const Tensor parts = split_axis(-3); // div matricies
            const Tensor *begin =
                reinterpret_cast<const Tensor *>(parts.data_ptr());
            const Tensor *end = begin + parts.numel();
            for (; begin != end; ++begin)
                begin->RowColSwap();
            const_cast<SizeRef &>(_size) = shape().transpose(-1, -2);
            return *this;
        }
        const size_value_t &rows = shape()[-2];
        const uint64_t mn1 = numel() - 1;
        _vals.execute_function(transposeRC_once, rows, mn1, _total_size);
        _size = shape().transpose(-1, -2);
        return *this;
    }
    if (dims() > 2) {
        size_value_t i_total = shape().flatten(0, -3)[0];
        size_value_t total = shape()[-1] * shape()[-2];
        const size_value_t &rows = shape()[-2];
        const size_value_t mn1 = total - 1;
#ifdef USE_PARALLEL
        const size_value_t &cols = shape()[-1];
#endif
        switch (dtype) {
        case DType::Float: {
            using value_type = float;
#ifdef USE_PARALLEL
            tbb::parallel_for(
                utils::calculateGrainSize1D(i_total),
                [&](const tbb::blocked_range<size_value_t> &range) {
                    for (size_value_t i = range.begin(); i != range.end();
                         ++i) {
#else
            for (size_value_t i = 0; i < i_total; ++i) {
#endif
                        value_type *first =
                            reinterpret_cast<value_type *>(data_ptr()) +
                            (i * total);
                        value_type *last = first + total;
                        transpose_RowColSwap_contiguous(first, last, rows, mn1,
                                                        total);
                    }
#ifdef USE_PARALLEL
                });
#endif
            _size = shape().transpose(-1, -2);
            return *this;
        }
        case DType::Double: {
            using value_type = double;
#ifdef USE_PARALLEL
            tbb::parallel_for(
                utils::calculateGrainSize1D(i_total),
                [&](const tbb::blocked_range<size_value_t> &range) {
                    for (uint32_t i = range.begin(); i != range.end(); ++i) {
#else
            for (size_value_t i = 0; i < i_total; ++i) {
#endif
                        value_type *first =
                            reinterpret_cast<value_type *>(data_ptr()) +
                            (i * total);
                        value_type *last = first + total;
                        transpose_RowColSwap_contiguous(first, last, rows, mn1,
                                                        total);
                    }
#ifdef USE_PARALLEL
                });
#endif
            _size = shape().transpose(-1, -2);
            return *this;
        }
#ifdef _HALF_FLOAT_SUPPORT_
        case DType::Float16: {
            using value_type = float16_t;
#ifdef USE_PARALLEL
            tbb::parallel_for(
                utils::calculateGrainSize1D(i_total),
                [&](const tbb::blocked_range<size_value_t> &range) {
                    for (uint32_t i = range.begin(); i != range.end(); ++i) {
#else
            for (size_value_t i = 0; i < i_total; ++i) {
#endif
                        value_type *first =
                            reinterpret_cast<value_type *>(data_ptr()) +
                            (i * total);
                        value_type *last = first + total;
                        transpose_RowColSwap_contiguous(first, last, rows, mn1,
                                                        total);
                    }
#ifdef USE_PARALLEL
                });
#endif
            _size = shape().transpose(-1, -2);
            return *this;
        }
        case DType::Complex32: {
            using value_type = complex_32;
#ifdef USE_PARALLEL
            tbb::parallel_for(
                utils::calculateGrainSize1D(i_total),
                [&](const tbb::blocked_range<size_value_t> &range) {
                    for (uint32_t i = range.begin(); i != range.end(); ++i) {
#else
            for (size_value_t i = 0; i < i_total; ++i) {
#endif
                        value_type *first =
                            reinterpret_cast<value_type *>(data_ptr()) +
                            (i * total);
                        value_type *last = first + total;
                        transpose_RowColSwap_contiguous(first, last, rows, mn1,
                                                        total);
                    }
#ifdef USE_PARALLEL
                });
#endif
            _size = shape().transpose(-1, -2);
            return *this;
        }
#endif
#ifdef _128_FLOAT_SUPPORT_
        case DType::Float128: {
            using value_type = float128_t;
#ifdef USE_PARALLEL
            tbb::parallel_for(
                utils::calculateGrainSize1D(i_total),
                [&](const tbb::blocked_range<size_value_t> &range) {
                    for (uint32_t i = range.begin(); i != range.end(); ++i) {
#else
            for (size_value_t i = 0; i < i_total; ++i) {
#endif
                        value_type *first =
                            reinterpret_cast<value_type *>(data_ptr()) +
                            (i * total);
                        value_type *last = first + total;
                        transpose_RowColSwap_contiguous(first, last, rows, mn1,
                                                        total);
                    }
#ifdef USE_PARALLEL
                });
#endif
            _size = shape().transpose(-1, -2);
            return *this;
        }
#endif
#ifdef __SIZEOF_INT128__
        case DType::int128: {
            using value_type = int128_t;
#ifdef USE_PARALLEL
            tbb::parallel_for(
                utils::calculateGrainSize1D(i_total),
                [&](const tbb::blocked_range<size_value_t> &range) {
                    for (uint32_t i = range.begin(); i != range.end(); ++i) {
#else
            for (size_value_t i = 0; i < i_total; ++i) {
#endif
                        value_type *first =
                            reinterpret_cast<value_type *>(data_ptr()) +
                            (i * total);
                        value_type *last = first + total;
                        transpose_RowColSwap_contiguous(first, last, rows, mn1,
                                                        total);
                    }
#ifdef USE_PARALLEL
                });
#endif
            _size = shape().transpose(-1, -2);
            return *this;
        }
        case DType::uint128: {
            using value_type = uint128_t;
#ifdef USE_PARALLEL
            tbb::parallel_for(
                utils::calculateGrainSize1D(i_total),
                [&](const tbb::blocked_range<size_value_t> &range) {
                    for (uint32_t i = range.begin(); i != range.end(); ++i) {
#else
            for (size_value_t i = 0; i < i_total; ++i) {
#endif
                        value_type *first =
                            reinterpret_cast<value_type *>(data_ptr()) +
                            (i * total);
                        value_type *last = first + total;
                        transpose_RowColSwap_contiguous(first, last, rows, mn1,
                                                        total);
                    }
#ifdef USE_PARALLEL
                });
#endif
            _size = shape().transpose(-1, -2);
            return *this;
        }
#endif
        case DType::Complex64: {
            using value_type = complex_64;
#ifdef USE_PARALLEL
            tbb::parallel_for(
                utils::calculateGrainSize1D(i_total),
                [&](const tbb::blocked_range<size_value_t> &range) {
                    for (uint32_t i = range.begin(); i != range.end(); ++i) {
#else
            for (size_value_t i = 0; i < i_total; ++i) {
#endif
                        value_type *first =
                            reinterpret_cast<value_type *>(data_ptr()) +
                            (i * total);
                        value_type *last = first + total;
                        transpose_RowColSwap_contiguous(first, last, rows, mn1,
                                                        total);
                    }
#ifdef USE_PARALLEL
                });
#endif
            _size = shape().transpose(-1, -2);
            return *this;
        }
        case DType::Complex128: {
            using value_type = complex_128;
#ifdef USE_PARALLEL
            tbb::parallel_for(
                utils::calculateGrainSize1D(i_total),
                [&](const tbb::blocked_range<size_value_t> &range) {
                    for (uint32_t i = range.begin(); i != range.end(); ++i) {
#else
            for (size_value_t i = 0; i < i_total; ++i) {
#endif
                        value_type *first =
                            reinterpret_cast<value_type *>(data_ptr()) +
                            (i * total);
                        value_type *last = first + total;
                        transpose_RowColSwap_contiguous(first, last, rows, mn1,
                                                        total);
                    }
#ifdef USE_PARALLEL
                });
#endif
            _size = shape().transpose(-1, -2);
            return *this;
        }
        case DType::int8: {
            using value_type = int8_t;
#ifdef USE_PARALLEL
            tbb::parallel_for(
                utils::calculateGrainSize1D(i_total),
                [&](const tbb::blocked_range<size_value_t> &range) {
                    for (uint32_t i = range.begin(); i != range.end(); ++i) {
#else
            for (size_value_t i = 0; i < i_total; ++i) {
#endif
                        value_type *first =
                            reinterpret_cast<value_type *>(data_ptr()) +
                            (i * total);
                        value_type *last = first + total;
                        transpose_RowColSwap_contiguous(first, last, rows, mn1,
                                                        total);
                    }
#ifdef USE_PARALLEL
                });
#endif
            _size = shape().transpose(-1, -2);
            return *this;
        }
        case DType::uint8: {
            using value_type = uint8_t;
#ifdef USE_PARALLEL
            tbb::parallel_for(
                utils::calculateGrainSize1D(i_total),
                [&](const tbb::blocked_range<size_value_t> &range) {
                    for (uint32_t i = range.begin(); i != range.end(); ++i) {
#else
            for (size_value_t i = 0; i < i_total; ++i) {
#endif
                        value_type *first =
                            reinterpret_cast<value_type *>(data_ptr()) +
                            (i * total);
                        value_type *last = first + total;
                        transpose_RowColSwap_contiguous(first, last, rows, mn1,
                                                        total);
                    }
#ifdef USE_PARALLEL
                });
#endif
            _size = shape().transpose(-1, -2);
            return *this;
        }
        case DType::int16: {
            using value_type = int16_t;
#ifdef USE_PARALLEL
            tbb::parallel_for(
                utils::calculateGrainSize1D(i_total),
                [&](const tbb::blocked_range<size_value_t> &range) {
                    for (uint32_t i = range.begin(); i != range.end(); ++i) {
#else
            for (size_value_t i = 0; i < i_total; ++i) {
#endif
                        value_type *first =
                            reinterpret_cast<value_type *>(data_ptr()) +
                            (i * total);
                        value_type *last = first + total;
                        transpose_RowColSwap_contiguous(first, last, rows, mn1,
                                                        total);
                    }
#ifdef USE_PARALLEL
                });
#endif
            _size = shape().transpose(-1, -2);
            return *this;
        }
        case DType::uint16: {
            using value_type = uint16_t;
#ifdef USE_PARALLEL
            tbb::parallel_for(
                utils::calculateGrainSize1D(i_total),
                [&](const tbb::blocked_range<size_value_t> &range) {
                    for (uint32_t i = range.begin(); i != range.end(); ++i) {
#else
            for (size_value_t i = 0; i < i_total; ++i) {
#endif
                        value_type *first =
                            reinterpret_cast<value_type *>(data_ptr()) +
                            (i * total);
                        value_type *last = first + total;
                        transpose_RowColSwap_contiguous(first, last, rows, mn1,
                                                        total);
                    }
#ifdef USE_PARALLEL
                });
#endif
            _size = shape().transpose(-1, -2);
            return *this;
        }
        case DType::int32: {
            using value_type = int32_t;
#ifdef USE_PARALLEL
            tbb::parallel_for(
                utils::calculateGrainSize1D(i_total),
                [&](const tbb::blocked_range<size_value_t> &range) {
                    for (uint32_t i = range.begin(); i != range.end(); ++i) {
#else
            for (size_value_t i = 0; i < i_total; ++i) {
#endif
                        value_type *first =
                            reinterpret_cast<value_type *>(data_ptr()) +
                            (i * total);
                        value_type *last = first + total;
                        transpose_RowColSwap_contiguous(first, last, rows, mn1,
                                                        total);
                    }
#ifdef USE_PARALLEL
                });
#endif
            _size = shape().transpose(-1, -2);
            return *this;
        }
        case DType::uint32: {
            using value_type = uint32_t;
#ifdef USE_PARALLEL
            tbb::parallel_for(
                utils::calculateGrainSize1D(i_total),
                [&](const tbb::blocked_range<size_value_t> &range) {
                    for (uint32_t i = range.begin(); i != range.end(); ++i) {
#else
            for (size_value_t i = 0; i < i_total; ++i) {
#endif
                        value_type *first =
                            reinterpret_cast<value_type *>(data_ptr()) +
                            (i * total);
                        value_type *last = first + total;
                        transpose_RowColSwap_contiguous(first, last, rows, mn1,
                                                        total);
                    }
#ifdef USE_PARALLEL
                });
#endif
            _size = shape().transpose(-1, -2);
            return *this;
        }
        case DType::int64: {
            using value_type = int64_t;
#ifdef USE_PARALLEL
            tbb::parallel_for(
                utils::calculateGrainSize1D(i_total),
                [&](const tbb::blocked_range<size_value_t> &range) {
                    for (uint32_t i = range.begin(); i != range.end(); ++i) {
#else
            for (size_value_t i = 0; i < i_total; ++i) {
#endif
                        value_type *first =
                            reinterpret_cast<value_type *>(data_ptr()) +
                            (i * total);
                        value_type *last = first + total;
                        transpose_RowColSwap_contiguous(first, last, rows, mn1,
                                                        total);
                    }
#ifdef USE_PARALLEL
                });
#endif
            _size = shape().transpose(-1, -2);
            return *this;
        }
        case DType::Bool: {
            using value_type = uint_bool_t;
#ifdef USE_PARALLEL
            tbb::parallel_for(
                utils::calculateGrainSize1D(i_total),
                [&](const tbb::blocked_range<size_value_t> &range) {
                    for (uint32_t i = range.begin(); i != range.end(); ++i) {
#else
            for (size_value_t i = 0; i < i_total; ++i) {
#endif
                        value_type *first =
                            reinterpret_cast<value_type *>(data_ptr()) +
                            (i * total);
                        value_type *last = first + total;
                        transpose_RowColSwap_contiguous(first, last, rows, mn1,
                                                        total);
                    }
#ifdef USE_PARALLEL
                });
#endif
            _size = shape().transpose(-1, -2);
            return *this;
        }
        case DType::TensorObj: {
            using value_type = Tensor;
#ifdef USE_PARALLEL
            tbb::parallel_for(
                utils::calculateGrainSize1D(i_total),
                [&](const tbb::blocked_range<size_value_t> &range) {
                    for (uint32_t i = range.begin(); i != range.end(); ++i) {
#else
            for (size_value_t i = 0; i < i_total; ++i) {
#endif
                        value_type *first =
                            reinterpret_cast<value_type *>(data_ptr()) +
                            (i * total);
                        value_type *last = first + total;
                        transpose_RowColSwap_contiguous(first, last, rows, mn1,
                                                        total);
                    }
#ifdef USE_PARALLEL
                });
#endif
            _size = shape().transpose(-1, -2);
            return *this;
        }
        }
        return *this;
    }
    const size_value_t &rows = shape()[-2];
    const size_value_t mn1 = numel() - 1;

    switch (dtype) {
    case DType::Float: {
        using value_type = float;
        value_type *first = reinterpret_cast<value_type *>(data_ptr());
        value_type *last = reinterpret_cast<value_type *>(data_ptr_end());
        transpose_RowColSwap_contiguous(first, last, rows, mn1, numel());
        _size = shape().transpose(-1, -2);
        return *this;
    }
    case DType::Double: {
        using value_type = double;
        value_type *first = reinterpret_cast<value_type *>(data_ptr());
        value_type *last = reinterpret_cast<value_type *>(data_ptr_end());
        transpose_RowColSwap_contiguous(first, last, rows, mn1, numel());
        _size = shape().transpose(-1, -2);
        return *this;
    }
#ifdef _HALF_FLOAT_SUPPORT_
    case DType::Float16: {
        using value_type = float16_t;
        value_type *first = reinterpret_cast<value_type *>(data_ptr());
        value_type *last = reinterpret_cast<value_type *>(data_ptr_end());
        transpose_RowColSwap_contiguous(first, last, rows, mn1, numel());
        _size = shape().transpose(-1, -2);
        return *this;
    }
    case DType::Complex32: {
        using value_type = complex_32;
        value_type *first = reinterpret_cast<value_type *>(data_ptr());
        value_type *last = reinterpret_cast<value_type *>(data_ptr_end());
        transpose_RowColSwap_contiguous(first, last, rows, mn1, numel());
        _size = shape().transpose(-1, -2);
        return *this;
    }
#endif
#ifdef _128_FLOAT_SUPPORT_
    case DType::Float128: {
        using value_type = float128_t;
        value_type *first = reinterpret_cast<value_type *>(data_ptr());
        value_type *last = reinterpret_cast<value_type *>(data_ptr_end());
        transpose_RowColSwap_contiguous(first, last, rows, mn1, numel());
        _size = shape().transpose(-1, -2);
        return *this;
    }
#endif
#ifdef __SIZEOF_INT128__
    case DType::int128: {
        using value_type = int128_t;
        value_type *first = reinterpret_cast<value_type *>(data_ptr());
        value_type *last = reinterpret_cast<value_type *>(data_ptr_end());
        transpose_RowColSwap_contiguous(first, last, rows, mn1, numel());
        _size = shape().transpose(-1, -2);
        return *this;
    }
    case DType::uint128: {
        using value_type = uint128_t;
        value_type *first = reinterpret_cast<value_type *>(data_ptr());
        value_type *last = reinterpret_cast<value_type *>(data_ptr_end());
        transpose_RowColSwap_contiguous(first, last, rows, mn1, numel());
        _size = shape().transpose(-1, -2);
        return *this;
    }
#endif
    case DType::Complex64: {
        using value_type = complex_64;
        value_type *first = reinterpret_cast<value_type *>(data_ptr());
        value_type *last = reinterpret_cast<value_type *>(data_ptr_end());
        transpose_RowColSwap_contiguous(first, last, rows, mn1, numel());
        _size = shape().transpose(-1, -2);
        return *this;
    }
    case DType::Complex128: {
        using value_type = complex_128;
        value_type *first = reinterpret_cast<value_type *>(data_ptr());
        value_type *last = reinterpret_cast<value_type *>(data_ptr_end());
        transpose_RowColSwap_contiguous(first, last, rows, mn1, numel());
        _size = shape().transpose(-1, -2);
        return *this;
    }
    case DType::int8: {
        using value_type = int8_t;
        value_type *first = reinterpret_cast<value_type *>(data_ptr());
        value_type *last = reinterpret_cast<value_type *>(data_ptr_end());
        transpose_RowColSwap_contiguous(first, last, rows, mn1, numel());
        _size = shape().transpose(-1, -2);
        return *this;
    }
    case DType::uint8: {
        using value_type = uint8_t;
        value_type *first = reinterpret_cast<value_type *>(data_ptr());
        value_type *last = reinterpret_cast<value_type *>(data_ptr_end());
        transpose_RowColSwap_contiguous(first, last, rows, mn1, numel());
        _size = shape().transpose(-1, -2);
        return *this;
    }
    case DType::int16: {
        using value_type = int16_t;
        value_type *first = reinterpret_cast<value_type *>(data_ptr());
        value_type *last = reinterpret_cast<value_type *>(data_ptr_end());
        transpose_RowColSwap_contiguous(first, last, rows, mn1, numel());
        _size = shape().transpose(-1, -2);
        return *this;
    }
    case DType::uint16: {
        using value_type = uint16_t;
        value_type *first = reinterpret_cast<value_type *>(data_ptr());
        value_type *last = reinterpret_cast<value_type *>(data_ptr_end());
        transpose_RowColSwap_contiguous(first, last, rows, mn1, numel());
        _size = shape().transpose(-1, -2);
        return *this;
    }
    case DType::int32: {
        using value_type = int32_t;
        value_type *first = reinterpret_cast<value_type *>(data_ptr());
        value_type *last = reinterpret_cast<value_type *>(data_ptr_end());
        transpose_RowColSwap_contiguous(first, last, rows, mn1, numel());
        _size = shape().transpose(-1, -2);
        return *this;
    }
    case DType::uint32: {
        using value_type = uint32_t;
        value_type *first = reinterpret_cast<value_type *>(data_ptr());
        value_type *last = reinterpret_cast<value_type *>(data_ptr_end());
        transpose_RowColSwap_contiguous(first, last, rows, mn1, numel());
        _size = shape().transpose(-1, -2);
        return *this;
    }
    case DType::int64: {
        using value_type = int64_t;
        value_type *first = reinterpret_cast<value_type *>(data_ptr());
        value_type *last = reinterpret_cast<value_type *>(data_ptr_end());
        transpose_RowColSwap_contiguous(first, last, rows, mn1, numel());
        _size = shape().transpose(-1, -2);
        return *this;
    }
    case DType::Bool: {
        using value_type = uint_bool_t;
        value_type *first = reinterpret_cast<value_type *>(data_ptr());
        value_type *last = reinterpret_cast<value_type *>(data_ptr_end());
        transpose_RowColSwap_contiguous(first, last, rows, mn1, numel());
        _size = shape().transpose(-1, -2);
        return *this;
    }
    case DType::TensorObj: {
        using value_type = Tensor;
        value_type *first = reinterpret_cast<value_type *>(data_ptr());
        value_type *last = reinterpret_cast<value_type *>(data_ptr_end());
        transpose_RowColSwap_contiguous(first, last, rows, mn1, numel());
        _size = shape().transpose(-1, -2);
        return *this;
    }
    }
    return *this;
}


Tensor Tensor::swap_axis(size_value_t dim, size_value_t a, size_value_t b) const{
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
    return functional::cat_unordered(split).view(shape());

}

Tensor& Tensor::swap_axis_(size_value_t dim, size_value_t a, size_value_t b){
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

Tensor Tensor::real() const {
    utils::THROW_EXCEPTION(
        DTypeFuncs::is_complex(dtype),
        "RuntimeError: Expected dtype to be a complex number but got $", dtype);

    ArrayVoid out_vals = _vals.get_bucket().is_strided()
                             ? _vals.copy_strides(true)
                             : _vals.bucket_all_indices();
#ifdef _HALF_FLOAT_SUPPORT_
    out_vals.dtype = (dtype == DType::Complex128  ? DType::Double
                      : dtype == DType::Complex64 ? DType::Float
                                                  : DType::Float16);
#else
    out_vals.dtype =
        (dtype == DType::Complex128 ? DType::Double : DType::Float);
#endif
    out_vals.get_bucket().dtype = out_vals.dtype;

    return Tensor(std::move(out_vals), shape());
}

Tensor Tensor::to_complex_from_real() const {
    utils::THROW_EXCEPTION(
#ifdef _HALF_FLOAT_SUPPORT_
        dtype == DType::Double || dtype == DType::Float ||
            dtype == DType::Float16,
#else
        dtype == DType::Double || dtype == DType::Float,
#endif
        "RuntimeError: Expected dtype to be a floating number but got $",
        dtype);
#ifdef _HALF_FLOAT_SUPPORT_
    Tensor output = ::nt::functional::zeros(
        shape(), (dtype == DType::Double  ? DType::Complex128
                  : dtype == DType::Float ? DType::Complex64
                                          : DType::Complex32));
#else
    Tensor output = ::nt::functional::zeros(
        shape(),
        (dtype == DType::Double ? DType::Complex128 : dtype == DType::Float));
#endif
    if (dtype == DType::Double) {
        complex_128 *start = reinterpret_cast<complex_128 *>(output.data_ptr());
        uint32_t type = _vals.get_bucket().iterator_type();
        if (type == 1) { // contiguous
            const double *begin = reinterpret_cast<const double *>(data_ptr());
            const double *end =
                reinterpret_cast<const double *>(data_ptr_end());
            for (; begin != end; ++begin, ++start) {
                start->real() = *begin;
            }
            return std::move(output);
        } else if (type == 2) {
            auto begin = _vals.get_bucket().cbegin_blocked<double>();
            auto end = _vals.get_bucket().cend_blocked<double>();
            for (; begin != end; ++begin, ++start) {
                start->real() = *begin;
            }
            return std::move(output);
        } else if (type == 3) {
            auto begin = _vals.get_bucket().cbegin_list<double>();
            auto end = _vals.get_bucket().cend_list<double>();
            for (; begin != end; ++begin, ++start) {
                start->real() = *begin;
            }

            return std::move(output);
        }
        return std::move(output);
    } else if (dtype == DType::Float) {
        complex_64 *start = reinterpret_cast<complex_64 *>(output.data_ptr());
        uint32_t type = _vals.get_bucket().iterator_type();
        if (type == 1) {
            const float *begin = reinterpret_cast<const float *>(data_ptr());
            const float *end = reinterpret_cast<const float *>(data_ptr_end());
            for (; begin != end; ++begin, ++start) {
                start->real() = *begin;
            }
            return std::move(output);
        } else if (type == 2) {
            auto begin = _vals.get_bucket().cbegin_blocked<float>();
            auto end = _vals.get_bucket().cend_blocked<float>();
            for (; begin != end; ++begin, ++start) {
                start->real() = *begin;
            }
            return std::move(output);
        } else if (type == 3) {
            auto begin = _vals.get_bucket().cbegin_list<float>();
            auto end = _vals.get_bucket().cend_list<float>();
            for (; begin != end; ++begin, ++start) {
                start->real() = *begin;
            }
            return std::move(output);
        }
        return std::move(output);
    }
#ifdef _HALF_FLOAT_SUPPORT_
    else if (dtype == DType::Float16) {
        complex_32 *start = reinterpret_cast<complex_32 *>(output.data_ptr());
        uint32_t type = _vals.get_bucket().iterator_type();
        if (type == 1) {
            const float16_t *begin =
                reinterpret_cast<const float16_t *>(data_ptr());
            const float16_t *end =
                reinterpret_cast<const float16_t *>(data_ptr_end());
            for (; begin != end; ++begin, ++start) {
                start->real() = *begin;
            }
            return std::move(output);
        } else if (type == 2) {
            auto begin = _vals.get_bucket().cbegin_blocked<float16_t>();
            auto end = _vals.get_bucket().cend_blocked<float16_t>();
            for (; begin != end; ++begin, ++start) {
                start->real() = *begin;
            }
            return std::move(output);
        } else if (type == 3) {
            auto begin = _vals.get_bucket().cbegin_list<float16_t>();
            auto end = _vals.get_bucket().cend_list<float16_t>();
            for (; begin != end; ++begin, ++start) {
                start->real() = *begin;
            }
            return std::move(output);
        }

        return std::move(output);
    }
#endif
    return std::move(output);
}

Tensor Tensor::imag() const {
    utils::THROW_EXCEPTION(
        DTypeFuncs::is_complex(dtype),
        "RuntimeError: Expected dtype to be a complex number but got $", dtype);
    ArrayVoid out_vals = _vals.get_bucket().is_strided()
                             ? _vals.copy_strides(true)
                             : _vals.bucket_all_indices();
#ifdef _HALF_FLOAT_SUPPORT_
    out_vals.dtype = (dtype == DType::Complex128  ? DType::Double
                      : dtype == DType::Complex64 ? DType::Float
                                                  : DType::Float16);
#else
    out_vals.dtype =
        (dtype == DType::Complex128 ? DType::Double : DType::Float);
#endif
    out_vals.get_bucket().dtype = out_vals.dtype;

    std::size_t complex_size = DTypeFuncs::size_of_dtype(dtype);
    std::size_t imag_size = DTypeFuncs::size_of_dtype(out_vals.dtype);
    // make sure the types are compatible
    utils::THROW_EXCEPTION((imag_size * 2) == complex_size,
                           "Expected to have a halfing value when going from "
                           "complex to imaginary");
    void **begin = out_vals.stride_begin();
    void **end = out_vals.stride_end();
    for (; begin != end; ++begin) {
        *begin = reinterpret_cast<uint8_t *>(*begin) + imag_size;
    }
    return Tensor(std::move(out_vals), shape());
}

Tensor Tensor::to_complex_from_imag() const {
    utils::THROW_EXCEPTION(
#ifdef _HALF_FLOAT_SUPPORT_
        dtype == DType::Double || dtype == DType::Float ||
            dtype == DType::Float16,
#else
        dtype == DType::Double || dtype == DType::Float,
#endif
        "RuntimeError: Expected dtype to be a floating number but got $",
        dtype);
#ifdef _HALF_FLOAT_SUPPORT_
    Tensor output = ::nt::functional::zeros(
        shape(), (dtype == DType::Double  ? DType::Complex128
                  : dtype == DType::Float ? DType::Complex64
                                          : DType::Complex32));
#else
    Tensor output = ::nt::functional::zeros(
        shape(),
        (dtype == DType::Double ? DType::Complex128 : dtype == DType::Float));
#endif
    if (dtype == DType::Double) {
        complex_128 *start = reinterpret_cast<complex_128 *>(output.data_ptr());
        uint32_t type = _vals.get_bucket().iterator_type();
        if (type == 1) { // contiguous
            const double *begin = reinterpret_cast<const double *>(data_ptr());
            const double *end =
                reinterpret_cast<const double *>(data_ptr_end());
            for (; begin != end; ++begin, ++start) {
                start->imag() = *begin;
            }
            return std::move(output);
        } else if (type == 2) {
            auto begin = _vals.get_bucket().cbegin_blocked<double>();
            auto end = _vals.get_bucket().cend_blocked<double>();
            for (; begin != end; ++begin, ++start) {
                start->imag() = *begin;
            }
            return std::move(output);
        } else if (type == 3) {
            auto begin = _vals.get_bucket().cbegin_list<double>();
            auto end = _vals.get_bucket().cend_list<double>();
            for (; begin != end; ++begin, ++start) {
                start->imag() = *begin;
            }

            return std::move(output);
        }
        return std::move(output);
    } else if (dtype == DType::Float) {
        complex_64 *start = reinterpret_cast<complex_64 *>(output.data_ptr());
        uint32_t type = _vals.get_bucket().iterator_type();
        if (type == 1) {
            const float *begin = reinterpret_cast<const float *>(data_ptr());
            const float *end = reinterpret_cast<const float *>(data_ptr_end());
            for (; begin != end; ++begin, ++start) {
                start->imag() = *begin;
            }
            return std::move(output);
        } else if (type == 2) {
            auto begin = _vals.get_bucket().cbegin_blocked<float>();
            auto end = _vals.get_bucket().cend_blocked<float>();
            for (; begin != end; ++begin, ++start) {
                start->imag() = *begin;
            }
            return std::move(output);
        } else if (type == 3) {
            auto begin = _vals.get_bucket().cbegin_list<float>();
            auto end = _vals.get_bucket().cend_list<float>();
            for (; begin != end; ++begin, ++start) {
                start->imag() = *begin;
            }
            return std::move(output);
        }
        return std::move(output);
    }
#ifdef _HALF_FLOAT_SUPPORT_
    else if (dtype == DType::Float16) {
        complex_32 *start = reinterpret_cast<complex_32 *>(output.data_ptr());
        uint32_t type = _vals.get_bucket().iterator_type();
        if (type == 1) {
            const float16_t *begin =
                reinterpret_cast<const float16_t *>(data_ptr());
            const float16_t *end =
                reinterpret_cast<const float16_t *>(data_ptr_end());
            for (; begin != end; ++begin, ++start) {
                start->imag() = *begin;
            }
            return std::move(output);
        } else if (type == 2) {
            auto begin = _vals.get_bucket().cbegin_blocked<float16_t>();
            auto end = _vals.get_bucket().cend_blocked<float16_t>();
            for (; begin != end; ++begin, ++start) {
                start->imag() = *begin;
            }
            return std::move(output);
        } else if (type == 3) {
            auto begin = _vals.get_bucket().cbegin_list<float16_t>();
            auto end = _vals.get_bucket().cend_list<float16_t>();
            for (; begin != end; ++begin, ++start) {
                start->imag() = *begin;
            }
            return std::move(output);
        }

        return std::move(output);
    }
#endif
    return std::move(output);
}

// I am going to remake this
// now that I can have a Tensor with a dtype of Tensor
// There may as well just be a Tensor with all the Tensor objects inside of it
// that way there is very little overhead, except for at the beggining

Tensor::Tensor(size_value_t i, const ArrayVoid &Arr, SizeRef &&_s)
    : _size({i}), _total_size(i), sub_tensor(false), _vals(i, DType::TensorObj),
      dtype(DType::TensorObj), stored_strides(nullptr) {
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
    : _size(std::move(_s)), _total_size(0), sub_tensor(false),
      _vals(std::move(Arr)), dtype(DType::Float),
      stored_strides(std::move(strides)) {
    _total_size = _vals.Size();
    dtype = _vals.dtype;
}

Tensor::Tensor(ArrayVoid Arr, SizeRef _s,
               const std::vector<size_value_t> &strides)
    : _size(std::move(_s)), _total_size(0), sub_tensor(false),
      _vals(std::move(Arr)), dtype(DType::Float),
      stored_strides(strides.size()) {
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
    output._size = SizeRef({static_cast<size_value_t>(result_ranges.size())});
    Tensor *begin = reinterpret_cast<Tensor *>(output.data_ptr());
    Tensor *end = begin + result_ranges.size();
    auto ra_begin = result_ranges.cbegin();
    for (; begin != end; ++begin, ++ra_begin)
        *begin = (*this)[*ra_begin];
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
	}
	return std::move(buckets); 
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
	}
	return std::move(buckets); 
}

Tensor Tensor::unfold(size_value_t dim, size_value_t size,
                      size_value_t step) const {
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
    utils::THROW_EXCEPTION(
        is_contiguous(),
        "Can only find end of data pointer on contiguous tensor");
    return (void *)(reinterpret_cast<uint8_t *>(data_ptr()) +
                    (_total_size * DTypeFuncs::size_of_dtype(dtype)));
}

const void *Tensor::data_ptr_end() const {
    utils::THROW_EXCEPTION(
        is_contiguous(),
        "Can only find end of data pointer on contiguous tensor");
    return (const void *)(reinterpret_cast<const uint8_t *>(data_ptr()) +
                          (_total_size * DTypeFuncs::size_of_dtype(dtype)));
}

// share from a specific point in memory
Tensor Tensor::div(size_value_t i) const {
    return Tensor(_vals.share_array(i, i), {i});
}

ArrayVoid &Tensor::arr_void() { return _vals; }
const ArrayVoid &Tensor::arr_void() const { return _vals; }

Tensor Tensor::repeat_(size_value_t amt) const {
    if (dtype == DType::TensorObj) {
        Tensor output = Tensor::makeNullTensorArray(amt * numel());
        this->_vals.cexecute_function<WRAP_DTYPES<DTypeEnum<DType::TensorObj>>>(
            [&output, &amt](auto start, auto end) {
                Tensor *begin = reinterpret_cast<Tensor *>(output.data_ptr());
                auto start_cpy = start;
                for (size_value_t i = 0; i < amt; ++i) {
                    for (; start != end; ++start, ++begin) {
                        *begin = *start;
                    }
                    start = start_cpy;
                }
            });
        return std::move(output);
    }
    Tensor output = Tensor::makeNullTensorArray(amt);
    Tensor *begin = reinterpret_cast<Tensor *>(output.data_ptr());
    Tensor *end = begin + output.numel();
    for (; begin != end; ++begin)
        *begin = *this;
    return functional::cat(output);
}

Tensor Tensor::repeat_(size_value_t dim, size_value_t amt) const {
    dim = dim < 0 ? dim + dims() : dim;
    if (dim == 0) {
        return this->repeat_(amt);
    }
    Tensor transposed = transpose(0, dim);
    return transposed.repeat_(amt).transpose(0, dim);
}

Tensor Tensor::expand(SizeRef s) const {
    if (shape() == s) {
        return *this;
    }
    utils::THROW_EXCEPTION(
        s.size() >= shape().size(),
        "Expected to expand with same dimensions but got $ compared to $",
        s.size(), dims());
    if (s.size() > shape().size())
        return unsqueeze_as(s).expand(s);
    std::vector<std::pair<size_value_t, size_value_t>>
        expandings; // which dimensions to repeat by how many
    for (int64_t i = 0; i < s.size(); ++i) {
        if (s[i] != shape()[i] && s[i] != 1) {
            utils::THROW_EXCEPTION(
                shape()[i] == 1,
                "The expanded size of the tensor ($) must match "
                "the existing size ($) at non-singleton dimension "
                "$.  Target sizes: $.  Tensor sizes: $",
                s[i], shape()[i], i, s, shape());
            expandings.push_back({i, s[i]});
        }
    }
    if (expandings.size() == 0) {
        return *this;
    }
    Tensor expanded = this->repeat_(expandings[0].first, expandings[0].second);
    for (size_t i = 1; i < expandings.size(); ++i)
        expanded = expanded.repeat_(expandings[i].first, expandings[i].second);
    return std::move(expanded);
}

Tensor Tensor::expand_as(const Tensor &t) const {
    if (shape() == t.shape()) {
        return *this;
    }
    return expand(t.shape());
}

// fill, subtract, add, multiply
Tensor &Tensor::subtract_(Scalar val) {
    _vals -= val;
    return *this;
}
Tensor &Tensor::subtract_(const Tensor &val) {
    return functional::subtract_(*this, val);
}
Tensor &Tensor::multiply_(Scalar val) {
    _vals *= val;
    return *this;
}
Tensor &Tensor::multiply_(const Tensor &val) {
    return functional::hadamard_multiply_this(*this, val);
}
Tensor &Tensor::divide_(Scalar val) {
    _vals /= val;
    return *this;
}
Tensor &Tensor::divide_(const Tensor &val) {
    return functional::divide_(*this, val);
}
Tensor &Tensor::fill_(Scalar val) {
    *this = val;
    return *this;
}
Tensor &Tensor::fill_(const Tensor &val) {
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
        _vals.transform_function([](auto &a, auto &b) { return b; }, val._vals);
        return *this;
    }
    _vals.for_each<DType::TensorObj>([&val](auto &inp) { inp = val; });
    return *this;
}

Tensor& Tensor::fill_diagonal_(Scalar c){
    if(dims() == 2){
        const int64_t& rows = shape()[-2];
        const int64_t& cols = shape()[-1];
        if(dtype == DType::TensorObj){
            _vals.execute_function<WRAP_DTYPES<DTypeEnum<DType::TensorObj> > >( [&c, &rows, &cols](auto begin, auto end){
                int64_t min_rows = std::min(rows, cols);
                for(int64_t r = 0; r < min_rows; ++r){
                    if (end < begin || begin == end) break; // Ensure within bounds
                    *begin = c;  // Assign diagonal element
                    begin += cols + 1;  // Move to next diagonal element
                }
            });
            return *this;
        }
        _vals.execute_function<WRAP_DTYPES<NumberTypesL, DTypeEnum<DType::Bool> > >( [&c, &rows, &cols](auto begin, auto end){
            using value_t = utils::IteratorBaseType_t<decltype(begin)>;
            auto v = c.to<value_t>();
            int64_t min_rows = std::min(rows, cols);
            for(int64_t r = 0; r < min_rows; ++r){
                if (end < begin || begin == end) break; // Ensure within bounds
                *begin = v;  // Assign diagonal element
                begin += cols + 1;  // Move to next diagonal element
            }
        });
        return *this;
    }
    if(dims() < 2 && dtype == DType::TensorObj){
        // _vals.execute_function<WRAP_DTYPES<DTypeEnum<DType::TensorObj> > >( [&c](auto begin, auto end){
        //     for(;begin != end; ++begin)
        //         begin->fill_diagonal_(c);
        //     // int64_t min_rows = std::min(rows, cols);
        //     // for(int64_t r = 0; r < min_rows; ++r){
        //     //     begin[r] = c;
        //     //     begin += cols;
        //     // }
        // });
        _vals.for_each<DType::TensorObj>([&c](auto &inp) { inp.fill_diagonal_(c); });
        return *this;
    }
    utils::throw_exception(dims() > 2, "cannot diagonally fill a tensor with dims less than 2, but got $", dims());
    Tensor split = this->split_axis(-3);
    Tensor* begin = reinterpret_cast<Tensor*>(split.data_ptr());
    Tensor* end = reinterpret_cast<Tensor*>(split.data_ptr_end());
    for(;begin != end; ++begin)
        begin->fill_diagonal_(c);
    return *this;
}

Tensor Tensor::diagonal() const {
    if(dims() == 2){
        const int64_t& rows = shape()[-2];
        const int64_t& cols = shape()[-1];
        const int64_t out_cols = std::min(rows, cols);
        ArrayVoid my_vals = _vals.bucket_all_indices();
        ArrayVoid new_vals = _vals.new_strides(out_cols);
        void **out_begin = new_vals.stride_begin();
        void **my_begin = my_vals.stride_begin();
        for(int64_t r = 0; r < out_cols; ++r, ++out_begin){
            *out_begin = my_begin[r * cols + r];
        }
        return Tensor(new_vals, {static_cast<size_value_t>(out_cols)});
    }
    if(dtype == DType::TensorObj){
        Tensor out = Tensor::makeNullTensorArray(numel());
        Tensor* out_begin = reinterpret_cast<Tensor*>(out.data_ptr());
        _vals.cexecute_function<WRAP_DTYPES<DTypeEnum<DType::TensorObj> > >( [&out_begin](auto begin, auto end){
            for(;begin != end; ++begin, ++out_begin)
                *out_begin = begin->diagonal();
        });
        return std::move(out); 
    }
    utils::throw_exception(dims() > 2, "cannot get diagonal from a tensor with dims less than 2, but got $", dims());
    const int64_t& rows = shape()[-2];
    const int64_t& cols = shape()[-1];
    const int64_t matrix_s = rows * cols;
    const int64_t batches = numel() / (rows * cols);
    const int64_t out_cols = std::min(rows, cols);
    ArrayVoid my_vals = _vals.bucket_all_indices();
    ArrayVoid new_vals = _vals.new_strides(out_cols * batches);
    void **out_begin = new_vals.stride_begin();
    void **my_begin = my_vals.stride_begin();
    for(int64_t b = 0; b < batches; ++b, my_begin += matrix_s){
        for(int64_t r = 0; r < out_cols; ++r, ++out_begin){
            *out_begin = my_begin[r * cols + r];
        }
    }
    std::vector<size_value_t> out_sh = shape().Vec();
    out_sh.pop_back();
    out_sh.back() = out_cols;
    return Tensor(new_vals, SizeRef(std::move(out_sh)));

}

Tensor &Tensor::add_(Scalar val) {
    _vals += val;
    return *this;
}
Tensor &Tensor::add_(const Tensor &val) { return functional::add_(*this, val); }

Tensor Tensor::operator==(Scalar c) const {
    return Tensor(_vals == c, shape());
}

Tensor Tensor::operator!=(Scalar c) const {
    return Tensor(_vals != c, shape());
}

Tensor Tensor::operator<=(Scalar c) const {
    return Tensor(_vals <= c, shape());
}

Tensor Tensor::operator>=(Scalar c) const {
    return Tensor(_vals >= c, shape());
}

Tensor Tensor::operator<(Scalar c) const { return Tensor(_vals < c, shape()); }

Tensor Tensor::operator>(Scalar c) const { return Tensor(_vals > c, shape()); }

CommaOperator Tensor::operator<<(Scalar s) {
    utils::throw_exception(is_contiguous(), "Must be contiguous to use comma operator");
    CommaOperator out(data_ptr(), data_ptr_end(), dtype);
    return out , s;
}


std::string_view Tensor::sv() const {
    utils::THROW_EXCEPTION(
        dtype == DType::uint8,
        "\nRuntimeError: Expected DType for string_view to be uint8 but got $",
        dtype);
    utils::THROW_EXCEPTION(is_contiguous(),
                           "Can only convert contiguous tensor to string_view");
    return std::string_view(reinterpret_cast<const char *>(data_ptr()),
                            numel());
}

Tensor Tensor::to_dtype(DType _dt) const {
    if (dtype == _dt)
        return *this;
    return Tensor(_vals.to(_dt), shape());
}

Tensor Tensor::to_device(DeviceType _dt) const {
    return Tensor(_vals.to(_dt), shape());
}

Tensor Tensor::Int() const { return Tensor(_vals.int32(), shape()); }

Tensor Tensor::Long() const { return Tensor(_vals.uint32(), shape()); }

Tensor Tensor::unsqueeze(size_value_t dim) const {
    dim = dim < 0 ? (dim + dims() + 1) : dim;
    std::vector<SizeRef::ArrayRefInt::value_type> Vec = shape().Vec();
    Vec.insert(Vec.begin() + dim, 1);
    return view(SizeRef(std::move(Vec)));
}

Tensor Tensor::unsqueeze_as(const Tensor &t) const {
    std::vector<SizeRef::ArrayRefInt::value_type> Vec = shape().Vec();
    while (Vec.size() < t.dims())
        Vec.insert(Vec.begin(), 1);
    return view(SizeRef(std::move(Vec)));
}

Tensor Tensor::unsqueeze_as(const SizeRef &s) const {
    std::vector<SizeRef::ArrayRefInt::value_type> Vec = shape().Vec();
    while (Vec.size() < s.size())
        Vec.insert(Vec.begin(), 1);
    return view(SizeRef(std::move(Vec)));
}

// now gets rid of all dimensions that are 1
Tensor Tensor::squeeze(utils::optional_list list) const {
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

inline static constexpr auto sumation_func = [](auto begin, auto end,
                                                ArrayVoid &outp) -> Scalar {
    return std::reduce(begin, end);
};

Tensor sum_one(const Tensor &t, Tensor::size_value_t dim) {
    if (dim == t.dims() || dim == (-1) * (t.dims() + 1))
        return t;
    dim = dim < 0 ? dim + t.dims() : dim;
    if (t.dtype == DType::TensorObj) {
        Tensor outp(t.shape(), DType::TensorObj);
        t.arr_void().transform_function<DType::TensorObj>(
            [&dim](const Tensor &output) -> Tensor { return output.sum(dim); },
            reinterpret_cast<Tensor *>(outp.data_ptr()));
        return std::move(outp);
    }
    if(dim != 0){
        return sum_one(t.transpose(0, dim), 0).transpose(0, dim);
    }
    Tensor::size_value_t total_size = t.shape()[0];
    Tensor split = t.split_axis(0);
    Tensor a = split[0].item<Tensor>().clone();
    const Tensor *begin = reinterpret_cast<const Tensor *>(split.data_ptr());
    // in the future, use a mutex to lock this and save individual indices
    // otherwise it returns the incorrect result
    // threading::preferential_parallel_for(
    //     threading::block_ranges<1>(1, split.numel()),
    //     [&a, &begin](const auto &range) {
    //         for (int64_t i = range.begin[0]; i < range.end[0]; ++i) {
    //             a += begin[i];
    //         }
    //     });
    const Tensor* end = reinterpret_cast<const Tensor*>(split.data_ptr_end());
    ++begin;
    for(;begin != end; ++begin){
        a += *begin;
    }
    auto Vec = t.shape().Vec();
    Vec[0] = 1;
    return a.view(SizeRef(std::move(Vec)));
}



Tensor Tensor::sum(utils::optional_list list, bool keepdim) const {
    if (dtype == DType::TensorObj) {
        Tensor outp(shape(), DType::TensorObj);
        _vals.transform_function<DType::TensorObj>(
            [&](const Tensor &output) -> Tensor {
                return output.sum(list, keepdim);
            },
            reinterpret_cast<Tensor *>(outp.data_ptr()));
        return std::move(outp);
    }
    if (!list) {
        Tensor outp(1, dtype);
        outp = _vals.cexecute_function<WRAP_DTYPES<NumberTypesL>>()(
            [](auto begin, auto end) -> Scalar {
                return mp::accumulate(
                    begin, end, utils::IteratorBaseType_t<decltype(begin)>(0));
            });
        if (keepdim) {
            std::vector<SizeRef::ArrayRefInt::value_type> Vec(dims());
            std::fill(Vec.begin(), Vec.end(), 1);
            outp = outp.view(SizeRef(std::move(Vec)));
        }
        return std::move(outp);
    }
    int64_t dim = list[0] < 0 ? list[0] + dims() : list[0];
    Tensor output = sum_one(*this, dim);
    for (auto begin = list->cbegin() + 1; begin != list->cend(); ++begin) {
        dim = *begin < 0 ? *begin + dims() : *begin;
        output = sum_one(output, dim);
    }
    if (!keepdim) {
        return output.squeeze();
    }
    return std::move(output);
}

Tensor Tensor::mean(utils::optional_list list, bool keepdim) const {
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
        size_value_t total_scale = shape().multiply();
        double div = 1.0 / (double)total_scale;
        Tensor output = sum(nullptr, keepdim);
        return output * div;
    }
    size_value_t total_scale = 0;
    for (const auto &dim : list) {
        total_scale += shape()[dim];
    }
    Tensor output = sum(list, keepdim);
    if (total_scale == 0) {
        output.arr_void().fill_(0);
        return std::move(output);
    }
    double div = 1.0 / (double)total_scale;
    output *= div;
    return std::move(output);
}

Tensor Tensor::sum_as(const SizeRef &s) const {
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
    utils::THROW_EXCEPTION(
        t.dims() == dims(),
        "Expected dims to be equal for sum as but got $ and $", t.dims(),
        dims());
    if (t.numel() == 1) {
        return sum();
    }
    return sum_as(t.shape());
}

result_types::max<Tensor, Tensor> max_(const Tensor &_x) {
    if (_x.dtype == DType::TensorObj) {
        result_types::max<Tensor, Tensor> output(
            Tensor(_x.shape(), DType::TensorObj),
            Tensor(_x.shape(), DType::TensorObj));
        _x.arr_void()
            .cexecute_function<WRAP_DTYPES<DTypeEnum<DType::TensorObj>>>(
                [&output](auto begin, auto end) {
                    Tensor *v_begin =
                        reinterpret_cast<Tensor *>(output.values.data_ptr());
                    Tensor *i_begin =
                        reinterpret_cast<Tensor *>(output.indices.data_ptr());
                    for (; begin != end; ++begin, ++v_begin, ++i_begin) {
                        result_types::max<Tensor, Tensor> o = begin->max();
                        *v_begin = o.values;
                        *i_begin = o.indices;
                    }
                });
        return std::move(output);
    }
    Tensor outp(1, _x.dtype);
    Tensor indices(_x.shape(), DType::Bool);
    indices.fill_(false);
    outp = _x.arr_void().cexecute_function<WRAP_DTYPES<NumberTypesL>>(
        [&indices](auto begin, auto end) -> Scalar {
            using value_t = utils::IteratorBaseType_t<decltype(begin)>;
            uint_bool_t *i_begin =
                reinterpret_cast<uint_bool_t *>(indices.data_ptr());
            value_t max_element = *begin;
            uint_bool_t *max_indice = i_begin;
            ++begin;
            for (; begin != end; ++begin, ++i_begin) {
                if (*begin > max_element) {
                    max_element = *begin;
                    max_indice = i_begin;
                }
            }
            *max_indice = uint_bool_t(true);
            return max_element;
        });
    return result_types::max<Tensor, Tensor>(std::move(outp),
                                             std::move(indices));
}

result_types::max<Tensor, Tensor> max_(const Tensor &_x,
                                       Tensor::size_value_t dim) {
    dim = dim < 0 ? dim + _x.dims() : dim;
    if (_x.dtype == DType::TensorObj) {
        result_types::max<Tensor, Tensor> output(
            Tensor(_x.shape(), DType::TensorObj),
            Tensor(_x.shape(), DType::TensorObj));
        _x.arr_void()
            .cexecute_function<WRAP_DTYPES<DTypeEnum<DType::TensorObj>>>(
                [&output, &dim](auto begin, auto end) {
                    Tensor *v_begin =
                        reinterpret_cast<Tensor *>(output.values.data_ptr());
                    Tensor *i_begin =
                        reinterpret_cast<Tensor *>(output.indices.data_ptr());
                    for (; begin != end; ++begin, ++v_begin, ++i_begin) {
                        result_types::max<Tensor, Tensor> o = begin->max(dim);
                        *v_begin = o.values;
                        *i_begin = o.indices;
                    }
                });
        return output;
    }
    Tensor bools(_x.shape(), DType::Bool);
    SizeRef o_shape = _x.shape().delete_index(dim);
    if (dim == _x.dims() - 1) {
        _x.arr_void().cexecute_function<WRAP_DTYPES<NumberTypesL>>(
            [&bools](auto begin, auto end) {
                using value_t = utils::IteratorBaseType_t<decltype(begin)>;
                uint_bool_t *b_begin =
                    reinterpret_cast<uint_bool_t *>(bools.data_ptr());
                const Tensor::size_value_t &rows = bools.shape()[-1];
                while (begin != end) {
                    uint_bool_t *b_max_ele = b_begin;
                    value_t max_ele = *begin;
                    auto current_end = begin + rows;
                    ++begin;
                    ++b_begin;
                    for (; begin != current_end; ++begin, ++b_begin) {
                        if (*begin > max_ele) {
                            max_ele = *begin;
                            b_max_ele = b_begin;
                        }
                    }
                    *b_max_ele = uint_bool_t(true);
                }
            });
        return result_types::max<Tensor, Tensor>((_x)[bools].view(o_shape),
                                                 std::move(bools));
    }
    Tensor bools_t = bools.transpose(dim, -1);
    Tensor _t = _x.transpose(dim, -1);
    _t.arr_void().cexecute_function<WRAP_DTYPES<NumberTypesL>>(
        [&bools_t](auto begin, auto end) {
            using value_t = utils::IteratorBaseType_t<decltype(begin)>;
            auto b_begin =
                bools_t.arr_void().get_bucket().begin<3, uint_bool_t>();
            const Tensor::size_value_t &rows = bools_t.shape()[-1];
            while (begin != end) {
                auto b_max_ele = b_begin;
                value_t max_ele = *begin;
                auto current_end = begin + rows;
                ++begin;
                ++b_begin;
                for (; begin != current_end; ++begin, ++b_begin) {
                    if (*begin > max_ele) {
                        max_ele = *begin;
                        b_max_ele = b_begin;
                    }
                }
                *b_max_ele = uint_bool_t(true);
            }
        });

    return result_types::max<Tensor, Tensor>((_x)[bools].view(o_shape),
                                             std::move(bools));
    /* if(dim == dims()-1){ */

    /* } */

    /* size_value_t total_size = shape().flatten(0,dim)[0]; */
    /* Tensor outp(shape()[my_range(0, dim)], dtype); */
    /* const Tensor split = this->split_axis(dim); */
    /* outp._vals.execute_function<WRAP_DTYPES<RealNumberTypesL>>()([](auto
     * begin, auto end, const Tensor* vals){ */
    /* 			using value_t =
     * utils::IteratorBaseType_t<decltype(begin)>; */
    /* 			for(;begin != end; ++begin, ++vals){ */
    /* 				*begin = vals->max().toScalar().to<value_t>();
     */
    /* 			} */
    /* 		}, reinterpret_cast<const Tensor*>(split.data_ptr())); */
    /* return std::move(outp); */
}

result_types::max<Tensor, Tensor> Tensor::max(utils::optional_list list) const {
    if (!list) {
        return max_(*this);
    }
    result_types::max<Tensor, Tensor> res_1 = max_(*this, list[0]);
    Tensor cur_indices = std::move(res_1.indices);
    SizeRef o_shape = shape().delete_index(list[0]);
    for (auto begin = list->cbegin() + 1; begin != list->cend(); ++begin) {
        result_types::max<Tensor, Tensor> res = max_(*this, *begin);
        cur_indices = cur_indices || res.indices;
        o_shape = o_shape.delete_index(*begin);
    }
    return result_types::max<Tensor, Tensor>((*this)[cur_indices].view(o_shape),
                                             std::move(cur_indices));
}

result_types::max<Tensor, Tensor> min_(const Tensor &_x) {
    if (_x.dtype == DType::TensorObj) {
        result_types::max<Tensor, Tensor> output(
            Tensor(_x.shape(), DType::TensorObj),
            Tensor(_x.shape(), DType::TensorObj));
        _x.arr_void()
            .cexecute_function<WRAP_DTYPES<DTypeEnum<DType::TensorObj>>>(
                [&output](auto begin, auto end) {
                    Tensor *v_begin =
                        reinterpret_cast<Tensor *>(output.values.data_ptr());
                    Tensor *i_begin =
                        reinterpret_cast<Tensor *>(output.indices.data_ptr());
                    for (; begin != end; ++begin, ++v_begin, ++i_begin) {
                        result_types::max<Tensor, Tensor> o = begin->min();
                        *v_begin = o.values;
                        *i_begin = o.indices;
                    }
                });
        return std::move(output);
    }
    Tensor outp(1, _x.dtype);
    Tensor indices(_x.shape(), DType::Bool);
    indices.fill_(false);
    outp = _x.arr_void().cexecute_function<WRAP_DTYPES<NumberTypesL>>()(
        [&indices](auto begin, auto end) -> Scalar {
            using value_t = utils::IteratorBaseType_t<decltype(begin)>;
            uint_bool_t *i_begin =
                reinterpret_cast<uint_bool_t *>(indices.data_ptr());
            value_t min_element = *begin;
            uint_bool_t *min_indice = i_begin;
            ++begin;
            for (; begin != end; ++begin, ++i_begin) {
                if (*begin < min_element) {
                    min_element = *begin;
                    min_indice = i_begin;
                }
            }
            *min_indice = uint_bool_t(true);
            return min_element;
        });
    return result_types::max<Tensor, Tensor>(std::move(outp),
                                             std::move(indices));
}

result_types::max<Tensor, Tensor> min_(const Tensor &_x,
                                       Tensor::size_value_t dim) {
    dim = dim < 0 ? dim + _x.dims() : dim;
    if (_x.dtype == DType::TensorObj) {
        result_types::max<Tensor, Tensor> output(
            Tensor(_x.shape(), DType::TensorObj),
            Tensor(_x.shape(), DType::TensorObj));
        _x.arr_void()
            .cexecute_function<WRAP_DTYPES<DTypeEnum<DType::TensorObj>>>(
                [&output, &dim](auto begin, auto end) {
                    Tensor *v_begin =
                        reinterpret_cast<Tensor *>(output.values.data_ptr());
                    Tensor *i_begin =
                        reinterpret_cast<Tensor *>(output.indices.data_ptr());
                    for (; begin != end; ++begin, ++v_begin, ++i_begin) {
                        result_types::max<Tensor, Tensor> o = begin->min(dim);
                        *v_begin = o.values;
                        *i_begin = o.indices;
                    }
                });
        return output;
    }
    Tensor bools(_x.shape(), DType::Bool);
    SizeRef o_shape = _x.shape().delete_index(dim);
    if (dim == _x.dims() - 1) {
        _x.arr_void().cexecute_function<WRAP_DTYPES<NumberTypesL>>(
            [&bools](auto begin, auto end) {
                using value_t = utils::IteratorBaseType_t<decltype(begin)>;
                uint_bool_t *b_begin =
                    reinterpret_cast<uint_bool_t *>(bools.data_ptr());
                const Tensor::size_value_t &rows = bools.shape()[-1];
                while (begin != end) {
                    uint_bool_t *b_min_ele = b_begin;
                    value_t min_ele = *begin;
                    auto current_end = begin + rows;
                    ++begin;
                    ++b_begin;
                    for (; begin != current_end; ++begin, ++b_begin) {
                        if (*begin < min_ele) {
                            min_ele = *begin;
                            b_min_ele = b_begin;
                        }
                    }
                    *b_min_ele = uint_bool_t(true);
                }
            });
        return result_types::max<Tensor, Tensor>((_x)[bools].view(o_shape),
                                                 std::move(bools));
    }
    Tensor bools_t = bools.transpose(dim, -1);
    Tensor _t = _x.transpose(dim, -1);
    _t.arr_void().cexecute_function<WRAP_DTYPES<NumberTypesL>>(
        [&bools_t](auto begin, auto end) {
            using value_t = utils::IteratorBaseType_t<decltype(begin)>;
            auto b_begin =
                bools_t.arr_void().get_bucket().begin<3, uint_bool_t>();
            const Tensor::size_value_t &rows = bools_t.shape()[-1];
            while (begin != end) {
                auto b_min_ele = b_begin;
                value_t min_ele = *begin;
                auto current_end = begin + rows;
                ++begin;
                ++b_begin;
                for (; begin != current_end; ++begin, ++b_begin) {
                    if (*begin < min_ele) {
                        min_ele = *begin;
                        b_min_ele = b_begin;
                    }
                }
                *b_min_ele = uint_bool_t(true);
            }
        });

    return result_types::max<Tensor, Tensor>((_x)[bools].view(o_shape),
                                             std::move(bools));
    /* if(dim == dims()-1){ */

    /* } */

    /* size_value_t total_size = shape().flatten(0,dim)[0]; */
    /* Tensor outp(shape()[my_range(0, dim)], dtype); */
    /* const Tensor split = this->split_axis(dim); */
    /* outp._vals.execute_function<WRAP_DTYPES<RealNumberTypesL>>()([](auto
     * begin, auto end, const Tensor* vals){ */
    /* 			using value_t =
     * utils::IteratorBaseType_t<decltype(begin)>; */
    /* 			for(;begin != end; ++begin, ++vals){ */
    /* 				*begin = vals->max().toScalar().to<value_t>();
     */
    /* 			} */
    /* 		}, reinterpret_cast<const Tensor*>(split.data_ptr())); */
    /* return std::move(outp); */
}

result_types::max<Tensor, Tensor> Tensor::min(utils::optional_list list) const {
    if (!list) {
        return min_(*this);
    }
    result_types::max<Tensor, Tensor> res_1 = min_(*this, list[0]);
    Tensor cur_indices = std::move(res_1.indices);
    SizeRef o_shape = shape().delete_index(list[0]);
    for (auto begin = list->cbegin() + 1; begin != list->cend(); ++begin) {
        result_types::max<Tensor, Tensor> res = min_(*this, *begin);
        cur_indices = cur_indices || res.indices;
        o_shape = o_shape.delete_index(*begin);
    }
    return result_types::max<Tensor, Tensor>((*this)[cur_indices].view(o_shape),
                                             std::move(cur_indices));
}

Tensor Tensor::exp() const {
    utils::THROW_EXCEPTION(
        dtype != DType::Bool,
        "\nRuntimeError: Tried running unsupported DType Bool "
        "with function exp()");
    return Tensor(_vals.exp(), shape());
}

Tensor &Tensor::exp_() {
    utils::THROW_EXCEPTION(
        dtype != DType::Bool,
        "\nRuntimeError: Tried running unsupported DType Bool "
        "with function exp_()");
    _vals.exp_();
    return *this;
}

// this was the function that made me implement the ability to choose a specific
// dtype for a for_each, execute, and transform_function
Tensor &Tensor::inverse_() {
    _vals.inverse_();
    dtype = _vals.dtype;
    return *this;
}

Tensor Tensor::inverse() const { return Tensor(_vals.inverse(), shape()); }

Tensor Tensor::pow(Scalar i) const { return Tensor(_vals.pow(i), shape()); }

Tensor &Tensor::pow_(Scalar i) {
    _vals.pow_(i);
    dtype = _vals.dtype;
    return *this;
}

Tensor &Tensor::clip_(Scalar a, Scalar b) {
    /* std::cout << "clip min: "<<a << */
    _vals.execute_function<WRAP_DTYPES<NumberTypesL>>()(
        [&a, &b](auto begin, auto end) {
            using value_t = utils::IteratorBaseType_t<decltype(begin)>;
            value_t lower = a.to<value_t>();
            value_t upper = b.to<value_t>();
            std::transform(begin, end, begin, [&lower, &upper](auto val) {
                return std::clamp(val, lower, upper);
            });
        });
    return *this;
}

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
    /* output._vals.execute_function<WRAP_DTYPES<NumberTypesL,
     * DTypeEnum<DType::Bool>>>([&v](auto begin, auto end){ */
    /* 	using value_type = typename std::remove_const<typename
     * decltype(begin)::value_type>::type; */
    /* 	utils::flatten_func<Scalar, N>(v, [&begin](const Scalar& a){*begin =
     * a.to<value_type>();}); */
    /* }); */
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

Tensor Tensor::clip(Scalar a, Scalar b) const {
    Tensor outp = this->clone();
    outp.clip_(a, b);
    return std::move(outp);
}

Tensor Tensor::pad(std::vector<size_value_t> p, const char *mode,
                   double value) const {
    utils::THROW_EXCEPTION(
        p.size() % 2 == 0,
        "RuntimeError: The size of the pad must have 2 per dimension");
    utils::THROW_EXCEPTION(
        (p.size() / 2) <= dims(),
        "RuntimeError: expected padding for at most $ dims but instead got $",
        dims(), int(p.size() / 2));

    std::vector<size_value_t> n_shape = shape().Vec();
    size_value_t i = 0;
    size_value_t last_dims = size_value_t(p.size() / 2);
    for (; i < (p.size() / 2); ++i) {
        n_shape[n_shape.size() - (i + 1)] += p[i * 2];
        n_shape[n_shape.size() - (i + 1)] += p[i * 2 + 1];
    }
    Tensor output(SizeRef(std::move(n_shape)), dtype);
    output = value;
    std::vector<nt::my_range> ranges(dims());
    auto begin = p.cbegin();
    size_value_t start = dims() - size_value_t(p.size() / 2);
    for (i = 0; i < dims(); ++i) {
        if (i < (start)) {
            ranges[i] = nt::my_range(0, shape()[i]);
            continue;
        }
        ranges[i] = nt::my_range(*begin, (-1) * size_value_t(*(begin + 1)));
        ++begin;
        ++begin;
    }
    output[ranges].fill_(*this);
    return std::move(output);
}

Tensor Tensor::flip(size_value_t dim) const {
    dim = dim < 0 ? dim + dims() : dim;
    if (dim == 0) {
        return functional::cat(this->split_axis(0).flip()).view(shape());
    }
    utils::THROW_EXCEPTION(
        dim < dims() && dim > 0,
        "RuntimeError: Expected input dim for flip to be less than $ but got $",
        dims(), dim);
    Tensor output(shape(), dtype);
    /* Tensor to_split = (dim == dims()-1) ? this->transpose(-1,-2) : *this; */
    Tensor my_tensors = this->split_axis(dim);
    Tensor out_tensors = output.split_axis(dim);
    const Tensor *begin =
        reinterpret_cast<const Tensor *>(my_tensors.data_ptr());
    Tensor *outp = reinterpret_cast<Tensor *>(out_tensors.data_ptr());
    if (dim > 0) {
        size_value_t a = dim == (dims() - 1)
                             ? shape().transpose(-1, -2).flatten(0, -3)[0]
                             : shape().flatten(0, dim - 1)[0];
        size_value_t b = size_value_t(out_tensors.shape()[0] / a);
        for (size_value_t i = 0; i < a; ++i, outp += b) {
            for (size_value_t j = b - 1; j >= 0; --j, ++begin) {
                (outp + j)->fill_(*begin);
            }
        }
        return output;
    }
    for (size_value_t i = out_tensors.numel() - 1; i >= 0; --i, ++begin) {
        outp[i].fill_(*begin);
    }
    return output;
}

Tensor Tensor::flip() const {
    Tensor output(shape(), dtype);
    _vals.cexecute_function(
        [](auto begin, auto end, ArrayVoid &v, const size_value_t &numel) {
            using value_t = utils::IteratorBaseType_t<decltype(begin)>;
            auto v_begin = v.get_bucket().begin_contiguous<value_t>();
            for (size_value_t i = numel - 1; i >= 0; --i, ++begin) {
                *(v_begin + i) = *begin;
            }
        },
        output._vals, output.numel());
    return output;
}

Tensor Tensor::flip_() const {
    ArrayVoid cpy1 = _vals.bucket_all_indices();
    ArrayVoid cpy = cpy1.copy_strides(false);
    void **my_strides = cpy1.stride_begin();
    void **out_strides = cpy.stride_begin();
    for (size_value_t i = numel() - 1; i >= 0; --i, ++my_strides) {
        out_strides[i] = *my_strides;
    }
    return Tensor(cpy, shape());
}

/* //there is an issue with dim = -1 */
/* //it seems as though maybe the transpose in split axis isn't it letting it
 * access the exact pointer values? */
/* //Which does not make sense */
/* //Wait, no, that does make sense */
/* //this is because the permute function automatically makes a new stride in
 * memory if the use_count > 1 */
/* //therefore, the solution would be to get the RowColSwap function working
 * again (which needs to happen anyways) */
/* Tensor Tensor::flip_(size_value_t dim){ */
/* 	dim = dim < 0 ? dim + dims() : dim; */
/* 	utils::THROW_EXCEPTION(dim < dims() && dim > 0, "RuntimeError: Expected
 * input dim for flip to be less than $ but got $", dims(), dim); */
/* 	Tensor output(_vals.copy_strides(false), shape()); */
/* 	Tensor my_tensors = (dim == (dims()-1)) ?
 * this->RowColSwap().split_axis(-2) : this->split_axis(dim); */
/* 	Tensor out_tensors = (dim == (dims()-1)) ?
 * output.RowColSwap().split_axis(-2) : output.split_axis(dim); */
/* 	tdtype_list<Tensor> begin = my_tensors.arr_void().tbegin<Tensor>(); */
/* 	Tensor* outp = reinterpret_cast<Tensor*>(out_tensors.data_ptr()); */
/* 	if(dim > 0){ */
/* 		uint32_t a = dim == (dims() - 1) ? shape().flatten(0,-3)[0] :
 * shape().flatten(0,dim-1)[0]; */
/* 		uint32_t b = uint32_t(out_tensors.shape()[0] / a); */
/* 		for(uint32_t i = 0; i < a; ++i, outp += b){ */
/* 			for(int32_t j = b-1; j >= 0; --j, ++begin){ */
/* 				Tensor& from = *begin; */
/* 				Tensor& to = (*(outp + j)); */
/* 				void** to_strides = to._vals.strides_begin(); */
/* 				void** to_strides_e = to._vals.strides_end(); */
/* 				void** from_strides =
 * from._vals.strides_begin(); */
/* 				void** from_strides_e =
 * from._vals.strides_end(); */
/* 				for(;from_strides != from_strides_e & to_strides
 * != to_strides_e; ++from_strides, ++to_strides) */
/* 					*to_strides = *from_strides; */
/* 			} */
/* 		} */
/* 		if(dim == (dims()-1)){ */
/* 			this->RowColSwap(); */
/* 			output.RowColSwap(); */
/* 		} */
/* 		return output; */
/* 	} */
/* 	for(int32_t i = out_tensors.numel()-1; i >= 0; --i, ++begin){ */
/* 		Tensor& to = outp[i]; */
/* 		Tensor& from = (*begin); */
/* 		void** to_strides = to._vals.strides_begin(); */
/* 		void** from_strides = from._vals.strides_begin(); */
/* 		void** from_strides_e = from._vals.strides_end(); */
/* 		for(;from_strides != from_strides_e; ++from_strides,
 * ++to_strides) */
/* 			*to_strides = *from_strides; */
/* 	} */
/* 	if(dim == (dims()-1)){ */
/* 		this->RowColSwap(); */
/* 		output.RowColSwap(); */
/* 	} */
/* 	return output; */

/* } */

/* Tensor Tensor::dilate_(size_value_t dil) const{ */
/* 	ArrayVoid cpy1 = _vals.bucket_all_indices(); */
/* 	ArrayVoid cpy = cpy1.copy_strides(false); */
/* 	void** my_strides = cpy1.stride_begin(); */
/* 	void** outp_strides = cpy.stride_begin(); */

/* 	size_value_t cols = shape()[-1]; */
/* 	size_value_t i_total = shape().multiply(-2); */
/* 	for(size_value_t i = 0; i < numel(); ++i, ++my_strides){ */
/* 		*outp_strides = *my_strides; */
/* 		if((i+1) % cols == 0){ */
/* 			if((i+1) % i_total == 0){outp_strides += 1; continue;}
 */
/* 			outp_strides += outp.shape().back()+(dil-1);continue; */
/* 		} */
/* 		outp_strides += dil; */
/* 	} */
/* 	return outp; */
/* } */

Tensor Tensor::dilate(size_value_t dil) const {
    if (dil == 0)
        return contiguous();
    utils::throw_exception(dims() >= 2, "Expected dim size to be greater than or equal to 2 for dilation but got $", dims());
    std::vector<size_value_t> vec = shape().Vec();
    /* dil -= 1; */
    vec.back() *= dil;
    vec.back() -= (dil - 1);
    vec[vec.size() - 2] *= dil;
    vec[vec.size() - 2] -= (dil - 1);
    Tensor outp = functional::zeros(SizeRef(vec), dtype);
    auto sh = shape();
    size_value_t back_add = (outp.shape().back()  * (dil-1)) + 1;
    _vals.cexecute_function<WRAP_DTYPES<AllTypesL>>(
        [&sh, &back_add, &dil](auto abegin, auto aend, void *obegin) {
            using value_t = utils::IteratorBaseType_t<decltype(abegin)>;
            size_value_t cols = sh[-1];
            size_value_t i_total = sh.multiply(-2);
            value_t *begin = reinterpret_cast<value_t *>(obegin);
            for (uint64_t i = 0; abegin != aend; ++abegin, ++i) {
                *begin = *abegin;
                if ((i + 1) % cols == 0) {
                    //if it is just at the end of the matrix, move to the next matrix
                    if ((i + 1) % i_total == 0) {
                        ++begin;
                        continue;
                    }
                    //otherwise, move down
                    //dil-1 rows, 
                    begin += back_add;
                    continue;
                }
                begin += dil;
            }
        },
        outp.data_ptr());
    return outp;
}

Tensor Tensor::dilate(size_value_t row_dil, size_value_t col_dil) const {
    if (row_dil == 0 && col_dil == 0)
        return contiguous();

    utils::throw_exception(dims() >= 2, "Expected dim size to be greater than or equal to 2 for dilation but got $", dims());

    std::vector<size_value_t> vec = shape().Vec();

    // Adjust shape for dilation (applies to the last two dimensions)
    vec[vec.size() - 1] *= col_dil; // Adjust columns
    vec[vec.size() - 1] -= (col_dil - 1);
    vec[vec.size() - 2] *= row_dil; // Adjust rows
    vec[vec.size() - 2] -= (row_dil - 1);

    Tensor outp = functional::zeros(SizeRef(vec), dtype);

    auto sh = shape();
    auto outp_shape = outp.shape();
    size_value_t rows = sh[-2]; // Original rows
    size_value_t cols = sh[-1]; // Original columns
    size_value_t num_batches = numel() / (rows * cols); // Total number of batches (product of all dims except last two)
    size_value_t output_cols = outp.shape().back();
    size_value_t back_add = (outp.shape().back()  * (row_dil-1)) + 1;
    // size_value_t back_add = (output_cols - cols * col_dil) + col_dil;

    _vals.cexecute_function<WRAP_DTYPES<AllTypesL>>(
        [&sh, &outp_shape, &num_batches, &rows, &cols, &output_cols, &back_add, &row_dil, &col_dil](auto abegin, auto aend, void *obegin) {
            using value_t = utils::IteratorBaseType_t<decltype(abegin)>;
            size_value_t cols = sh[-1];
            size_value_t i_total = sh.multiply(-2);
            value_t *begin = reinterpret_cast<value_t *>(obegin);
            for (uint64_t i = 0; abegin != aend; ++abegin, ++i) {
                *begin = *abegin;
                if ((i + 1) % cols == 0) {
                    if ((i + 1) % i_total == 0) {
                        ++begin;
                        continue;
                    }
                    begin += back_add;
                    continue;
                }
                begin += col_dil;
            }
        },
        outp.data_ptr());
    return outp;
}

// template<typename value_t>
// void process_dilate(std::vector<Tensor::size_value_t>& multiplies,
//                     std::vector<Tensor::size_value_t>& back_adds,
//                     std::vector<Tensor::size_value_t>& dils,
//                     const SizeRef& shape,
//                     const uint64_t& i, 
//                     value_t*& begin){
//     if(!((i+1) % multiplies[multiplies.size()-2] == 0)){begin += dils.back();return;}
    
// }

//Tensor Tensor::dilate(std::vector<size_value_t> dils) const {
//    if(std::all_of(dils.cbegin(), dils.cend(), [](const size_value_t& val){return val <= 1;}))
//            return contiguous();

//    utils::throw_exception(!std::any_of(dils.cbegin(), dils.cend(), 
//                                        [](const size_value_t& val){val < 1;}), 
//                           "cannot dilate less than 1 at any dimension");
//    utils::throw_exception(dims() >= dils.size(), 
//                           "Expected dim size to be greater than or equal to $ for dilation but got $", 
//                           dils.size(), dims());

//    std::vector<size_value_t> vec = shape().Vec();

//    // Adjust shape for dilation (applies to the last two dimensions)
//    auto v_rbegin = vec.rbegin();
//    auto d_rbegin = dils.crbegin();
//    auto d_rend = dils.crend();
//    for(;d_rbegin != d_rend; ++d_rbegin, ++v_rbegin){
//        *v_rbegin *= *d_rbegin;
//        *v_rbegin -= (*d_drbegin - 1);
//    }

//    Tensor outp = functional::zeros(SizeRef(vec), dtype);
//    //size_value_t back_add = (outp.shape().back()  * (row_dil-1)) + 1;

//    auto sh = shape();
//    auto outp_shape = outp.shape();
//    std::vector<size_value_t> multiplies(dils.size());
//    for (int i = dils.size() - 1; i >= 1; --i) {
//        multiplies[dils.size() - 1 - i] = x.shape().multiply(-1 * i);
//    }
//    multiplies.back() = 0;
//    std::vector<size_value_t> back_adds(dils.size());
//    for(int i = 0; i < dils.size(); ++i){
//        back_adds[i] =  multiplies[i] * (dils[i]-1) + 1; 
//    }

//    _vals.cexecute_function<WRAP_DTYPES<AllTypesL>>(
//        [&sh, &outp_shape, &dils, &multiplies](auto abegin, auto aend, void *obegin) {
//            using value_t = utils::IteratorBaseType_t<decltype(abegin)>;
//            size_value_t cols = sh[-1];
//            size_value_t i_total = sh.multiply(-2);
//            value_t *begin = reinterpret_cast<value_t *>(obegin);
//            for (uint64_t i = 0; abegin != aend; ++abegin, ++i) {
//                *begin = *abegin;
//                if ((i + 1) % cols == 0) {
//                    if ((i + 1) % i_total == 0) {
//                        ++begin;
//                        continue;
//                    }
//                    begin += back_add;
//                    continue;
//                }
//                begin += col_dil;
//            }
//        },
//        outp.data_ptr());
//    return outp;
//}



Tensor Tensor::dilate(size_value_t channel_dil, size_value_t row_dil, size_value_t col_dil) const {
    if ((row_dil == 0 || row_dil == 1) && (col_dil == 0 || col_dil == 1) && (channel_dil == 0 || channel_dil == 1))
        return contiguous();

    utils::throw_exception(row_dil >= 1 && col_dil >= 1 && channel_dil >= 1,
                           "Cannot dilate less than 1 but got dilations of {$, $, $}",
                           channel_dil, row_dil, col_dil);

    utils::throw_exception(dims() >= 3, "Expected dim size to be greater than or equal to 3 for 3D dilation but got $", dims());

    std::vector<size_value_t> vec = shape().Vec();

    // Adjust shape for dilation (applies to the last two dimensions)
    vec[vec.size() - 1] *= col_dil; // Adjust columns
    vec[vec.size() - 1] -= (col_dil - 1);
    vec[vec.size() - 2] *= row_dil; // Adjust rows
    vec[vec.size() - 2] -= (row_dil - 1);
    vec[vec.size() - 3] *= channel_dil; // Adjust channels
    vec[vec.size() - 3] -= (channel_dil - 1);

    Tensor outp = functional::zeros(SizeRef(vec), dtype);

    auto sh = shape();
    auto outp_shape = outp.shape();
    size_value_t channels = sh[-3]; // Original channels
    size_value_t rows = sh[-2]; // Original rows
    size_value_t cols = sh[-1]; // Original columns

    // size_value_t num_batches = numel() / (rows * cols); // Total number of batches (product of all dims except last two)
    
    size_value_t output_cols = outp.shape().back();
    size_value_t col_back_add = 1;
    size_value_t row_back_add = (outp.shape().back()  * (row_dil-1)) + 1;
    size_value_t channel_back_add = ((outp.shape()[-1] * outp.shape()[-2]) * channel_dil - 1) + 1;
    // size_value_t back_add = (output_cols - cols * col_dil) + col_dil;

    _vals.cexecute_function<WRAP_DTYPES<AllTypesL>>(
        [&sh, &outp_shape, &channels, &rows, &cols, 
            &col_back_add, &row_back_add, &channel_back_add, 
            &channel_dil, &row_dil, &col_dil](auto abegin, auto aend, void *obegin) {
            using value_t = utils::IteratorBaseType_t<decltype(abegin)>;
            size_value_t cols = sh[-1];
            size_value_t mat_total = sh.multiply(-2);
            size_value_t batched_mat_total = sh.multiply(-3);
            value_t *begin = reinterpret_cast<value_t *>(obegin);
            for (uint64_t i = 0; abegin != aend; ++abegin, ++i) {
                *begin = *abegin;
                if ((i + 1) % cols == 0) {
                    if ((i + 1) % mat_total == 0) {
                        if((i + 1) % batched_mat_total == 0){
                            begin++;
                            continue;
                        }
                        begin += channel_back_add;
                        continue;
                    }
                    begin += row_back_add;
                    continue;
                }
                begin += col_dil;
            }
        },
        outp.data_ptr());
    return outp;
}


Tensor Tensor::undilate(size_value_t dil) const {
        return this->undilate(dil, dil);
}

Tensor Tensor::undilate(size_value_t row_dil, size_value_t col_dil) const {
    if ((row_dil == 0 || row_dil == 1) && (col_dil == 0 || col_dil == 1)) {
        return contiguous();
    }
    utils::throw_exception(dims() >= 2, "Expected dim size to be greater than or equal to 2 for undilation but got $", dims());
    utils::throw_exception(row_dil >= 1 && col_dil >= 1,
                           "Cannot dilate less than 1 but got dilations of {$, $}",
                           row_dil, col_dil);

    std::vector<size_value_t> vec = shape().Vec();
    vec.back() = (vec.back() + (col_dil - 1)) / col_dil;
    vec[vec.size() - 2] = (vec[vec.size() - 2] + (row_dil - 1)) / row_dil;
    Tensor outp = functional::zeros(SizeRef(vec), dtype);

    auto sh = shape();
    size_value_t original_cols = sh.back();
    size_value_t original_rows = sh[-2];
    size_value_t matrix_size = original_rows * original_cols;
    size_value_t batches = numel() / matrix_size;
    size_value_t row_add = (sh[-2] * (row_dil - 1));  // Adjust row dilation
    size_value_t col_add = (sh.back() * (col_dil - 1));          // Adjust column dilation
    // size_value_t back_add = sh.back() + (dil - 1);
    _vals.cexecute_function<WRAP_DTYPES<AllTypesL>>(
        [&original_rows, &batches, &matrix_size, &original_cols, &row_dil, &col_dil](auto abegin, auto aend, void *obegin) {
            // std::cout << "called undilate function"<<std::endl;
            using value_t = utils::IteratorBaseType_t<decltype(abegin)>;
            value_t *begin = reinterpret_cast<value_t *>(obegin);
            // std::cout << "starting undilate for loop"<<std::endl;
            // int64_t b_counter = 0;
            for(int64_t b = 0; b < batches; ++b, abegin += matrix_size){
                auto cur_begin = abegin;
                for(int64_t r = 0; r < original_rows; r += row_dil, cur_begin += (original_cols * row_dil)){
                    auto mBegin = cur_begin;
                    for(int64_t c = 0; c < original_cols; c += col_dil, mBegin += col_dil){
                        *begin++ = *mBegin;
                    }
                }
            }
        },
        outp.data_ptr());

    return outp;
}

Tensor Tensor::undilate(size_value_t chan_dil, size_value_t row_dil, size_value_t col_dil) const {
    // If no dilation, return contiguous tensor
    if ((row_dil == 0 || row_dil == 1) && (col_dil == 0 || col_dil == 1) && (chan_dil == 0 || chan_dil == 1)) {
        return contiguous();
    }

    // Check dimensionality and validate dilation values
    utils::throw_exception(dims() >= 3, "Expected dim size to be greater than or equal to 3 for 3D undilation but got $", dims());
    utils::throw_exception(row_dil >= 1 && col_dil >= 1 && chan_dil >= 1,
                           "Cannot dilate less than 1 but got dilations of {$, $, $}",
                           row_dil, col_dil, chan_dil);

    // Calculate the new shape after undilation for each dimension
    std::vector<size_value_t> vec = shape().Vec();
    vec[vec.size() - 3] = (vec[vec.size() - 3] + (chan_dil - 1)) / chan_dil; // Channel dimension
    vec[vec.size() - 2] = (vec[vec.size() - 2] + (row_dil - 1)) / row_dil;    // Row dimension
    vec.back() = (vec.back() + (col_dil - 1)) / col_dil;                        // Column dimension

    Tensor outp = functional::zeros(SizeRef(vec), dtype);

    auto sh = shape();
    size_value_t original_cols = sh.back();
    size_value_t original_rows = sh[-2];
    size_value_t original_channels = sh[-3];
    size_value_t matrix_size = original_rows * original_cols;
    size_value_t channel_size = matrix_size * original_channels;
    size_value_t batches = numel() / channel_size;
    // size_value_t back_add = sh.back() + (dil - 1);
    _vals.cexecute_function<WRAP_DTYPES<AllTypesL>>(
        [&original_channels, &original_rows, &batches, &channel_size, &matrix_size, &original_cols, &chan_dil, &row_dil, &col_dil]
            (auto abegin, auto aend, void *obegin) {
            // std::cout << "called undilate function"<<std::endl;
            using value_t = utils::IteratorBaseType_t<decltype(abegin)>;
            value_t *begin = reinterpret_cast<value_t *>(obegin);
            // std::cout << "starting undilate for loop"<<std::endl;
            // int64_t b_counter = 0;
            for(int64_t b = 0; b < batches; ++b, abegin += channel_size){
                auto cur_begin = abegin;
                for(int64_t d = 0; d < original_channels; d += chan_dil, cur_begin += matrix_size * chan_dil){
                    auto dBegin = cur_begin;
                    for(int64_t r = 0; r < original_rows; r += row_dil, dBegin += (original_cols * row_dil)){
                        auto mBegin = dBegin;
                        for(int64_t c = 0; c < original_cols; c += col_dil, mBegin += col_dil){
                            *begin++ = *mBegin;
                        }
                    }
                }
            }
        },
        outp.data_ptr());
    return std::move(outp);
}


Tensor Tensor::undilate_(size_value_t dil) const {
    return this->undilate_(dil, dil);
    // if (dil == 0) {
    //     return contiguous();
    // }

    // // Calculate the original shape before dilation
    // std::vector<size_value_t> vec = shape().Vec();
    // vec.back() = (vec.back() + (dil - 1)) / dil;
    // vec[vec.size() - 2] = (vec[vec.size() - 2] + (dil - 1)) / dil;
    // SizeRef outp_shape(std::move(vec));
    // /* Tensor outp = functional::zeros(SizeRef(vec), dtype); */

    // ArrayVoid cpy1 = _vals.bucket_all_indices();
    // ArrayVoid cpy = cpy1.new_strides(outp_shape.multiply());
    // void **my_strides = cpy1.stride_begin();
    // void **outp_strides = cpy.stride_begin();

    // size_value_t cols = shape()[-1];
    // size_value_t i_total = shape().multiply(-2);
    // size_value_t outp_cols = vec.back();
    // size_value_t outp_i_total = vec[vec.size() - 2];

    // for (size_value_t i = 0; i < numel(); ++i, ++my_strides) {
    //     // Check if the current element should be part of the original tensor
    //     if ((i % (outp_cols * dil)) % dil == 0 &&
    //         (i / (outp_cols * dil)) % dil == 0) {
    //         *outp_strides = *my_strides;
    //         ++outp_strides;
    //     }
    // }

    // return Tensor(std::move(cpy), std::move(outp_shape));
}

Tensor Tensor::undilate_(size_value_t row_dil, size_value_t col_dil) const {
    if ((row_dil == 0 || row_dil == 1) && (col_dil == 0 || col_dil == 1)) {
        return *this;
    }
    utils::throw_exception(dims() >= 2, "Expected dim size to be greater than or equal to 2 for undilation but got $", dims());
    utils::throw_exception(row_dil >= 1 && col_dil >= 1,
                           "Cannot dilate less than 1 but got dilations of {$, $}",
                           row_dil, col_dil);

    std::vector<size_value_t> vec = shape().Vec();
    vec.back() = (vec.back() + (col_dil - 1)) / col_dil;
    vec[vec.size() - 2] = (vec[vec.size() - 2] + (row_dil - 1)) / row_dil;
    SizeRef outp_shape(std::move(vec));
    
    ArrayVoid cpy1 = _vals.bucket_all_indices();
    ArrayVoid cpy = cpy1.new_strides(outp_shape.multiply());
    void **my_strides = cpy1.stride_begin();
    void **outp_strides = cpy.stride_begin();


    auto sh = shape();
    size_value_t original_cols = sh.back();
    size_value_t original_rows = sh[-2];
    size_value_t matrix_size = original_rows * original_cols;
    size_value_t batches = numel() / matrix_size;
    for(int64_t b = 0; b < batches; ++b, my_strides += matrix_size){
        void **cur_begin = my_strides;
        for(int64_t r = 0; r < original_rows; r += row_dil, cur_begin += (original_cols * row_dil)){
            void **mBegin = cur_begin;
            for(int64_t c = 0; c < original_cols; c += col_dil, mBegin += col_dil){
                *outp_strides++ = *mBegin;
            }
        }
    }

    return Tensor(std::move(cpy), std::move(outp_shape));
}

Tensor Tensor::undilate_(size_value_t chan_dil, size_value_t row_dil, size_value_t col_dil) const {
    if ((row_dil == 0 || row_dil == 1) && (col_dil == 0 || col_dil == 1) && (chan_dil == 0 || chan_dil == 1)) {
        return *this;
    }

    // Check dimensionality and validate dilation values
    utils::throw_exception(dims() >= 3, "Expected dim size to be greater than or equal to 3 for 3D undilation but got $", dims());
    utils::throw_exception(row_dil >= 1 && col_dil >= 1 && chan_dil >= 1,
                           "Cannot dilate less than 1 but got dilations of {$, $, $}",
                           row_dil, col_dil, chan_dil);

    // Calculate the new shape after undilation for each dimension
    std::vector<size_value_t> vec = shape().Vec();
    vec[vec.size() - 3] = (vec[vec.size() - 3] + (chan_dil - 1)) / chan_dil; // Channel dimension
    vec[vec.size() - 2] = (vec[vec.size() - 2] + (row_dil - 1)) / row_dil;    // Row dimension
    vec.back() = (vec.back() + (col_dil - 1)) / col_dil;                        // Column dimension
    SizeRef outp_shape(std::move(vec));
    
    ArrayVoid cpy1 = _vals.bucket_all_indices();
    ArrayVoid cpy = cpy1.new_strides(outp_shape.multiply());
    void **my_strides = cpy1.stride_begin();
    void **outp_strides = cpy.stride_begin();


    auto sh = shape();
    size_value_t original_cols = sh.back();
    size_value_t original_rows = sh[-2];
    size_value_t original_channels = sh[-3];
    size_value_t matrix_size = original_rows * original_cols;
    size_value_t channel_size = matrix_size * original_channels;
    size_value_t batches = numel() / channel_size;

    for(int64_t b = 0; b < batches; ++b, my_strides += channel_size){
        void **cur_begin = my_strides;
        for(int64_t d = 0; d < original_channels; d += chan_dil, cur_begin += matrix_size * chan_dil){
            void **dBegin = cur_begin;
            for(int64_t r = 0; r < original_rows; r += row_dil, dBegin += (original_cols * row_dil)){
                void **mBegin = dBegin;
                for(int64_t c = 0; c < original_cols; c += col_dil, mBegin += col_dil){
                    *outp_strides++ = *mBegin;
                }
            }
        }
    }

    return Tensor(std::move(cpy), std::move(outp_shape));
}

// a second look at this, this is a really bad idea below

// this version only makes 1 extra value in the amount of a certain element
// then it only expands the size of the actual void** ptr
/* Tensor Tensor::dilate_mem_(size_value_t dil) const{ */
/* 	std::vector<size_value_t> vec = shape().Vec(); */
/* 	/1* dil -= 1; *1/ */
/* 	vec.back() *= dil; */
/* 	vec.back() -= (dil-1); */
/* 	vec[vec.size()-2] *= dil; */
/* 	vec[vec.size()-2] -= (dil-1); */
/* 	ArrayVoid outp_arr_b(1, dtype); */
/* 	outp_arr_b = 1; */
/* 	SizeRef outp_shape(std::move(vec)); */
/* 	ArrayVoid outp_arr = outp_arr_b.new_stride(outp_shape.multiply()); */
/* 	void** outp_strides = outp_arr.strides_begin(); */
/* 	void** my_strides = _vals.strides_cbegin(); */
/* 	size_value_t cols = shape()[-1]; */
/* 	size_value_t i_total = shape().multiply(-2); */
/* 	void* ptr = outp_arr.data_ptr(); */
/* 	for(size_value_t i = 0; i < numel(); ++i, ++my_strides){ */
/* 		*outp_strides = *my_strides; */
/* 		if((i+1) % cols == 0){ */
/* 			if((i+1) % i_total == 0){ */
/* 				*outp_strides = ptr; */
/* 				outp_strides += 1; */
/* 				continue; */
/* 			} */
/* 			for(size_value_t j = 0; j < outp_shape.back() + (dil -
 * 1); ++j, ++outp_strides){ */
/* 				*outp_strides = ptr; */
/* 			} */
/* 			continue; */
/* 		} */
/* 		outp_strides += dil; */
/* 	} */
/* 	return Tensor(outp_arr, std::move(outp_shape)); */

/* } */

}; // namespace nt
