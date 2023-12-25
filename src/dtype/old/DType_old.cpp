#include "DType.h"
#include "DType_enum.h"
#include <__nullptr>
#include <_types/_uint32_t.h>
#include <_types/_uint8_t.h>
#include <complex>
#include <functional>
#include <typeinfo>
#include <iostream>
#include <type_traits>
#include <utility>

namespace nt{

uint_bool_t::uint_bool_t(const bool &val)
	:value(val ? 1 : 0)
{}

uint_bool_t::uint_bool_t(const uint_bool_t &inp)
	:value(inp.value)
{}

uint_bool_t::uint_bool_t(uint_bool_t&& inp)
	:value(std::exchange(inp.value, 0)){}

uint_bool_t& operator=(const bool &val){value = val ? 1 : 0; return *this;}
uint_bool_t& operator=(const uint_bool_t &val){value = val.value;}
uint_bool_t& operator=(uint_bool_t&& val){value = val.value;}

std::ostream& operator<<(std::ostream &out, const uint_bool_t &data){
	out << bool(data.value);
	return out;
}

d_type::d_type(const int32_t _i)
	:type(DType::Integer),
	data(_i)
{}

d_type::d_type(const float _f)
	:type(DType::Float),
	data(_f)
{}

d_type::d_type(const double _d)
	:type(DType::Double),
	data(_d)
{}

d_type::d_type(const uint32_t _l)
	:type(DType::Long),
	data(_l)
{}

d_type::d_type(const Tensor &_t)
	:type(DType::TensorObj),
	data(std::cref(_t))
{}

d_type::d_type(const int32_t _i, DType _t){set(_i, _t);}
d_type::d_type(const float _i, DType _t){set(_i, _t);}
d_type::d_type(const double _i, DType _t){set(_i, _t);}
d_type::d_type(const uint32_t _i, DType _t){set(_i, _t);}
d_type::d_type(const d_type& _t)
	:type(_t.type), data(_t.data)
{}

d_type& d_type::operator=(const d_type& _t){
	data = _t.data;
	type = _t.type;
	return *this;
}

template<typename T>
T d_type::cast_num() const{
	switch(type){
		case DType::Integer:
			return (T)std::get<0>(data);
		case DType::Long:
			return (T)std::get<3>(data);
		case DType::Double:
			return (T)std::get<2>(data);
		case DType::Float:
			return (T)std::get<1>(data);
		default:
			return 0;
	}
}

template uint32_t d_type::cast_num<uint32_t>() const;
template int d_type::cast_num<int>() const;
template double d_type::cast_num<double>() const;
template float d_type::cast_num<float>() const;


const std::type_info& d_type::m_type() const noexcept{
	switch(type){
		case DType::Integer:
			return typeid(int32_t);
		case DType::Float:
			return typeid(float);
		case DType::Double:
			return typeid(double);
		case DType::Long:
			return typeid(uint32_t);
		case DType::Complex64:
			return typeid(std::complex<float>);
		case DType::Complex128:
			return typeid(std::complex<double>);
		case DType::uint8:
			return typeid(uint8_t);
		case DType::int8:
			return typeid(int8_t);
		case DType::int16:
			return typeid(int16_t);
		case DType::uint16:
			return typeid(uint16_t);
		case DType::LongLong:
			return typeid(int64_t);
		case DType::Bool:
			return typeid(uint_bool_t);
		case DType::TensorObj:
			return typeid(Tensor);
		default:
			return typeid(void);
	}
}

void d_type::set(const int32_t _i, DType _t){
	type = _t;
	switch(_t){
		case DType::Integer:
			data = (int32_t)_i;
			break;
		case DType::Long:
			data = (uint32_t)_i;
			break;
		case DType::Double:
			data = (double)_i;
			break;
		case DType::Float:
			data = (float)_i;
			break;
		case DType::cfloat:
			data = std::complex<float>(_i);
			break;
		case DType::cdouble:
			data = std::complex<double>(_i);
			break;
		case DType::uint8:
			data = uint8_t(_i);
			break;
		case DType::int8:
			data = int8_t{_i};
			break;
		case DType::int16:
			data = int16_t(_i);
			break;
		case DType::uint16:
			data = uint16_t(_i);
			break;
		case DType::int64:
			data = int64_t(_i);
			break;
		case DType::Bool:
			data = uint_bool_t(_i > 0 : true ? false);
			break;
		default:
			break;
	}
}
void d_type::set(const float _i, DType _t){
	type = _t;
	switch(_t){
		case DType::Integer:
			data = (int32_t)_i;
			break;
		case DType::Long:
			data = (uint32_t)_i;
			break;
		case DType::Double:
			data = (double)_i;
			break;
		case DType::Float:
			data = (float)_i;
			break;
		case DType::cfloat:
			data = std::complex<float>(_i);
			break;
		case DType::cdouble:
			data = std::complex<double>(_i);
			break;
		case DType::uint8:
			data = uint8_t(_i);
			break;
		case DType::int8:
			data = int8_t{_i};
			break;
		case DType::int16:
			data = int16_t(_i);
			break;
		case DType::uint16:
			data = uint16_t(_i);
			break;
		case DType::int64:
			data = int64_t(_i);
			break;
		case DType::Bool:
			data = uint_bool_t(_i > 0 : true ? false);
			break;
		default:
			break;
	}
}
void d_type::set(const double _i, DType _t){
	type = _t;
	switch(_t){
		case DType::Integer:
			data = (int32_t)_i;
			break;
		case DType::Long:
			data = (uint32_t)_i;
			break;
		case DType::Double:
			data = (double)_i;
			break;
		case DType::Float:
			data = (float)_i;
			break;
		case DType::cfloat:
			data = std::complex<float>(_i);
			break;
		case DType::cdouble:
			data = std::complex<double>(_i);
			break;
		case DType::uint8:
			data = uint8_t(_i);
			break;
		case DType::int8:
			data = int8_t{_i};
			break;
		case DType::int16:
			data = int16_t(_i);
			break;
		case DType::uint16:
			data = uint16_t(_i);
			break;
		case DType::int64:
			data = int64_t(_i);
			break;
		case DType::Bool:
			data = uint_bool_t(_i > 0 : true ? false);
			break;
		default:
			break;
	}

}

void d_type::set(const uint32_t _i, DType _t){
	type = _t;
	switch(_t){
		case DType::Integer:
			data = (int32_t)_i;
			break;
		case DType::Long:
			data = (uint32_t)_i;
			break;
		case DType::Double:
			data = (double)_i;
			break;
		case DType::Float:
			data = (float)_i;
			break;
		case DType::cfloat:
			data = std::complex<float>(_i);
			break;
		case DType::cdouble:
			data = std::complex<double>(_i);
			break;
		case DType::uint8:
			data = uint8_t(_i);
			break;
		case DType::int8:
			data = int8_t{_i};
			break;
		case DType::int16:
			data = int16_t(_i);
			break;
		case DType::uint16:
			data = uint16_t(_i);
			break;
		case DType::int64:
			data = int64_t(_i);
			break;
		case DType::Bool:
			data = uint_bool_t(_i > 0 : true ? false);
			break;
		default:
			break;
	}
}

std::ostream& operator<< (std::ostream &out, DType const& data) {
	switch(data){
		case DType::Float:
		      out << "DTypeFloat32";
		      break;
		case DType::Double:
		      out << "DTypeFloat64";
		      break;
		case DType::Complex64:
		      out << "DTypeComplex64";
		      break;
		case DType::Complex128:
		      out << "DTypeComplex128";
		      break;
		case DType::Byte:
		      out << "DTypeByteSigned8";
		      break;
		case DType::Short:
			out << "DTypeShortSigned16";
			break;
		case DType::UnsignedShort:
			out << "DTypeUnsignedShort16";
		case DType::Long:
		      out << "DTypeUnsignedLong32";
		      break;
		case DType::Integer:
		      out << "DTypeInteger32";
		      break;
		case DType::LongLong:
		      out << "DTypeInteger64";
		      break;
		case DType::Bool:
		      out << "DTypeBool";
		      break;
		case DType::TensorObj:
		      out << "DTypeTensor";
		      break;
		default:
		      out << "UnknownType";
		      break;
	}
	return out;
}

d_type_list::d_type_list(var_t property)
	:data(property)
{set_type();}

d_type_reference::d_type_reference(var_t property)
	:data(property)
{set_type();}

void d_type_list::set_type(){
	const size_t index = data.index();
	if(index == 0)
		type = DTypeConst::Float;
	else if(index == 1)
		type = DTypeConst::Double;
	else if(index == 2)
		type = DTypeConst::Complex64;
	else if(index == 3)
		type = DTypeConst::Complex128;
	else if(index == 4)
		type = DTypeConst::uint8;
	else if(index == 5)
		type = DTypeConst::int8;
	else if(index == 6)
		type = DTypeConst::int16;
	else if(index == 7)
		type = DTypeConst::uint16;
	else if(index == 8)
		type = DTypeConst::Long;
	else if(index == 9)
		type = DTypeConst::LongLong;
	else if(index == 10)
		type = DTypeConst::Bool;
	else if(index == 11)
		type = DTypeConst::TensorObj
	else if(index == 12)
		type = DTypeConst::ConstFloat;
	else if(index == 13)
		type = DTypeConst::ConstDouble;
	else if(index == 14)
		type = DTypeConst::ConstComplex64;
	else if(index == 15)
		type = DTypeConst::ConstComplex128;
	else if(index == 16)
		type = DTypeConst::ConstUint8;
	else if(index == 17)
		type = DTypeConst::ConstInt8;
	else if(index == 18)
		type = DTypeConst::ConstInt16;
	else if(index == 19)
		type = DTypeConst::ConstUint16;
	else if(index == 20)
		type = DTypeConst::ConstLong;
	else if(index == 21)
		type = DTypeConst::ConstLongLong;
	else if(index == 22)
		type = DTypeConst::ConstBool;
	else if(index == 23)
		type = DTypeConst::ConstTensorObj;
	
	
}

void d_type_reference::set_type(){
	const size_t index = data.index();
	if(index == 0)
		type = DTypeConst::Float;
	else if(index == 1)
		type = DTypeConst::Double;
	else if(index == 2)
		type = DTypeConst::Complex64;
	else if(index == 3)
		type = DTypeConst::Complex128;
	else if(index == 4)
		type = DTypeConst::uint8;
	else if(index == 5)
		type = DTypeConst::int8;
	else if(index == 6)
		type = DTypeConst::int16;
	else if(index == 7)
		type = DTypeConst::uint16;
	else if(index == 8)
		type = DTypeConst::Integer;
	else if(index == 9)
		type = DTypeConst::Long;
	else if(index == 10)
		type = DTypeConst::LongLong;
	else if(index == 11)
		type = DTypeConst::Bool;
	else if(index == 12)
		type = DTypeConst::TensorObj
	else if(index == 13)
		type = DTypeConst::ConstFloat;
	else if(index == 14)
		type = DTypeConst::ConstDouble;
	else if(index == 15)
		type = DTypeConst::ConstComplex64;
	else if(index == 16)
		type = DTypeConst::ConstComplex128;
	else if(index == 17)
		type = DTypeConst::ConstUint8;
	else if(index == 18)
		type = DTypeConst::ConstInt8;
	else if(index == 19)
		type = DTypeConst::ConstInt16;
	else if(index == 20)
		type = DTypeConst::ConstUint16;
	else if(index == 21)
		type = DTypeConst::ConstInteger;
	else if(index == 22)
		type = DTypeConst::ConstLong;
	else if(index == 23)
		type = DTypeConst::ConstLongLong;
	else if(index == 24)
		type = DTypeConst::ConstBool;
	else if(index == 25)
		type = DTypeConst::ConstTensorObj;
}

const bool d_type_reference::is_const() const{
	
	switch(type){
		case DTypeConst::Float:
			return false;
		case DTypeConst::Double:
			return false;
		case DTypeConst::Complex64:
			return false;
		case DTypeConst::Complex128:
			return false;
		case DTypeConst::uint8:
			return false;
		case DTypeConst::int8:
			return false;
		case DTypeConst::int16:
			return false;
		case DTypeConst::uint16:
			return false;
		case DTypeConst::Integer:
			return false;
		case DTypeConst::Long:
			return false;
		case DTypeConst::LongLong:
			return false;
		case DTypeConst::Bool:
			return false;
		case DTypeConst::TensorObj
			return false;
		case DTypeConst::ConstFloat:
			return true;
		case DTypeConst::ConstDouble:
			return true;
		case DTypeConst::ConstComplex64:
			return true;
		case DTypeConst::ConstComplex128:
			return true;
		case DTypeConst::ConstUint8:
			return true;
		case DTypeConst::ConstInt8:
			return true;
		case DTypeConst::ConstInt16:
			return true;
		case DTypeConst::ConstUint16:
			return true;
		case DTypeConst::ConstInteger:
			return true;
		case DTypeConst::ConstLong:
			return true;
		case TypeConst::ConstLongLong:
			return true;
		case TypeConst::ConstBool:
			return true;
		case DTypeConst::ConstTensorObj:
			return true;
	}
}

template<typename T>
const T& d_type_reference::item() const{
	if(is_const())
		return std::get<std::reference_wrapper<const T>>(data).get();
	return std::get<std::reference_wrapper<T>>(data).get();
}

template<typename T>
T& d_type_reference::get() const{
	return std::get<std::reference_wrapper<T>>(data).get();
}

template const int64_t& d_type_reference::item<int64_t>() const;
template const uint32_t& d_type_reference::item<uint32_t>() const;
template const int32_t& d_type_reference::item<int32_t>() const;
template const uint16_t& d_type_reference::item<uint16_t>() const;
template const int16_t& d_type_reference::item<int16_t>() const;
template const uint8_t& d_type_reference::item<uint8_t>() const;
template const int8_t& d_type_reference::item<int8_t>() const;
template const double& d_type_reference::item<double>() const;
template const float& d_type_reference::item<float>() const;
template const std::complex<double>& d_type_reference::item<std::complex<double> >() const;
template const std::complex<float>& d_type_reference::item<std::complex<float> >() const;
template const Tensor& d_type_reference::item<Tensor>() const;
template const uint_bool_t& d_type_reference::item<uint_bool_t>() const;


template int64_t& d_type_reference::get<int64_t>();
template uint32_t& d_type_reference::get<uint32_t>();
template int32_t& d_type_reference::get<int32_t>();
template uint16_t& d_type_reference::get<uint16_t>();
template int16_t& d_type_reference::get<int16_t>();
template uint8_t& d_type_reference::get<uint8_t>();
template int8_t& d_type_reference::get<int8_t>();
template double& d_type_reference::get<double>();
template float& d_type_reference::get<float>();
template std::complex<double>& d_type_reference::get<std::complex<double> >();
template std::complex<float>& d_type_reference::get<std::complex<float> >();
template Tensor& d_type_reference::get<Tensor>();
template uint_bool_t& d_type_reference::get<uint_bool_t>();

template const int64_t& d_type_reference::get<const int64_t>();
template const uint32_t& d_type_reference::get<const uint32_t>();
template const int32_t& d_type_reference::get<const int32_t>();
template const uint16_t& d_type_reference::get<const uint16_t>();
template const int16_t& d_type_reference::get<const int16_t>();
template const uint8_t& d_type_reference::get<const uint8_t>();
template const int8_t& d_type_reference::get<const int8_t>();
template const double& d_type_reference::get<const double>();
template const float& d_type_reference::get<const float>();
template const std::complex<double>& d_type_reference::get<const std::complex<double> >();
template const std::complex<float>& d_type_reference::get<const std::complex<float> >();
template const Tensor& d_type_reference::get<const Tensor>();
template const uint_bool_t& d_type_reference::get<const uint_bool_t>();


d_type_reference& d_type_reference::operator=(const d_type_reference& inp){
	switch(type){
		case DTypeConst::Float:
			std::get<0>(data).get() = inp.item<float>();
			break;
		case DTypeConst::Double:
			std::get<1>(data).get() = inp.item<double>();
			break;
		case DTypeConst::Complex64:
			std::get<2>(data).get() = inp.item<std::complex<float>>();
			break;
		case DTypeConst::Complex128:
			std::get<3>(data).get() = inp.item<std::complex<double>>();
			break;
		case DTypeConst::uint8:
			std::get<4>(data).get() = inp.item<uint8_t>();
			break;
		case DTypeConst::int8:
			std::get<5>(data).get() = inp.item<int8_t>();
			break;
		case DTypeConst::int16:
			std::get<6>(data).get() = inp.item<int16_t>();
			break;
		case DTypeConst::uint16:
			std::get<7>(data).get() = inp.item<uint16_t>();
			break;
		case DTypeConst::Integer:
			std::get<8>(data).get() = inp.item<int32_t>();
			break;
		case DTypeConst::Long:
			std::get<9>(data).get() = inp.item<uint32_t>();
			break;
		case DTypeConst::LongLong:
			std::get<10>(data).get() = inp.item<int64_t>();
			break;
		case DTypeConst::Bool:
			std::get<11>(data).get() = inp.item<uint_bool_t>();
			break;
		case DTypeConst::TensorObj:
			std::get<12>(data).get() = inp.item<Tensor>();
			break;
		default:
			break;
	}
	return *this;
}

d_type_reference& d_type_reference::operator=(const int32_t& inp){
	switch(type){
		case DTypeConst::Float:
			std::get<0>(data).get() = inp;
			break;
		case DTypeConst::Double:
			std::get<1>(data).get() = inp;
			break;
		case DTypeConst::Complex64:
			std::get<2>(data).get() = inp;
			break;
		case DTypeConst::Complex128:
			std::get<3>(data).get() = inp;
			break;
		case DTypeConst::uint8:
			std::get<4>(data).get() = inp;
			break;
		case DTypeConst::int8:
			std::get<5>(data).get() = inp;
			break;
		case DTypeConst::int16:
			std::get<6>(data).get() = inp;
			break;
		case DTypeConst::uint16:
			std::get<7>(data).get() = inp;
			break;
		case DTypeConst::Integer:
			std::get<8>(data).get() = inp;
			break;
		case DTypeConst::Long:
			std::get<9>(data).get() = inp;
			break;
		case DTypeConst::LongLong:
			std::get<10>(data).get() = inp;
			break;
		case DTypeConst::Bool:
			std::get<11>(data).get() = inp;
			break;
		case DTypeConst::TensorObj:
			std::get<12>(data).get() = inp;
			break;
		default:
			break;
	}
	return *this;
}

d_type_reference& d_type_reference::operator=(const float& inp){
	switch(type){
		case DTypeConst::Float:
			std::get<0>(data).get() = inp;
			break;
		case DTypeConst::Double:
			std::get<1>(data).get() = inp;
			break;
		case DTypeConst::Complex64:
			std::get<2>(data).get() = inp;
			break;
		case DTypeConst::Complex128:
			std::get<3>(data).get() = inp;
			break;
		case DTypeConst::uint8:
			std::get<4>(data).get() = inp;
			break;
		case DTypeConst::int8:
			std::get<5>(data).get() = inp;
			break;
		case DTypeConst::int16:
			std::get<6>(data).get() = inp;
			break;
		case DTypeConst::uint16:
			std::get<7>(data).get() = inp;
			break;
		case DTypeConst::Integer:
			std::get<8>(data).get() = inp;
			break;
		case DTypeConst::Long:
			std::get<9>(data).get() = inp;
			break;
		case DTypeConst::LongLong:
			std::get<10>(data).get() = inp;
			break;
		case DTypeConst::Bool:
			std::get<11>(data).get() = inp;
			break;
		case DTypeConst::TensorObj:
			std::get<12>(data).get() = inp;
			break;
		default:
			break;
	}
	return *this;
}


d_type_reference& d_type_reference::operator=(const double& inp){
	switch(type){
		case DTypeConst::Float:
			std::get<0>(data).get() = inp;
			break;
		case DTypeConst::Double:
			std::get<1>(data).get() = inp;
			break;
		case DTypeConst::Complex64:
			std::get<2>(data).get() = inp;
			break;
		case DTypeConst::Complex128:
			std::get<3>(data).get() = inp;
			break;
		case DTypeConst::uint8:
			std::get<4>(data).get() = inp;
			break;
		case DTypeConst::int8:
			std::get<5>(data).get() = inp;
			break;
		case DTypeConst::int16:
			std::get<6>(data).get() = inp;
			break;
		case DTypeConst::uint16:
			std::get<7>(data).get() = inp;
			break;
		case DTypeConst::Integer:
			std::get<8>(data).get() = inp;
			break;
		case DTypeConst::Long:
			std::get<9>(data).get() = inp;
			break;
		case DTypeConst::LongLong:
			std::get<10>(data).get() = inp;
			break;
		case DTypeConst::Bool:
			std::get<11>(data).get() = inp;
			break;
		case DTypeConst::TensorObj:
			std::get<12>(data).get() = inp;
			break;
		default:
			break;
	}
	return *this;

}

d_type_reference& d_type_reference::operator=(const uint32_t& inp){
	switch(type){
		case DTypeConst::Float:
			std::get<0>(data).get() = inp;
			break;
		case DTypeConst::Double:
			std::get<1>(data).get() = inp;
			break;
		case DTypeConst::Complex64:
			std::get<2>(data).get() = inp;
			break;
		case DTypeConst::Complex128:
			std::get<3>(data).get() = inp;
			break;
		case DTypeConst::uint8:
			std::get<4>(data).get() = inp;
			break;
		case DTypeConst::int8:
			std::get<5>(data).get() = inp;
			break;
		case DTypeConst::int16:
			std::get<6>(data).get() = inp;
			break;
		case DTypeConst::uint16:
			std::get<7>(data).get() = inp;
			break;
		case DTypeConst::Integer:
			std::get<8>(data).get() = inp;
			break;
		case DTypeConst::Long:
			std::get<9>(data).get() = inp;
			break;
		case DTypeConst::LongLong:
			std::get<10>(data).get() = inp;
			break;
		case DTypeConst::Bool:
			std::get<11>(data).get() = inp;
			break;
		case DTypeConst::TensorObj:
			std::get<12>(data).get() = inp;
			break;
		default:
			break;
	}
	return *this;
}

d_type_reference& d_type_reference::operator=(const std::complex<double>& inp){
	if(type == DTypeConst::cdouble)
		std::get<3> = inp;
	return *this;
}

d_type_reference& d_type_reference::operator=(const std::complex<float>& inp){
	if(type == DTypeConst::cdouble)
		std::get<2> = inp;
	return *this;
}


d_type_reference& d_type_reference::operator=(const Tensor &t){
	if(type == DTypeConst::TensorObj)
		std::get<12>(data).get() = t;
	return *this;
}

d_type_reference& d_type_reference::operator=(Tensor&& t){
	if(type == Type::TensorObj)
		std::get<12>(data).get() = std::move(t);
	return *this;
}

inline d_type_list& d_type_list::operator++(){
	std::visit([](auto&& val){++val;},data);
	return *this;

}

inline d_type_list& d_type_list::operator+=(uint32_t i){
	std::visit([i](auto&& val){val += i;},data);
	return *this;
}


inline d_type_list d_type_list::operator++(int){
	d_type_list tmp = *this;
	++(*this);
	return tmp;
}

/* bool d_type_list::operator==(const d_type_list& b) const{ */
/* 	if(b.type != type){ */
/* 		return false; */
/* 	} */
/* 	switch(type){ */
/* 		case Type::Integer: */
/* 			return std::get<0>(b.data) == std::get<0>(data); */
/* 		case Type::Float: */
/* 			return std::get<1>(b.data) == std::get<1>(data); */
/* 		case Type::Double: */
/* 			return std::get<2>(b.data) == std::get<2>(data); */
/* 		case Type::Long: */
/* 			return std::get<3>(b.data) == std::get<3>(data); */
/* 		case Type::ConstInteger: */
/* 			return std::get<4>(b.data) == std::get<4>(data); */
/* 		case Type::ConstFloat: */
/* 			return std::get<5>(b.data) == std::get<5>(data); */
/* 		case Type::ConstDouble: */
/* 			return std::get<6>(b.data) == std::get<6>(data); */
/* 		case Type::ConstLong: */
/* 			return std::get<7>(b.data) == std::get<7>(data); */
/* 		default: */
/* 			return false; */
/* 	} */
/* } */


inline bool d_type_list::operator==(const d_type_list &b) const{
	return std::visit([](auto a1, auto a2) -> bool{
			if constexpr (std::is_same_v<std::decay_t<decltype(a1)>, std::decay_t<decltype(a2)>>){
				return a1 == a2;
			}
			return false;
			}, data, b.data);
}
inline bool d_type_list::operator!=(const d_type_list &b) const {
	return std::visit([](auto a1, auto a2) -> bool{
			if constexpr (std::is_same_v<std::decay_t<decltype(a1)>, std::decay_t<decltype(a2)>>){
				return a1 != a2;
			}
			return true;
			}, data, b.data);
	return !(b == *this);
}

/* bool d_type_list::operator!=(const d_type_list &b) const {return !(b == *this);} */

inline d_type_reference d_type_list::operator*() const{
	return d_type_reference(
			std::visit([](auto&& arg) -> d_type_reference::var_t {
				using T = std::decay_t<decltype(arg)>;
				if(std::is_const<T>::value)
					return std::cref(*(arg));
				return std::ref(*(arg));
				},data)
			);
}

template<typename T>
inline T d_type_reference::operator+(const T& inp) const{
	return item<T>() + inp;
}

template int64_t d_type_reference::operator+(const int64_t&) const;
template uint16_t d_type_reference::operator+(const uint16_t&) const;
template int16_t d_type_reference::operator+(const int16_t&) const
template uint8_t d_type_reference::operator+(const uint8_t&) const;
template int8_t d_type_reference::operator+(const int8_t&) const;
template uint32_t d_type_reference::operator+(const uint32_t&) const;
template int32_t d_type_reference::operator+(const int32_t&) const;
template float d_type_reference::operator+(const float&) const;
template double d_type_reference::operator+(const double&) const;
template std::complex<float> d_type_reference::operator+(const std::complex<float>&) const;
template std::complex<double> d_type_reference::operator+(const std::complex<double>&) const;
template Tensor d_type_reference::operator+(const Tensor&) const;

template<typename T>
d_type_reference& d_type_reference::operator+=(const T &inp){
	std::get<std::reference_wrapper<T>>(data).get() += inp;
	return *this;
}

template d_type_reference& d_type_reference::operator+=(const uint32_t&);
template d_type_reference& d_type_reference::operator+=(const int32_t&);
template d_type_reference& d_type_reference::operator+=(const uint16_t&);
template d_type_reference& d_type_reference::operator+=(const int16_t&);
template d_type_reference& d_type_reference::operator+=(const uint8_t&);
template d_type_reference& d_type_reference::operator+=(const int8_t&);
template d_type_reference& d_type_reference::operator+=(const float&);
template d_type_reference& d_type_reference::operator+=(const double&);
template d_type_reference& d_type_reference::operator+=(const std::complex<float>&);
template d_type_reference& d_type_reference::operator+=(const std::complex<double>&);
template d_type_reference& d_type_reference::operator+=(const Tensor&);

d_type_list d_type_list::operator+(uint32_t i) const{
	return d_type_list(std::visit([i](auto arg) -> var_t {return arg + i;}, data));
}

d_type_reference d_type_list::operator[](uint32_t i) const{
	return d_type_reference(
			std::visit([i](auto arg) -> d_type_reference::var_t {
				using T = std::decay_t<decltype(arg)>;
				if(std::is_const<T>::value)
					return std::cref(*(arg + i));
				return std::ref(*(arg + i));
				},data)
			);
}

void* d_type_list::operator->(){
	switch(type){
		case DTypeConst::Float:
			return std::get<0>(data);
		case DTypeConst::Double:
			return std::get<1>(data);
		case DTypeConst::Complex64:
			return std::get<2>(data);
		case DTypeConst::Complex128:
			return std::get<3>(data);
		case DTypeConst::uint8:
			return std::get<4>(data);
		case DTypeConst::int8:
			return std::get<5>(data);
		case DTypeConst::int16:
			return std::get<6>(data);
		case DTypeConst::uint16:
			return std::get<7>(data);
		case DTypeConst::Integer:
			return std::get<8>(data);
		case DTypeConst::Long:
			return std::get<9>(data);
		case DTypeConst::LongLong:
			return std::get<10>(data);
		case DTypeConst::Bool:
			return std::get<11>(data);
		case DTypeConst::TensorObj:
			return std::get<12>(data);
		default:
			return nullptr;
	}
}

const void* d_type_list::operator->() const{
	switch(type){
		case DTypeConst::Float:
			return std::get<0>(data);
		case DTypeConst::Double:
			return std::get<1>(data);
		case DTypeConst::Complex64:
			return std::get<2>(data);
		case DTypeConst::Complex128:
			return std::get<3>(data);
		case DTypeConst::uint8:
			return std::get<4>(data);
		case DTypeConst::int8:
			return std::get<5>(data);
		case DTypeConst::int16:
			return std::get<6>(data);
		case DTypeConst::uint16:
			return std::get<7>(data);
		case DTypeConst::Integer:
			return std::get<8>(data);
		case DTypeConst::Long:
			return std::get<9>(data);
		case DTypeConst::LongLong:
			return std::get<10>(data);
		case DTypeConst::Bool:
			return std::get<11>(data);
		case DTypeConst::TensorObj:
			return std::get<12>(data);
		default:
			return nullptr;
	}
}


const void* d_type_list::g_ptr() const{
	switch(type){
		case DTypeConst::Float:
			return std::get<0>(data);
		case DTypeConst::Double:
			return std::get<1>(data);
		case DTypeConst::Complex64:
			return std::get<2>(data);
		case DTypeConst::Complex128:
			return std::get<3>(data);
		case DTypeConst::uint8:
			return std::get<4>(data);
		case DTypeConst::int8:
			return std::get<5>(data);
		case DTypeConst::int16:
			return std::get<6>(data);
		case DTypeConst::uint16:
			return std::get<7>(data);
		case DTypeConst::Integer:
			return std::get<8>(data);
		case DTypeConst::Long:
			return std::get<9>(data);
		case DTypeConst::LongLong:
			return std::get<10>(data);
		case DTypeConst::Bool:
			return std::get<11>(data);
		case DTypeConst::TensorObj:
			return std::get<12>(data);
		default:
			return nullptr;
	}
}

void* d_type_list::g_ptr(){
	switch(type){
		case DTypeConst::Float:
			return std::get<0>(data);
		case DTypeConst::Double:
			return std::get<1>(data);
		case DTypeConst::Complex64:
			return std::get<2>(data);
		case DTypeConst::Complex128:
			return std::get<3>(data);
		case DTypeConst::uint8:
			return std::get<4>(data);
		case DTypeConst::int8:
			return std::get<5>(data);
		case DTypeConst::int16:
			return std::get<6>(data);
		case DTypeConst::uint16:
			return std::get<7>(data);
		case DTypeConst::Integer:
			return std::get<8>(data);
		case DTypeConst::Long:
			return std::get<9>(data);
		case DTypeConst::LongLong:
			return std::get<10>(data);
		case DTypeConst::Bool:
			return std::get<11>(data);
		case DTypeConst::TensorObj:
			return std::get<12>(data);
		default:
			return nullptr;
	}
}


DType d_type_list::d_type() const{
	switch(type){
		case DTypeConst::Float:
			return DType::Float;
		case DTypeConst::Double:
			return DType::Double;
		case DTypeConst::Complex64:
			return DType::cfloat;
		case DTypeConst::Complex128:
			return DType::cdouble;
		case DTypeConst::uint8:
			return DType::uint8;
		case DTypeConst::int8:
			return DType::int8;
		case DTypeConst::int16:
			return DType::int16;
		case DTypeConst::uint16:
			return DType::uint16;
		case DTypeConst::Integer:
			return DType::Integer;
		case DTypeConst::Long:
			return DType::Long;
		case DTypeConst::LongLong:
			return DType::LongLong;
		case DTypeConst::Bool:
			return DType::Bool;
		case DTypeConst::TensorObj:
			return DType::TensorObj;
		case DTypeConst::ConstFloat:
			return DType::Float;
		case DTypeConst::ConstDouble:
			return DType::Double;
		case DTypeConst::ConstComplex64:
			return DType::cfloat;
		case DTypeConst::ConstComplex128:
			return DType::cdouble;
		case DTypeConst::ConstUint8:
			return DType::uint8;
		case DTypeConst::ConstInt8:
			return DType::int8;
		case DTypeConst::ConstInt16:
			return DType::int16;
		case DTypeConst::ConstUint16:
			return DType::uint16;
		case DTypeConst::ConstInteger:
			return DType::Integer;
		case DTypeConst::ConstLong:
			return DType::Long;
		case DTypeConst::ConstLongLong:
			return DType::LongLong;
		case DTypeConst::ConstBool:
			return DType::Bool;
		case DTypeConst::ConstTensorObj:
			return DType::TensorObj;

		default:
			return DType::Integer;
	}
}

std::ostream& operator << (std::ostream& out, const d_type_reference &ref){
	switch(type){
		case DTypeConst::Float:
			out << std::get<0>(data).get();
			break;
		case DTypeConst::Double:
			out << std::get<1>(data).get();
			break;
		case DTypeConst::Complex64:
			out << std::get<2>(data).get();
			break;
		case DTypeConst::Complex128:
			out << std::get<3>(data).get();
			break;
		case DTypeConst::uint8:
			out << std::get<4>(data).get();
			break;
		case DTypeConst::int8:
			out << std::get<5>(data).get();
			break;
		case DTypeConst::int16:
			out << std::get<6>(data).get();
			break;
		case DTypeConst::uint16:
			out << std::get<7>(data).get();
			break;
		case DTypeConst::Integer:
			out << std::get<8>(data).get();
			break;
		case DTypeConst::Long:
			out << std::get<9>(data).get();
			break;
		case DTypeConst::LongLong:
			out << std::get<10>(data).get();
			break;
		case DTypeConst::Bool:
			out << std::get<11>(data).get();
			break;
		case DTypeConst::TensorObj:
			out << std::get<12>(data).get();
			break;
		default:
			break;
	}
	return out;
}


}
