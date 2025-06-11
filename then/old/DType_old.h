#ifndef DTYPE_H
#define DTYPE_H
#include "DType_enum.h"
namespace nt{
struct uint_bool_t;
struct d_type;
class d_type_reference;
class d_type_list;
}



#include <functional>
#include <iostream>
#include <typeinfo>
#include <variant>
#include "ArrayVoid.h"
#include "../Tensor.h"
#include <complex>


namespace nt{
std::ostream& operator<< (std::ostream &out, DType const& data);

struct uint_bool_t{
	unsigned value : 1;
	uint_bool_t(const bool&);
	uint_bool_t(const uint_bool_t&);
	uint_bool_t(uint_bool_t&&);
	uint_bool_t& operator=(const bool&);
	uint_bool_t& operator=(const uint_bool_t&);
};
std::ostream& operator<<(std::ostream &out, const uint_bool_t &data);

struct d_type{
	DType type;
	using var_t = std::variant<float,
					double,
					std::complex<float>,
					std::complex<double>,
					uint8_t,
					int8_t,
					int16_t,
					uint16_t,
					int32_t,
					uint32_t,
					int64_t,
					uint_bool_t,
					std::reference_wrapper<const Tensor> >;
	var_t data;
	d_type(const std::complex<float> _cf);
	d_type(const std::complex<double> _cd);
	d_type(const int32_t _i);
	d_type(const float _f);
	d_type(const double _d);
	d_type(const uint32_t _l);
	d_type(const Tensor& _t);
	d_type(const int32_t _i, DType _t);
	d_type(const float _i, DType _t);
	d_type(const double _i, DType _t);
	d_type(const uint32_t _i, DType _t);
	d_type(const d_type& _t);
	d_type(d_type&& _t);
	d_type& operator=(const d_type& _t);
	d_type& operator=(d_type&& _t);
	template<typename T>
	T cast_num() const;
	const std::type_info& m_type() const noexcept;
	private:
		void set(const int32_t _i, DType _t);
		void set(const float _i, DType _t);
		void set(const double _i, DType _t);
		void set(const uint32_t _i, DType _t);


};


class d_type_reference{
	
	void set_type();
	/* DType type; */
	DTypeConst type;
	public:
	using var_t = std::variant<std::reference_wrapper<float>,
					std::reference_wrapper<double>,
					std::reference_wrapper<std::complex<float> >,
					std::reference_wrapper<std::complex<double> >,
					std::reference_wrapper<uint8_t>,
					std::reference_wrapper<int8_t>,
					std::reference_wrapper<int16_t>,
					std::reference_wrapper<uint16_t>,
					std::reference_wrapper<int32_t>,
					std::reference_wrapper<uint32_t>,
					std::reference_wrapper<int64_t>,
					std::reference_wrapper<uint_bool_t>,
					std::reference_wrapper<Tensor>,
					std::reference_wrapper<const float>,
					std::reference_wrapper<const double>,
					std::reference_wrapper<const std::complex<float> >,
					std::reference_wrapper<const std::complex<double> >,
					std::reference_wrapper<const uint8_t>,
					std::reference_wrapper<const int8_t>,
					std::reference_wrapper<const int16_t>,
					std::reference_wrapper<const uint16_t>,
					std::reference_wrapper<const int32_t>,
					std::reference_wrapper<const uint32_t>,
					std::reference_wrapper<const int64_t>,
					std::reference_wrapper<const uint_bool_t>,
					std::reference_wrapper<const Tensor> >;		


		d_type_reference(var_t);
		const bool is_const() const;
		template<typename T>
		const T& item() const;
		template<typename T>
		T& get() const;
		d_type_reference& operator=(const d_type_reference& inp);
		d_type_reference& operator=(const int32_t&);
		d_type_reference& operator=(const float&);
		d_type_reference& operator=(const double&);
		d_type_reference& operator=(const uint32_t&);
		d_type_reference& operator=(const std::complex<double>&);
		d_type_reference& operator=(const std::complex<float>&);
		d_type_reference& operator=(const Tensor&);
		d_type_reference& operator=(Tensor&&);
		template<typename T>
		T operator+(const T&) const;
		template<typename T>
		d_type_reference& operator+=(const T&);
		friend std::ostream& operator<< (std::ostream& out, const d_type_reference&);

	private:
		var_t data;
	
};

class d_type_list{
	void set_type();
	public:
		DTypeConst type;
		using var_t = std::variant<float*,
					double*,
					std::complex<float>*,
					std::complex<double>*,
					uint8_t*,
					int8_t*,
					int16_t*,
					uint16_t*,
					int32_t*,
					uint32_t*,
					int64_t*,
					uint_bool_t*,
					Tensor*,
					const float*,
					const double*,
					const std::complex<float>*,
					const std::complex<double>*,
					const uint8_t*,
					const int8_t*,
					const int16_t*,
					const uint16_t*,
					const int32_t*,
					const uint32_t*,
					const int64_t*,
					const uint_bool_t*,
					const Tensor* >;	
		var_t data;
		d_type_list(var_t);
		d_type_list& operator++();
		d_type_list operator++(int);
		d_type_list& operator+=(uint32_t);
		bool operator==(const d_type_list&) const;
		bool operator!=(const d_type_list&) const;
		d_type_reference operator*() const;
		d_type_list operator+(uint32_t i) const;
		d_type_reference operator[](uint32_t i) const;
		void* operator->();
		const void* operator->() const;
		void* g_ptr();
		const void* g_ptr() const;
		DType d_type() const;

};

}
#endif
