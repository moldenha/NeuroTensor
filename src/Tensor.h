#ifndef TENSOR_H_
#define TENSOR_H_

namespace nt{
class Tensor;
}




#include <memory.h>
#include <memory>
#include <sys/_types/_int8_t.h>
#include <vector>
#include <iostream>
#include "refs/SizeRef.h"
#include <string.h>
#include <initializer_list>
#include <variant>
#include "dtype/DType.h"
#include "dtype/DType_enum.h"
#include "dtype/DType_list.h"
#include "dtype/ArrayVoid.h"
#include "dtype/Scalar.h"
#include "utils/utils.h"
#include "CustomOperator.h"
#include <type_traits>
/* #include "Itterator.h" */

namespace nt{
namespace detect_num_tensor {
	template<class T, class...Ts>
	constexpr bool is_charlike(tag_t<T>, Ts&&...){ return false; }
	constexpr bool is_charlike( tag_t<double> ){ return true; }
	constexpr bool is_charlike( tag_t<float> ){ return true; }
	constexpr bool is_charlike( tag_t<int32_t> ){ return true; }
	constexpr bool is_charlike( tag_t<uint32_t> ){ return true;}
	constexpr bool is_charlike( tag_t<int64_t> ){return true;}
	constexpr bool is_charlike( tag_t<int16_t> ){ return true; }
	constexpr bool is_charlike( tag_t<uint16_t> ){ return true; }
	constexpr bool is_charlike( tag_t<int8_t> ){ return true; }
	constexpr bool is_charlike( tag_t<uint8_t> ){ return true; }
	constexpr bool is_charlike( tag_t<Tensor> ){ return true; }
	template<class T>
	constexpr bool detect=is_charlike(tag<T>);
};

class Tensor{
	ArrayVoid _vals;
	SizeRef _size;
	uint32_t _total_size;
	const bool sub_tensor;
	/* Tensor(float*, const std::vector<long long>&); */
	/* Tensor(const float*, const float*, const std::vector<long long>&); */
	Tensor(ArrayVoid, SizeRef);
	/* Tensor(ArrayVoid, std::shared_ptr<SizeRef>); */
	Tensor(uint32_t, const ArrayVoid&, SizeRef&&);
	public:
		DType dtype;
		Tensor(DType _dt = DType::Float);
		Tensor(SizeRef, DType _dt = DType::Float);
		Tensor(std::string_view);
		Tensor(const Tensor&);
		Tensor(Tensor&&);
		
		template<std::size_t N>
		static Tensor FromInitializer(typename utils::NestedInitializerLists_type<Scalar, N>::type v, DType dt=DType::Float);
		Tensor& operator++();
		Tensor& operator=(Scalar);
		Tensor& operator=(const Tensor&);
		Tensor& operator=(Tensor&&);
		Tensor& set_(const Tensor&);
		
		Tensor operator>=(const Tensor&) const;
		Tensor operator<=(const Tensor&) const;
		Tensor operator==(const Tensor&) const;
		Tensor operator>=(Scalar) const;
		Tensor operator<=(Scalar) const;
		Tensor operator==(Scalar) const;

		Tensor operator>(const Tensor&) const;
		Tensor operator<(const Tensor&) const;
		Tensor operator>(Scalar) const;
		Tensor operator<(Scalar) const;


		Tensor& operator+=(Scalar);
		Tensor& operator+=(const Tensor&);
		Tensor operator+(const Scalar) const;
		Tensor operator+(const Tensor&) const;

		Tensor operator*(const Scalar) const;
		Tensor operator*(const Tensor&) const;
		Tensor& operator*=(Scalar);
		Tensor& operator*=(const Tensor&);

		Tensor& operator/=(Scalar);
		Tensor& operator/=(const Tensor&);
		Tensor operator/(Scalar) const;
		Tensor operator/(const Tensor&) const;


		Tensor operator-(Scalar) const;
		Tensor operator-(const Tensor&) const;
		Tensor& operator-=(Scalar);
		Tensor& operator-=(const Tensor&);
		
		Tensor operator-() const;
		/* inline Tensor operator@(const Tensor& b) const {return functional::matmult(*this, b);} */

		Tensor& _fill(Scalar);
		Tensor& _fill(const Tensor& val);
		Tensor& _add(Scalar val);
		Tensor& _add(const Tensor& val);
		Tensor& _subtract(Scalar val);
		Tensor& _subtract(const Tensor& val);
		Tensor& _multiply(Scalar val);
		Tensor& _multiply(const Tensor& val);
		Tensor& _divide(Scalar val);
		Tensor& _divide(const Tensor& val);
		Scalar toScalar() const;
		
		Tensor contiguous() const;
		template<typename T>
		const T& item() const;
		template<typename T>
		T& item();
		const bool is_contiguous() const;
		const uint32_t contig_count() const;
		const size_t dims() const;
		const SizeRef& shape() const;
		Tensor operator[](int32_t);
		const Tensor operator[](int32_t) const;
		Tensor operator[](const my_range&);
		const Tensor operator[](const my_range&) const;
		Tensor operator[](const Tensor&);
		/* template<std::is_integral ...Ts> */
		/* Tensor operator[](Ts... ts); */
		/* template<std::is_integral ...Ts> */
		/* const Tensor operator[](Ts... ts) const; */
		Tensor operator[](std::vector<my_range>);
		void print() const;
		void* data_ptr();
		const void* data_ptr() const;
		Tensor unfold(int32_t, uint32_t, uint32_t) const;
		/* void** stack_data_ptr(); */
		inline const uint32_t numel() const {return _total_size;}
		Tensor view(SizeRef) const;
		Tensor unsqueeze() const;
		Tensor squeeze() const;
		Tensor permute(std::vector<uint32_t>) const;
		inline std::vector<typename SizeRef::ArrayRefInt::value_type> strides() const {return shape().strides();}
		Tensor transpose(int8_t, int8_t) const;
		/* Tensor unfold(int8_t, uint32_t, uint8_t) const; */
		Tensor flatten(int8_t, int8_t) const;
		dtype_list val_begin();
		dtype_list val_end();
		const_dtype_list val_cbegin() const;
		const_dtype_list val_cend() const;
		friend std::ostream& operator << (std::ostream &out, const Tensor&);
		Tensor div(uint32_t);
		Tensor split_axis(int8_t);
		const Tensor split_axis(int8_t) const;
		const Tensor split_axis_1_() const;
		ArrayVoid& arr_void();
		const ArrayVoid& arr_void() const;
		std::string_view sv() const;
		Tensor to_dtype(DType) const;
		Tensor Int() const;
		Tensor Long() const;
		Tensor& RowColSwap();
		const Tensor& RowColSwap() const;
		Tensor& RowColSwap_contiguous();
		const Tensor& RowColSwap_contiguous() const;
		Tensor real() const;
		Tensor imag() const;
		Tensor sum() const;
		Tensor max() const;
		Tensor sum(int32_t) const;
		Tensor max(int32_t) const;
		Tensor exp() const;
		Tensor& exp_();
		Tensor& inverse_();
		Tensor inverse() const;
		Tensor clip(Scalar, Scalar) const;
		Tensor& clip_(Scalar, Scalar);
		//this is going to pad based on what is given
		//for example if the tuple is : {1,1}, then pad the last dimension with 1 up and 1 down
		Tensor pad(std::vector<uint32_t>, const char* mode="constant", double value=0) const;
		Tensor flip() const;
		Tensor flip(int32_t dim) const; //contiguous copies of the original tensor
		Tensor flip_();
		Tensor flip_(int32_t dim); // this returns a version where the stride was changed, makes it easier for things like a convolution backprop without taking more memory
		Tensor dilate_(uint32_t dil) const;
		//dilate by basically adding that many zeros between each row and collumn
		//the above has a memory thing where it takes part of its memory from the original tensor
		//there is also a contiguous version below
		Tensor dilate(uint32_t dil) const;
		Tensor dilate_mem_(uint32_t dil) const;

};

/* template<std::is_integral ...Ts> */
/* Tensor Tensor::operator[](Ts... ts){ */
	
/* } */
/* template<std::is_integral ...Ts> */
/* const Tensor operator[](Ts... ts) const; */


Tensor operator+(Scalar s, const Tensor& t);
Tensor operator-(Scalar s, const Tensor& t);
Tensor operator*(Scalar s, const Tensor& t);
Tensor operator/(Scalar s, const Tensor& t);


}

#include "functional/functional.h"
#endif
