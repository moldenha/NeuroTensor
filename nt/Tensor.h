#ifndef NT_TENSOR_H__
#define NT_TENSOR_H__
#include "types/TensorDeclare.h"
#include "dtype/compatible/DType_compatible.h"

#include <memory.h>
#include <memory>
#include <cmath>

#include <vector>
#include <iostream>
#include "refs/SizeRef.h"
#include <string.h>
#include <initializer_list>
#include <variant>
#include "dtype/DType.h"
#include "dtype/DType_enum.h"
#include "dtype/ArrayVoid.h"
#include "dtype/Scalar.h"
#include "utils/utils.h"
#include "utils/optional_list.h"
#include "utils/CommaOperator.h"
#include "utils/api_macro.h"
#include "utils/collect_ri.hpp"

/* #include "CustomOperator.h" */
#include <type_traits>
/* #include "Itterator.h" */

namespace nt{
/* namespace detect_num_tensor { */
/* 	template<class T, class...Ts> */
/* 	constexpr bool is_charlike(tag_t<T>, Ts&&...){ return false; } */
/* 	constexpr bool is_charlike( tag_t<double> ){ return true; } */
/* 	constexpr bool is_charlike( tag_t<float> ){ return true; } */
/* 	constexpr bool is_charlike( tag_t<int32_t> ){ return true; } */
/* 	constexpr bool is_charlike( tag_t<uint32_t> ){ return true;} */
/* 	constexpr bool is_charlike( tag_t<int64_t> ){return true;} */
/* 	constexpr bool is_charlike( tag_t<int16_t> ){ return true; } */
/* 	constexpr bool is_charlike( tag_t<uint16_t> ){ return true; } */
/* 	constexpr bool is_charlike( tag_t<int8_t> ){ return true; } */
/* 	constexpr bool is_charlike( tag_t<uint8_t> ){ return true; } */
/* 	constexpr bool is_charlike( tag_t<Tensor> ){ return true; } */
/* 	template<class T> */
/* 	constexpr bool detect=is_charlike(tag<T>); */
/* }; */
class TensorIterator;

namespace result_types{
	template<typename T, typename G>
	struct NEUROTENSOR_API max{ 
		T values; 
		G indices;
		explicit max(const T& a, const G& b) :values(a), indices(b) {}
		explicit max(T&& a, G&& b) :values(std::move(a)), indices(std::move(b)) {}
		max(const max<T,G>& m) : values(m.values), indices(m.indices) {}
		max(max<T,G>&& m):values(std::move(m.values)), indices(std::move(m.indices)) {}
	};
	template<typename T, typename G>
	inline std::ostream& operator<<(std::ostream& out, const max<T,G>& m){
		return out << "values: "<<m.values << std::endl << "indices: "<<m.indices;
	}
}


class NEUROTENSOR_API Tensor final{
	friend class ArrayVoid;
	friend class Bucket;
    friend class SparseTensor;
	public:
		using size_value_t = typename SizeRef::ArrayRefInt::value_type;
	private:
		/* friend class layers::AttributeAccess; */
		/* friend ::nt::functional::as_strided(Tensor, const SizeRef, const SizeRef, const int64_t); */
		ArrayVoid _vals;
		SizeRef _size;
		size_value_t _total_size;
		intrusive_tracked_list<size_value_t> stored_strides;
        bool _is_mutable;
		/* Tensor(float*, const std::vector<long long>&); */
		/* Tensor(const float*, const float*, const std::vector<long long>&); */
		/* Tensor(ArrayVoid, std::shared_ptr<SizeRef>); */
		Tensor(size_value_t, const ArrayVoid&, SizeRef&&);
		Tensor(ArrayVoid, SizeRef, intrusive_tracked_list<size_value_t>);

		inline void collectIntegers(std::vector<int64_t>& a) const {;}
		template<typename... Args>
		inline void collectIntegers(std::vector<int64_t>& a, int64_t i, Args... args) const {
			a.push_back(i);
			collectIntegers(a, args...);
		}

		inline void nullify(){
			stored_strides.nullify();
			_size.nullify();
			_vals.nullify();
			_total_size = 0;

		}
		Tensor view_Tensor_vector(std::vector<size_value_t> v) const;
        inline bool is_sub_tensor() const {return _vals.get_bucket().is_sub_memory() && !this->is_null();}
	public:
        Tensor();
		Tensor(DType _dt);
		Tensor(SizeRef, DType _dt = DType::Float);
		Tensor(ArrayVoid, SizeRef);
		Tensor(ArrayVoid, SizeRef, const std::vector<size_value_t>&);
		Tensor(std::string_view);
		Tensor(const Tensor&);
		Tensor(Tensor&&);
        Tensor(std::initializer_list<Tensor>);
		explicit Tensor(std::nullptr_t);
		explicit Tensor(Scalar);
		void swap(Tensor&);
		inline const DeviceType& device() const noexcept {return _vals.device_type();}

		template<std::size_t N>
		static Tensor FromInitializer(typename utils::NestedInitializerLists_type<Scalar, N>::type v, DType dt=DType::Float);
        inline const DType& dtype() const noexcept {return _vals.dtype();}
		Tensor& operator++();
		Tensor& operator=(Scalar);
		Tensor& operator=(const Tensor&);
		Tensor& operator=(Tensor&&);
		Tensor& set_(const Tensor&);
		
		Tensor operator>=(const Tensor&) const;
		Tensor operator<=(const Tensor&) const;
		Tensor operator==(const Tensor&) const;
		Tensor operator!=(const Tensor&) const;
		Tensor operator>=(Scalar) const;
		Tensor operator<=(Scalar) const;
		Tensor operator==(Scalar) const;
		Tensor operator!=(Scalar) const;
		Tensor operator&&(Tensor) const;
		Tensor operator||(Tensor) const;

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
	    
        Tensor operator%(Scalar) const;

		Tensor operator-() const;
	

		Tensor& fill_(Scalar);
		Tensor& fill_(const Tensor& val);
        Tensor& fill_diagonal_(Scalar);
        Tensor diagonal() const;
		Tensor& add_(Scalar val);
		Tensor& add_(const Tensor& val);
		Tensor& subtract_(Scalar val);
		Tensor& subtract_(const Tensor& val);
		Tensor& multiply_(Scalar val);
		Tensor& multiply_(const Tensor& val);
		Tensor& divide_(Scalar val);
		Tensor& divide_(const Tensor& val);
		Scalar toScalar() const;
        CommaOperator operator<<(Scalar s);   

		Tensor clone() const;
        Tensor conditional_mutate_clone() const; //only clone if mutatable, otherwise, return *this
		Tensor contiguous() const;
		template<typename T>
		const T& item() const;
		template<typename T>
		T& item();
        template<typename T>
        T& item(const std::vector<size_value_t>&);
        template<typename T>
        const T& item(const std::vector<size_value_t>&) const;
        template<typename T, typename... Args>
        inline T& item(int64_t i, Args... args){
            std::vector<size_value_t> s;
			collectIntegers(s, i, args...);
			return this->item<T>(s);
        }
        template<typename T, typename... Args>
        inline const T& item(int64_t i, Args... args) const {
            std::vector<size_value_t> s;
			collectIntegers(s, i, args...);
			return this->item<T>(s);
        }

		inline bool is_contiguous() const {return _vals.is_contiguous();}
		inline bool is_empty() const {return _vals.is_empty() && _total_size == 0;}
		inline bool is_null() const {return _vals.is_null();}
		inline int64_t contig_count() const {return _vals.use_count();}
	    int64_t dims() const;
		const SizeRef& shape() const;
		Tensor operator[](size_value_t);
		const Tensor operator[](size_value_t) const;
		Tensor operator[](const range_&);
		const Tensor operator[](const range_&) const;
		Tensor operator[](const Tensor&) const;
		const Tensor operator[](std::vector<range_>) const;
		Tensor operator[](std::vector<range_>);
		const Tensor operator[](std::vector<size_value_t>) const;
		Tensor operator[](std::vector<size_value_t>);
        Tensor index_except(int64_t dim, int64_t excluding_index) const; 

        template<typename Arg, typename... Args>
        inline Tensor operator()(Arg&& i, Args&&... args) const {
            auto vec = utils::collect_integers_or_ranges<size_value_t>(std::forward<Arg>(i), std::forward<Args>(args)...);
            return (*this)[vec];
        }
		void print() const;
		void* data_ptr(); // _vals.strides()[0] <- important distinction btwn what vals and this will return this tensors beggining ptr, versus vals will return the shared pointers first
		const void* data_ptr() const;
		void* data_ptr_end(); // _vals.strides()[-1] <- important distinction btwn what vals and this will return
		const void* data_ptr_end() const;
		inline bool occupy_same_memory(const Tensor& t) const noexcept {
			return t._vals.occupy_same_memory(t._vals);
		}


		/* void** stack_data_ptr(); */
		inline const size_value_t& numel() const {return _total_size;}
		Tensor view(SizeRef) const;
		template<typename... Args>
		inline Tensor view(int64_t i, Args&&... args) const {
			std::vector<int64_t> s;
			collectIntegers(s, i, args...);
			size_value_t n = 1;
			bool is_neg = false;
			size_value_t neg_index = 0;
			for(size_value_t i = 0; i < s.size(); ++i){
				if(s[i] < 0){
					utils::throw_exception(is_neg == false, "already had negative value in shape at index $" , neg_index);
					is_neg = true;
					neg_index = i;
					continue;
				}
				n *= s[i];
			}
			if(is_neg){
				utils::throw_exception(_total_size % n == 0, "shape must be divisible by what has been given, $ is not divisible by $", _total_size, n);
				s[neg_index] = _total_size / n;
			}
			std::vector<size_value_t> n_shape(s.size());
            size_value_t n_shape_size = static_cast<size_value_t>(s.size());
			for(size_value_t i = 0; i < n_shape_size; ++i)
				n_shape[i] = s[i];
			return view(SizeRef(std::move(n_shape)));
		}
		//this is a function meant to change the view of all the tensors inside of a DType::TensorObj tensor
		template<typename... Args>
		inline Tensor view_Tensors(int i, Args&&... args) const {
			std::vector<size_value_t> s;
			collectIntegers(s, i, args...);
			return view_Tensor_vector(std::move(s));
			
		}
		Tensor view_Tensors(SizeRef) const;
        Tensor transpose_Tensors(size_value_t, size_value_t) const;

		Tensor unsqueeze(size_value_t dim = 0) const;
		Tensor unsqueeze_as(const Tensor&) const;
		Tensor unsqueeze_as(const SizeRef&) const;
		Tensor squeeze(utils::optional_list list = nullptr) const;
		Tensor permute(std::vector<size_value_t>) const;
		inline std::vector<size_value_t> strides() const {return shape().strides();}
		inline std::vector<size_value_t> getChangedStrides() const {
			if(stored_strides){
				return std::vector<size_value_t>(stored_strides.get(), stored_strides.get() + (dims() + 1));
			}
			return strides();
		}
		inline Tensor& set_stored_strides(const std::vector<size_value_t>& nS){
			stored_strides = intrusive_tracked_list<size_value_t>(nS.size());
			for(size_t i = 0; i < nS.size(); ++i){stored_strides[i] = nS[i];}
			return *this;
		}
		Tensor transpose(size_value_t, size_value_t) const;
        //this is a function where you can for example swap rows
        //if you wanted to swap rows 1 & 2 this would be equivalent to t.swap_axis(-2, 1, 2)
        Tensor swap_axis(size_value_t dim, size_value_t a, size_value_t b) const;
        //swap axis on self
        Tensor& swap_axis_(size_value_t dim, size_value_t a, size_value_t b);
		Tensor unfold(size_value_t, size_value_t, size_value_t) const;
		/* Tensor fold(size_value_t dim, size_value_t size, size_value_t step, const SizeRef& output_shape) const; */
		Tensor flatten(size_value_t, size_value_t) const;
		Tensor unflatten(size_value_t, size_value_t) const;
		friend std::ostream& operator << (std::ostream &out, const Tensor&);
		Tensor div(size_value_t) const;
		/* Tensor split_axis(size_value_t); */ //replaced by just a const version
						       //felt it was okay to be const becaust it is just a stride/view change
		/* Tensor split_axis_experimental(size_value_t); */
		Tensor split_axis(std::vector<range_>) const;
		Tensor split_axis(size_value_t) const;
		Tensor split_axis_1() const;
		/* const Tensor split_axis(std::vector<range_>) const; */
		ArrayVoid& arr_void();
		const ArrayVoid& arr_void() const;
		std::string_view sv() const;
		Tensor to_dtype(DType) const;
		Tensor to_device(DeviceType) const;
		inline Tensor to(DType dt) const {return to_dtype(dt);}
		inline Tensor to(DeviceType dt) const {return to_device(dt);}
		Tensor Int() const;
		Tensor Long() const;
		Tensor& RowColSwap(); //this contiguously in memory changes the rows and collumns
                              // output will be contiguous
                              // look at functional::row_col_swap_ for implementation
		Tensor& RowColSwap_Tensors(); //this is for a tensor that has the dtype tensor obj
					      //it swaps all the rows and collumns of the sub tensors
		Tensor real() const;
		Tensor to_complex_from_real() const;
		Tensor imag() const;
		Tensor to_complex_from_imag() const;
		Tensor sum(utils::optional_list list = nullptr, bool keepdim=false) const;
		Tensor mean(utils::optional_list list = nullptr, bool keepdim=false) const;
		result_types::max<Tensor, Tensor> max(utils::optional_list list = nullptr) const;
		result_types::max<Tensor, Tensor> min(utils::optional_list list = nullptr) const;
		/* result_types::max<Tensor, Tensor> max() const; */
		Tensor sum_as(const SizeRef&) const;
		Tensor sum_as(const Tensor&) const;
		/* result_types::max<Tensor, Tensor> max(size_value_t) const; */
		Tensor exp() const;
		Tensor& exp_();
		Tensor pow(Scalar) const;
		Tensor& pow_(Scalar);
		Tensor& inverse_();
		Tensor inverse() const;
		Tensor clip(Scalar, Scalar) const;
		Tensor& clip_(Scalar, Scalar);
		inline Tensor force_contiguity() const {return Tensor(_vals.force_contiguity(), shape());}	
		//this is going to pad based on what is given
		//for example if the tuple is : {1,1}, then pad the last dimension with 1 up and 1 down
		Tensor pad(std::vector<size_value_t>, const char* mode="constant", Scalar value=0) const;
        Tensor unpad(std::vector<size_value_t>) const;
        Tensor flip(utils::optional_list list = nullptr) const;
        Tensor flip_view(utils::optional_list list = nullptr) const;
        Tensor undilate_(std::vector<size_value_t>) const;
        template<typename... Args>
        inline Tensor undilate_(size_value_t i, Args... args) const {
            std::vector<size_value_t> vec;
            vec.reserve(sizeof...(Args) + 1);
            utils::collect_integers_impl(vec, i, std::forward<Args>(args)...);
            return this->undilate_(std::move(vec)); 
        }
        Tensor undilate(std::vector<size_value_t>) const;
        template<typename... Args>
        inline Tensor undilate(size_value_t i, Args... args) const {
            std::vector<size_value_t> vec;
            vec.reserve(sizeof...(Args) + 1);
            utils::collect_integers_impl(vec, i, std::forward<Args>(args)...);
            return this->undilate(std::move(vec)); 
        }
        Tensor dilate(std::vector<size_value_t>) const;
        template<typename... Args>
        inline Tensor dilate(size_value_t i, Args... args) const {
            std::vector<size_value_t> vec;
            vec.reserve(sizeof...(Args) + 1);
            utils::collect_integers_impl(vec, i, std::forward<Args>(args)...);
            return this->dilate(std::move(vec)); 
        }

		//Tensor undilate_(size_value_t dil) const;
		//Tensor undilate_(size_value_t, size_value_t) const;
		//Tensor undilate_(size_value_t, size_value_t, size_value_t) const;
		//Tensor undilate(size_value_t dil) const;
        //Tensor undilate(size_value_t row_dil, size_value_t col_dil) const;
        //Tensor undilate(size_value_t chan_dil, size_value_t row_dil, size_value_t col_dil) const;
		////dilate by basically adding that many zeros between each row and collumn
		////the above has a memory thing where it takes part of its memory from the original tensor
		////there is also a contiguous version below
		//Tensor dilate(size_value_t dil) const;
		//Tensor dilate(size_value_t dil_r, size_value_t dil_c) const;
		//Tensor dilate(size_value_t chan_dil, size_value_t dil_r, size_value_t dil_c) const;
		Tensor repeat_(size_value_t amt) const; //this is the amount to repeat wthout copying memory by
		/* Tensor dilate_mem_(size_value_t dil) const; */
		Tensor repeat_(size_value_t dim, size_value_t amt) const;
		Tensor expand(SizeRef s) const;
		Tensor expand_as(const Tensor&) const;
        inline const bool& is_mutable() const {return _is_mutable;}
        inline Tensor& set_mutability(bool mut) {
            //this is important for things like autograd updates
            //and making sure undefined behavior doesn't happen
            utils::throw_exception(!(_is_mutable == false && mut == true),
                                   "Once a tensor is marked as immutable, it cannot be marked as mutable"
                                   "please clone the tensor");
            _is_mutable = mut; 
            return *this;
        }
        Tensor& force_mutable_function(std::function<void(Tensor&)>);


		//this is only available for if the dtype is DType::TensorObj
		//otherwise would have to split the tensor by the first dimension,
		//      and then iteratively increment a numerical value
		//that can just be done by the user, and the user can then split by which ever dimension they choose
		//may decide to change this in the future
		class NEUROTENSOR_API Iterator{
			public:
				explicit Iterator(const Tensor& p, int64_t index) : ptr(p), index(index) {}
				inline const Tensor operator*() {return ptr.dtype() == DType::TensorObj ? ptr[index].item<Tensor>() : ptr[index];}
				inline const Iterator& operator++() {++index;return *this;}
				inline bool operator!=(const Iterator& other) const {return other.index != index;}
			private:
				const Tensor& ptr;
				int64_t index;
		};

		inline Iterator begin() const { 
			return Iterator(*this, 0);
		}
		inline Iterator end() const {
			return Iterator(*this, shape()[0]);
		}
		
		/* inline static Tensor makeNullTensor(DType dt){ */
		/* 	return Tensor(ArrayVoid::makeEmptyArray(dt), {0}); */
		/* } */

		inline static Tensor makeNullTensorArray(int64_t num){ return Tensor({num}, DType::TensorObj); }

		inline static Tensor Null(){ return Tensor(nullptr); }
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


// Specialization of std::swap for nt::Tensor
// #include "functional/functional.h"
#include "functional/tensor_get.h"
// #include "sparse/SparseTensor.h"

namespace std {
    inline void swap(::nt::Tensor& lhs, ::nt::Tensor& rhs) {
        lhs.swap(rhs); // Call your custom swap function
    }
    inline ::nt::Tensor exp(const ::nt::Tensor& x){return x.exp();}
    inline ::nt::Tensor pow(const ::nt::Tensor& x, ::nt::Scalar s){return x.pow(s);}
}


#endif // _NT_TENSOR_H_
