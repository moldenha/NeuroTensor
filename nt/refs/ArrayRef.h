#ifndef NT_ARRAY_REF_H__
#define NT_ARRAY_REF_H__

//I believe the C10 library has something similar
//this is to handle the shape variable inside of tensors

#include <iterator>
#include <memory.h>
#include <memory>
#include <vector>
#include <array>
#include <initializer_list>
#include "../intrusive_ptr/intrusive_ptr.hpp"
#include "../utils/type_traits.h"
#include "../utils/api_macro.h"
#include "../memory/meta_allocator.h"

namespace nt{

template<typename T>
inline void ArrayRefDeleteNothing(T*){;}

template<typename T>
class ArrayRef;

template<typename T>
NEUROTENSOR_API std::ostream& operator<<(std::ostream& os, const ArrayRef<T>& data);

template<typename T>
class NEUROTENSOR_API ArrayRef{
	std::unique_ptr<T[], void(*)(T*)> _vals;
	size_t _total_size;
	bool _empty;
	public:
		using iterator = const T*;
		using const_iterator = const T*;
		using size_type = size_t;
		using value_type = T;
		using reverse_iterator = std::reverse_iterator<iterator>;
		ArrayRef();
		ArrayRef(const ArrayRef<T> &Arr);
		ArrayRef(ArrayRef<T>&& Arr);
		// Constructor for general types
		template<typename U = T, typename std::enable_if_t<!std::is_same_v<U, std::nullptr_t> && std::is_convertible_v<U, T>, bool> = true>
		ArrayRef(const U& OneEle)
		:_vals(MetaNewArr(T, 1), MetaFreeArr<T>), _total_size(1), _empty(false)
		{_vals[0] = OneEle;}
		// Constructor for nullptr type
		template<typename U = T, typename std::enable_if_t<std::is_same_v<U, std::nullptr_t>, bool> = true>
		ArrayRef(const U& OneEle)
		:_vals(nullptr, ArrayRefDeleteNothing<T>), _total_size(0), _empty(true)
		{}
		ArrayRef(const T *data, size_t length);
		ArrayRef(const std::vector<T> &Vec);
		template<size_t N>
		ArrayRef(const std::array<T, N> &Arr);
		template<size_t N>
		ArrayRef(const T (&Arr)[N]);
        template<typename U, typename std::enable_if_t<!std::is_same_v<U, T>, bool> = true>
        ArrayRef(const std::initializer_list<U> &Vec)
        :_vals(MetaNewArr(T, Vec.size()), MetaFreeArr<T>), _total_size(Vec.size()), _empty(false)
        {
            const U* begin = Vec.begin();
            for(size_t i = 0; i < _total_size; ++i, ++begin)
                _vals[i] = static_cast<T>(*begin);
        }
		ArrayRef(const std::initializer_list<T> &Vec);
		ArrayRef(const std::unique_ptr<T[], void(*)(T*)>&, size_t);
		ArrayRef(std::unique_ptr<T[], void(*)(T*)>&&, size_t);
		ArrayRef<T>& operator=(const ArrayRef<T>&);
		ArrayRef<T>& operator=(ArrayRef<T>&&);
		bool operator==(const ArrayRef<T>&) const;
		bool operator!=(const ArrayRef<T>&) const;
		const T* data() const;
		size_t size() const;
		const T& front() const;
		const T& back() const;
		ArrayRef<T> clone() const;
		T& front();
		T& back();
		iterator begin() const;
		iterator end() const;
		const_iterator cbegin() const;
		const_iterator cend() const;
		reverse_iterator rbegin() const;
		reverse_iterator rend() const;
		bool empty() const;
		const T& operator[](size_t index) const;
		T& operator[](size_t index);
		const T& at(size_t index) const;
		T multiply() const;
		ArrayRef<T> pop_front() const;
		std::vector<T> to_vec() const;
		ArrayRef<T> permute(const std::vector<uint32_t>&) const;
		value_type* d_data();
		friend std::ostream& operator<< <> (std::ostream& out, const ArrayRef& obj);
		inline void nullify(){
			_vals.reset(nullptr);
			_total_size = 0;
			_empty = true;
		}
		inline void swap(ArrayRef& other){
			std::swap(other._vals, _vals);
			std::swap(_total_size, other._total_size);
			std::swap(_empty, other._empty);
		}
		inline static ArrayRef<T> zeros(size_t len){
			std::unique_ptr<T[], void(*)(T*)> ptr(MetaNewArr(T, len), MetaFreeArr<T>);
			std::fill(ptr.get(), ptr.get() + len, 0);
			return ArrayRef<T>(std::move(ptr), len);
		}
};


}


// Specialization of std::swap for nt::ArrayRef
namespace std {
    template<typename T>
    inline void swap(::nt::ArrayRef<T>& lhs, ::nt::ArrayRef<T>& rhs) {
        lhs.swap(rhs); // Call your custom swap function
    }
}
#endif
