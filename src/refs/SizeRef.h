#ifndef SIZE_REF_H_
#define SIZE_REF_H_

namespace nt{
class SizeRef;
}

#include "ArrayRef.h"



#include <array>
#include <initializer_list>
#include "../dtype/ranges.h"
#include <cassert>


namespace nt{

class SizeRef{
	public:
		using ArrayRefInt = ArrayRef<int64_t>;
		using value_type = int64_t;
	private:
		ArrayRefInt _sizes;
		template<typename T>
		inline int64_t get_index_size_t(size_t index, T first){
			static_assert(std::is_same<T, size_t>::value, "Arguments must be size_t");
			++index;
			return first * multiply(index);
		}
		template<typename T, typename... Args>
		inline value_type get_index_size_t(size_t index, T first, Args... rest){
			static_assert(std::is_same<T, size_t>::value, "Arguments must be size_t");
			++index;
			return (first * multiply(index)) + get_index_size_t(index, rest...);
		}
	public:
		/* SizeRef(ArrayRefInt); */
		SizeRef(const value_type &OneEle);
		SizeRef(const ArrayRefInt::value_type *data, size_t length);
		SizeRef(const std::vector<ArrayRefInt::value_type> &Vec);
		SizeRef(std::vector<value_type>&& Vec);
		template<size_t N>
		SizeRef(const std::array<ArrayRefInt::value_type, N> &Arr);
		template<size_t N>
		SizeRef(const ArrayRefInt::value_type (&Arr)[N]);
		SizeRef(const std::initializer_list<ArrayRefInt::value_type> &Vec);
		SizeRef(ArrayRefInt&&);
		SizeRef(const ArrayRefInt&);

		SizeRef(const SizeRef&);
		SizeRef(SizeRef&&);
		SizeRef& operator=(const SizeRef&);
		SizeRef& operator=(SizeRef&&);
		const bool operator==(const SizeRef&) const;
		const bool operator!=(const SizeRef&) const;
		const value_type* data() const;
		const size_t size() const;
		const value_type& front() const;
		const value_type& back() const;
		ArrayRefInt::iterator begin() const;
		ArrayRefInt::iterator end() const;
		ArrayRefInt::const_iterator cbegin() const;
		ArrayRefInt::const_iterator cend() const;
		ArrayRefInt::reverse_iterator rbegin() const;
		ArrayRefInt::reverse_iterator rend() const;
		bool empty() const;
		const ArrayRefInt::value_type& operator[](value_type) const;
		SizeRef redo_index(value_type, value_type) const;
		SizeRef delete_index(value_type) const;
		SizeRef operator[](my_range) const;
		SizeRef range(value_type, value_type) const;
		/* ArrayRefInt::value_type& operator[](int16_t); */
		ArrayRefInt::value_type multiply(value_type i =0) const;
		ArrayRefInt::value_type unsigned_multiply(value_type i =0) const; //this will check to see if a dimension is negative
		SizeRef permute(const std::vector<value_type> &Vec) const;
		SizeRef transpose(value_type, value_type) const;
		/* std::pair<const std::vector<ArrayRefInt::value_type>, SizeRef> get_index_order(const std::vector<ArrayRefInt::value_type>&) const; */
		SizeRef pop_front() const;
		/* const ArrayRefInt::value_type to_index(const std::vector<ArrayRefInt::value_type>&) const; */
		/* const ArrayRefInt::value_type permute_to_index(const std::vector<ArrayRefInt::value_type>&, const std::vector<ArrayRefInt::value_type>&, const SizeRef&) const; */
		std::vector<value_type> strides() const;
		SizeRef flatten(value_type, value_type) const;
		SizeRef unflatten(value_type, value_type) const;
		inline SizeRef clone() const {return SizeRef(_sizes.clone());}
		inline const ArrayRefInt& original_ref() const {return _sizes;}
		inline ArrayRefInt arr() const {return _sizes.clone();}

		template<typename... Args>
		inline value_type get_index(Args... args){
			constexpr size_t num_args = sizeof...(Args);
			assert(num_args <= size());
			/* utils::throw_exception(num_args <= size(), "expected to have at most $ arguments but got $", size(), num_args); */
			return get_index_size_t(0, args...);
		}

		friend std::ostream& operator<< (std::ostream& out, const SizeRef&);
		std::vector<value_type> Vec() const;
		inline void swap(SizeRef& other){_sizes.swap(other._sizes);}
		inline void nullify(){_sizes.nullify();}

};

}


// Specialization of std::swap for nt::SizeRef
namespace std {
    inline void swap(::nt::SizeRef& lhs, ::nt::SizeRef& rhs) {
        lhs.swap(rhs); // Call your custom swap function
    }
}

#endif
