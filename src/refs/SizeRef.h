#ifndef SIZE_REF_H_
#define SIZE_REF_H_

namespace nt{
class SizeRef;
}

#include "ArrayRef.h"



#include <array>
#include <initializer_list>
#include "../dtype/ranges.h"


namespace nt{

class SizeRef{
	public:
		using ArrayRefInt = ArrayRef<uint32_t>;
		using value_type = uint32_t;
	private:
		ArrayRefInt _sizes;
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
		const ArrayRefInt::value_type& operator[](int16_t) const;
		SizeRef redo_index(int16_t, value_type) const;
		SizeRef operator[](my_range) const;
		/* ArrayRefInt::value_type& operator[](int16_t); */
		ArrayRefInt::value_type multiply(int16_t i =0) const;
		SizeRef permute(const std::vector<uint32_t> &Vec) const;
		SizeRef transpose(int8_t, int8_t) const;
		/* std::pair<const std::vector<ArrayRefInt::value_type>, SizeRef> get_index_order(const std::vector<ArrayRefInt::value_type>&) const; */
		SizeRef pop_front() const;
		/* const ArrayRefInt::value_type to_index(const std::vector<ArrayRefInt::value_type>&) const; */
		/* const ArrayRefInt::value_type permute_to_index(const std::vector<ArrayRefInt::value_type>&, const std::vector<ArrayRefInt::value_type>&, const SizeRef&) const; */
		std::vector<value_type> strides() const;
		SizeRef flatten(int8_t, int8_t) const;
		SizeRef unflatten(int8_t, int8_t) const;

		friend std::ostream& operator<< (std::ostream& out, const SizeRef&);
		std::vector<value_type> Vec() const;

};

}

#endif
