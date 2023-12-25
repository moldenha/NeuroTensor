#ifndef _DTYPE_LIST_H_
#define _DTYPE_LIST_H_

#include <_types/_uint32_t.h>
#include <_types/_uint64_t.h>
#include <iterator>
#include <vector>

namespace nt{
class const_dtype_list;
class dtype_list;
template<typename T>
class tdtype_list{
	public:
		using iterator_category = std::forward_iterator_tag;
		using difference_type = std::ptrdiff_t;
		using reference = T&;
		using pointer = T*;
		using value_type = T;

		tdtype_list(void** _ptr);
		tdtype_list<T>& operator++();
		tdtype_list<T> operator++(int);
		const bool operator==(const tdtype_list<T>& b) const;
		const bool operator!=(const tdtype_list<T>& b) const;
		reference operator*();
		reference operator[](const uint32_t);
		tdtype_list<T>& operator+=(const uint32_t);
		tdtype_list<T> operator+(const uint32_t) const;
	private:
		void** m_ptr;

};
}

#include "DType_enum.h"
#include "DType.h"
#include "Scalar.h"
#include <functional>

namespace nt{


class const_dtype_list{
	public:
		using iterator_category = std::forward_iterator_tag;
		using difference_type = std::ptrdiff_t;
		using reference = ConstScalarRef;
		using pointer = void**;
		using value_type = void;

		const_dtype_list(pointer, DType);
		const_dtype_list& operator++(); // std::static_cast<uint8_t*>(ptr) += add_val;
		const_dtype_list operator++(int);
		const_dtype_list& operator+=(uint32_t);
		bool operator==(const const_dtype_list&) const;
		bool operator!=(const const_dtype_list&) const;
		ConstScalarRef operator*();
	
	private:
		DType dtype;
		pointer m_ptr;
		const size_t add_val; // this is going to be equal to the size of the dtype
		const size_t get_val_add(DType);


};

class dtype_list{
	public:
		using iterator_category = std::forward_iterator_tag;
		using difference_type = std::ptrdiff_t;
		using reference = ScalarRef;
		using pointer = void**;
		using value_type = void;

		dtype_list(pointer, DType);
		dtype_list& operator++(); // std::static_cast<uint8_t*>(ptr) += add_val;
		dtype_list operator++(int);
		dtype_list& operator+=(uint32_t);
		bool operator==(const dtype_list&) const;
		bool operator!=(const dtype_list&) const;
		ScalarRef operator*();
		void set(const Scalar);
	
		/* std::functional<void(void**, const Scalar&)> set_func; */
		/* std::functional<ScalarRef(void**)> ref_func; */
	private:
		DType dtype;
		pointer m_ptr;
		const size_t add_val; // this is going to be equal to the size of the dtype
		const size_t get_val_add(DType);
		/* void set_functions(); */

};

}


#endif
