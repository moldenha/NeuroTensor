#ifndef ITTERATOR_H_
#define ITTERATOR_H_

namespace nt{
class static_tensor_iterator;
}

#include "Tensor.h"
#include "SizeRef.h"
#include "dtype/DType.h"
#include <_types/_uint32_t.h>


namespace nt{
template<typename IteratorValueType> class static_mat_iterator{
	public:
		using iterator_category = std::forward_iterator_tag;
		using difference_type   = std::ptrdiff_t;
		using value_type        = IteratorValueType;
		using pointer           = IteratorValueType*;
		using reference         = IteratorValueType&;
		
		explicit static_mat_iterator<IteratorValueType>(pointer ptr);
		static_mat_iterator<IteratorValueType>& operator++();
		static_mat_iterator<IteratorValueType> operator++(int);
		bool operator==(const static_mat_iterator<IteratorValueType>& b);
		bool operator!=(const static_mat_iterator<IteratorValueType>& b);
		reference operator*() const;
		reference operator[](uint32_t col) const;
		pointer operator->() const;
		pointer m_pt() const;
		pointer operator+(int i);
	private:
		pointer m_ptr;
};

class static_tensor_iterator{
	public:
		using iterator_category = std::forward_iterator_tag;
		using difference_type   = std::ptrdiff_t;
		
		explicit static_tensor_iterator(d_type_list ptr, SizeRef s);
		explicit static_tensor_iterator(d_type_list ptr, SizeRef s, uint32_t va);
		static_tensor_iterator& operator++();
		static_tensor_iterator operator++(int);
		bool operator==(const static_tensor_iterator& b) const;
		bool operator!=(const static_tensor_iterator& b) const;
		Tensor operator*();
		Tensor operator[](uint32_t col);
		Tensor operator->();
		d_type_list& m_pt();
		Tensor operator+(int i);
	private:
		d_type_list m_ptr;
		uint32_t val_add;
		std::shared_ptr<SizeRef> _size;


};

}

#endif
