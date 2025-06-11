#ifndef _DTYPE_ARRAYTENSOR_H
#define _DTYPE_ARRAYTENSOR_H

namespace nt{
class ArrayTensorTypes;
}

#include "DType.h"
#include "_DTYPE_ARRAYINT.h"
#include <cstddef>
#include <iterator>
#include <memory.h>
#include "../Tensor.h"

namespace nt{

class ArrayTensorTypes : public ArrayIntTypes{
	public:
		using value_t = Tensor;
	private:
		value_t* my_end(void* begin);
		const value_t* my_end_c(const void* begin) const;
	public:
		ArrayTensorTypes();
		void set(const d_type& val, void* begin) override;
		std::shared_ptr<void> make_shared(size_t _size) override;
		void* end_ptr(void*) override;
		const void* cend_ptr(const void*) const override;
		std::shared_ptr<void> share_from(const std::shared_ptr<void>& ptr, uint32_t x) const override;
		void make_size(size_t _size) override;
		d_type_list begin(void*) override;
		d_type_list end(void*) override;
		d_type_list cbegin(const void*) const override;
		d_type_list cend(const void*) const override;



};


}


#endif
