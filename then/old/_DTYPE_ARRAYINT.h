#ifndef _DTYPE_ARRAYINT_H
#define _DTYPE_ARRAYINT_H
namespace nt{
class ArrayIntTypes;
}

#include "DType.h"

#include <cstddef>
#include <iterator>
#include <memory.h>

namespace nt{


class ArrayIntTypes{
	public:
		using value_t = int;
		size_t size;
	private:
		value_t* my_end(void* begin);
		const value_t* my_end_c(const void* begin) const;
	public:
		ArrayIntTypes();
		virtual void set(const d_type& val, void* begin);
		virtual std::shared_ptr<void> make_shared(size_t _size);
		virtual void make_size(size_t _size);
		virtual void* end_ptr(void* ptr);
		virtual const void* cend_ptr(const void* ptr) const;
		virtual std::shared_ptr<void> share_from(const std::shared_ptr<void>& ptr, uint32_t x) const;
		virtual d_type_list begin(void*);
		virtual d_type_list end(void*);
		virtual d_type_list cbegin(const void*) const;
		virtual d_type_list cend(const void*) const;
};
}

#endif
