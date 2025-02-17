#ifndef _DTYPE_ARRAYLONG_H
#define _DTYPE_ARRAYLONG_H
namespace nt{
class ArrayLongTypes;
}
#include "DType.h"
#include "_DTYPE_ARRAYINT.h"

#include <memory.h>

namespace nt{

class ArrayLongTypes: public ArrayIntTypes{
	public:
		using value_t = uint32_t;
		size_t size;
	private:
		value_t* my_end(void* begin);
		const value_t* my_end_c(const void* begin) const;
	public:
		ArrayLongTypes();
		void set(const d_type& val, void* begin) override;
		std::shared_ptr<void> make_shared(size_t _size) override;
		void make_size(size_t _size) override;
		void* end_ptr(void* ptr) override;
		const void* cend_ptr(const void* ptr) const override;
		std::shared_ptr<void> share_from(const std::shared_ptr<void>& ptr, uint32_t x) const override;
		d_type_list begin(void*) override;
		d_type_list end(void*) override;
		d_type_list cbegin(const void*) const override;
		d_type_list cend(const void*) const override;
};

}

#endif
