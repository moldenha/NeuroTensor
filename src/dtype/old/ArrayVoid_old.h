#ifndef ARRAY_VOID_H
#define ARRAY_VOID_H
namespace nt {
class ArrayVoid;
}

#include <_types/_uint32_t.h>
#include <memory>

#include "DType_enum.h"
#include "_DTYPE_ARRAYINT.h"

namespace nt{

class ArrayVoid{
	std::unique_ptr<ArrayIntTypes> typed;
	std::shared_ptr<void> _vals;
	uint32_t size;
	std::unique_ptr<ArrayIntTypes> get_typed(DType _type) const;
	ArrayVoid(const std::shared_ptr<void>&, std::unique_ptr<ArrayIntTypes>&&, uint32_t, DType);
	public:
		DType dtype;
		ArrayVoid(uint32_t, DType);
		ArrayVoid(void*, uint32_t, DType);
		ArrayVoid& operator=(const ArrayVoid&);
		ArrayVoid& operator=(ArrayVoid&&);
		ArrayVoid(const ArrayVoid&);
		ArrayVoid(ArrayVoid&&);
		const uint32_t Size() const;
		const void* data_ptr() const;
		void* data_ptr();
		void operator=(const d_type&);
		std::shared_ptr<void> share_part(uint32_t) const;
		ArrayVoid share_array(uint32_t) const;
		ArrayVoid share_array(uint32_t, uint32_t) const;
		void* begin_ptr();
		void* end_ptr();
		const void* cbegin_ptr() const;
		const void* cend_ptr() const;
		d_type_list begin();
		d_type_list end();
		d_type_list cbegin() const;
		d_type_list cend() const;
		const uint32_t use_count() const;
};

}
#endif
