#ifndef _DTYPE_ARRAY_H
#define _DTYPE_ARRAY_H
namespace nt{
class ArrayIntTypes;
class ArrayDoubleTypes;
class ArrayFloatTypes;
class ArrayLongTypes;
class ArrayTensorTypes;
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

class ArrayDoubleTypes: public ArrayIntTypes{
	public:
		using value_t = double;
		size_t size;
	private:
		value_t* my_end(void* begin);
		const value_t* my_end_c(const void* begin) const;
	public:
		ArrayDoubleTypes();
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
class ArrayFloatTypes: public ArrayIntTypes{
	public:
		using value_t = float;
		size_t size;
	private:
		value_t* my_end(void* begin);
		const value_t* my_end_c(const void* begin) const;

	public:
		ArrayFloatTypes();
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

#include "../Tensor.h"
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
