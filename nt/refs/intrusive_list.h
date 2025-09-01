#ifndef NT_INTRUSIVE_LIST_H__
#define NT_INTRUSIVE_LIST_H__
#include "../intrusive_ptr/intrusive_ptr.hpp"
#include "../memory/meta_allocator.h"


namespace nt{
//more of a const sized list
//this is a vector-like object to be held inside of an intrusive ptr
//but holds a constant size
//simple, really just to hold like a list of elements
template<typename T>
class intrusive_list : public intrusive_ptr_target{
	T* _ptr;
	int64_t _size;
	public:
		intrusive_list()
			:_ptr(nullptr), _size(0)
		{}
		intrusive_list(int64_t elements)
			:_ptr(elements == 0 ? nullptr : MetaNewArr(T, elements)), _size(elements)
		{}
		intrusive_list(int64_t elements, T element)
			:intrusive_list(elements)
		{
			std::for_each(_ptr, _ptr + _size, [&element](T& e){e = element;});
		}
		intrusive_list(std::initializer_list<T> ls)
			:intrusive_list(ls.size())
		{
			std::copy(ls.begin(), ls.end(), _ptr);
		}
		intrusive_list(const intrusive_list& l)
			:_ptr(l._ptr), _size(l._size)
		{}
		intrusive_list(intrusive_list&& l)
			:_ptr(l._ptr), _size(l._size)
		{
			l._size = 0;
			// delete[] l._ptr;
		}

		inline ~intrusive_list() { if(_ptr != nullptr){ MetaFreeArr<T>(_ptr); } }
		inline const int64_t& size() const noexcept {return _size;}
		inline const bool empty() const noexcept {return _ptr == nullptr;}
		inline T* ptr() noexcept {return _ptr;}
		inline const T* ptr() const noexcept {return _ptr;}
		inline T& operator[](int64_t element) noexcept {return _ptr[element];}
		inline const T& operator[](int64_t element) const noexcept {return _ptr[element];}
		inline T* begin() noexcept {return _ptr;}
		inline const T* begin() const noexcept {return _ptr;}
		inline const T* cbegin() const noexcept {return _ptr;}
		inline T* end() noexcept {return _ptr + _size;}
		inline const T* end() const noexcept {return _ptr + _size;}
		inline const T* cend() const noexcept {return _ptr + _size;}
		inline T& at(int64_t n) noexcept {return _ptr[n];}
		inline const T& at(int64_t n) const noexcept {return _ptr[n];}
		inline intrusive_list& operator=(std::initializer_list<T> ls) noexcept {
			if(!empty()){
                MetaFreeArr<T>(_ptr);
			}
			_size = ls.size();
			_ptr = MetaNewArr(T, _size);
			std::copy(ls.begin(), ls.end(), _ptr);
			return *this;
		}

};




}

#endif //_NT_INTRUSIVE_LIST_H_
