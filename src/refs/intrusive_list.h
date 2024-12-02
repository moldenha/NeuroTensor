#ifndef _NT_INTRUSIVE_LIST_H_
#define _NT_INTRUSIVE_LIST_H_
#include "../intrusive_ptr/intrusive_ptr.hpp"


namespace nt{
//more of a const sized list
template<typename T>
class intrusive_list : public intrusive_ptr_target{
	T* _ptr;
	int64_t _size;
	public:
		intrusive_list() :_ptr(nullptr), _size(0) {}
		intrusive_list(int64_t n) :_ptr(new T[n]), _size(n) {}
		intrusive_list(int64_t n, T element)
			:intrusive_list(n)
		{
			std::for_each(_ptr, _ptr + _size, [&element](T& e){e = element;});
		}
		inline const int64_t& size() const noexcept {return _size;}
		inline const bool empty() const noexcept {return (_size == 0);}
		inline T* ptr() noexcept {return _ptr;}
		inline T* end() noexcept {return _ptr + _size;}
		inline const T* ptr() const noexcept {return _ptr;}
		inline T& at(int64_t n) noexcept {return _ptr[n];}
		inline const T& at(int64_t n) const noexcept {return _ptr[n];}
};


}

#endif //_NT_INTRUSIVE_LIST_H_
