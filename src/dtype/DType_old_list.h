#ifndef _DTYPE_LIST_H_
#define _DTYPE_LIST_H_



#include <_types/_uint32_t.h>
#include <iterator>
#include <vector>
#include <memory>
#include "../intrusive_ptr/intrusive_ptr.hpp"

namespace nt{

template<typename T>
class tdtype_list;
template<typename T>
class tdtype_contiguous;

template<typename T>
class tdtype_contiguous{
	friend class tdtype_list<T>;
	public:
		using iterator_category = std::forward_iterator_tag;
		using difference_type = std::ptrdiff_t;
		using reference = T&;
		using pointer = T*;
		using value_type = T;

		explicit tdtype_contiguous(pointer _ptr) :m_ptr(_ptr) {}
		inline tdtype_contiguous<T>& operator++(){++m_ptr; return *this;}
		inline tdtype_contiguous<T> operator++(int){
			tdtype_contiguous tmp(*this);
			++(*this);
			return tmp;
		}
		inline const bool operator==(const tdtype_contiguous<T>& b) const {return b.m_ptr == m_ptr;}
		inline const bool operator!=(const tdtype_contiguous<T>& b) const {return b.m_ptr != m_ptr;}
		inline const bool operator==(const tdtype_list<T>& b) const {return b.ptr_ == m_ptr;}
		inline const bool operator!=(const tdtype_list<T>& b) const {return b.ptr_ != m_ptr;}
		inline std::ptrdiff_t operator-(const tdtype_contiguous<T>& b) const {return m_ptr - b.m_ptr;}
		inline reference operator*() {return *m_ptr;}
		inline pointer operator->() const {return m_ptr;}
		inline reference operator[](const uint64_t i) {return m_ptr[i];}
		inline tdtype_contiguous<T>& operator+=(const uint64_t i) {m_ptr += i; return *this;}
		inline tdtype_contiguous<T> operator+(const uint64_t i) const {return tdtype_contiguous<T>(m_ptr + i);}
	private:
		pointer m_ptr;

};

template<typename T>
class tdtype_list{
	friend class tdtype_contiguous<T>;
	public:
		using iterator_category = std::forward_iterator_tag;
		using difference_type = std::ptrdiff_t;
		using reference = T&;
		using pointer = T*;
		using value_type = T;
		using intrusive_ptr_type = typename std::conditional<std::is_const<T>::value, const intrusive_ptr<void>*, intrusive_ptr<void>*>::type;;
		using size_ptr_type = typename std::conditional<std::is_const<T>::value, const uint64_t*, uint64_t*>::type;;
		
		explicit tdtype_list(intrusive_ptr_type ptrs, size_ptr_type sizes, const uint64_t cur_bucket, const uint64_t cur_data_index, const uint64_t max_bucket_size)
			:sizes(sizes),
			current_bucket(cur_bucket),
			data_index(cur_data_index),
			ptrs_(ptrs),
			ptr_(reinterpret_cast<T*>(ptrs_[cur_bucket].get()) + cur_data_index),
			maxBS(max_bucket_size)
		{}
		inline tdtype_list<T>& operator++(){
			++data_index;
			if(current_bucket < maxBS && data_index >= sizes[current_bucket]){return increment_bucket();} 
			++ptr_; 
			return*this;
		}
		inline tdtype_list<T> operator++(int){
			tdtype_list<T> tmp(ptrs_, sizes, current_bucket, data_index, maxBS);
			++(*this);
			return tmp;
		}

		inline const bool operator==(const tdtype_list<T>& b) const {return b.ptr_ == ptr_;}
		inline const bool operator==(const tdtype_contiguous<T>& b) const {return b.m_ptr == ptr_;}
		inline const bool operator!=(const tdtype_list<T>& b) const {return b.ptr_ != ptr_;}
		inline const bool operator!=(const tdtype_contiguous<T>& b) const {return b.m_ptr != ptr_;}
		inline reference operator*() {return *ptr_;}
		reference operator[](uint64_t i);
		tdtype_list<T>& operator+=(uint64_t i);
		tdtype_list<T> operator+(uint64_t i);
		std::ptrdiff_t operator-(const tdtype_list<T>&) const;
		inline pointer operator->() const {return ptr_;}
		inline tdtype_list<T>& operator=(const tdtype_list<T>& dt){
			sizes = dt.sizes;
			current_bucket = dt.current_bucket;
			data_index = dt.data_index;
			ptrs_ = dt.ptrs_;
			ptr_ = dt.ptr_;
			uint64_t& mutableX = const_cast<uint64_t&>(maxBS);
			mutableX = dt.maxBS;
			return *this;
		}


		

	private:
		size_ptr_type sizes;
		uint64_t current_bucket, data_index;
		intrusive_ptr_type ptrs_;
		pointer ptr_;
		const uint64_t maxBS;
		inline void set_ptr(){ptr_ = reinterpret_cast<pointer>(ptrs_[current_bucket].get()) + data_index;}
		inline tdtype_list& increment_bucket(){++current_bucket;data_index=0;ptr_ = reinterpret_cast<pointer>(ptrs_[current_bucket].get());return *this;}

};

namespace utils{

template<typename T>
struct ItteratorBaseType{
	using type = T;
};

template<typename T>
struct ItteratorBaseType<T*>{
	using type = T;
};

template<typename T>
struct ItteratorBaseType<tdtype_list<T> >{
	using type = T;
};

template<typename T>
struct ItteratorBaseType<const T*>{
	using type = T;
};

template<typename T>
struct ItteratorBaseType<tdtype_list<const T> >{
	using type = T;
};


template<typename T>
struct ItteratorBaseType<tdtype_contiguous<const T> >{
	using type = T;
};

template<typename T>
struct ItteratorBaseType<tdtype_contiguous<T> >{
	using type = T;
};

template<typename T>
using ItteratorBaseType_t = typename ItteratorBaseType<T>::type;


}


}


#endif
