#ifndef _NT_BUCKET_ITERATOR_H_
#define _NT_BUCKET_ITERATOR_H_

#include <iterator>
#include <unistd.h>

namespace nt{

template<typename T>
class BucketIterator_blocked; //this is when the strides are in blocks, but not contiguous

template<typename T>
class BucketIterator_list; // this is when the strides are not in blocks


//this has been replaced by a standard T* pointer (faster, no overhead, etc.)
/* template<typename T> */
/* class BucketIterator_contiguous{ */
/* 	friend class BucketIterator_list<T>; */
/* 	friend class BucketIterator_blocked<T>; */
	
/* 	public: */
/* 		using iterator_category = std::forward_iterator_tag; */
/* 		using difference_type = std::ptrdiff_t; */
/* 		using reference = T&; */
/* 		using pointer = T*; */
/* 		using value_type = T; */

/* 		explicit BucketIterator_contiguous(pointer _ptr) :m_ptr(_ptr) {} */
/* 		inline BucketIterator_contiguous& operator++(){++m_ptr; return *this;} */
/* 		inline BucketIterator_contiguous operator++(int){ */
/* 			BucketIterator_contiguous tmp(*this); */
/* 			++(*this); */
/* 			return tmp; */
/* 		} */
/* 		inline const bool operator!=(const BucketIterator_contiguous& b) const {return m_ptr != b.m_ptr;} */
/* 		inline const bool operator==(const BucketIterator_contiguous& b) const {return m_ptr == b.m_ptr;} */
/* 		inline const bool operator!=(const BucketIterator_list<T>& b) const {return m_ptr != *b.m_ptr;} */
/* 		inline const bool operator==(const BucketIterator_list<T>& b) const {return m_ptr == *b.m_ptr;} */
/* 		inline const bool operator!=(const BucketIterator_blocked<T>& b) const {return m_ptr != b.current_ptr;} */
/* 		inline const bool operator==(const BucketIterator_blocked<T>& b) const {return m_ptr == b.current_ptr;} */
/* 		inline const bool operator<(const BucketIterator_contiguous<T>& b) const {return m_ptr < b.m_ptr;} */
/* 		inline std::ptrdiff_t operator-(const BucketIterator_contiguous<T>& b) const {return m_ptr - b.m_ptr;} */
/* 		inline reference operator*() {return *m_ptr;} */
/* 		inline pointer operator->() const {return m_ptr;} */
/* 		inline reference operator[](const uint64_t i) {return m_ptr[i];} */
/* 		inline BucketIterator_contiguous& operator+=(const uint64_t i){m_ptr += i; return *this;} */
/* 		inline BucketIterator_contiguous operator+(const uint64_t i) const {return BucketIterator_contiguous(m_ptr+i);} */
/* 		inline operator pointer() const {return m_ptr;} */

/* 	private: */
/* 		pointer m_ptr; */
/* }; */

/* template<typename T> */
/* using BucketIterator_contiguous<T> = T*; */

template<typename T>
class BucketIterator_list{
	friend class BucketIterator_blocked<T>;
	public:
		using iterator_category = std::forward_iterator_tag;
		using difference_type = std::ptrdiff_t;
		using reference = T&;
		using pointer = T*;
		using value_type = T;
		using store_type = std::remove_const_t<T>**;

		explicit BucketIterator_list(store_type _ptr) :m_ptr(_ptr) {}
		inline BucketIterator_list& operator++() noexcept {++m_ptr; return *this;}
		inline BucketIterator_list operator++(int) noexcept {
			BucketIterator_list tmp(*this);
			++(*this);
			return tmp;
		}
		inline const bool operator!=(const BucketIterator_list& b) const noexcept {return m_ptr != b.m_ptr;}
		inline const bool operator==(const BucketIterator_list& b) const noexcept {return m_ptr == b.m_ptr;}
		inline const bool operator!=(const T*& b) const noexcept {return *m_ptr != b;}
		inline const bool operator==(const T*& b) const noexcept {return *m_ptr == b;}
		inline const bool operator!=(const BucketIterator_blocked<T>& b) const noexcept {return *m_ptr != b.current_ptr;}
		inline const bool operator==(const BucketIterator_blocked<T>& b) const noexcept {return *m_ptr == b.current_ptr;}
		inline const bool operator<(const BucketIterator_list& b) const noexcept {return m_ptr < b.m_ptr;}
		inline const bool operator<=(const BucketIterator_list& b) const noexcept {return m_ptr <= b.m_ptr;}
		inline std::ptrdiff_t operator-(const BucketIterator_list& b) const noexcept {return m_ptr - b.m_ptr;}
		inline reference operator*() noexcept {return **m_ptr;}
		inline pointer operator->() const noexcept {return *m_ptr;}
		inline reference operator[](const std::ptrdiff_t i) noexcept {return *m_ptr[i];}
		inline reference operator[](const std::ptrdiff_t i) const noexcept {return *m_ptr[i];}
		inline BucketIterator_list& operator+=(const std::ptrdiff_t i) noexcept {m_ptr += i; return *this;}
		inline BucketIterator_list operator+(std::ptrdiff_t i) noexcept {return BucketIterator_list(m_ptr + i);}

		inline friend bool operator!=(const T*& a, const BucketIterator_list<T>& b) noexcept {
			return b.current_ptr != a;
		}
		inline friend bool operator==(const T*& a, const BucketIterator_list<T>& b) noexcept {
			return b.current_ptr == a;
		}
	private:
		store_type m_ptr;
};


//within these operations, where recursion can be used it is, and where if statements can be taken out they are
//this is to reduce code complexity, and optimize as much as possible
//most of these operations are used thousands of times, so it is important they are as fast as can be
//some changes because now max stride num is _stride_size / 2 - 1
template<typename T>
class BucketIterator_blocked{
	friend class BucketIterator_list<T>;

	public:
		using iterator_category = std::forward_iterator_tag;
		using difference_type = std::ptrdiff_t;
		using reference = T&;
		using pointer = T*;
		using value_type = T;
		using store_type = std::remove_const_t<T>**;

		explicit BucketIterator_blocked(store_type _ptr, pointer _current, int64_t stride_num, int64_t index=0) : m_ptr(_ptr), current_ptr(_current), stride_num(stride_num), current_stride(index) {}
		//this operation is obviously used a lot
		//so in order to optimize it, all the if statements were taken out, in order to reduce branching
		//this way it is just a few simple comparison operations, some additions, and a multiplication
		//all of this can be optimized by the compiler and turned into loop unrolling
		//considering using _mm_prefetch to load the next block into cache
		inline BucketIterator_blocked& operator++() noexcept {
			//this version uses no branching
			//this offers an optimized way to achieve this
			++current_ptr;
			
			//avoiding brainching using a conditional move
			const bool is_end_of_block = (current_ptr == m_ptr[1] && current_stride < stride_num);

			//use a conditional increment to avoid an explicit `if`.
			current_stride += is_end_of_block;
			
			//_mm_prefetch loads the next block of memory into cache
			//making access and use faster
			/*if (is_end_of_block) {
			 *     _mm_prefetch(m_ptr + 2, _MM_HINT_T0);
			 *     m_ptr += 2;
			 *     current_ptr = *m_ptr;
			 *}
			 *
			 */

			//adjust pointers when the current block is exhausted.
			m_ptr += 2 * is_end_of_block; //if `is_end_of_block` is true, increment by 2, else no change.
			current_ptr = (is_end_of_block ? m_ptr[0] : current_ptr);
			return *this;
		}
		inline BucketIterator_blocked operator++(int) noexcept{
			BucketIterator_blocked tmp(*this);
			++(*this);
			return tmp;
		}
		inline const bool operator!=(const BucketIterator_blocked& b) const noexcept {return current_ptr != b.current_ptr;}
		inline const bool operator==(const BucketIterator_blocked& b) const noexcept {return current_ptr == b.current_ptr;}
		inline const bool operator!=(const T*& b) const noexcept {return current_ptr != b;}
		inline const bool operator==(const T*& b) const noexcept {return current_ptr == b;}
		inline const bool operator!=(const BucketIterator_list<T>& b) const noexcept {return current_ptr != *b.m_ptr;}
		inline const bool operator==(const BucketIterator_list<T>& b) const noexcept {return current_ptr == *b.m_ptr;}
		inline const bool operator<(const BucketIterator_blocked& b) const noexcept {return current_stride < b.current_stride || current_ptr < b.current_ptr;}
		inline const bool operator<=(const BucketIterator_blocked& b) const noexcept {return current_stride < b.current_stride || current_ptr <= b.current_ptr;}
		//if they are at equal strides it is just pointer arithmatic
		//if this stride is less than b's, just multiply the result again
		//otherwise, get b's current block size, iterate it to the next block, and then add that to this repeated process
		inline std::ptrdiff_t operator-(const BucketIterator_blocked& b) const noexcept {
			return (current_stride == b.current_stride) ? current_ptr - b.current_ptr : 
				((current_stride < b.current_stride) ? (-1) * (b - (*this)) : 
				 (b.m_ptr[1] - b.current_ptr) + ((*this) - b.get_next_block())); 
		}
		inline reference operator*() noexcept {return *current_ptr;}
		inline pointer operator->() noexcept {return current_ptr;}
		inline reference operator[](std::ptrdiff_t i){
			pointer store = current_ptr + i;
			const bool is_end_of_block = (store >= m_ptr[1] && current_stride < stride_num);
			return (is_end_of_block) ? this->get_next_block()[i - (m_ptr[1] - current_ptr)] : *store;
		}
		inline reference operator[](std::ptrdiff_t i) const{
			const pointer store = current_ptr + i;
			const bool is_end_of_block = (store >= m_ptr[1] && current_stride < stride_num);
			return (is_end_of_block) ? this->get_next_block()[i - (m_ptr[1] - current_ptr)] : *store;
		}
		inline BucketIterator_blocked& operator+=(std::ptrdiff_t i){
			pointer store = current_ptr;
			current_ptr += i;
			const bool is_end_of_block = (current_ptr >= m_ptr[1] && current_stride < stride_num);
			return (is_end_of_block) ? this->iterate_next_block() += (i - (m_ptr[1] - store)) : *this;
		}
		inline BucketIterator_blocked operator+(std::ptrdiff_t i) const noexcept {
			pointer store = current_ptr + i;
			const bool is_end_of_block = (store >= m_ptr[1] && current_stride < stride_num);
			return (is_end_of_block) ? this->get_next_block() + (i - (m_ptr[1] - current_ptr)) : BucketIterator_blocked(m_ptr, store, stride_num, current_stride);
		}
		inline friend const bool same_block(const BucketIterator_blocked& b, const BucketIterator_blocked& a) noexcept {
			//to see if they are in the same block of contiguous memory
			if(b.current_stride == a.current_stride)
				return true;
			/* //these last 2 are for if one of them is an created using end_blocked */
			/* if(b.current_stride == a.stride_num && a.current_stride == (b.current_stride-1)) */
			/* 	return true; */
			/* if(a.current_stride == b.stride_num && b.current_stride == (a.current_stride-1)) */
			/* 	return true; */
			return false;

		}
		inline const int64_t& get_current_stride() const noexcept {return current_stride;}
		inline operator const std::remove_const_t<pointer>() const noexcept { //gives the pointer of the iterator
			return current_ptr;
		}
		inline operator pointer() noexcept {
			return current_ptr;
		}
		inline std::ptrdiff_t block_size(int64_t block=0) const noexcept { //gives the size of the current contiguous block
			if(block == 0){return m_ptr[1] - current_ptr;} //amount left in this block
			return m_ptr[block*2+1] - m_ptr[block*2];
		}
		template<size_t N>
		inline bool block_size_left() const noexcept{
			return (m_ptr[1] - current_ptr) >= N && (m_ptr[1] > current_ptr);
		}
		inline friend uint64_t block_diff(const BucketIterator_blocked& a, const BucketIterator_blocked& b) noexcept {
			if(b.current_stride == a.current_stride)
				return 0;
			//next 2 are for if a or b is an end() condition
			/* if(b.current_stride == b.stride_num && a.current_stride == b.stride_num-1) */
			/* 	return 0; */
			/* if(a.current_stride == a.stride_num && b.current_stride == a.stride_num-1) */
			/* 	return 0; */
			//this gives the number of blocks between this block and the next block
			return b.current_stride > a.current_stride ? b.current_stride - a.current_stride : a.current_stride - b.current_stride;
		}
		inline T* block_end() const noexcept {
			return m_ptr[1];
		}
		inline T* block_end() noexcept {
			return m_ptr[1];
		}
		inline BucketIterator_blocked<T>& iterate_next_block() noexcept {
			m_ptr += 2;
			current_ptr = m_ptr[0];
			++current_stride;
			return *this;
		}
		inline BucketIterator_blocked<T> get_next_block() const noexcept {
			return BucketIterator_blocked<T>(m_ptr + 2, m_ptr[2], stride_num, current_stride+1);
		}
		inline friend bool operator!=(T*& a, BucketIterator_blocked<T>& b) noexcept {
			return b.current_ptr != a;
		}
		inline friend bool operator==(T*& a, BucketIterator_blocked<T>& b) noexcept {
			return b.current_ptr == a;
		}
	private:
		/* explicit BucketIterator_blocked(store_type _pt, int64_t stride_n, int64_t curStride) */
		/* 	:m_ptr(_pt), */
		/* 	stride_num(stride_n), */
		/* 	current_stride(curStride) */
		/* 	{} */
			/* :m_ptr(_pt), */
			/* current_ptr(cur), */

		store_type m_ptr;
		pointer current_ptr;
		int64_t current_stride;
		int64_t stride_num;



};


/* template<typename T> */
/* inline bool operator!=(BucketIterator_contiguous<T>& a, BucketIterator_blocked<T>& b){ */
/* 	return b.current_ptr != a; */
/* } */

namespace utils{

template<typename T>
struct IteratorBaseType{
	using type = T;
};

template<typename T>
struct IteratorBaseType<T*>{
	using type = T;
};

template<typename T>
struct IteratorBaseType<BucketIterator_list<T> >{
	using type = T;
};
template<typename T>
struct IteratorBaseType<BucketIterator_list<const T> >{
	using type = T;
};


template<typename T>
struct IteratorBaseType<const T*>{
	using type = T;
};

/* template<typename T> */
/* struct IteratorBaseType<BucketIterator_contiguous<const T> >{ */
/* 	using type = T; */
/* }; */

/* template<typename T> */
/* struct IteratorBaseType<T* >{ */
/* 	using type = T; */
/* }; */

template<typename T>
struct IteratorBaseType<BucketIterator_blocked<T> >{
	using type = T;
};
template<typename T>
struct IteratorBaseType<BucketIterator_blocked<const T> >{
	using type = T;
};


template<typename T>
using IteratorBaseType_t = typename IteratorBaseType<T>::type;


template<typename T>
struct iterator_is_contiguous{
	static constexpr bool value = false;
};

template<typename T>
struct iterator_is_contiguous<T*>{
	static constexpr bool value = true;
};

template<typename T>
struct iterator_is_contiguous<const T*>{
	static constexpr bool value = true;
};


template<typename T>
inline static constexpr bool iterator_is_contiguous_v = iterator_is_contiguous<T>::value;


template<typename T>
struct iterator_is_blocked{
	static constexpr bool value = false;
};

template<typename T>
struct iterator_is_blocked<BucketIterator_blocked<T>>{
	static constexpr bool value = true;
};

template<typename T>
struct iterator_is_blocked<BucketIterator_blocked<const T>>{
	static constexpr bool value = true;
};


template<typename T>
inline static constexpr bool iterator_is_blocked_v = iterator_is_blocked<T>::value;


template<typename T>
struct iterator_is_list{
	static constexpr bool value = false;
};

template<typename T>
struct iterator_is_list<BucketIterator_list<T>>{
	static constexpr bool value = true;
};

template<typename T>
struct iterator_is_list<BucketIterator_list<const T>>{
	static constexpr bool value = true;
};


template<typename T>
inline static constexpr bool iterator_is_list_v = iterator_is_list<T>::value;


} //nt::utils::


} // nt::

#endif //_NT_BUCKET_ITERATOR_H_
