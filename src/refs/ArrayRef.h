#ifndef ARRAY_REF_H_
#define ARRAY_REF_H_

#include <_types/_uint16_t.h>
#include <_types/_uint32_t.h>
#include <iterator>
#include <memory.h>
#include <vector>
#include <array>
#include <initializer_list>

namespace nt{

template<typename T>
class ArrayRef;

template<typename T>
std::ostream& operator<<(std::ostream& os, const ArrayRef<T>& data);

template<typename T>
class ArrayRef{
	std::unique_ptr<T[]> _vals;
	size_t _total_size;
	bool _empty;
	public:
		using iterator = const T*;
		using const_iterator = const T*;
		using size_type = size_t;
		using value_type = T;
		using reverse_iterator = std::reverse_iterator<iterator>;
		ArrayRef();
		ArrayRef(const ArrayRef<T> &Arr);
		ArrayRef(ArrayRef<T>&& Arr);
		ArrayRef(const T &OneEle);
		ArrayRef(const T *data, size_t length);
		ArrayRef(const std::vector<T> &Vec);
		template<size_t N>
		ArrayRef(const std::array<T, N> &Arr);
		template<size_t N>
		ArrayRef(const T (&Arr)[N]);
		ArrayRef(const std::initializer_list<T> &Vec);
		ArrayRef(std::unique_ptr<T[]>, size_t);
		ArrayRef<T>& operator=(const ArrayRef<T>&);
		ArrayRef<T>& operator=(ArrayRef<T>&&);
		const bool operator==(const ArrayRef<T>&) const;
		const bool operator!=(const ArrayRef<T>&) const;
		const T* data() const;
		size_t size() const;
		const T& front() const;
		const T& back() const;
		T& front();
		T& back();
		iterator begin() const;
		iterator end() const;
		const_iterator cbegin() const;
		const_iterator cend() const;
		reverse_iterator rbegin() const;
		reverse_iterator rend() const;
		bool empty() const;
		const T& operator[](int16_t index) const;
		T& operator[](int16_t index);
		const T& at(uint16_t index) const;
		const T multiply() const;
		ArrayRef<T> pop_front() const;
		std::vector<T> to_vec() const;
		ArrayRef<T> permute(const std::vector<uint32_t>&) const;
		value_type* d_data();
		friend std::ostream& operator<< <> (std::ostream& out, const ArrayRef& obj);
};


}
#endif
