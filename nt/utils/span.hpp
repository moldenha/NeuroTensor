// this is kind of a copy of the c++20 std::span
// This is useful for not copying memory and can add extra speedups when needed
// think of this as kind of a constant view of memory
// NOTE:This does not make sure memory is allocated or deallocated, it is up to the user to make sure that memory does not go out of sope
//

#ifndef NT_UTILS_SPAN_HPP__
#define NT_UTILS_SPAN_HPP__

#include "type_traits.hpp"

namespace nt{

template<class T> class span{
    public:
        using element_type = T;
        using value_type = type_traits::remove_cvref_t<T>;
        using size_type = int64_t;
        using pointer = T*;
        using const_pointer = const T*;
        using reference = T&;
        using const_reference = const T&;

    private:
        const_pointer data_;
        size_type size_;
    public:
        span() = delete;
        constexpr span(const_pointer data, size_type size)
            :data_(data), size_(size) {}
        constexpr span(const_pointer first, const_pointer end)
            :data_(first), size_(end - first) {}
        constexpr span(std::initializer_list<element_type> l)
            :span(l.begin(), l.end()) {}
        constexpr span(const span& s)
            :data_(s.data_), size_(s.size_)
        {}
        span(span&& s)
            :data_(s.data_), size_(s.size_)
        {
            const_cast<T*&>(s.data_) = nullptr;
            s.size_ = 0;
        }
        span(const std::vector<T>& v)
            :data_(v.data()), size_(static_cast<int64_t>(v.size()))
        {}

        span(const std::vector<T>& v, int64_t n_size)
            :data_(v.data()), size_(n_size)
        {}

        inline span& operator=(const span& s){
            const_cast<T*&>(data_) = s.data_;
            size_ = s.size_;
            return *this;
        }
        inline span& operator=(span&& s){
            const_cast<T*&>(data_) = s.data_;
            size_ = s.size_;
            s.data_ = nullptr;
            s.size_ = 0;
            return *this;
        }
        
        inline const_pointer data() const noexcept {return data_;}
        inline const size_type& size() const noexcept {return size_;}
        inline const_pointer begin() const noexcept {return data_;}
        inline const_pointer cbegin() const noexcept {return data_;}
        inline const_pointer end() const noexcept {return data_ + size_;}
        inline const_pointer cend() const noexcept {return data_ + size_;}
        inline const_reference front() const noexcept {return data_[0];}
        inline const_reference back() const noexcept {return data_[size_-1];}
        inline const_reference operator[](int64_t i) const noexcept {return data_[i];}
        inline size_type size_bytes() const noexcept {return size_ * sizeof(T);}
        inline bool empty() const noexcept {return size_ == 0 || data_ == nullptr;}
        inline span first(const size_type& vals) const noexcept {return span(data_, vals)};
        inline span last(const size_type& vals) const noexcept {return span(data_ + vals, cend());}
        inline span subspan(const size_type& start, const size_value& end) const noexcept {
            return span(data_ + start, data_ + end);
        }


};

}

#endif
