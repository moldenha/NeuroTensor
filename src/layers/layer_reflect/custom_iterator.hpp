#ifndef _NT_REFLECT_CUSTOM_ITERATOR_HPP_
#define _NT_REFLECT_CUSTOM_ITERATOR_HPP_

#include <vector>
#include <unordered_set>
#include <typeindex>
#include <any>
#include <tuple>
#include <utility>
#include <type_traits>

namespace nt{
namespace reflect{
namespace detail{

class custom_any_iterator{
	std::unordered_map<std::type_index, std::vector<std::any> > references;

	template<typename T>
	inline void add_reference(const T& ref){
		references[std::type_index(typeid(T))].push_back(std::ref(const_cast<T&>(ref)));
	}
	template<typename T, typename... Args>
	inline void add_reference(const T& ref, const Args&... args){
		add_reference(ref);
		(add_reference(args), ...);
	}

	template<std::size_t I = 0, typename... Args>
	inline void add_tuple(std::tuple<Args...>& t){
		if constexpr (I < std::tuple_size_v<std::tuple<Args...>>){
			using type = std::remove_reference_t<std::remove_cv_t<std::tuple_element_t<I, std::tuple<Args...>>>>;
			references[std::type_index(typeid(type))].push_back(std::ref(const_cast<type&>(std::get<I>(t))));
			add_tuple<I+1>(t);
		}
	}

	template<typename T>
	inline void add_a_reference(T& ref){
		references[std::type_index(typeid(T))].push_back(std::ref(ref));
	}
	template<typename T, typename... Args>
	inline void add_references(T& ref, Args&... args){
		add_a_reference(ref);
		(add_references(args), ...);
	}
	public:
		template<typename... Args>
		explicit custom_any_iterator(std::tuple<Args...>& t){
			references.reserve(std::tuple_size_v<std::tuple<Args...>>);
			add_tuple(t);
		}

		template<typename... Args>
		explicit custom_any_iterator(Args&... args){
			references.reserve(sizeof...(Args));
			add_references(args...);
			
		}
		
		explicit custom_any_iterator(){;}

		inline std::vector<std::any>& get_references(std::type_index index) noexcept {return references[index];}
};


template<typename T>
inline std::vector<std::reference_wrapper<T> > anyToRefVec(std::vector<std::any>& vec){
	std::vector<std::reference_wrapper<T> > outp;
	outp.reserve(vec.size());
	for(auto& element : vec){
		outp.push_back(std::any_cast<std::reference_wrapper<T> >(element));
	}
	return std::move(outp);
}



template<typename T>
class custom_typed_iterator {
    std::vector<std::reference_wrapper<T>> references;

    // Private helper function to add a reference
    inline void add_reference(const T& ref) {
        references.push_back(std::ref(const_cast<T&>(ref)));
    }

    // Template function to add multiple references
    template<typename... Args>
    inline void add_reference(const T& ref, const Args&... args) {
        add_reference(ref);
        (add_reference(args), ...);
    }

    // Template function to add tuple elements as references
    template<std::size_t I = 0, typename... Args>
    inline void add_tuple(std::tuple<Args...>& t) {
        if constexpr (I < std::tuple_size_v<std::tuple<Args...>>) {
            using type = std::remove_reference_t<std::remove_cv_t<std::tuple_element_t<I, std::tuple<Args...>>>>;
            if constexpr (std::is_same_v<type, T>) {
                references.push_back(std::ref(const_cast<T&>(std::get<I>(t))));
            }
            add_tuple<I + 1>(t);
        }
    }

public:
    // Default constructor
    explicit custom_typed_iterator() = default;

    // Constructor taking a tuple
    template<typename... Args>
    explicit custom_typed_iterator(std::tuple<Args...>& t) {
        references.reserve(std::tuple_size_v<std::tuple<Args...>>);
        add_tuple(t);
    }

    custom_typed_iterator(custom_typed_iterator&& it)
	    :references(std::move(it.references))
    {}

    custom_typed_iterator(const custom_typed_iterator& it)
	    :references(it.references)
    {}
	
    custom_typed_iterator(custom_any_iterator any_it)
	    :references(anyToRefVec<T>(any_it.get_references(std::type_index(typeid(T)))))
    {}
    // Iterator class for the custom_typed_iterator
    class Iterator {
    public:
        explicit Iterator(typename std::vector<std::reference_wrapper<T>>::iterator it) : _it(it) {}
        
        // Dereference operator
        inline T& operator*() { return _it->get(); }
        
        // Pre-increment operator
        inline Iterator& operator++() { ++_it; return *this; }
        
        // Inequality operator
        inline bool operator!=(const Iterator& other) const { return _it != other._it; }

    private:
        typename std::vector<std::reference_wrapper<T>>::iterator _it;
    };
    
    // Iterator class for the custom_typed_iterator
    class CIterator {
    public:
        explicit CIterator(typename std::vector<std::reference_wrapper<T>>::const_iterator it) : _it(it) {}
        
        // Dereference operator
        inline const T& operator*() { return _it->get(); }
        
        // Pre-increment operator
        inline CIterator& operator++() { ++_it; return *this; }
        
        // Inequality operator
        inline bool operator!=(const CIterator& other) const { return _it != other._it; }

    private:
        typename std::vector<std::reference_wrapper<T>>::const_iterator _it;
    };


    // Begin iterator
    inline Iterator begin() {
        return Iterator(references.begin());
    }

    // End iterator
    inline Iterator end() {
        return Iterator(references.end());
    }

    // Begin iterator
    inline CIterator begin() const noexcept {
        return CIterator(references.cbegin());
    }

    // End iterator
    inline CIterator end() const noexcept {
        return CIterator(references.cend());
    }

    inline void extend(custom_typed_iterator&& it){
	references.reserve(it.references.size());
	references.insert(references.end(), it.references.begin(), it.references.end());
    }
};

template<typename T>
inline std::ostream& operator<<(std::ostream& os, const custom_typed_iterator<T>& it){
	for(const auto& var : it){
		os << var << ',';
	}
	return os;
}



}}} //nt::reflect::detail::

#endif //_NT_REFLECT_CUSTOM_ITERATOR_HPP_
