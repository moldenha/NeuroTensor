#ifndef _NT_REFLECT_CUSTOM_ITERATOR_MAP_HPP_
#define _NT_REFLECT_CUSTOM_ITERATOR_MAP_HPP_

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

class custom_any_map{
	std::unordered_map<std::type_index, std::map<std::string, std::any> > references;

	template<typename T>
	inline void add_reference(const std::string& name, const T& ref){
		references[std::type_index(typeid(T))].insert({name, std::ref(const_cast<T&>(ref))});
	}
	template<typename T, typename... Args>
	inline void add_reference(const std::string& name, const T& ref, const Args&... args){
		add_reference(name, ref);
		(add_reference(args), ...);
	}

	template<typename T>
	inline void add_references(std::vector<std::string>::const_iterator& name, T& ref){
		references[std::type_index(typeid(T))].insert({*name, std::ref(ref)});
		++name;
	}
	template<typename T, typename... Args>
	inline void add_references(std::vector<std::string>::const_iterator& name, T& ref, Args&... args){
		add_references(name, ref);
		(add_references(name, args), ...);
	}

	template<std::size_t I = 0, typename... Args>
	inline void add_tuple(std::vector<std::string>& names, std::tuple<Args...>& t){
		if constexpr (I < std::tuple_size_v<std::tuple<Args...>>){
			using type = std::remove_reference_t<std::remove_cv_t<std::tuple_element_t<I, std::tuple<Args...>>>>;
			references[std::type_index(typeid(type))].insert({
					names[I], std::ref(const_cast<type&>(std::get<I>(t)))
					});
			add_tuple<I+1>(names, t);
		}
	}
public:
    explicit custom_any_map() {;}

    template<typename... Args>
    explicit custom_any_map(std::vector<std::string> names, std::tuple<Args...>& t){
        assert(names.size() == std::tuple_size_v<std::tuple<Args...>>);
        add_tuple(names, t);
    }
    template<typename... Args>
    explicit custom_any_map(std::vector<std::string> names, Args&... args){
        assert(sizeof...(Args) == names.size());
        auto begin = names.cbegin();
        add_references(begin, args...);
    }

    inline size_t size() const {return references.size();}

    inline std::map<std::string, std::any>& get_references(std::type_index index) noexcept {return references[index];}
    inline std::vector<std::type_index> keys() const {
        std::vector<std::type_index> out;
        out.reserve(references.size());
        for(const auto& [key, map] : references)
            out.push_back(key);
        return std::move(out);
    }
};

template<typename T>
inline std::map<std::string, std::reference_wrapper<T> > anyToRefMap(std::map<std::string, std::any>& vec) noexcept {
	std::map<std::string, std::reference_wrapper<T> > outp;
	for(auto& [name, ref] : vec){
		outp.insert({name, std::any_cast<std::reference_wrapper<T> >(ref)});
	}
	return std::move(outp);
}

template <typename T>
class custom_typed_map {
    std::map<std::string, std::reference_wrapper<T>> references;

    // Private helper function to add a reference
    inline void add_reference(const std::string& name, const T& ref) {
        references.insert({name, std::ref(const_cast<T&>(ref))});
    }

    // Template function to add multiple references
    template<typename... Args>
    inline void add_reference(const std::string& name, const T& ref, const Args&... args) {
        add_reference(name, ref);
        (add_reference(args), ...);
    }

    // Template function to add tuple elements as references
    template<std::size_t I = 0, typename... Args>
    inline void add_tuple(std::vector<std::string>& names, std::tuple<Args...>& t) {
	/* assert(names.size() == sizeof(Args...)); */
        if constexpr (I < std::tuple_size_v<std::tuple<Args...>>) {
            using type = std::remove_reference_t<std::remove_cv_t<std::tuple_element_t<I, std::tuple<Args...>>>>;
            if constexpr (std::is_same_v<type, T>) {
                references.insert({names[I], std::ref(const_cast<T&>(std::get<I>(t)))});
            }
            add_tuple<I + 1>(names, t);
        }
    }

public:
    // Default constructor
    explicit custom_typed_map() = default;

    // Constructor taking a tuple
    template<typename... Args>
    explicit custom_typed_map(std::vector<std::string> names, std::tuple<Args...>& t) {
	assert(names.size() == std::tuple_size_v<std::tuple<Args...>>);
        /* references.reserve(std::tuple_size_v<std::tuple<Args...>>); */
        add_tuple(names, t);
    }
    explicit custom_typed_map(custom_any_map any_it)
	    :references(anyToRefMap<T>(any_it.get_references(std::type_index(typeid(T)))))
    {}
    // Iterator class for the custom_typed_iterator
    class Iterator {
    public:
        explicit Iterator(typename std::map<std::string, std::reference_wrapper<T>>::iterator it) : _it(it) {}
        
        // Dereference operator
        inline std::pair<const std::string&, T&> operator*() { return std::pair<const std::string&, T&>(_it->first, _it->second.get()); }
        
        // Pre-increment operator
        inline Iterator& operator++() { ++_it; return *this; }
        
        // Inequality operator
        inline bool operator!=(const Iterator& other) const { return _it != other._it; }

    private:
        typename std::map<std::string, std::reference_wrapper<T>>::iterator _it;
    };
    
    // Iterator class for the custom_typed_iterator
    class ConstIterator {
    public:
        explicit ConstIterator(typename std::map<std::string, std::reference_wrapper<T>>::const_iterator it) : _it(it) {}
        
        // Dereference operator
        inline std::pair<const std::string&, const T&> operator*() { return std::pair<const std::string&, const T&>(_it->first, _it->second.get()); }
        
        // Pre-increment operator
        inline ConstIterator& operator++() { ++_it; return *this; }
        
        // Inequality operator
        inline bool operator!=(const ConstIterator& other) const { return _it != other._it; }

    private:
        typename std::map<std::string, std::reference_wrapper<T>>::const_iterator _it;
    };
    
    inline size_t size() const {
	return references.size();
    }
    // Begin iterator
    inline Iterator begin() noexcept {
        return Iterator(references.begin());
    }

    // End iterator
    inline Iterator end() noexcept {
        return Iterator(references.end());
    }
    // Begin iterator
    inline ConstIterator begin() const noexcept {
        return ConstIterator(references.cbegin());
    }

    // End iterator
    inline ConstIterator end() const noexcept {
        return ConstIterator(references.cend());
    }
    custom_typed_map(custom_typed_map&& it)
	    :references(std::move(it.references))
    {}

    custom_typed_map(const custom_typed_map& it)
	    :references(it.references)
    {}

    inline void extend(custom_typed_map&& it){
	/* references.reserve(it.references.size()); */
	references.insert(it.references.begin(), it.references.end());
    }
    
    inline std::map<std::string, std::reference_wrapper<T>>& get_references() {return references;}

    inline void extend_unique(custom_typed_map&& it, std::string to_add){
	for(auto ele : it){
		auto& [name, element] = ele;
		std::string key = to_add + name;
		uint32_t index = 0;
		while(references.find(key) != references.end()){
			++index;
			key = to_add + name + "("+std::to_string(index)+")";
		}
		references.insert({key, element});
	}
    }

    inline void extend(std::map<std::string, std::reference_wrapper<T> >& map){
	    references.insert(map.begin(), map.end());
    }

};

template<typename T>
inline std::ostream& operator<<(std::ostream& stream, const custom_typed_map<T>& map) noexcept {
	for(auto ele : map){
		stream << "{" << ele.first << ',' << ele.second <<"} ";
	}
	return stream;
}

}}} //nt::reflect::detail::

#endif //_NT_REFLECT_CUSTOM_ITERATOR_MAP_HPP_
