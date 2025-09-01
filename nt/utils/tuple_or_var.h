#ifndef NT_UTILS_TUPLE_OR_VAR_H__
#define NT_UTILS_TUPLE_OR_VAR_H__

#include <array>
#include <vector>
#include "utils.h"

namespace nt::utils{


template<typename T, std::size_t Size>
class tuple_or_var{
public:
    using tuple_t = typename generate_tuple_type_n<T, Size>::type;
private:
    int16_t _index;
    tuple_t tuple;
    template<std::size_t... Is>
    inline static tuple_t repeat_element_impl(T val, std::index_sequence<Is...>) {
        return tuple_t(std::make_tuple(((void)Is, val)...));
    }

    template<typename T2, std::size_t... Is>
    inline static tuple_t store_initializer_list_impl(T2 begin, std::index_sequence<Is...>){
        return tuple_t(std::make_tuple(static_cast<T>(begin[Is])...));
    }
    template<typename T2>
    inline static tuple_t store_initializer_list(std::initializer_list<T2> ls){
        throw_exception(ls.size() == Size, "Expected initializer list for tuple to have $ variables, instead got $", Size, ls.size());
        return store_initializer_list_impl(ls.begin(), std::make_index_sequence<Size>{});
    }
	public:
        template<typename T2, std::enable_if_t<std::is_convertible_v<T2, T>, bool> = true>
		tuple_or_var(const T2 val) : tuple(repeat_element_impl(static_cast<T>(val), std::make_index_sequence<Size>{})), _index(0) {} 
	    explicit tuple_or_var(generate_tuple_type_n<T, Size> _tuple): tuple(_tuple), _index(1) {}
        template<typename T2>
        tuple_or_var(std::initializer_list<T2> ls) : tuple(store_initializer_list(ls)), _index(1) {}        
        template<typename T2, std::enable_if_t<std::is_convertible_v<T2, T>, bool> = true>
        inline tuple_or_var& operator=(T2 x){
            this->_index = 0;
            tuple = repeat_element_impl(static_cast<T>(x), std::make_index_sequence<Size>{});
            return *this;
        }
        template<typename T2>
        inline tuple_or_var& operator=(std::initializer_list<T2> ls){
            this->_index = 1;
            tuple = store_initializer_list(ls);
            return *this;
        }
        template<std::size_t J>
        T& get(){return std::get<J>(tuple);}
        template<std::size_t J>
        const T& get() const {return std::get<J>(tuple);}
        
        const tuple_t& get_tup() const {return tuple;}
        const int16_t& index() { return _index;}
        
		// bool operator==(const int64_t x) const ;
		// bool operator!=(const int64_t x) const ;
		// bool operator > (const int64_t x) const ;
		// bool operator < (const int64_t x) const ;
		// bool operator >= (const int64_t x) const ;
		// bool operator <= (const int64_t x) const ;
};

}

// Inject std::get support
namespace std {
template<typename T, std::size_t Size>
struct tuple_size<::nt::utils::tuple_or_var<T, Size>> : std::integral_constant<std::size_t, Size> {};

template<std::size_t I, typename T, std::size_t Size>
struct tuple_element<I, ::nt::utils::tuple_or_var<T, Size>> {
    using type = T;
};

template<std::size_t I, typename T, std::size_t Size>
T& get(::nt::utils::tuple_or_var<T, Size>& v) {
    return v.template get<I>();
}

template<std::size_t I, typename T, std::size_t Size>
const T& get(const ::nt::utils::tuple_or_var<T, Size>& v) {
    return v.template get<I>();
}

template<std::size_t I, typename T, std::size_t Size>
T&& get(::nt::utils::tuple_or_var<T, Size>&& v) {
    return std::move(v.template get<I>());
}


}


#endif
