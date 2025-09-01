//certain type traits not included in the default std namespace for c++17
#ifndef NT_UTILS_TYPE_TRAITS_H__
#define NT_UTILS_TYPE_TRAITS_H__

#include <type_traits>
#include <functional>

//need to make a neurotensor standard library
//where basically I re-make the type_traits into this
namespace nt::type_traits{
template<class T>
struct remove_cvref
{
    using type = std::remove_cv_t<std::remove_reference_t<T>>;
};


template< class T >
using remove_cvref_t = typename remove_cvref<T>::type;


//this is like std::reference_wrapper
//but specifically for r values
template<typename T>
struct rvalue_wrapper{
	T t;
	explicit rvalue_wrapper(T &&t):t(std::forward<T>(t)) {}
	template<typename... U> T&& operator()(U&& ...){
		return std::forward<T>(t);
	}
};

template<typename T>
rvalue_wrapper<T> rvref(T&& val){return rvalue_wrapper<T>(std::forward(val));}
template<typename T>
rvalue_wrapper<const T> crvref(const T&& val){return rvalue_wrapper<const T>(std::forward(val));}



}

namespace std{
template<typename T>
struct is_bind_expression<::nt::type_traits::rvalue_wrapper<T> > : std::true_type {};
}
#endif //_NT_UTILS_TYPE_TRAITS_H_
