//certain type traits not included in the default std namespace for c++17
#ifndef _NT_UTILS_TYPE_TRAITS_H_
#define _NT_UTILS_TYPE_TRAITS_H_

#include <type_traits>
namespace std{
template<class T>
struct remove_cvref
{
    using type = remove_cv_t<remove_reference_t<T>>;
};


template< class T >
using remove_cvref_t = typename remove_cvref<T>::type;

}

#endif //_NT_UTILS_TYPE_TRAITS_H_
