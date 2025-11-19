// certain type traits not included in the default std namespace for c++17
// certain platfoms and certain compilers were found to support different versions of type traits for some reason
// So, this has everything the way that neurotensor needs it
//
// certain things added:
//  - Differentiation between lvalue reference and rvalue reference
//  - ways to hold rvalue reference and lvalue (for now use std::reference_wrapper) reference seperately
//  - remove_cvref_t <- not in all versions, guarenteed here
//  - is_in <- standard way to check if a type is contained in a parameter pack
//  - is_decay_same <- if 2 types both decayed are the same
//  - is_in_decay <- standard way to check if a type is contained in a parameter pack (but decay each type)
//  
    

#ifndef NT_UTILS_TYPE_TRAITS_H__
#define NT_UTILS_TYPE_TRAITS_H__

#include <type_traits>
#include <functional>
#include <uchar.h>

//need to make a neurotensor standard library
//where basically I re-make the type_traits into this
namespace nt::type_traits{

template<class T, T v>
struct integral_constant
{
    static constexpr T value = v;
    using value_type = T;
    using type = integral_constant; // using injected-class-name
    constexpr operator value_type() const noexcept { return value; }
    constexpr value_type operator()() const noexcept { return value; } // since c++14
};

template< bool B >
using bool_constant = integral_constant<bool, B>;

using true_type = bool_constant<true>;
using false_type = bool_constant<false>;

using nullptr_t = decltype(nullptr);

template<class T1, class T2>
struct is_same : false_type{};
template<class T>
struct is_same<T, T> : true_type{};
template<class T1, class T2>
inline constexpr bool is_same_v = is_same<T1, T2>::value;


template<class T>
struct remove_cv{
    using type = T;
};
template<class T>
struct remove_cv<const T>{
    using type = T;
};
template<class T>
struct remove_cv<volatile T>{
    using type = T;
};
template<class T>
struct remove_cv<const volatile T>{
    using type = T;
};

template<class T>
struct remove_reference{
    using type = T;
};

template<class T>
struct remove_reference<T&>{
    using type = T;
};

template<class T>
struct remove_reference<T&&>{
    using type = T;
};

template<class T>
using remove_cv_t = typename remove_cv<T>::type;
template<class T>
using remove_reference_t = typename remove_reference<T>::type;
template<class T>
struct remove_cvref
{
    using type = remove_cv_t<remove_reference_t<T>>;
};


template< class T >
using remove_cvref_t = typename remove_cvref<T>::type;

template<class T>
struct is_void : is_same<remove_cvref_t<T>, void> {};

template<class T>
inline constexpr bool is_void_v = is_void<T>::value;


template<class...>
using void_t = void;


template<class T>
struct is_null_pointer : is_same<remove_cvref_t<T>, nullptr_t> {};
template<class T>
inline constexpr bool is_null_pointer_v = is_null_pointer<T>::value;

template<class T>
struct is_pointer : false_type {};
template<class T>
struct is_pointer<T*> : true_type {};
template<class T>
struct is_pointer<const T*> : true_type {};
template<class T>
struct is_pointer<volatile T*> : true_type {};
template<class T>
struct is_pointer<const volatile T*> : true_type {};

template<class T>
inline constexpr bool is_pointer_v = is_pointer<T>::value;



// Use:
// if int is in the types <float, int, double, char>
// it would be:
// is_in<int, float, int, double, char>::value
//
template<class T, class... Ts>
struct is_in : bool_constant<(is_same_v<T, Ts> || ...)> {};

template<class T, typename... Rest>
inline constexpr bool is_in_v = is_in<T, Rest...>::value;

template<class T>
struct is_integral : is_in<remove_cv_t<T>, bool, char, char16_t, char32_t, wchar_t, short, int, long, long long, int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t> {};

template<class T>
inline constexpr bool is_integral_v = is_integral<T>::value;

template<class T>
struct is_floating_point : is_in<remove_cv_t<T>, float, double, long double> {};
template<class T>
inline constexpr bool is_floating_point_v = is_floating_point<T>::value;


template<class T>
struct is_array : false_type {};
template<class T>
struct is_array<T[]> : true_type {};
template<class T, std::size_t N>
struct is_array<T[N]> : true_type {};
template< class T >
constexpr bool is_array_v = is_array<T>::value;


// to implement this natively, there are a lot of macros involved that are specific to each platform
// for that reason, the c++ type_traits is_enum is used here, and should work fine after c++11 
// and neurotensor only supports c++17 and up
// So, for that reason, (unless it needs to be changed) this will just follow the standard c++ is_enum
template<class T>
struct is_enum : std::is_enum<T> {};
template<class T>
constexpr bool is_enum_v = is_enum<T>::value;


// to implement this natively, there are a lot of macros involved that are specific to each platform
// for that reason, the c++ type_traits is_union is used here, and should work fine after c++11 
// and neurotensor only supports c++17 and up
// So, for that reason, (unless it needs to be changed) this will just follow the standard c++ is_enum
template<typename T>
struct is_union : std::is_union<T> {};
template<typename T>
constexpr bool is_union_v = is_union<T>::value;


namespace is_class_detail{

template<class T>
integral_constant<bool, !is_union_v<T>> test(int T::*);

template<class>
false_type test(...);

}

template<class T>
struct is_class : decltype(is_class_detail::test<T>(nullptr)) {};
template<class T>
inline constexpr bool is_class_v = is_class<T>::value;


// pretty long and complicated implementation
// may do it myself in future
// if more specializations needed, will add
template<class T>
struct is_function : std::is_function<T>{};
template<class T>
inline constexpr bool is_function_v = is_function<T>::value;


template<class T>
struct is_lvalue_reference : false_type {};
template<class T>
struct is_lvalue_reference<T&> : true_type {};
template<class T>
struct is_lvalue_reference<const T&> : true_type {};
template<class T>
struct is_lvalue_reference<volatile T&> : true_type {};
template<class T>
struct is_lvalue_reference<const volatile T&> : true_type {};
template<class T>
inline constexpr bool is_lvalue_refernece_v = is_lvalue_reference<T>::value;

template<class T>
struct is_cv_lvalue_reference : false_type {};
template<class T>
struct is_cv_lvalue_reference<const T&> : true_type {};
template<class T>
struct is_cv_lvalue_reference<volatile T&> : true_type {};
template<class T>
struct is_cv_lvalue_reference<const volatile T&> : true_type {};
template<class T>
inline constexpr bool is_cv_lvalue_refernece_v = is_cv_lvalue_reference<T>::value;

template<class T>
struct is_rvalue_reference : false_type {};
template<class T>
struct is_rvalue_reference<T&&> : true_type {};
template<class T>
struct is_rvalue_reference<const T&&> : true_type {};
template<class T>
struct is_rvalue_reference<volatile T&&> : true_type {};
template<class T>
struct is_rvalue_reference<const volatile T&&> : true_type {};
template<class T>
inline constexpr bool is_rvalue_refernece_v = is_rvalue_reference<T>::value;

template<class T>
struct is_cv_rvalue_reference : false_type {};
template<class T>
struct is_cv_rvalue_reference<const T&&> : true_type {};
template<class T>
struct is_cv_rvalue_reference<volatile T&&> : true_type {};
template<class T>
struct is_cv_rvalue_reference<const volatile T&&> : true_type {};
template<class T>
inline constexpr bool is_cv_rvalue_refernece_v = is_cv_rvalue_reference<T>::value;

template<class T>
struct is_member_pointer_helper : false_type {};
template<class T, class U>
struct is_member_pointer_helper<T U::*> : true_type {};
template<class T>
struct is_member_pointer : is_member_pointer_helper<remove_cv_t<T>> {};
template<class T>
inline constexpr bool is_member_pointer_v = is_member_pointer<T>::value;

template<class T>
struct is_member_function_pointer_helper : false_type{};
template<class T, class U>
struct is_member_function_pointer_helper<T U::*> : is_function<T> {};
template<class T>
struct is_member_function_pointer : 
    is_member_function_pointer_helper<remove_cv_t<T>> {};
template<class T>
inline constexpr bool is_member_function_pointer_v = is_member_function_pointer<T>::value;

template<class T>
struct is_member_object_pointer : 
                        integral_constant<
                            bool,
                            is_member_pointer_v<T> && 
                            !is_member_function_pointer_v<T>
                        > {};
template<class T>
inline constexpr bool is_member_object_pointer_v = is_member_object_pointer<T>::value;


#define NT_MULT_CONDITION_TYPE_(name, conditions)\
template<class T>\
struct name : integral_constant<bool,\
                                    conditions> {};\
template<class T>\
inline constexpr bool name##_v = name<T>::value;


#define NT_IS_SAME_V_MACRO_(a, b) is_same_v<a, b>

NT_MULT_CONDITION_TYPE_(is_arithmetic, is_integral_v<T> || is_floating_point_v<T>);
NT_MULT_CONDITION_TYPE_(is_fundamental, is_arithmetic_v<T> || is_void_v<T> || NT_IS_SAME_V_MACRO_(nullptr_t, remove_cv_t<T>));
NT_MULT_CONDITION_TYPE_(is_scalar,
                            is_arithmetic_v<T> || is_enum_v<T> || is_pointer_v<T> || is_member_pointer_v<T>
                            || is_null_pointer_v<T>);

NT_MULT_CONDITION_TYPE_(is_object,
                        is_scalar_v<T> || is_array_v<T> || is_union_v<T> || is_class_v<T>);

NT_MULT_CONDITION_TYPE_(is_compound, !is_fundamental_v<T>);

template<class T>
struct is_reference_helper : false_type{};
template<class T>
struct is_reference_helper<T&> : true_type{};
template<class T>
struct is_reference_helper<T&&> : true_type{};
template<class T>
struct is_reference : is_reference_helper<remove_cv_t<T>> {};
template<class T>
inline constexpr bool is_reference_v = is_reference<T>::value;


template<class T> struct is_const : false_type {};
template<class T> struct is_const<const T> : true_type {};
template<class T> inline constexpr bool is_const_v = is_const<T>::value;

template<class T> struct is_volatile : false_type {};
template<class T> struct is_volatile<volatile T> : true_type {};
template<class T> inline constexpr bool is_volatile_v = is_volatile<T>::value;


template<bool B, typename T, typename F>
struct conditional{ using type = T; };
template<typename T, typename F>
struct conditional<false, T, F> { using type = F; };
template<bool B, typename T, typename F>
using conditional_t = typename conditional<B, T, F>::type;

template<class T>
struct type_identity { using type = T; };
template<class T>
using type_identity_t = typename type_identity<T>::type;

namespace add_pointer_detail{

template<class T>
auto try_add_pointer(int)
  -> type_identity<remove_reference_t<T>*>; // usual case

template<class T>
auto try_add_pointer(...)
  -> type_identity<T>; // unusual case (cannot form remove_reference<T>::type*)

}

template<class T> struct add_pointer : decltype(add_pointer_detail::try_add_pointer<T>(0)) {};
template<class T>
using add_pointer_t = typename add_pointer<T>::type;



template<class T>
struct remove_extent { using type = T; };
 
template<class T>
struct remove_extent<T[]> { using type = T; };
 
template<class T, std::size_t N>
struct remove_extent<T[N]> { using type = T; };

template<class T>
using remove_extent_t = typename remove_extent<T>::type;

template<class T>
struct decay {
private:
    using U = remove_reference_t<T>;
public:
    using type = conditional_t< 
        is_array_v<U>,
        add_pointer_t<remove_extent_t<U>>,
        conditional_t< 
            is_function_v<U>,
            add_pointer_t<U>,
            remove_cv_t<U>
        >
    >;
};

template<class T>
using decay_t = typename decay<T>::type;

template<class A, class B>
struct is_decay_same : is_same<decay_t<A>, decay_t<B>> {};

template<class A, class B>
inline constexpr bool is_decay_same_v = is_decay_same<A, B>::value;

template<class T, class... Ts>
struct is_decay_in : bool_constant<(is_decay_same_v<T, Ts> || ...)> {};

template<class T, typename... Rest>
inline constexpr bool is_decay_in_v = is_decay_in<T, Rest...>::value;

// make has tytpe
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

// this is basically std::reference_wrapper, as it does take an l-value reference
// look at std::reference_wrapper for future implementation
// template<typename T>
// struct lvalue_wrapper{
//     T* t;
//     explicit lvalue_wrapper(T& t):t(&t) {}
//     template<typename... U> T& operator()(U&& ...){
//         return *t;
//     }

// };

template<class T> 
struct unwrap_reference{
    using type = T;
};

template<class U>
struct unwrap_reference<rvalue_wrapper<U>>{
    using type = U;
};

}

namespace std{
template<typename T>
struct is_bind_expression<::nt::type_traits::rvalue_wrapper<T> > : std::true_type {};
}

#undef NT_MULT_CONDITION_TYPE_
#undef NT_IS_SAME_V_MACRO_ 
#endif //_NT_UTILS_TYPE_TRAITS_H_
