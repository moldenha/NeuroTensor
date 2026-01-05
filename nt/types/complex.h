/*
 * This is a specific NeuroTensor defined complex number
 * This has been done in order to control exactly what can and cannot be done with complex types
 * It also adds constexpr capabilities extending the functionality a little
 * Should be treated no different from std::complex for most cases
 *
*/

#ifndef NT_TYPES_COMPLEX_H__
#define NT_TYPES_COMPLEX_H__

namespace nt{
template<typename T>
class my_complex;
}

//silence depreciation warnings for certain needed headers
#ifdef _MSC_VER
#ifndef _SILENCE_CXX17_C_HEADER_DEPRECATION_WARNING
#define _SILENCE_CXX17_C_HEADER_DEPRECATION_WARNING
#endif

#ifndef _SILENCE_ALL_CXX17_DEPRECATION_WARNINGS
#define _SILENCE_ALL_CXX17_DEPRECATION_WARNINGS
#endif

#ifndef _SILENCE_NONFLOATING_COMPLEX_DEPRECATION_WARNING
#define _SILENCE_NONFLOATING_COMPLEX_DEPRECATION_WARNING
#endif

#endif

#include <tuple>
#include <type_traits>
#include <complex>
#include "../utils/type_traits.h"
#include "../utils/always_inline_macro.h"
#include "../convert/floating.h"


// specific specialization for is_complex
namespace nt::type_traits{

template<class T>
struct is_complex : false_type {};
template<class Sub>
struct is_complex<::nt::my_complex<Sub>> : true_type {};
template<class T>
inline constexpr bool is_complex_v = is_complex<T>::value;

}


namespace nt{


namespace complex_details{

template<typename T>
NT_ALWAYS_INLINE auto ensure_floating(const T& val){
    static_assert(type_traits::is_floating_point_v<type_traits::decay_t<T>>
                || type_traits::is_integral_v<type_traits::decay_t<T>>,
                  "error, complex types can input floating and integral types into floating complex");
    if constexpr (type_traits::is_floating_point_v<type_traits::decay_t<T>>){
        return val;
    }else if constexpr(type_traits::is_integral_v<type_traits::decay_t<T>>){
        return static_cast<float>(val);
    }
}

}


#define NT_MAKE_COMPLEX_OPERATOR(op)\
template<typename U, std::enable_if_t<type_traits::is_arithmetic_v<U>, bool> = true>\
inline my_complex operator op (const U& val) const&{\
    return my_complex(re op convert::floating_convert<T>(complex_details::ensure_floating(val)), im);\
}\
template<typename U, std::enable_if_t<!std::is_same_v<U, T>, bool> = true>\
inline my_complex operator op (const my_complex<U>& val) const& {\
    return my_complex(re op convert::floating_convert<T>(val.re), im op convert::floating_convert<T>(val.im));\
}\
constexpr my_complex operator op (const my_complex& val) const& {\
    return my_complex(re op val.re, im op val.im);\
}

#define NT_MAKE_COMPLEX_THIS_OPERATOR(op)\
template<typename U, std::enable_if_t<type_traits::is_arithmetic_v<U>, bool> = true>\
inline my_complex& operator op (const U& val){\
    re op convert::floating_convert<T>(complex_details::ensure_floating(val));\
    return *this;\
}\
template<typename U, std::enable_if_t<!std::is_same_v<U, T>, bool> = true>\
inline my_complex& operator op (const my_complex<U>& val){\
    re op convert::floating_convert<T>(val.re);\
    im op convert::floating_convert<T>(val.im);\
    return *this;\
}\
constexpr my_complex& operator op (const my_complex& val) {\
    re op val.re;\
    im op val.im;\
    return *this;\
}


template<typename T>
class my_complex{
    static_assert(type_traits::is_floating_point_v<type_traits::decay_t<T>>);
    T re, im;
    template <std::size_t Index, typename U>
    friend constexpr U get_complex(const my_complex<U>& obj) noexcept;

    template <std::size_t Index, typename U>
    friend constexpr U& get_complex(my_complex<U>& obj) noexcept;

    template <std::size_t Index, typename U>
    friend constexpr U&& get_complex(my_complex<U>&& obj) noexcept;

    template<typename> friend class my_complex;

public:
    using value_type = T;
    constexpr my_complex()
            :re(0), im(0)
        {;}
    constexpr my_complex(T ele)
             :re(ele), im(ele) 
        {;}
    constexpr my_complex(const T& r, const T& i)
            :re(r), im(i)
        {;}
    constexpr my_complex(const std::complex<T>& cp)
        :re(cp.real()), im(cp.imag())
        {;}

    constexpr my_complex(const my_complex&) = default;
    constexpr my_complex(my_complex&&) = default;
    
    template<typename U, std::enable_if_t<type_traits::is_floating_point_v<U>, bool> = true>
    my_complex(const U& ele)
    :re(convert::floating_convert<T>(ele)), im(convert::floating_convert<T>(ele))
    {;}
    
    template<typename U, std::enable_if_t<type_traits::is_integral_v<U>, bool> = true>
    my_complex(const U& ele)
    :re(convert::floating_convert<T>(complex_details::ensure_floating(ele))), im(convert::floating_convert<T>(complex_details::ensure_floating(ele)))
    {;}

    template<typename U, std::enable_if_t<type_traits::is_integral_v<U>, bool> = true>
    my_complex(const U& real, const U& imag)
    :re(convert::floating_convert<T>(complex_details::ensure_floating(real))), im(convert::floating_convert<T>(complex_details::ensure_floating(imag)))
    {;}

    template<typename U, std::enable_if_t<type_traits::is_floating_point_v<U>, bool> = true>
    my_complex(const U& real, const U& imag)
    :re(convert::floating_convert<T>(real)), im(convert::floating_convert<T>(imag))
    {;}

    template<typename U, std::enable_if_t<type_traits::is_floating_point_v<U> && !type_traits::is_same_v<U, T>, bool> = true>
    my_complex(const my_complex<U>& other)
    :my_complex(other.re, other.im)
    {;}

    template<typename U, std::enable_if_t<type_traits::is_floating_point_v<U> && !type_traits::is_same_v<U, T>, bool> = true>
    my_complex(my_complex<U>&& other)
    :my_complex(other.re, other.im)
    {other.re = 0; other.im = 0;}



    NT_MAKE_COMPLEX_OPERATOR(+);
    NT_MAKE_COMPLEX_OPERATOR(-);
    NT_MAKE_COMPLEX_OPERATOR(/);
    NT_MAKE_COMPLEX_OPERATOR(*);

    NT_MAKE_COMPLEX_THIS_OPERATOR(*=);
    NT_MAKE_COMPLEX_THIS_OPERATOR(+=);
    NT_MAKE_COMPLEX_THIS_OPERATOR(-=);
    NT_MAKE_COMPLEX_THIS_OPERATOR(/=);
    
    inline constexpr const T& real() const noexcept {return re;}
    inline constexpr const T& imag() const noexcept {return im;}
    inline constexpr T& real() noexcept {return re;}
    inline constexpr T& imag() noexcept {return im;}
    inline constexpr my_complex& operator++() {++re; ++im; return *this;}
    inline constexpr my_complex operator++(int){
        my_complex tmp = *this;
        ++(*this);
        return tmp;
    }

    inline constexpr bool operator==(const my_complex& c) const {return (re == c.re) && (im == c.im);}
    inline constexpr bool operator!=(const my_complex& c) const {return (re != c.re) || (im != c.im);}
    inline constexpr operator std::complex<T>() const {return std::complex<T>(re, im);}
    template<typename X>
    inline operator my_complex<X>() const {return my_complex<X>(convert::floating_convert<X>(re), convert::floating_convert<X>(im));}
    
    inline my_complex& operator=(const my_complex& other) noexcept {
        re = other.re;
        im = other.im;
        return *this;
    }

    inline my_complex& operator=(my_complex&& other) noexcept {
        re = other.re;
        im = other.im;
        other.re = 0;
        other.im = 0;
        return *this;
    }

    template<typename U>
    inline my_complex& operator=(const my_complex<U>& other){
        return *this = my_complex(other);
    }

    template<typename U>
    inline my_complex& operator=(my_complex<U>&& other){
        return *this = my_complex(std::forward<my_complex<U>>(other));
    }
    

    inline constexpr my_complex inverse() const {
        return my_complex(re == 0 ? 0 : 1.0/re, im == 0 ? 0 : 1.0/im);
    }
    inline constexpr my_complex& inverse_() {
        return *this = inverse();
    }
    

    inline constexpr bool operator<(const my_complex& c) const{
        return (re < c.re) && (im < c.im);
    }

    inline constexpr bool operator>(const my_complex& c) const{
        return (re > c.re) && (im > c.im);
    }

    inline constexpr bool operator>=(const my_complex& c) const{
        return (re >= c.re) && (im >= c.im);
    }

    inline constexpr bool operator<=(const my_complex& c) const{
        return (re <= c.re) && (im <= c.im);
    }
    inline constexpr my_complex operator-() const noexcept { return my_complex(-re, -im);}
    inline friend std::ostream& operator<<(std::ostream& os, const my_complex<T>& c){return os << c.re <<" + "<<c.im<<"i";}

};

#undef NT_MAKE_COMPLEX_OPERATOR
#undef NT_MAKE_COMPLEX_THIS_OPERATOR 


#define NT_MAKE_COMPLEX_OPERATOR(op)\
template<typename U, typename T>\
inline nt::my_complex<T> operator op (const U& ele, const nt::my_complex<T>& val){\
    return nt::my_complex<T>(ele) op val;\
}


NT_MAKE_COMPLEX_OPERATOR(+);
NT_MAKE_COMPLEX_OPERATOR(/);
NT_MAKE_COMPLEX_OPERATOR(-);
NT_MAKE_COMPLEX_OPERATOR(*);

#undef NT_MAKE_COMPLEX_OPERATOR

using complex_64 = my_complex<float>;
using complex_128 = my_complex<double>;
using complex_32 = my_complex<float16_t>;


template <std::size_t Index, typename T>
inline constexpr T get_complex(const my_complex<T>& c) noexcept {
    static_assert(Index < 2, "Index out of bounds for nt::my_complex");
    if constexpr (Index == 0) {
        return c.re;  // Access allowed due to 'friend' declaration
    } else {
        return c.im;
    }
}

template <std::size_t Index, typename T>
inline constexpr T& get_complex(my_complex<T>& c) noexcept {
    static_assert(Index < 2, "Index out of bounds for nt::my_complex");
    if constexpr (Index == 0) {
        return c.re;  // Access allowed due to 'friend' declaration
    } else {
        return c.im;
    }
}

template <std::size_t Index, typename T>
inline constexpr T&& get_complex(my_complex<T>&& c) noexcept {
    static_assert(Index < 2, "Index out of bounds for nt::my_complex");
    if constexpr (Index == 0) {
        return std::move(c.re);  // Access allowed due to 'friend' declaration
    } else {
        return std::move(c.im);
   }
}


static_assert( std::is_trivially_copy_constructible_v<my_complex<float>>
              && std::is_trivially_copy_constructible_v<my_complex<double>> 
              && std::is_trivially_copy_constructible_v<my_complex<::nt::float16_t>> ,
              "Need my_complex to be trivially copy constructible");

}




// Enable ADL for `get`
namespace std {
// Overloads of std::get for nt::my_complex
template <std::size_t Index, typename T>
inline constexpr T get(const nt::my_complex<T>& c) noexcept {
	return nt::get_complex<Index>(c);
}

template <std::size_t Index, typename T>
inline constexpr T& get(nt::my_complex<T>& c) noexcept {
	return nt::get_complex<Index>(c);
}

template <std::size_t Index, typename T>
inline constexpr T&& get(nt::my_complex<T>&& c) noexcept {
	return nt::get_complex<Index>(c);
}


template <typename T>
struct tuple_size<nt::my_complex<T>> : std::integral_constant<size_t, 2> {};

template <size_t Index, typename T>
struct tuple_element<Index, nt::my_complex<T>> {
    using type = T;
};

}  // namespace std


#endif
