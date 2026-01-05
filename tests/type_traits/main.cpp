#include "../../nt/utils/type_traits.h"
#include <math.h>

#define assert_true(val) static_assert(val, "Expected true");
#define assert_false(val) static_assert(!val, "Expected false");
#define assert_same_type(vala, valb) static_assert(is_same_v<vala, valb>, "Expected same type");

void foo();

static_assert
(
    nt::type_traits::is_void_v<void> == true and
    nt::type_traits::is_void_v<const void> == true and
    nt::type_traits::is_void_v<volatile void> == true and
    nt::type_traits::is_void_v<const volatile void> == true and
    nt::type_traits::is_void_v<nt::type_traits::void_t<>> == true and
    nt::type_traits::is_void_v<void*> == false and
    nt::type_traits::is_void_v<int> == false and
    nt::type_traits::is_void_v<decltype(foo)> == false and
    nt::type_traits::is_void_v<nt::type_traits::is_void<void>> == false
);

static_assert(nt::type_traits::is_null_pointer_v<decltype(nullptr)>);
static_assert(!nt::type_traits::is_null_pointer_v<int*>);
static_assert(!nt::type_traits::is_pointer_v<decltype(nullptr)>);
static_assert(nt::type_traits::is_pointer_v<int*>);

static_assert(nt::type_traits::is_in_v<int, float, int, double, char>);  
static_assert(!nt::type_traits::is_in_v<int, float, double, char>);  


class A {};
static_assert(!nt::type_traits::is_floating_point_v<A>);
 
static_assert(nt::type_traits::is_floating_point_v<float>);
static_assert(!nt::type_traits::is_floating_point_v<float&>);
static_assert(nt::type_traits::is_floating_point_v<double>);
static_assert(!nt::type_traits::is_floating_point_v<double&>);
static_assert(!nt::type_traits::is_floating_point_v<int>);


static_assert
(
    nt::type_traits::is_integral_v<float> == false &&
    nt::type_traits::is_integral_v<int*> == false &&
    nt::type_traits::is_integral_v<int> == true &&
    nt::type_traits::is_integral_v<const int> == true &&
    nt::type_traits::is_integral_v<bool> == true &&
    nt::type_traits::is_integral_v<char> == true
);
 
static_assert(nt::type_traits::is_integral_v<A> == false);
 
struct B { int x:4; };
static_assert(nt::type_traits::is_integral_v<B> == false);
using BF = decltype(B::x); // bit-field's type
static_assert(nt::type_traits::is_integral_v<BF> == true);
 
enum E : int {};
static_assert(nt::type_traits::is_integral_v<E> == false);
 
template <class T>
constexpr T same(T i)
{
    static_assert(nt::type_traits::is_integral<T>::value, "Integral required.");
    return i;
}
static_assert(same('"') == 042);
 
static_assert(nt::type_traits::is_array<A>::value == false);
static_assert(nt::type_traits::is_array<A[]>::value == true);
static_assert(nt::type_traits::is_array<A[3]>::value == true);
 
static_assert(nt::type_traits::is_array<float>::value == false);
static_assert(nt::type_traits::is_array<int>::value == false);
static_assert(nt::type_traits::is_array<int[]>::value == true);
static_assert(nt::type_traits::is_array<int[3]>::value == true);
static_assert(nt::type_traits::is_array<std::array<int, 3>>::value == false);

static_assert(nt::type_traits::is_enum<E>::value == true);
static_assert(nt::type_traits::is_enum_v<E> == true);
static_assert(nt::type_traits::is_enum_v<A> == false);

static_assert(nt::type_traits::is_class_v<A>);
static_assert(!nt::type_traits::is_class_v<E>);
static_assert(nt::type_traits::is_class_v<B>);
static_assert(nt::type_traits::is_class_v<struct EXAMPLE_IS_CLASS_STRUCT>, "incomplete struct");
static_assert(nt::type_traits::is_class_v<class EXAMPLE_IS_CLASS_CLASS>, "incomplete struct");


void mem_obj_test(){
class C {};
static_assert(nt::type_traits::is_member_object_pointer_v<int(C::*)>);
static_assert(!nt::type_traits::is_member_object_pointer_v<int(C::*)()>);
}

void mem_func_test(){
class C { public: void member() {} };
static_assert(nt::type_traits::is_member_function_pointer_v<decltype(&C::member)>);
}

void mem_is_test(){
    static_assert(!nt::type_traits::is_member_pointer_v<int*>);
    struct C{
        int i{42};
        int foo(){return 0xF00;}
    };
    using mem_int_ptr_t = int C::*;
    using mem_fun_ptr_t = int (C::*)();
    static_assert(nt::type_traits::is_member_pointer_v<mem_int_ptr_t>);
    static_assert(nt::type_traits::is_member_pointer_v<mem_fun_ptr_t>);
}
template<typename TypeA, typename TypeB>
constexpr bool is_decay_equ = nt::type_traits::is_same_v<nt::type_traits::decay_t<TypeA>, TypeB>;

void decay_test(){
// decay_t test


     
    static_assert
    (
        is_decay_equ<int, int> &&
        ! is_decay_equ<int, float> &&
        is_decay_equ<int&, int> &&
        is_decay_equ<int&&, int> &&
        is_decay_equ<const int&, int> &&
        is_decay_equ<int[2], int*> &&
        ! is_decay_equ<int[4][2], int*> &&
        ! is_decay_equ<int[4][2], int**> &&
        is_decay_equ<int[4][2], int(*)[2]> &&
        is_decay_equ<int(int), int(*)(int)>
    );
    
    
    static_assert
    (
        nt::type_traits::is_decay_same_v<int, int> &&
        ! nt::type_traits::is_decay_same_v<int, float> &&
        nt::type_traits::is_decay_same_v<int&, int> &&
        nt::type_traits::is_decay_same_v<int&&, int> &&
        nt::type_traits::is_decay_same_v<const int&, int> &&
        nt::type_traits::is_decay_same_v<int[2], int*> &&
        ! nt::type_traits::is_decay_same_v<int[4][2], int*> &&
        ! nt::type_traits::is_decay_same_v<int[4][2], int**> &&
        nt::type_traits::is_decay_same_v<int[4][2], int(*)[2]> &&
        nt::type_traits::is_decay_same_v<int(int), int(*)(int)>
    );

    static_assert(
        nt::type_traits::is_decay_in_v<int, int&, float&, float, int*>
    );

    static_assert(
        nt::type_traits::is_decay_in_v<int, const volatile int&&, float, const double>
    );

    static_assert(
        nt::type_traits::is_decay_in_v<int, const volatile int&&, float, const double>
    );

    static_assert(
        !nt::type_traits::is_decay_in_v<int, int[3][2], float, const double>
    );

}


#ifndef NT_GET_DEFINE_FLOATING_DTYPES_ 
#define NT_GET_DEFINE_FLOATING_DTYPES_(NT_CUR_FUNC__, ...)\
    NT_CUR_FUNC__(float, Float32, Float)\
    NT_CUR_FUNC__(double, Float64, Double)\

#define NT_GET_DEFINE_FLOATING_DTYPES_OTHER_(NT_CUR_FUNC__, ...)\
    NT_CUR_FUNC__(float, Float32, Float, __VA_ARGS__)\
    NT_CUR_FUNC__(double, Float64, Double, __VA_ARGS__)\


#endif
 
#ifndef NT_GET_DEFINE_SIGNED_INTEGER_DTYPES_ 

#define NT_GET_DEFINE_SIGNED_INTEGER_DTYPES_(NT_CUR_FUNC__)\
    NT_CUR_FUNC__(int8_t, int8, Char)\
    NT_CUR_FUNC__(int16_t, int16, Short)\
    NT_CUR_FUNC__(int32_t, int32, Integer)\
    NT_CUR_FUNC__(int64_t, int64, Long)\


#define NT_GET_DEFINE_SIGNED_INTEGER_DTYPES_OTHER_(NT_CUR_FUNC__, ...)\
    NT_CUR_FUNC__(int8_t, int8, Char, __VA_ARGS__)\
    NT_CUR_FUNC__(int16_t, int16, Short, __VA_ARGS__)\
    NT_CUR_FUNC__(int32_t, int32, Integer, __VA_ARGS__)\
    NT_CUR_FUNC__(int64_t, int64, Long, __VA_ARGS__)\

#endif


namespace nt::math{

template<typename T>
T sqrt(const T& val){return std::sqrt(val);}

template<typename T, typename U>
T pow(const T& a, const U& b){
    return std::pow(a, b);
}

}


#define NT_CHECK_SQRT_FN_(type, name_a, name_b)\
    static_assert(std::is_invocable_r_v<type, decltype(static_cast<type (*)(const type&)>(::nt::math::sqrt)), const type&>, \
                  "Error, sqrt with type " #type " is not invocable!"); \

NT_GET_DEFINE_FLOATING_DTYPES_(NT_CHECK_SQRT_FN_); 
NT_GET_DEFINE_SIGNED_INTEGER_DTYPES_(NT_CHECK_SQRT_FN_); 

#define NT_CHECK_POW_FN_(type, name_a, name_b)\
    static_assert(std::is_invocable_r_v<type, decltype(static_cast<type (*)(const type&, const type&)>(::nt::math::pow)),  \
        const type&, const type&>, \
        "Error, pow with type " #type " is not invocable!"); \

#define NT_CHECK_POW_FN_RETURN_INTEGER_HELPER_(type, name_a, name_b, original_type)\
    static_assert(std::is_invocable_r_v<original_type, decltype(static_cast<original_type (*)(const original_type&, const type&)>(::nt::math::pow)),  \
        const type&, const type&>, \
        "Error, pow with type " #original_type " and integer " #type "is not invocable!"); \

#define NT_CHECK_POW_FN_RETURN_INTEGER_(type, name_a, name_b)\
    NT_GET_DEFINE_SIGNED_INTEGER_DTYPES_OTHER_(NT_CHECK_POW_FN_RETURN_INTEGER_HELPER_, type)\



NT_GET_DEFINE_FLOATING_DTYPES_(NT_CHECK_POW_FN_); 
NT_GET_DEFINE_SIGNED_INTEGER_DTYPES_(NT_CHECK_POW_FN_); 
NT_GET_DEFINE_FLOATING_DTYPES_(NT_CHECK_POW_FN_RETURN_INTEGER_); 
NT_GET_DEFINE_SIGNED_INTEGER_DTYPES_(NT_CHECK_POW_FN_RETURN_INTEGER_); 




int main(){
    using namespace nt::type_traits;
    static_assert(is_same_v<int, int>, "Expected ints to be the same");
    static_assert(!is_same_v<int, float>, "Expected int and float to be different");
    assert_same_type(remove_cv_t<const float>, float);
    assert_same_type(remove_cv_t<volatile float>, float);
    assert_same_type(remove_cv_t<const volatile float>, float);
    assert_same_type(remove_cv_t<float>, float);
    assert_same_type(remove_reference_t<float>, float);
    assert_same_type(remove_reference_t<float&>, float);
    assert_same_type(remove_cvref_t<const float>, float);
    assert_same_type(remove_cvref_t<volatile float>, float);
    assert_same_type(remove_cvref_t<const volatile float>, float);

    return 0;
}

#undef assert_true
#undef assert_false
#undef assert_same_type
