#ifndef NT_DTYPE_H__
#define NT_DTYPE_H__

#include "DType_enum.h"
#include "../utils/api_macro.h"
#include "../utils/always_inline_macro.h"

namespace nt {

template <DType... Rest> struct VariadicArgCount {
    static constexpr int value = 0;
};

template <DType dt, DType... Rest> struct VariadicArgCount<dt, Rest...> {
    static constexpr int value = 1 + VariadicArgCount<Rest...>::value;
};

template <DType... Rest> constexpr DType FirstVariadicDType = DType::Bool;

template <DType dt, DType... Rest>
constexpr DType FirstVariadicDType<dt, Rest...> = dt;

template <DType dt, DType... Rest> struct DTypeEnum {
    using next_wrapper = DTypeEnum<Rest...>;
    static constexpr DType next = dt;
    static constexpr bool done = false;
};

template <DType T> struct DTypeEnum<T> {
    using next_wrapper = DTypeEnum<T>;
    static constexpr DType next = T;
    static constexpr bool done = true;
};

// the input template is DTypeEnum or DTypeEnum
template <typename T, typename... Rest> struct WRAP_DTYPES {
    static constexpr DType next = T::next;
    using next_wrapper =
        std::conditional_t<T::done, WRAP_DTYPES<Rest...>,
                           WRAP_DTYPES<typename T::next_wrapper, Rest...>>;
    static constexpr bool done = false;
};

template <typename T> struct WRAP_DTYPES<T> {
    static constexpr DType next = T::next;
    using next_wrapper =
        std::conditional_t<T::done, WRAP_DTYPES<T>,
                           WRAP_DTYPES<typename T::next_wrapper>>;
    static constexpr bool done = T::done;
};

template <typename T> inline constexpr bool is_in_dtype_enum(DType dt) {
    if (dt != T::next) {
        if (T::done) {
            return false;
        }
        return is_in_dtype_enum<typename T::next_wrapper>(dt);
    } else {
        return true;
    }
}

namespace DTypeFuncs {
template <typename Test, template <typename...> class Ref>
struct is_specialization : std::false_type {};
template <template <typename...> class Ref, typename... Args>
struct is_specialization<Ref<Args...>, Ref> : std::true_type {};

template <class T> struct is_wrapped_dtype {
    static constexpr bool value = is_specialization<T, WRAP_DTYPES>::value;
};

// universal declarations to any dtype
template <DType dt> 
NT_ALWAYS_INLINE bool is_in(const DType inp){return inp == dt;}

template <DType dt, DType M, DType... Rest> 
NT_ALWAYS_INLINE bool is_in(const DType inp) {
    return (inp == dt) ? true : is_in<M, Rest...>(inp);
}

template <typename T,
          std::enable_if_t<DTypeFuncs::is_wrapped_dtype<T>::value, bool> = true>
NT_ALWAYS_INLINE bool is_in(const DType &inp) {
    if (inp != T::next) {
        if (T::done)
            return false;
        return is_in<typename T::next_wrapper>(inp);
    }
    return true;
}

template <DType dt, DType M, DType... Rest> struct is_in_t {
    static constexpr bool value =
        (dt == M) ? true : is_in_t<dt, Rest...>::value;
};

template <DType dt, DType M> struct is_in_t<dt, M> {
    static constexpr bool value = (dt == M);
};

template <DType dt, DType M, DType... Rest>
inline constexpr bool is_in_v = is_in_t<dt, M, Rest...>::value;

template <DType dt, DType M>
inline constexpr bool is_in_v<dt, M> = is_in_t<dt, M>::value;

template <DType dt> inline constexpr bool is_in_v<dt, DType::Bool> = false;

} // namespace DTypeFuncs
} // namespace nt
// #include "../Tensor.h"
#include "../types/Types.h"
#include "../types/TensorDeclare.h"
#include "../utils/utils.h"
#include "compatible/DType_compatible.h"
#include <complex>
#include <memory>
#include <stdlib.h>
#include <type_traits>

namespace nt {
NEUROTENSOR_API std::ostream &operator<<(std::ostream &out, DType const &data);
NEUROTENSOR_API std::ostream &operator<<(std::ostream &out, const uint_bool_t &data);

namespace DTypeFuncs {

template <DType dt> 
inline std::ostream &print_dtypes(std::ostream &os) {
    return os << dt << "}";
}
template <DType dt, DType M, DType... Rest>
inline std::ostream &print_dtypes(std::ostream &os) {
    os << dt << ",";
    return print_dtypes<M, Rest...>(os);
}

template <DType... Rest> bool 
inline check_dtypes(const char *str, const DType dtype) {
    bool outp = is_in<Rest...>(dtype);
    if (!outp) {
        std::cout << str << "() was expected to support {";
        std::cout << print_dtypes<Rest...> << " but instead got " << dtype
                  << std::endl;
    }
    return outp;
}

template <typename...> struct all_dtype;

template <> struct all_dtype<> : std::true_type {};

template <typename T, typename... Rest>
struct all_dtype<T, Rest...>
    : std::integral_constant<bool, std::is_same_v<T, DType> &&
                                       all_dtype<Rest...>::value> {};

template <class... DTs>
inline constexpr bool all_dtype_v = all_dtype<DTs...>::value;

template <class T> 
inline void is_same(DType a, bool &outp, T b){
    if constexpr(std::is_same_v<T, DType>){
		if(a == b) outp = true;
	}
}

inline constexpr std::size_t size_of_dtype_p(const DType &d){return sizeof(std::uintptr_t);}

template<typename T>
inline bool is_in(T&& a){return false;} 
template <typename T, typename T2, class... DTs>  
inline bool is_in(T&& dt, T2&& sub_dt, DTs&&... dts) {
    static_assert(std::is_same_v<std::decay_t<T>, std::decay_t<T2>>, "Expected to get same types for is_in");
    return dt == sub_dt || is_in(std::forward<T>, std::forward<DTs>(dts)...);
}


NEUROTENSOR_API std::size_t size_of_dtype(const DType &);
NEUROTENSOR_API bool can_convert(const DType &, const DType &);

template <DType dt = DType::Integer>
NEUROTENSOR_API void initialize_strides(void **ptrs, void *cast, const std::size_t &s,
                        const DType &ds);
NEUROTENSOR_API bool is_unsigned(const DType &dt);
NEUROTENSOR_API bool is_integer(const DType &dt);
NEUROTENSOR_API bool is_floating(const DType &dt);
NEUROTENSOR_API bool is_complex(const DType &dt);
NEUROTENSOR_API DType complex_size(const std::size_t &s);
NEUROTENSOR_API DType floating_size(const std::size_t &s);
NEUROTENSOR_API DType integer_size(const std::size_t &s);
NEUROTENSOR_API DType unsigned_size(const std::size_t &s);
NEUROTENSOR_API uint8_t dtype_int_code(const DType &);
NEUROTENSOR_API DType code_int_dtype(const uint8_t &);

} // namespace DTypeFuncs
} // namespace nt
#endif
