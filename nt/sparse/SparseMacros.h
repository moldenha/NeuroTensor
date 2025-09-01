#ifndef NT_SPARSE_MACROS_H__
#define NT_SPARSE_MACROS_H__
#include "../utils/numargs_macro.h"

#if defined(__GNUC__)
  #define NT_SUPPRESS_VA_ARGS_WARNING_PUSH \
    _Pragma("GCC diagnostic push") \
    _Pragma("GCC diagnostic ignored \"-Wvariadic-macros\"")

  #define NT_SUPPRESS_VA_ARGS_WARNING_POP \
    _Pragma("GCC diagnostic pop")
#elif defined(_MSC_VER) && defined(_WIN32)
    #define NT_SUPPRESS_VA_ARGS_WARNING_PUSH\
        __pragma(warning(push))\
        __pragma(warning(disable:C5100))
  #define NT_SUPPRESS_VA_ARGS_WARNING_POP \
        __pragma(warning(pop))
#else
  #define NT_SUPPRESS_VA_ARGS_WARNING_PUSH
  #define NT_SUPPRESS_VA_ARGS_WARNING_POP
#endif

NT_SUPPRESS_VA_ARGS_WARNING_PUSH

#define _NT_SPARSE_RUN_SINGLE_FUNCTION_EMPTY_1(func, begin, end, __VA_ARGS__) func(begin, end)
#define _NT_SPARSE_RUN_SINGLE_FUNCTION_EMPTY_0(func, begin, end, ...) func(begin, end, __VA_ARGS__)

#define _NT_SPARSE_RUN_SINGLE_FUNCTION_(dtype, func, data, ...)                \
    auto begin =                                                               \
        details::SMDenseIterator<DTypeFuncs::dtype_to_type_t<dtype>>(data, false);      \
    auto end =                                                                 \
        detail::SMDenseIterator<DTypeFuncs::dtype_to_type_t<dtype>>(data, true);       \
    _NT_GLUE_(_NT_SPARSE_RUN_SINGLE_FUNCTION_EMPTY_, _NT_IS_EMPTY_(__VA_ARGS__))(func, begin, end, __VA_ARGS__);
    // func<detail::SMDenseIterator<DTypeFuncs::dtype_to_type_t<dtype>>>(begin, end, __VA_ARGS__);



//128 bit types except for complex taken out
#define _NT_SPARSE_RUN_FUNCTION_(dtype, func, data, ...)                       \
    switch (dtype) {                                                           \
    case DType::Float: {                                                       \
        _NT_SPARSE_RUN_SINGLE_FUNCTION_(DType::Float, func, data, __VA_ARGS__) \
    break;																		\
	}                                                                          \
    case DType::Double: {                                                      \
        _NT_SPARSE_RUN_SINGLE_FUNCTION_(DType::Double, func, data,             \
                                        __VA_ARGS__)                           \
    break;																		\
	}                                                                          \
    case DType::Complex64: {                                                   \
        _NT_SPARSE_RUN_SINGLE_FUNCTION_(DType::Complex64, func, data,          \
                                        __VA_ARGS__)                           \
    break;																		\
	}                                                                          \
    case DType::Complex128: {                                                  \
        _NT_SPARSE_RUN_SINGLE_FUNCTION_(DType::Complex128, func, data,         \
                                        __VA_ARGS__)                           \
    break;																		\
	}                                                                          \
    case DType::int8: {                                                        \
        _NT_SPARSE_RUN_SINGLE_FUNCTION_(DType::int8, func, data, __VA_ARGS__)  \
    break;																		\
	}                                                                          \
    case DType::uint8: {                                                        \
        _NT_SPARSE_RUN_SINGLE_FUNCTION_(DType::uint8, func, data, __VA_ARGS__)  \
    break;																		\
	}                                                                          \
    case DType::int16: {                                                       \
        _NT_SPARSE_RUN_SINGLE_FUNCTION_(DType::int16, func, data, __VA_ARGS__) \
    break;																		\
	}                                                                          \
    case DType::uint16: {                                               \
        _NT_SPARSE_RUN_SINGLE_FUNCTION_(DType::uint16, func, data,      \
                                        __VA_ARGS__)                           \
    break;																		\
	}                                                                          \
    case DType::uint32: {                                                        \
        _NT_SPARSE_RUN_SINGLE_FUNCTION_(DType::uint32, func, data, __VA_ARGS__)  \
    break;																		\
	}                                                                          \
    case DType::int32: {                                                     \
        _NT_SPARSE_RUN_SINGLE_FUNCTION_(DType::int32, func, data,            \
                                        __VA_ARGS__)                           \
    break;																		\
	}                                                                          \
    case DType::int64: {                                                    \
        _NT_SPARSE_RUN_SINGLE_FUNCTION_(DType::int64, func, data,           \
                                        __VA_ARGS__)                           \
    break;																		\
	}                                                                          \
    case DType::int128: {                                                    \
        _NT_SPARSE_RUN_SINGLE_FUNCTION_(DType::int128, func, data,           \
                                        __VA_ARGS__)                           \
    break;																		\
	} \
    case DType::uint128: {                                                    \
        _NT_SPARSE_RUN_SINGLE_FUNCTION_(DType::uint128, func, data,           \
                                        __VA_ARGS__)                           \
    break;																		\
	} \
    case DType::Float128: {                                                    \
        _NT_SPARSE_RUN_SINGLE_FUNCTION_(DType::Float128, func, data,           \
                                        __VA_ARGS__)                           \
    break;																		\
	} \
    case DType::Bool: {                                                        \
        _NT_SPARSE_RUN_SINGLE_FUNCTION_(DType::Bool, func, data, __VA_ARGS__)  \
    break;																		\
	}                                                                          \
    case DType::Float16: {                                                     \
        _NT_SPARSE_RUN_SINGLE_FUNCTION_(DType::Float16, func, data,            \
                                        __VA_ARGS__)                           \
    break;																		\
	}                                                                          \
    case DType::Complex32: {                                                   \
        _NT_SPARSE_RUN_SINGLE_FUNCTION_(DType::Complex32, func, data,          \
                                        __VA_ARGS__)                           \
    break;																		\
	}                                                                          \
    default : std::cout << "Error unsupported sparse DType " << dtype;         \
        break;                                                                 \
    }

#define _NT_SPARSE_RUN_SINGLE_CONST_FUNCTION_(dtype, func, data, ...)          \
    using ValueType = DTypeFuncs::dtype_to_type_t<static_cast<DType>(dtype)>;  \
    auto begin = details::SMDenseIterator<const ValueType>(data, false);       \
    auto end = details::SMDenseIterator<const ValueType>(data, true);          \
    _NT_GLUE_(_NT_SPARSE_RUN_SINGLE_FUNCTION_EMPTY_, _NT_IS_EMPTY_(__VA_ARGS__))(func, begin, end, __VA_ARGS__);
    // if constexpr (sizeof...(__VA_ARGS__) > 0){                                 \
    // func<details::SMDenseIterator<const ValueType>>(begin, end, __VA_ARGS__);  \
    // }else{                                                                     \
    //     func<details::SMDenseIterator<const ValueType>>(begin, end);           \
    //}

#define _NT_SPARSE_RUN_CONST_FUNCTION_(dtype, func, data, ...)                 \
    switch (dtype) {                                                           \
    case DType::Float: {                                                       \
        _NT_SPARSE_RUN_SINGLE_CONST_FUNCTION_(DType::Float, func, data,__VA_ARGS__);                     \
    break;																		\
	}                                                                          \
    case DType::Double: {                                                      \
        _NT_SPARSE_RUN_SINGLE_CONST_FUNCTION_(DType::Double, func, data,       \
                                              __VA_ARGS__)                     \
    break;																		\
	}                                                                          \
    case DType::Complex64: {                                                   \
        _NT_SPARSE_RUN_SINGLE_CONST_FUNCTION_(DType::Complex64, func, data,    \
                                              __VA_ARGS__)                     \
    break;																		\
	}                                                                          \
    case DType::Complex128: {                                                  \
        _NT_SPARSE_RUN_SINGLE_CONST_FUNCTION_(DType::Complex128, func, data,   \
                                              __VA_ARGS__)                     \
    break;																		\
	}                                                                          \
    case DType::int8: {                                                        \
        _NT_SPARSE_RUN_SINGLE_CONST_FUNCTION_(DType::int8, func, data,         \
                                              __VA_ARGS__)                     \
    break;																		\
	}                                                                          \
    case DType::uint8: {                                                        \
        _NT_SPARSE_RUN_SINGLE_CONST_FUNCTION_(DType::uint8, func, data,         \
                                              __VA_ARGS__)                     \
    break;																		\
	}                                                                          \
    case DType::int16: {                                                       \
        _NT_SPARSE_RUN_SINGLE_CONST_FUNCTION_(DType::int16, func, data,        \
                                              __VA_ARGS__)                     \
    break;																		\
	}                                                                          \
    case DType::uint16: {                                               \
        _NT_SPARSE_RUN_SINGLE_CONST_FUNCTION_(DType::uint16, func,      \
                                              data, __VA_ARGS__)               \
    break;																		\
	}                                                                          \
    case DType::uint32: {                                                        \
        _NT_SPARSE_RUN_SINGLE_CONST_FUNCTION_(DType::uint32, func, data,         \
                                              __VA_ARGS__)                     \
    break;																		\
	}                                                                          \
    case DType::int32: {                                                     \
        _NT_SPARSE_RUN_SINGLE_CONST_FUNCTION_(DType::int32, func, data,      \
                                              __VA_ARGS__)                     \
    break;																		\
	}                                                                          \
    case DType::int64: {                                                    \
        _NT_SPARSE_RUN_SINGLE_CONST_FUNCTION_(DType::int64, func, data,     \
                                              __VA_ARGS__)                     \
    break;																		\
	}                                                                          \
    case DType::int128: {                                                    \
        _NT_SPARSE_RUN_SINGLE_CONST_FUNCTION_(DType::int128, func, data,     \
                                              __VA_ARGS__)                     \
    break;																		\
	}                                                                          \
    case DType::uint128: {                                                    \
        _NT_SPARSE_RUN_SINGLE_CONST_FUNCTION_(DType::uint128, func, data,     \
                                              __VA_ARGS__)                     \
    break;																		\
	}                                                                          \
    case DType::Float128: {                                                    \
        _NT_SPARSE_RUN_SINGLE_CONST_FUNCTION_(DType::Float128, func, data,     \
                                              __VA_ARGS__)                     \
    break;																		\
	}                                                                          \
    case DType::Bool: {                                                        \
        _NT_SPARSE_RUN_SINGLE_CONST_FUNCTION_(DType::Bool, func, data,         \
                                              __VA_ARGS__)                     \
    break;																		\
	}                                                                          \
    case DType::Float16: {                                                     \
        _NT_SPARSE_RUN_SINGLE_CONST_FUNCTION_(DType::Float16, func, data,      \
                                              __VA_ARGS__)                     \
    break;																		\
	}                                                                          \
    case DType::Complex32: {                                                   \
        _NT_SPARSE_RUN_SINGLE_CONST_FUNCTION_(DType::Complex32, func, data,    \
                                              __VA_ARGS__)                     \
    break;																		\
	}                                                                          \
    default : std::cout << "Error unknown DType " << dtype;                    \
        break;                                                                 \
    }


NT_SUPPRESS_VA_ARGS_WARNING_POP

#undef NT_SUPPRESS_VA_ARGS_WARNING_PUSH
#undef NT_SUPPRESS_VA_ARGS_WARNING_POP

#endif
