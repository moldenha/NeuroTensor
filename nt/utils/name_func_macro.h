//this is just the macro used to get the name of a function
#ifndef __NT_UTILS_NAME_FUNC_MACRO_H__
#define __NT_UTILS_NAME_FUNC_MACRO_H__


#ifndef __NT_FUNCTION_NAME__
    #if defined(_MSC_VER) //microsoft visual studios
        #define __NT_FUNCTION_NAME__ __func__
    #elif defined(_WIN32) || defined(_WIN64)  // Windows
        #define __NT_FUNCTION_NAME__   __FUNCTION__
    #elif defined(__GNUC__) || defined(__clang__)  // GCC or Clang (Linux/macOS)
        #define __NT_FUNCTION_NAME__   __builtin_FUNCTION()
    #else  // Other compilers or platforms
        #define __NT_FUNCTION_NAME__   __func__
    #endif
#endif


#endif
