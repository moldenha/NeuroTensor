#ifndef __NT_MACRO_HEADERS_FILE_H__
#define __NT_MACRO_HEADERS_FILE_H__

#include "name_func_macro.h"
#include "numargs_macro.h"


#ifndef NT_VLA
    #if defined(__GNUC__) && !defined(__clang__) || defined(__clang__)
        #define NT_VLA(type, name, size) type name[size]
    #else
        #define NT_VLA(type, name, size) type* name = new type[size]
    #endif

    #if defined(__GNUC__) && !defined(__clang__) || defined(__clang__)
        #define NT_VLA_DEALC(name)
    #else
        #define NT_VLA_DEALC(name) delete[] name
    #endif
#endif //NT_VLA


#endif
