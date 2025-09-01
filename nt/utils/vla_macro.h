#ifndef NT_VLA
#if defined(__clang__) && !defined(__APPLE__)
    #define NT_VLA(type, name, size) \
        _Pragma("clang diagnostic push") \
        _Pragma("clang diagnostic ignored \"-Wvla-cxx-extension\"") \
        type name[size]; \
        _Pragma("clang diagnostic pop")
#elif defined(__GNUC__) || (defined(__clang__) && defined(__APPLE__))
    #define NT_VLA(type, name, size) type name[size]
#else
#include "../memory/meta_allocator.h"
    #define NT_VLA(type, name, size) type* name = MetaNewArr(type, size);
#endif

#if defined(__GNUC__) && !defined(__clang__) || defined(__clang__)
    #define NT_VLA_DEALC(name)
#else
    #define NT_VLA_DEALC(name) MetaFreeArr(name); // should automatically detect template
#endif
#endif //NT_VLA
