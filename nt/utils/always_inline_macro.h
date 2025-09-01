//always inline:
#ifndef NT_ALWAYS_INLINE

#ifdef _MSC_VER
    #define NT_ALWAYS_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
    #define NT_ALWAYS_INLINE inline __attribute__((always_inline))
#else
    #define NT_ALWAYS_INLINE inline 
    //otherwise default to just inline
#endif
#endif
