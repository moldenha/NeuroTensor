#ifndef NEUROTENSOR_API

// The reason for NEUROTENSOR_{name} is because this is very CMake at compile-time specific definitions
// The difference in the name is to make it easier to deduce
#ifdef NEUROTENSOR_DYNAMIC
// Define a macro to handle platform-specific import/export directives
#if defined(_WIN32) || defined(__CYGWIN__) // For Windows
    #ifdef NEUROTENSOR_EXPORTS // Defined when building the DLL
        #define NEUROTENSOR_API __declspec(dllexport)
    #else // Defined when using the DLL
        #define NEUROTENSOR_API __declspec(dllimport)
    #endif
#else // For Unix-like systems (Linux, macOS, etc.)
    #define NEUROTENSOR_API __attribute__((visibility("default"))) // Export by default
#endif

#else //NEUROTENSOR_STATIC
#define NEUROTENSOR_API
#endif


#endif
