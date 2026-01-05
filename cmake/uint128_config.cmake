# Check for compiler support for __uint128_t
include(CheckCCompilerFlag)

# Set a flag to check for __uint128_t support
check_c_compiler_flag("-m128" HAS_UINT128)

if(NOT HAS_UINT128)
    # Check for GCC compiler
    if(CMAKE_C_COMPILER_ID STREQUAL "GNU")
        message(STATUS "Was gnu")

        set(HAS_UINT128 TRUE)
    else()
        # Check if __SIZEOF_INT128__ is defined in C++ code
        message(STATUS "Checking compilation")
        check_cxx_source_compiles("#include <cstddef>\nint main() { return sizeof(__int128) > 0; }" HAS_UINT128)
    endif()
endif()

if(NOT HAS_UINT128)
    message(STATUS "Compiler does not support __uint128_t, using custom uint128_t library")

    # Include the directory for the uint128_t header
    include_directories(third_party/uint128_t)
    list(APPEND TYPES_SOURCES 
        third_party/uint128_t/uint128_t.cpp)
else()
    message(STATUS "Compiler supports __uint128_t")
endif()


