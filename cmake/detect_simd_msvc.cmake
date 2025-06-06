# Only run on MSVC
if(NOT MSVC)
    return()
endif()

message(STATUS "Checking SIMD support using CPUID...")

include(CheckCXXSourceRuns)

set(SIMD_DETECTOR_SRC "${CMAKE_BINARY_DIR}/check_simd.cpp")

file(WRITE "${SIMD_DETECTOR_SRC}" "
#include <iostream>
#include <intrin.h>
int main() {
    int info[4];
    __cpuid(info, 0);
    if (info[0] >= 7) {
        __cpuidex(info, 7, 0);
        if (info[1] & (1 << 16)) { std::cout << \"AVX512\"; return 0; }
        if (info[1] & (1 << 5)) { std::cout << \"AVX2\"; return 0; }
    }
    __cpuid(info, 1);
    if (info[2] & (1 << 28)) { std::cout << \"AVX\"; return 0; }
    std::cout << \"NONE\";
    return 0;
}
")

execute_process(
    COMMAND ${CMAKE_CXX_COMPILER} /nologo /EHsc /O2 /Fecheck_simd.exe "${SIMD_DETECTOR_SRC}"
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
    RESULT_VARIABLE COMP_RESULT
)

if(COMP_RESULT EQUAL 0)
    execute_process(
        COMMAND check_simd.exe
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
        OUTPUT_VARIABLE SIMD_OUT
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(SIMD_OUT STREQUAL "AVX512")
        set(SIMD_FLAG "/arch:AVX512")
    elseif(SIMD_OUT STREQUAL "AVX2")
        set(SIMD_FLAG "/arch:AVX2")
    elseif(SIMD_OUT STREQUAL "AVX")
        set(SIMD_FLAG "/arch:AVX")
    else()
        set(SIMD_FLAG "")
    endif()

    message(STATUS "Detected SIMD level: ${SIMD_OUT} -> using flag: ${SIMD_FLAG}")
    add_compile_options(${SIMD_FLAG})

else()
    message(WARNING "SIMD detection failed — not adding SIMD flags.")
endif()

