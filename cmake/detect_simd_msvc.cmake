if(MSVC)
    message(STATUS "Compiler is MSVC, performing CPUID check...")

    include(CheckCXXSourceRuns)

    set(SIMD_TEST_SOURCE "
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
    }")

    file(WRITE "${CMAKE_BINARY_DIR}/check_simd.cpp" "${SIMD_TEST_SOURCE}")

    try_run(
        COMPILED_AND_RAN COMPILE_OK
        "${CMAKE_BINARY_DIR}" "${CMAKE_BINARY_DIR}/check_simd.cpp"
        CMAKE_FLAGS -DCMAKE_CXX_STANDARD=17
        COMPILE_OUTPUT_VARIABLE COMPILE_OUT
        RUN_OUTPUT_VARIABLE SIMD_OUT
    )

    if(COMPILE_OK AND COMPILED_AND_RAN)
        string(STRIP "${SIMD_OUT}" SIMD_OUT)
        message(STATUS "Detected SIMD level: ${SIMD_OUT}")
        if(SIMD_OUT STREQUAL "AVX512")
            add_compile_options(/arch:AVX512)
        elseif(SIMD_OUT STREQUAL "AVX2")
            add_compile_options(/arch:AVX2)
        elseif(SIMD_OUT STREQUAL "AVX")
            add_compile_options(/arch:AVX)
        else()
            message(WARNING "SIMD detection ran successfully but no supported instruction set was detected.")
        endif()
    else()
        message(WARNING "SIMD detection failed (compile or run failure). Output: ${COMPILE_OUT}")
    endif()
endif()
