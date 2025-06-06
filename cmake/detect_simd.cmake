# Universal SIMD detection for GCC, Clang, and MSVC
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    include(detect_simd_gcc_clang.cmake)
elseif(MSVC)
    include(detect_simd_msvc.cmake)
endif()
