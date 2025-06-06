# Universal SIMD detection for GCC, Clang, and MSVC
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    include(${CMAKE_SOURCE_DIR}/cmake/detect_simd_gcc_clang.cmake)
elseif(MSVC)
    include(${CMAKE_SOURCE_DIR}/cmake/detect_simd_msvc.cmake)
endif()
