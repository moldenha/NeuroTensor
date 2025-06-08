# Detect build type (Debug/Release/RelWithDebInfo/MinSizeRel)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

# Allow user to enable ASan manually
option(ENABLE_ASAN "Enable Address Sanitizer" OFF)

# Platform check (to avoid using unsupported flags)
if(NOT DEFINED NEUROTENSOR_CROSSCOMPILING_TO_WINDOWS)
  set(NEUROTENSOR_CROSSCOMPILING_TO_WINDOWS FALSE)
endif()

# Platform check (to avoid using unsupported flags)
if(NOT DEFINED NEUROTENSOR_CROSSCOMPILING_TO_WINDOWS)
  set(NEUROTENSOR_CROSSCOMPILING_TO_WINDOWS FALSE)
endif()

# Optimization based on build type
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  set(COMMON_CXX_FLAGS "${COMMON_CXX_FLAGS} -O3")
  set(COMMON_C_FLAGS   "${COMMON_C_FLAGS} -O3")
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(COMMON_CXX_FLAGS "${COMMON_CXX_FLAGS} -O0 -g")
  set(COMMON_C_FLAGS   "${COMMON_C_FLAGS} -O0 -g")
endif()


# Address Sanitizer (only on supported platforms)
if(ENABLE_ASAN AND NOT NEUROTENSOR_CROSSCOMPILING_TO_WINDOWS)
  set(COMMON_CXX_FLAGS "${COMMON_CXX_FLAGS} -fsanitize=address")
  set(COMMON_C_FLAGS   "${COMMON_C_FLAGS} -fsanitize=address")
  message(STATUS "AddressSanitizer ENABLED")
elseif(ENABLE_ASAN)
  message(WARNING "AddressSanitizer is not supported when cross-compiling to Windows; disabling it")
endif()

#these files will automatically detect and add the correct simd instruction set
include(${CMAKE_SOURCE_DIR}/cmake/detect_simd.cmake)


#this makes all the macros in NeuroTensor a lot more likely to work and a lot easier to use for reflection purposes
if (MSVC)
    add_compile_options(/Zc:preprocessor)
    add_compile_options(/bigobj)
    add_compile_options(/MP-) 
    add_compile_definitions(_SILENCE_CXX17_C_HEADER_DEPRECATION_WARNING) # disable depreciation warnings for MSVC
endif()


set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DUSE_PARALLEL")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_PARALLEL")
