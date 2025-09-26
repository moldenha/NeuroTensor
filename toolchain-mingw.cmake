set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_PROCESSOR x86_64)

# Compilers
set(CMAKE_C_COMPILER x86_64-w64-mingw32-gcc)
set(CMAKE_CXX_COMPILER x86_64-w64-mingw32-g++)

# Needed to make sure certain libraries installed on the machine for mac are not found for mingw 
set(CMAKE_SYSROOT "/usr/local/Cellar/mingw-w64/11.0.1_1/toolchain-x86_64")
set(CMAKE_FIND_ROOT_PATH ${CMAKE_SYSROOT})
set(CMAKE_INCLUDE_PATH "${CMAKE_FIND_ROOT_PATH}/include")
set(CMAKE_LIBRARY_PATH "${CMAKE_FIND_ROOT_PATH}/lib")

set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)

# Disable TBB autodetection (ensure system TBB is not picked up when cross-compiling)
unset(TBB_DIR CACHE)
set(TBB_DIR "" CACHE PATH "Force CMake to ignore system TBB" FORCE)

# Tell NeuroTensor we are cross-compiling to Windows
set(NEUROTENSOR_CROSSCOMPILING_TO_WINDOWS TRUE CACHE BOOL "Cross-compiling to Windows from non-Windows host")

# Add AVX/AVX2 support
set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS} -mavx -mavx2 -mfma -mf16c")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx -mavx2 -mfma -mf16c")

# Make TBB non-strict (optional dependency)
set(TBB_STRICT OFF CACHE BOOL "Allow builds without TBB")

# Prevent CMake from searching Mac system paths
# set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
# set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
# set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM ONLY)
