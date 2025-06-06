if(MSVC)
    message(STATUS "Compiler is MSVC, performing CPUID check...")

include(CheckCXXSourceCompiles)

# Check AVX512
set(CMAKE_REQUIRED_FLAGS "/arch:AVX512")
check_cxx_source_compiles("
#include <immintrin.h>
int main() {
    __m512 x = _mm512_set1_ps(1.0f);
    return 0;
}" HAS_AVX512)

# Check AVX2
set(CMAKE_REQUIRED_FLAGS "/arch:AVX2")
check_cxx_source_compiles("
#include <immintrin.h>
int main() {
    __m256i x = _mm256_set1_epi32(1);
    return 0;
}" HAS_AVX2)

# Check AVX
set(CMAKE_REQUIRED_FLAGS "/arch:AVX")
check_cxx_source_compiles("
#include <immintrin.h>
int main() {
    __m256 x = _mm256_set1_ps(1.0f);
    return 0;
}" HAS_AVX)

# Set compile options based on detection
if(HAS_AVX512)
    message(STATUS "Detected AVX512")
    add_compile_options(/arch:AVX512)
elseif(HAS_AVX2)
    message(STATUS "Detected AVX2")
    add_compile_options(/arch:AVX2)
elseif(HAS_AVX)
    message(STATUS "Detected AVX")
    add_compile_options(/arch:AVX)
else()
    message(WARNING "No SIMD support detected")
endif()

endif()
