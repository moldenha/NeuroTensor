cmake_minimum_required(VERSION 3.10)
project(NeuroTensor)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -fsanitize=address -g -DUSE_PARALLEL -O3")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native -fsanitize=address")
set(CMAKE_VERBOSE_MAKEFILE OFF) 

# Find TBB
# find_package(TBB REQUIRED)
find_package(TBB QUIET)

if (TBB_FOUND)
    # If TBB is found, link it
    message(STATUS "Found system-installed TBB")
    include_directories(${TBB_INCLUDE_DIRS})
    set(TBB_LIB TBB::tbb)
else()
    # If TBB is not found, build it from the third-party directory
    message(STATUS "System-installed TBB not found, building from source")
    add_subdirectory(third_party/tbb)
    set(TBB_LIB TBB::tbb)  # Use the built TBB target
    include_directories(${CMAKE_SOURCE_DIR}/third_party/tbb/include)
    link_directories(${CMAKE_BINARY_DIR}/third_party/tbb/build/src)

endif()


find_package(Matplot++ QUIET)

if (Matplot++_FOUND)
    message(STATUS "Found system-installed Matplot++")
    include_directories(${Matplot++_INCLUDE_DIRS})
    set(MATPLOT_LIB Matplot++::matplot)
else()
    message(STATUS "System-installed Matplot++ not found, using source")
    add_subdirectory(third_party/matplot)
    include_directories(${CMAKE_SOURCE_DIR}/third_party/matplot/source/matplot)
    set(MATPLOT_LIB matplot)
endif()
    
    

#add_subdirectory(third_party/xnetwork)

# set_target_properties(tbb PROPERTIES COMPILE_FLAGS "-fno-sanitize=address" LINK_FLAGS "-fno-sanitize=address")

#MKL_FLAGS := -I/opt/intel/oneapi/mkl/latest/include -L/opt/intel/oneapi/mkl/latest/lib -Wl,-rpath,/opt/intel/oneapi/mkl/latest/lib -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl
# Find MKL
# set(MKLROOT "/opt/intel/oneapi/mkl/latest")
# find_package(MKL REQUIRED)
# set(MKL_INCLUDE_DIR "${MKLROOT}/include")


#Find Boost
# find_package(Boost REQUIRED COMPONENTS system filesystem iostreams)



# Include directories
include_directories(
    # ${MKL_INCLUDE_DIR}
    # ${TBB_INCLUDE_DIRS}
    # ${Boost_INCLUDE_DIRS}
    # ${CMAKE_SOURCE_DIR}/third_party/tbb/include
    ${CMAKE_SOURCE_DIR}/third_party/simde
    ${CMAKE_SOURCE_DIR}/third_party/half
    ${CMAKE_SOURCE_DIR}/third_party/stb
    #For matrix and vector linear algebra calculations
    #This incorperates most of the nt::linalg namespace
    ${CMAKE_SOURCE_DIR}/third_party/eigen
)


set(CORE_SOURCES
    src/refs/ArrayRef.cpp
    src/refs/SizeRef.cpp
    src/utils/utils.cpp
    src/utils/optional.cpp
    src/utils/CommaOperator.cpp
    src/Tensor.cpp
)

set(MATMULT_SOURCES
	src/functional/matmult/nt_matmult_float.cpp
	src/functional/matmult/nt_matmult_float16.cpp
	src/functional/matmult/nt_matmult_double.cpp
	src/functional/matmult/nt_matmult_float128.cpp
	src/functional/matmult/nt_matmult_complex32.cpp
	src/functional/matmult/nt_matmult_complex64.cpp
	src/functional/matmult/nt_matmult_complex128.cpp
	src/functional/matmult/nt_matmult_int128.cpp
	src/functional/matmult/nt_matmult_uint128.cpp
	src/functional/matmult/nt_matmult_int64.cpp
	src/functional/matmult/nt_matmult_uint64.cpp
	src/functional/matmult/nt_matmult_int32.cpp
	src/functional/matmult/nt_matmult_uint32.cpp
	src/functional/matmult/nt_matmult_int16.cpp
	src/functional/matmult/nt_matmult_uint16.cpp
	src/functional/matmult/nt_matmult_int8.cpp
	src/functional/matmult/nt_matmult_uint8.cpp
)

set(FUNCTIONAL_SOURCES
	src/functional/functional.cpp
	src/functional/functional_operator.cpp
	src/functional/load.cpp
	src/functional/save.cpp
	src/functional/functional_matmult.cpp
	src/functional/functional_fold.cpp
	src/functional/functional_conv.cpp
	src/functional/wrbits.cpp
    src/functional/numpy.cpp
)


#split linalg into different functions
#this reduces compile times when specific functions need to be changed
#would like to do this for all general function in the future and current 
set(LINALG_SOURCES
    src/linalg/SVD.cpp
    src/linalg/QR.cpp
    src/linalg/inv.cpp
    src/linalg/column_space.cpp
    src/linalg/process/SVD.cpp
    src/linalg/process/QR.cpp
    src/linalg/process/inv.cpp
    src/linalg/process/null_space.cpp
    src/linalg/process/column_space.cpp
    src/linalg/adjugate.cpp
    src/linalg/determinant.cpp
    src/linalg/norm.cpp
    src/linalg/null_space.cpp
    src/linalg/pivots.cpp
    src/linalg/eye.cpp
    src/linalg/independent.cpp
)


set(TYPES_SOURCES
    src/dtype/ArrayVoid.cpp
    src/dtype/DType.cpp
    src/dtype/DType_operators.cpp
    src/dtype/ranges.cpp
    src/dtype/Scalar.cpp
    src/types/Types.cpp
    src/convert/Convert.cpp
)

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


set(PERMUTE_SOURCES
    src/permute/permute_old.cpp
    src/permute/permute.cpp
)

#this also includes autograds and layers
set(AI_SOURCES
    src/layers/Layer.cpp
    src/layers/LNN.cpp
    src/layers/TensorGrad.cpp
    src/layers/functional.cpp
    src/layers/Loss.cpp
    src/layers/Optimizers.cpp
    src/layers/Module.cpp
    #ncps sources
    src/layers/ncps/lstm_cell.cpp
    src/layers/ncps/wiring/wiring.cpp
    src/layers/ncps/cfc/wired_cfc_cell.cpp
    src/layers/ncps/cfc/cfc_cell.cpp
    src/layers/ncps/cfc/cfc.cpp
    src/layers/ncps/ltc/ltc.cpp
    src/layers/ncps/ltc/ltc_cell.cpp

)

set(MP_SOURCES
    src/mp/Threading.cpp
)

set(TDA_SOURCES
    src/tda/old_tda/Simplex.cpp
    src/tda/old_tda/Simplex2d.cpp
    src/tda/old_tda/Shapes.cpp
    src/tda/old_tda/Basis.cpp
    src/tda/old_tda/Points.cpp
    src/tda/old_tda/Points2d.cpp
    src/tda/old_tda/KDTree.cpp
    # src/tda/old_tda/refinement/refine.cpp <- unreleased
    src/tda/old_tda/BatchBasis.cpp
    src/tda/old_tda/BatchKDTree.cpp
    src/tda/old_tda/BatchPoints.cpp
    src/tda/BasisOverlapping.cpp
    src/tda/Boundaries.cpp
    src/tda/Homology.cpp
    src/tda/MatrixReduction.cpp
    src/tda/PlotDiagrams.cpp
    src/tda/Points.cpp
    src/tda/SimplexConstruct.cpp
    src/tda/SimplexRadi.cpp
)

set(SPARSE_SOURCES
    src/sparse/SparseTensor.cpp
)

set(IMAGES_SOURCES
    src/images/image.cpp
)

set(MEMORY_SOURCES
    src/memory/bucket.cpp
    src/memory/device.cpp
)



# Source files
set(SOURCES
    ${MATMULT_SOURCES}
    ${CORE_SOURCES}
    ${FUNCTIONAL_SOURCES}
    ${LINALG_SOURCES}
    ${TYPES_SOURCES}
    ${PERMUTE_SOURCES}
    ${AI_SOURCES}
    ${MP_SOURCES}
    ${TDA_SOURCES}
    ${SPARSE_SOURCES}
    ${IMAGES_SOURCES}
    ${MEMORY_SOURCES}
)

# Add executable
add_executable(main.out main.cpp ${SOURCES})





# Link libraries
target_link_libraries(main.out
    # -L${MKL_LIB_PATH}
    # ${MKL_LIBS}
    # ${Boost_LIBRARIES}
    ${TBB_LIB}
    ${MATPLOT_LIB}
    pthread
)

# Set RPATH for runtime
set_target_properties(main.out PROPERTIES
	INSTALL_RPATH_USE_LINK_PATH TRUE
)

# Install target (optional)
install(TARGETS main.out DESTINATION bin)

