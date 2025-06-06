cmake_minimum_required(VERSION 3.10)
project(NeuroTensor LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
include(${CMAKE_SOURCE_DIR}/cmake/flags.cmake)
set(CMAKE_VERBOSE_MAKEFILE OFF) 

message(STATUS "Root path is " ${CMAKE_FIND_ROOT_PATH})
include(${CMAKE_SOURCE_DIR}/cmake/tbb_config.cmake)


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
    ${CMAKE_SOURCE_DIR}/third_party/nifti_clib/znzlib
    ${CMAKE_SOURCE_DIR}/third_party/nifti_clib/niftilib

    #Boost files
    #Mainly for 128 bit data types
    ${CMAKE_SOURCE_DIR}/third_party/boost_config/include/
    ${CMAKE_SOURCE_DIR}/third_party/multiprecision/include/

)


# Include NIfTI macros
include(${CMAKE_SOURCE_DIR}/third_party/nifti_clib/cmake/nifti_macros.cmake)

# Add znzlib for compressed NIfTI support
add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/nifti_clib/znzlib)

# Add NIfTI core library (creates niftiio target)
add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/nifti_clib/niftilib)

set(CORE_SOURCES
    nt/refs/ArrayRef.cpp
    nt/refs/SizeRef.cpp
    nt/utils/utils.cpp
    nt/utils/optional.cpp
    nt/utils/CommaOperator.cpp
    nt/Tensor.cpp
)

set(MATMULT_SOURCES
	nt/functional/cpu/matmult/nt_matmult_float.cpp
	nt/functional/cpu/matmult/nt_matmult_float16.cpp
	nt/functional/cpu/matmult/nt_matmult_double.cpp
	nt/functional/cpu/matmult/nt_matmult_float128.cpp
	nt/functional/cpu/matmult/nt_matmult_complex32.cpp
	nt/functional/cpu/matmult/nt_matmult_complex64.cpp
	nt/functional/cpu/matmult/nt_matmult_complex128.cpp
	nt/functional/cpu/matmult/nt_matmult_int128.cpp
	nt/functional/cpu/matmult/nt_matmult_uint128.cpp
	nt/functional/cpu/matmult/nt_matmult_int64.cpp
	nt/functional/cpu/matmult/nt_matmult_uint64.cpp
	nt/functional/cpu/matmult/nt_matmult_int32.cpp
	nt/functional/cpu/matmult/nt_matmult_uint32.cpp
	nt/functional/cpu/matmult/nt_matmult_int16.cpp
	nt/functional/cpu/matmult/nt_matmult_uint16.cpp
	nt/functional/cpu/matmult/nt_matmult_int8.cpp
	nt/functional/cpu/matmult/nt_matmult_uint8.cpp
)

set(FUNCTIONAL_SOURCES
	nt/functional/tensor_files/colim_transform.cpp
	nt/functional/tensor_files/combine.cpp
	nt/functional/tensor_files/conv.cpp
	nt/functional/tensor_files/fill.cpp
	nt/functional/tensor_files/fused.cpp
	nt/functional/tensor_files/save_load.cpp
	nt/functional/tensor_files/matmult.cpp
	nt/functional/tensor_files/normalize.cpp
	nt/functional/tensor_files/numpy.cpp
	nt/functional/tensor_files/operators.cpp
	nt/functional/tensor_files/rand.cpp
	nt/functional/tensor_files/min_max.cpp
	nt/functional/tensor_files/compare.cpp
	nt/functional/tensor_files/ranges.cpp
	nt/functional/tensor_files/softmax.cpp
	nt/functional/tensor_files/sum_exp_log.cpp
	nt/functional/tensor_files/mesh.cpp
	nt/functional/tensor_files/repeat.cpp
	nt/functional/tensor_files/stride.cpp
	nt/functional/tensor_files/index.cpp
	nt/functional/tensor_files/padding.cpp
	nt/functional/tensor_files/dropout.cpp
	nt/functional/tensor_files/activation_functions.cpp
	nt/functional/tensor_files/trig.cpp
	nt/functional/tensor_files/split.cpp
	nt/functional/tensor_files/unique.cpp
	nt/functional/tensor_files/combinations.cpp
	nt/functional/tensor_files/print.cpp
	nt/functional/tensor_files/transpose.cpp
	nt/functional/tensor_files/complex.cpp
	nt/functional/tensor_files/flip.cpp
	nt/functional/tensor_files/dilate.cpp
	nt/functional/tensor_files/convert.cpp
	nt/functional/tensor_files/sort.cpp

    nt/functional/tensor_files/pooling/avg_pool1d.cpp
    nt/functional/tensor_files/pooling/avg_pool2d.cpp
    nt/functional/tensor_files/pooling/avg_pool3d.cpp
    nt/functional/tensor_files/pooling/lp_pool1d.cpp
    nt/functional/tensor_files/pooling/lp_pool2d.cpp
    nt/functional/tensor_files/pooling/lp_pool3d.cpp
    nt/functional/tensor_files/pooling/max_pool1d.cpp
    nt/functional/tensor_files/pooling/max_pool2d.cpp
    nt/functional/tensor_files/pooling/max_pool3d.cpp
    nt/functional/tensor_files/pooling/fractional.cpp
    
    nt/functional/cpu/unfold1d.cpp
    nt/functional/cpu/unfold2d.cpp
    nt/functional/cpu/unfold3d.cpp
    nt/functional/cpu/fold.cpp
    nt/functional/cpu/unfold1d_backward.cpp
    nt/functional/cpu/unfold2d_backward.cpp
    nt/functional/cpu/unfold3d_backward.cpp
    nt/functional/cpu/fold_backward.cpp
    nt/functional/cpu/rand.cpp
    nt/functional/cpu/fused.cpp
    nt/functional/cpu/operators.cpp
    nt/functional/cpu/normalize.cpp
    nt/functional/cpu/min_max.cpp
    nt/functional/cpu/compare.cpp
    nt/functional/cpu/softmax.cpp
    nt/functional/cpu/sum_exp_log.cpp
    nt/functional/cpu/fractional_pooling.cpp
    nt/functional/cpu/activation_functions.cpp
    nt/functional/cpu/trig.cpp
    nt/functional/cpu/sort.cpp
    nt/functional/cpu/sort_tensor.cpp
    nt/functional/cpu/unique.cpp
    nt/functional/cpu/complex.cpp
    nt/functional/cpu/convert.cpp
    nt/functional/cpu/fill.cpp
    nt/functional/cpu/mesh.cpp

)


#split linalg into different functions
#this reduces compile times when specific functions need to be changed
#would like to do this for all general function in the future and current 
set(LINALG_SOURCES
    nt/linalg/SVD.cpp
    nt/linalg/QR.cpp
    nt/linalg/inv.cpp
    nt/linalg/column_space.cpp
    nt/linalg/process/SVD.cpp
    nt/linalg/process/QR.cpp
    nt/linalg/process/inv.cpp
    nt/linalg/process/null_space.cpp
    nt/linalg/process/column_space.cpp
    nt/linalg/adjugate.cpp
    nt/linalg/determinant.cpp
    nt/linalg/norm.cpp
    nt/linalg/null_space.cpp
    nt/linalg/pivots.cpp
    nt/linalg/eye.cpp
    nt/linalg/independent.cpp
)

set(FMRI_SOURCES
    nt/fmri/tensor_files/save_load.cpp
)

set(TYPES_SOURCES
    nt/dtype/ArrayVoid.cpp
    nt/dtype/DType.cpp
    nt/dtype/DType_operators.cpp
    nt/dtype/ranges.cpp
    nt/dtype/Scalar.cpp
    nt/types/Types.cpp
    nt/convert/Convert.cpp
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


#this also includes autograds and layers
set(AI_SOURCES
    nt/nn/Layer.cpp
    nt/nn/LNN.cpp
    nt/nn/TensorGrad.cpp
    nt/nn/ScalarGrad.cpp
    nt/nn/functional/functional.cpp
    nt/nn/functional/pooling.cpp
    nt/nn/functional/non_linear.cpp
    nt/nn/Loss.cpp
    nt/nn/Optimizers.cpp
    nt/nn/Module.cpp
    #ncps sources
    nt/nn/ncps/lstm_cell.cpp
    nt/nn/ncps/wiring/wiring.cpp
    nt/nn/ncps/cfc/wired_cfc_cell.cpp
    nt/nn/ncps/cfc/cfc_cell.cpp
    nt/nn/ncps/cfc/cfc.cpp
    nt/nn/ncps/ltc/ltc.cpp
    nt/nn/ncps/ltc/ltc_cell.cpp
    #individual layers
    nt/nn/layers/BatchNorm1D.cpp
    nt/nn/layers/Conv1D.cpp
    nt/nn/layers/Conv2D.cpp
    nt/nn/layers/Conv3D.cpp
    nt/nn/layers/ConvTranspose1D.cpp
    nt/nn/layers/ConvTranspose2D.cpp
    nt/nn/layers/ConvTranspose3D.cpp
    nt/nn/layers/Dropout.cpp
    nt/nn/layers/Fold.cpp
    nt/nn/layers/Functional.cpp
    nt/nn/layers/Identity.cpp
    nt/nn/layers/Linear.cpp
    nt/nn/layers/SymmetricBilinear.cpp
    nt/nn/layers/Softplus.cpp
    nt/nn/layers/Unfold1D.cpp
    nt/nn/layers/Unfold2D.cpp
    nt/nn/layers/Unfold3D.cpp
    nt/nn/layers/AdaptiveAvgPool1D.cpp
    nt/nn/layers/AdaptiveAvgPool2D.cpp
    nt/nn/layers/AdaptiveAvgPool3D.cpp
    nt/nn/layers/AdaptiveLPPool1D.cpp
    nt/nn/layers/AdaptiveLPPool2D.cpp
    nt/nn/layers/AdaptiveLPPool3D.cpp
    nt/nn/layers/AdaptiveMaxPool1D.cpp
    nt/nn/layers/AdaptiveMaxPool2D.cpp
    nt/nn/layers/AdaptiveMaxPool3D.cpp
    nt/nn/layers/AvgPool1D.cpp
    nt/nn/layers/AvgPool2D.cpp
    nt/nn/layers/AvgPool3D.cpp
    nt/nn/layers/FractionalMaxPool2D.cpp
    nt/nn/layers/FractionalMaxPool3D.cpp
    nt/nn/layers/LPPool1D.cpp
    nt/nn/layers/LPPool2D.cpp
    nt/nn/layers/LPPool3D.cpp
    nt/nn/layers/MaxPool1D.cpp
    nt/nn/layers/MaxPool2D.cpp
    nt/nn/layers/MaxPool3D.cpp
    nt/nn/layers/MaxUnPool1D.cpp
    nt/nn/layers/MaxUnPool2D.cpp
    nt/nn/layers/MaxUnPool3D.cpp
)

set(MP_SOURCES
    nt/mp/Threading.cpp
)

set(TDA_SOURCES
    nt/tda/old_tda/Simplex.cpp
    nt/tda/old_tda/Simplex2d.cpp
    nt/tda/old_tda/Shapes.cpp
    nt/tda/old_tda/Basis.cpp
    nt/tda/old_tda/Points.cpp
    nt/tda/old_tda/Points2d.cpp
    nt/tda/old_tda/KDTree.cpp
    # nt/tda/old_tda/refinement/refine.cpp <- unreleased
    nt/tda/old_tda/BatchBasis.cpp
    nt/tda/old_tda/BatchKDTree.cpp
    nt/tda/old_tda/BatchPoints.cpp
    nt/tda/BasisOverlapping.cpp
    nt/tda/Boundaries.cpp
    nt/tda/Homology.cpp
    nt/tda/MatrixReduction.cpp
    nt/tda/cpu/MatrixReduction.cpp
    nt/tda/PlotDiagrams.cpp
    nt/tda/Points.cpp
    nt/tda/SimplexConstruct.cpp
    nt/tda/SimplexRadi.cpp
    #learned persistent homology
    nt/tda/nn/distance.cpp
    nt/tda/nn/filtration.cpp
    nt/tda/nn/laplacian.cpp
    nt/tda/nn/boundaries.cpp
    nt/tda/nn/loss.cpp
)

set(SPARSE_SOURCES
    nt/sparse/SparseTensor.cpp
    nt/sparse/SparseMatrix.cpp
    nt/sparse/SparseDataMatrix.cpp
)

set(IMAGES_SOURCES
    nt/images/image.cpp
)

set(MEMORY_SOURCES
    nt/memory/bucket.cpp
    nt/memory/device.cpp
)



# Source files
set(SOURCES
    ${MATMULT_SOURCES}
    ${CORE_SOURCES}
    ${FUNCTIONAL_SOURCES}
    ${LINALG_SOURCES}
    ${TYPES_SOURCES}
    ${AI_SOURCES}
    ${MP_SOURCES}
    ${TDA_SOURCES}
    ${FMRI_SOURCES}
    ${SPARSE_SOURCES}
    ${IMAGES_SOURCES}
    ${MEMORY_SOURCES}
)

# Add NeuroTensor library
add_library(neurotensor STATIC ${SOURCES})
target_include_directories(neurotensor PUBLIC ${CMAKE_SOURCE_DIR})

# Optional: shared instead of static
# add_library(neurotensor SHARED ${SOURCES})

# Link libraries
target_link_libraries(neurotensor
    ${TBB_LIB}
    ${MATPLOT_LIB}
    pthread
    niftiio
    znz
)

# Install the library and headers
install(TARGETS neurotensor
    EXPORT neurotensorTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

file(GLOB_RECURSE NT_HEADERS ${CMAKE_SOURCE_DIR}/nt/*.h ${CMAKE_SOURCE_DIR}/nt/*.hpp)
install(FILES ${NT_HEADERS} DESTINATION include/nt)
# install(DIRECTORY ${CMAKE_SOURCE_DIR}/nt DESTINATION include)

# Set RPATH for runtime
# MKL Usage depreciated, so this is no longer needed
# set_target_properties(main.out PROPERTIES
# 	INSTALL_RPATH_USE_LINK_PATH TRUE
# )

# Option to build main
option(BUILD_MAIN "Build main executable (main.cpp)" ON)

if(BUILD_MAIN)
    add_executable(main.out main.cpp)
    target_link_libraries(main.out PRIVATE neurotensor)

    # Optional RPATH
    # set_target_properties(main.out PROPERTIES
    #     INSTALL_RPATH_USE_LINK_PATH TRUE
    # )

    install(TARGETS main.out DESTINATION bin)
endif()


