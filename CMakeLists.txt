cmake_minimum_required(VERSION 3.10)
project(NeuroTensor LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/flags.cmake)
set(CMAKE_VERBOSE_MAKEFILE OFF) 

message(STATUS "Root path is " ${CMAKE_FIND_ROOT_PATH})
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/tbb_config.cmake)


 find_package(Matplot++ QUIET)

 if (Matplot++_FOUND)
     message(STATUS "Found system-installed Matplot++")
     include_directories(${Matplot++_INCLUDE_DIRS})
     set(MATPLOT_LIB Matplot++::matplot)
 else()
     message(STATUS "System-installed Matplot++ not found, using source")
     include_directories(third_party/matplot/source)
     add_subdirectory(third_party/matplot)
     set(MATPLOT_LIB matplot)
 endif()






# Include directories
include_directories(
    # ${MKL_INCLUDE_DIR}
    # ${TBB_INCLUDE_DIRS}
    # ${Boost_INCLUDE_DIRS}
    # ${CMAKE_SOURCE_DIR}/third_party/tbb/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/simde
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/half
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/stb
    #For matrix and vector linear algebra calculations
    #This incorperates most of the nt::linalg namespace
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/nifti_clib/znzlib
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/nifti_clib/niftilib

    #Boost files
    #Mainly for 128 bit data types
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/boost_config/include/
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/multiprecision/include/

)


# Include NIfTI macros
include(${CMAKE_CURRENT_SOURCE_DIR}/third_party/nifti_clib/cmake/nifti_macros.cmake)

# Add znzlib for compressed NIfTI support
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third_party/nifti_clib/znzlib)

# Add NIfTI core library (creates niftiio target)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third_party/nifti_clib/niftilib)



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
    
)

set(FUNCTIONAL_CPU_SOURCES
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
    nt/linalg/adjugate.cpp
    nt/linalg/determinant.cpp
    nt/linalg/norm.cpp
    nt/linalg/pivots.cpp
    nt/linalg/eye.cpp
    nt/linalg/independent.cpp
)

set(NULL_SPACE_SOURCES
    nt/linalg/null_space.cpp

    # nt/linalg/process/null_space.cpp
    # null_space broken up into different types for compilation reasons
    nt/linalg/process/null_space/null_space_complex_double.cpp 
    nt/linalg/process/null_space/null_space_double.cpp         
    nt/linalg/process/null_space/null_space_int16.cpp          
    nt/linalg/process/null_space/null_space_int64.cpp          
    nt/linalg/process/null_space/null_space_uint16.cpp         
    nt/linalg/process/null_space/null_space_uint8.cpp
    nt/linalg/process/null_space/null_space_complex_float.cpp  
    nt/linalg/process/null_space/null_space_float.cpp          
    nt/linalg/process/null_space/null_space_int32.cpp          
    nt/linalg/process/null_space/null_space_int8.cpp           
    nt/linalg/process/null_space/null_space_uint32.cpp
)

set(COLUMN_SPACE_SOURCES
    nt/linalg/column_space.cpp

    # nt/linalg/process/column_space.cpp
    # column_space broken up into different types for compilation reasons
    nt/linalg/process/column_space/column_space_complex_double.cpp 
    nt/linalg/process/column_space/column_space_double.cpp         
    nt/linalg/process/column_space/column_space_int16.cpp          
    nt/linalg/process/column_space/column_space_int64.cpp          
    nt/linalg/process/column_space/column_space_uint16.cpp         
    nt/linalg/process/column_space/column_space_uint8.cpp
    nt/linalg/process/column_space/column_space_complex_float.cpp  
    nt/linalg/process/column_space/column_space_float.cpp          
    nt/linalg/process/column_space/column_space_int32.cpp          
    nt/linalg/process/column_space/column_space_int8.cpp           
    nt/linalg/process/column_space/column_space_uint32.cpp

)

set(INV_SOURCES
    nt/linalg/inv.cpp

    # nt/linalg/process/inv.cpp
    # inv broken up into different types for compilation reasons
    nt/linalg/process/inv/inv_complex_double.cpp 
    nt/linalg/process/inv/inv_double.cpp         
    nt/linalg/process/inv/inv_int16.cpp          
    nt/linalg/process/inv/inv_int64.cpp          
    nt/linalg/process/inv/inv_uint16.cpp         
    nt/linalg/process/inv/inv_uint8.cpp
    nt/linalg/process/inv/inv_complex_float.cpp  
    nt/linalg/process/inv/inv_float.cpp          
    nt/linalg/process/inv/inv_int32.cpp          
    nt/linalg/process/inv/inv_int8.cpp           
    nt/linalg/process/inv/inv_uint32.cpp

)

set(SVD_SOURCES
    nt/linalg/SVD.cpp
    #svd broken up into different types for compilation reasons
    nt/linalg/process/svd/svd_complex_double.cpp 
    nt/linalg/process/svd/svd_double.cpp         
    nt/linalg/process/svd/svd_int16.cpp          
    nt/linalg/process/svd/svd_int64.cpp          
    nt/linalg/process/svd/svd_uint16.cpp         
    nt/linalg/process/svd/svd_uint8.cpp
    nt/linalg/process/svd/svd_complex_float.cpp  
    nt/linalg/process/svd/svd_float.cpp          
    nt/linalg/process/svd/svd_int32.cpp          
    nt/linalg/process/svd/svd_int8.cpp           
    nt/linalg/process/svd/svd_uint32.cpp
)

set(QR_SOURCES
    nt/linalg/QR.cpp
    
    # nt/linalg/process/QR.cpp
    #qr broken up into different types for compilation reasons
    nt/linalg/process/qr/qr_complex_double.cpp 
    nt/linalg/process/qr/qr_double.cpp         
    nt/linalg/process/qr/qr_int16.cpp          
    nt/linalg/process/qr/qr_int64.cpp          
    nt/linalg/process/qr/qr_uint16.cpp         
    nt/linalg/process/qr/qr_uint8.cpp
    nt/linalg/process/qr/qr_complex_float.cpp  
    nt/linalg/process/qr/qr_float.cpp          
    nt/linalg/process/qr/qr_int32.cpp          
    nt/linalg/process/qr/qr_int8.cpp           
    nt/linalg/process/qr/qr_uint32.cpp

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



# # Source files
# set(SOURCES
#     ${MATMULT_SOURCES}
#     ${CORE_SOURCES}
#     ${TYPES_SOURCES}
#     ${FUNCTIONAL_SOURCES}
#     ${LINALG_SOURCES}
#     ${AI_SOURCES}
#     ${MP_SOURCES}
#     ${TDA_SOURCES}
#     ${FMRI_SOURCES}
#     ${SPARSE_SOURCES}
#     ${IMAGES_SOURCES}
#     ${MEMORY_SOURCES}
# )


set(NEUROTENSOR_SOURCES_ALL
    ${MATMULT_SOURCES}
    ${CORE_SOURCES}
    ${TYPES_SOURCES}
    ${FUNCTIONAL_SOURCES}
    ${FUNCTIONAL_CPU_SOURCES}
    ${LINALG_SOURCES}
    ${AI_SOURCES}
    ${MP_SOURCES}
    ${TDA_SOURCES}
    ${FMRI_SOURCES}
    ${SPARSE_SOURCES}
    ${IMAGES_SOURCES}
    ${MEMORY_SOURCES}
    ${SVD_SOURCES}
    ${QR_SOURCES}
    ${INV_SOURCES}
    ${COLUMN_SPACE_SOURCES}
    ${NULL_SPACE_SOURCES}
)


 

add_library(nt_core_obj OBJECT ${CORE_SOURCES})
target_include_directories(nt_core_obj PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(nt_core_obj PUBLIC ${TBB_LIB} pthread)
add_library(nt_core STATIC $<TARGET_OBJECTS:nt_core_obj>)
target_include_directories(nt_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_library(nt_types STATIC ${TYPES_SOURCES})
target_include_directories(nt_types PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(nt_types PRIVATE ${TBB_LIB} pthread)


# Create object libraries to isolate object files
add_library(nt_functional_obj OBJECT ${FUNCTIONAL_SOURCES})
target_include_directories(nt_functional_obj PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(nt_functional_obj PRIVATE ${TBB_LIB} pthread)

add_library(nt_functional_cpu_obj OBJECT ${FUNCTIONAL_CPU_SOURCES})
target_include_directories(nt_functional_cpu_obj PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(nt_functional_cpu_obj PRIVATE ${TBB_LIB} pthread)
# -- The above is to avoid duplicate file/object names --

add_library(nt_functional STATIC $<TARGET_OBJECTS:nt_functional_obj>)
target_include_directories(nt_functional PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_library(nt_functional_cpu STATIC $<TARGET_OBJECTS:nt_functional_cpu_obj>)
target_include_directories(nt_functional_cpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_library(nt_matmult STATIC ${MATMULT_SOURCES})
target_include_directories(nt_matmult PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(nt_matmult PRIVATE ${TBB_LIB} pthread)

add_library(nt_linalg STATIC ${LINALG_SOURCES})
target_include_directories(nt_linalg PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(nt_linalg PRIVATE ${TBB_LIB} pthread)

add_library(nt_svd STATIC ${SVD_SOURCES})
target_include_directories(nt_svd PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_library(nt_qr STATIC ${QR_SOURCES})
target_include_directories(nt_qr PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_library(nt_inv STATIC ${INV_SOURCES})
target_include_directories(nt_inv PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_library(nt_column_space STATIC ${COLUMN_SPACE_SOURCES})
target_include_directories(nt_column_space PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_library(nt_null_space STATIC ${NULL_SPACE_SOURCES})
target_include_directories(nt_null_space PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_library(nt_ai STATIC ${AI_SOURCES})
target_include_directories(nt_ai PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# -- Handling TDA Source Objects

add_library(nt_tda_obj OBJECT ${TDA_SOURCES})
target_include_directories(nt_tda_obj PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(nt_tda_obj PUBLIC ${MATPLOT_LIB}) # Reconfiguring matplot lib part of tda objects
target_link_libraries(nt_tda_obj PRIVATE ${TBB_LIB} pthread)

# -- TDA_SOURCES Have Duplicate Filenames like Points and MatrixReduction

add_library(nt_tda STATIC $<TARGET_OBJECTS:nt_tda_obj>)
target_include_directories(nt_tda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# target_link_libraries(nt_tda ${MATPLOT_LIB})



add_library(nt_sparse STATIC ${SPARSE_SOURCES})
target_include_directories(nt_sparse PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(nt_sparse PRIVATE ${TBB_LIB} pthread)

add_library(nt_images STATIC ${IMAGES_SOURCES})
target_include_directories(nt_images PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_library(nt_memory STATIC ${MEMORY_SOURCES})
target_include_directories(nt_memory PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(nt_memory PRIVATE ${TBB_LIB} pthread)

add_library(nt_multi_processing STATIC ${MP_SOURCES})
target_include_directories(nt_multi_processing PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(nt_multi_processing PUBLIC ${TBB_LIB} pthread)

add_library(nt_fmri STATIC ${FMRI_SOURCES})
target_include_directories(nt_fmri PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(nt_fmri PRIVATE niftiio znz)

# add_library(nt_third_party STATIC nt/utils/utils_systems/dummy_third.cpp)
# target_include_directories(nt_third_party PUBLIC ${CMAKE_SOURCE_DIR})



# Determine config directory (Release/Debug/etc.) — safe for both single/multi-config
set(CONFIG_DIR ${CMAKE_CFG_INTDIR}) # Will be "Release" on MSVC, "." on single-config generators

set(NEUROTENSOR_DEPS
    nt_core
    nt_types
    nt_functional
    nt_functional_cpu
    nt_matmult
    nt_linalg
    nt_svd
    nt_qr
    nt_inv
    nt_column_space
    nt_null_space
    nt_ai
    nt_tda
    nt_sparse
    nt_images
    nt_memory
    nt_multi_processing
    nt_fmri
)

#if(UNIX)
##Generate full paths with prefix/suffix
#foreach(dep IN LISTS NEUROTENSOR_DEPS)
#    list(APPEND NEUROTENSOR_LIBS $<TARGET_FILE:${dep}>)
#endforeach()
#else()
foreach(dep IN LISTS NEUROTENSOR_DEPS)
    list(APPEND NEUROTENSOR_LIBS
        "${CMAKE_BINARY_DIR}/${CONFIG_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}${dep}${CMAKE_STATIC_LIBRARY_SUFFIX}"
    )
endforeach()
# endif()

#Looking for tbb static library

# file(GLOB_RECURSE FOUND_ALL_NEUROTENSOR_LIBRARY_FILES
#     "${CMAKE_BINARY_DIR}/${CONFIG_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}tbb${CMAKE_STATIC_LIBRARY_SUFFIX}"
#     "${CMAKE_BINARY_DIR}/${CONFIG_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}matplot${CMAKE_STATIC_LIBRARY_SUFFIX}"
#     "${CMAKE_BINARY_DIR}/${CONFIG_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}znz${CMAKE_STATIC_LIBRARY_SUFFIX}"
#     "${CMAKE_BINARY_DIR}/${CONFIG_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}niftiio${CMAKE_STATIC_LIBRARY_SUFFIX}"
#     # If you had multiple patterns, you'd list them:
#     # "${SEARCH_START_DIR}/another_pattern.log"
#     # CONFIGURE_DEPENDS # Consider adding this for CMake 3.11+ for better re-configuration
# )



set(NEUROTENSOR_ALL_LIBRARY_FILES_EXIST TRUE)
foreach(file_path IN LISTS NEUROTENSOR_LIBS)
    if(NOT EXISTS "${file_path}")
        set(NEUROTENSOR_ALL_LIBRARY_FILES_EXIST FALSE)
        message(STATUS "Cannot find file " ${file_path})
    endif()
endforeach()

message(STATUS "Looking for ${CMAKE_STATIC_LIBRARY_PREFIX}tbb${CMAKE_STATIC_LIBRARY_SUFFIX}")
file(GLOB_RECURSE NEUROTENSOR_TBB_STATIC_LIBRARY_FILES "${CMAKE_BINARY_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}tbb${CMAKE_STATIC_LIBRARY_SUFFIX}*")
message(STATUS "Looking for ${CMAKE_STATIC_LIBRARY_PREFIX}matplot${CMAKE_STATIC_LIBRARY_SUFFIX}")
file(GLOB_RECURSE NEUROTENSOR_MATPLOT_STATIC_LIBRARY_FILES "${CMAKE_BINARY_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}matplot${CMAKE_STATIC_LIBRARY_SUFFIX}*")
message(STATUS "Looking for ${CMAKE_STATIC_LIBRARY_PREFIX}znz${CMAKE_STATIC_LIBRARY_SUFFIX}")
file(GLOB_RECURSE NEUROTENSOR_ZNZ_STATIC_LIBRARY_FILES "${CMAKE_BINARY_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}znz${CMAKE_STATIC_LIBRARY_SUFFIX}*")
message(STATUS "Looking for ${CMAKE_STATIC_LIBRARY_PREFIX}niftiio${CMAKE_STATIC_LIBRARY_SUFFIX}")
file(GLOB_RECURSE NEUROTENSOR_NIFTIIO_STATIC_LIBRARY_FILES "${CMAKE_BINARY_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}NIFTIIO${CMAKE_STATIC_LIBRARY_SUFFIX}*")


if(NEUROTENSOR_TBB_STATIC_LIBRARY_FILES)
    list(GET NEUROTENSOR_TBB_STATIC_LIBRARY_FILES 0 NEUROTENSOR_TBB_STATIC_LIBRARY_FILE)
    list(APPEND NEUROTENSOR_LIBS NEUROTENSOR_TBB_STATIC_LIBRARY_FILE)
    message(STATUS "Found tbb at " ${NEUROTENSOR_TBB_STATIC_LIBRARY_FILE}) 
else()
message(STATUS "TBB File not found")
endif()

if(NEUROTENSOR_MATPLOT_STATIC_LIBRARY_FILES)
    list(GET NEUROTENSOR_MATPLOT_STATIC_LIBRARY_FILES 0 NEUROTENSOR_MATPLOT_STATIC_LIBRARY_FILE)
    list(APPEND NEUROTENSOR_LIBS NEUROTENSOR_MATPLOT_STATIC_LIBRARY_FILE)
    message(STATUS "Found matplot at " ${NEUROTENSOR_MATPLOT_STATIC_LIBRARY_FILE}) 
else()
message(STATUS "matplot File not found")
endif()

if(NEUROTENSOR_ZNZ_STATIC_LIBRARY_FILES)
    list(GET NEUROTENSOR_ZNZ_STATIC_LIBRARY_FILES 0 NEUROTENSOR_ZNZ_STATIC_LIBRARY_FILE)
    list(APPEND NEUROTENSOR_LIBS NEUROTENSOR_ZNZ_STATIC_LIBRARY_FILE)
    message(STATUS "Found znz at " ${NEUROTENSOR_ZNZ_STATIC_LIBRARY_FILE}) 
else()
message(STATUS "znz File not found")
endif()

if(NEUROTENSOR_NIFTIIO_STATIC_LIBRARY_FILES)
    list(GET NEUROTENSOR_NIFTIIO_STATIC_LIBRARY_FILES 0 NEUROTENSOR_NIFTIIO_STATIC_LIBRARY_FILE)
    list(APPEND NEUROTENSOR_LIBS NEUROTENSOR_NIFTIIO_STATIC_LIBRARY_FILE)
    message(STATUS "Found niftiio at " ${NEUROTENSOR_NIFTIIO_STATIC_LIBRARY_FILE}) 
else()
    message(STATUS "Niftiio File not found")
endif()


if(NEUROTENSOR_ALL_LIBRARY_FILES_EXIST )
    message(STATUS "Building neurotensor from static libraries")
    message(STATUS "Looking for object files")
    if (WIN32)
        set(OBJECT_FILE_EXT "*.obj")
    else()
        set(OBJECT_FILE_EXT "*.o")
    endif()
    

    # Collect object files
    file(GLOB_RECURSE NEUROTENSOR_COMPILED_STATIC_OBJECT_FILES
        RELATIVE "${CMAKE_BINARY_DIR}"
        "${CMAKE_BINARY_DIR}/${OBJECT_FILE_EXT}"
    )

    # Output archive name (e.g., libneurotensor.a or neurotensor.lib)
    set(OUTPUT_ARCHIVE "${CMAKE_BINARY_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}neurotensor${CMAKE_STATIC_LIBRARY_SUFFIX}")

    # Create archive using CMAKE_AR
    add_custom_command(
        OUTPUT "${OUTPUT_ARCHIVE}"
        COMMAND "${CMAKE_AR}" rcs "${OUTPUT_ARCHIVE}" ${NEUROTENSOR_COMPILED_STATIC_OBJECT_FILES}
        DEPENDS ${NEUROTENSOR_COMPILED_STATIC_OBJECT_FILES}
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
        COMMENT "Archiving NeuroTensor object files into ${OUTPUT_ARCHIVE}"
        VERBATIM
    )

    # Declare IMPORTED static library target
    add_library(neurotensor STATIC IMPORTED GLOBAL)
    set_target_properties(neurotensor PROPERTIES
        IMPORTED_LOCATION "${OUTPUT_ARCHIVE}"
        LINKER_LANGUAGE CXX
    )

    # Trigger archive creation before the imported target is usable
    add_custom_target(neurotensor_archive DEPENDS "${OUTPUT_ARCHIVE}")
    add_dependencies(neurotensor neurotensor_archive)
    # add_library(neurotensor STATIC
    #                    ${NEUROTENSOR_COMPILED_STATIC_OBJECT_FILES}
    #                     )
    # set_target_properties(neurotensor PROPERTIES LINKER_LANGUAGE CXX)
else()
    message(WARNING "Cannot find all library files, building dummy for neurotensor")
    add_library(neurotensor STATIC nt/utils/utils_systems/dummy.cpp)
endif()



# Third-party libs (adapt path as needed)
# These will be linked because they are built with neurotensor
# list(APPEND NEUROTENSOR_LIBS
#     $<TARGET_FILE:tbb>
#     $<TARGET_FILE:tbbmalloc>
#     $<TARGET_FILE:tbbmalloc>
#     $<TARGET_FILE:niftiio>
#     $<TARGET_FILE:znz>
#     $<TARGET_FILE:matplot++>
# )




# message(STATUS "NEUROTENSOR_LIBS = ${NEUROTENSOR_LIBS}")

# # Cross-platform merging of static libs into one archive
# if(APPLE)
#   add_custom_command(TARGET neurotensor POST_BUILD
#     COMMAND libtool -static -o $<TARGET_FILE:neurotensor> 
#                                 ${NEUROTENSOR_LIBS}
#                                     $<TARGET_FILE:tbb>
#                                     $<TARGET_FILE:tbbmalloc>
#                                     $<TARGET_FILE:niftiio>
#                                     $<TARGET_FILE:znz>
#                                     $<TARGET_FILE:matplot>

#     COMMENT "🔧 macOS: Merging static libraries into libneurotensor.a with libtool"
#   )
# # elseif(UNIX)
# #   file(GLOB_RECURSE OBJECT_FILES "${CMAKE_BINARY_DIR}/*.o")
# #   add_custom_command(TARGET neurotensor POST_BUILD
# #     COMMAND rm -f $<TARGET_FILE:neurotensor>
# #     COMMAND ar crus $<TARGET_FILE:neurotensor> ${OBJECT_FILES}
# #     COMMAND echo "📦 Running ranlib on final archive"
# #     COMMAND ranlib $<TARGET_FILE:neurotensor>
# #     # COMMENT "🔧 Merging all .o files into libneurotensor.a with ar"    # Above command is to generate symbol index 
# #     # COMMENT "If you recieve an error that libtool is not installed run apt-get install libtool"
# #     # COMMAND libtool -static -o $<TARGET_FILE:neurotensor> 
# #     #                             ${NEUROTENSOR_LIBS}
# #     #                                 $<TARGET_FILE:tbb>
# #     #                                 $<TARGET_FILE:tbbmalloc>
# #     #                                 $<TARGET_FILE:niftiio>
# #     #                                 $<TARGET_FILE:znz>
# #     #                                 $<TARGET_FILE:matplot>
# #     # COMMENT "🔧 Linux: Merged static libraries into libneurotensor.a with libtool"
# #   )
# elseif(WIN32)
#   add_custom_command(TARGET neurotensor POST_BUILD
#     COMMAND lib.exe /OUT:$<TARGET_FILE:neurotensor> 
#                         ${NEUROTENSOR_LIBS}
#                         $<TARGET_FILE:tbb>
#                         $<TARGET_FILE:tbbmalloc>
#                         $<TARGET_FILE:niftiio>
#                         $<TARGET_FILE:znz>
#                         $<TARGET_FILE:matplot>

#     COMMENT "🔧 Windows: Merging static libraries into libneurotensor.lib with lib.exe"
#   )
# endif()


# target_link_libraries(neurotensor
#     nt_core
#     nt_types
#     nt_functional
#     nt_functional_cpu
#     nt_matmult
#     nt_linalg
#     nt_svd
#     nt_qr
#     nt_inv
#     nt_column_space
#     nt_null_space
#     nt_ai
#     nt_tda
#     nt_sparse
#     nt_images
#     nt_memory
#     nt_multi_processing
#     nt_fmri
#     nt_third_party
# )

# target_include_directories(neurotensor PUBLIC ${CMAKE_SOURCE_DIR})

# Optional: shared instead of static
# add_library(neurotensor SHARED ${SOURCES})



# Install the library and headers
install(TARGETS neurotensor
    EXPORT neurotensorTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

file(GLOB_RECURSE NT_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/nt/*.h ${CMAKE_CURRENT_SOURCE_DIR}/nt/*.hpp)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/nt/ DESTINATION include/nt FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")

#installing matplot header files
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/third_party/matplot/source DESTINATION include
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/third_party/simde/simde DESTINATION include
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/third_party/half/half DESTINATION include
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")


install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/third_party/multiprecision/include/ DESTINATION include
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/third_party/boost_config/include/ DESTINATION include
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")


# install(DIRECTORY ${CMAKE_SOURCE_DIR}/third_party/ DESTINATION include/third_party
#         FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")

# install(FILES ${NT_HEADERS} DESTINATION include/nt)
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
    if(NEUROTENSOR_ALL_LIBRARY_FILES_EXIST )
        target_link_libraries(main.out PRIVATE neurotensor)

    else()
        target_link_libraries(main.out PRIVATE ${TBB_LIB} ${MATPLOT_LIB} pthread znz niftiio ${NEUROTENSOR_DEPS})
    endif()

    # target_link_libraries(main.out PRIVATE neurotensor)

    # Optional RPATH
    # set_target_properties(main.out PROPERTIES
    #     INSTALL_RPATH_USE_LINK_PATH TRUE
    # )

    install(TARGETS main.out DESTINATION bin)
endif()


