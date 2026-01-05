include(${CMAKE_CURRENT_LIST_DIR}/base_dir.cmake)

# Include directories
include_directories(
    # ${MKL_INCLUDE_DIR}
    # ${TBB_INCLUDE_DIRS}
    # ${Boost_INCLUDE_DIRS}
    # ${CMAKE_LIST_DIR}/third_party/tbb/include
    ${BASE_DIR}/third_party/simde
    ${BASE_DIR}/third_party/half
    ${BASE_DIR}/third_party/stb
    #For matrix and vector linear algebra calculations
    #This incorperates most of the nt::linalg namespace
    ${BASE_DIR}/third_party/eigen
    ${BASE_DIR}/third_party/nifti_clib/znzlib
    ${BASE_DIR}/third_party/nifti_clib/niftilib

    #Boost files
    #Mainly for 128 bit data types
    ${BASE_DIR}/third_party/boost_config/include/
    ${BASE_DIR}/third_party/multiprecision/include/

)


# Include NIfTI macros
include(${BASE_DIR}/third_party/nifti_clib/cmake/nifti_macros.cmake)

# Add znzlib for compressed NIfTI support
add_subdirectory(${BASE_DIR}/third_party/nifti_clib/znzlib)

# Add NIfTI core library (creates niftiio target)
add_subdirectory(${BASE_DIR}/third_party/nifti_clib/niftilib)

