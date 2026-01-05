cmake_minimum_required(VERSION 3.10)
project(NeuroTensor LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/flags.cmake)
set(CMAKE_VERBOSE_MAKEFILE OFF) 

message(STATUS "Root path is " ${CMAKE_FIND_ROOT_PATH})
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/tbb_config.cmake)
# ${CMAKE_BINARY_DIR}/third_party/matplot/source/matplot
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/matplot_config.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/include_directories.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/core_sources.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/matmult_sources.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/functional_sources.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/functional_cpu_sources.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/linalg_sources.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/null_space_sources.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/column_space_sources.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/inv_sources.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/svd_sources.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/qr_sources.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/fmri_sources.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/types_sources.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/uint128_config.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/ai_sources.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/mp_sources.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/tda_sources.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/sparse_sources.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/images_sources.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/memory_sources.cmake)


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

# This is going to make a list that contains all the filenames of the libraries that should be
#   wrapped into neurotensor

foreach(dep IN LISTS NEUROTENSOR_DEPS)
    list(APPEND NEUROTENSOR_LIBS
        "${CMAKE_BINARY_DIR}/${CONFIG_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}${dep}${CMAKE_STATIC_LIBRARY_SUFFIX}"
    )
endforeach()


# This is going to look for all the library files that exist
# if one of them doesn't -> then they don't all exist and mark which ones don't
set(NEUROTENSOR_UNMADE_LIBRARY_FILES "") # empty list
set(NEUROTENSOR_UNMADE_LIBRARIES "") # empty list
set(NEUROTENSOR_ALL_LIBRARY_FILES_EXIST TRUE)
foreach(file_path dep IN ZIP_LISTS NEUROTENSOR_LIBS NEUROTENSOR_DEPS)
    if(NOT EXISTS "${file_path}")
        set(NEUROTENSOR_ALL_LIBRARY_FILES_EXIST FALSE)
        list(APPEND NEUROTENSOR_UNMADE_LIBRARY_FILES file_path) 
        list(APPEND NEUROTENSOR_UNMADE_LIBRARIES dep) 
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
    list(APPEND NEUROTENSOR_UNMADE_LIBRARIES ${TBB_LIB})
    message(STATUS "TBB File not found")
endif()

if(NEUROTENSOR_MATPLOT_STATIC_LIBRARY_FILES)
    list(GET NEUROTENSOR_MATPLOT_STATIC_LIBRARY_FILES 0 NEUROTENSOR_MATPLOT_STATIC_LIBRARY_FILE)
    list(APPEND NEUROTENSOR_LIBS NEUROTENSOR_MATPLOT_STATIC_LIBRARY_FILE)
    message(STATUS "Found matplot at " ${NEUROTENSOR_MATPLOT_STATIC_LIBRARY_FILE}) 
else()
    list(APPEND NEUROTENSOR_UNMADE_LIBRARIES ${MATPLOT_LIB})
message(STATUS "matplot File not found")
endif()

if(NEUROTENSOR_ZNZ_STATIC_LIBRARY_FILES)
    list(GET NEUROTENSOR_ZNZ_STATIC_LIBRARY_FILES 0 NEUROTENSOR_ZNZ_STATIC_LIBRARY_FILE)
    list(APPEND NEUROTENSOR_LIBS NEUROTENSOR_ZNZ_STATIC_LIBRARY_FILE)
    message(STATUS "Found znz at " ${NEUROTENSOR_ZNZ_STATIC_LIBRARY_FILE}) 
else()
    list(APPEND NEUROTENSOR_UNMADE_LIBRARIES znz)
message(STATUS "znz File not found")
endif()

if(NEUROTENSOR_NIFTIIO_STATIC_LIBRARY_FILES)
    list(GET NEUROTENSOR_NIFTIIO_STATIC_LIBRARY_FILES 0 NEUROTENSOR_NIFTIIO_STATIC_LIBRARY_FILE)
    list(APPEND NEUROTENSOR_LIBS NEUROTENSOR_NIFTIIO_STATIC_LIBRARY_FILE)
    message(STATUS "Found niftiio at " ${NEUROTENSOR_NIFTIIO_STATIC_LIBRARY_FILE}) 
else()
    list(APPEND NEUROTENSOR_UNMADE_LIBRARIES niftiio)
    message(STATUS "Niftiio File not found")
endif()

option(BUILD_ARCHIVE_FORCE "Force the building of an archive even if all library files dont exist, build them" OFF)

function(build_missing_neurotensor_deps)
    # Inputs: list of dep names, list of files, boolean flag
    set(all_exist ${NEUROTENSOR_ALL_LIBRARY_FILES_EXIST})

    if(all_exist)
        message(STATUS "All NeuroTensor libraries already built.")
        return()
    endif()

    message(STATUS "Some NeuroTensor libraries are missing. Building missing ones...")

    foreach(dep IN LISTS NEUROTENSOR_UNMADE_LIBRARIES)
        if(TARGET ${dep})
            message(STATUS "Building missing library target: ${dep}")

            add_custom_target(build_${dep}
                COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target ${dep}
                COMMENT "Building missing dependency ${dep}"
            )

            # Run immediately when function is called
            add_dependencies(build_missing_deps build_${dep})

        else()
            message(WARNING "Dependency target '${dep}' does not exist in this project!")
        endif()
    endforeach()
endfunction()

#If build archive force is true, it will build everything not already built, and then have the main target built
if(BUILD_ARCHIVE_FORCE)
    build_missing_neurotensor_deps()
    set(NEUROTENSOR_ALL_LIBRARY_FILES_EXIST TRUE)
endif()
    


if(NEUROTENSOR_ALL_LIBRARY_FILES_EXIST)
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



if( NEUROTENSOR_ALL_LIBRARY_FILES_EXIST)
install(FILES "${OUTPUT_ARCHIVE}" DESTINATION lib)
else()
# Install the library and headers
install(TARGETS neurotensor
    EXPORT neurotensorTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
endif()

file(GLOB_RECURSE NT_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/nt/*.h ${CMAKE_CURRENT_SOURCE_DIR}/nt/*.hpp)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/nt/ DESTINATION include/nt FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")

#installing matplot header files
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/third_party/matplot/source DESTINATION include
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")
install(DIRECTORY ${CMAKE_BINARY_DIR}/third_party/matplot/source/matplot DESTINATION include
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
        target_include_directories(main.out PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
        target_include_directories(main.out
            PRIVATE
                ${CMAKE_BINARY_DIR}/third_party/matplot/source/matplot
        )

    else()
        target_link_libraries(main.out PRIVATE ${TBB_LIB} ${MATPLOT_LIB} pthread znz niftiio 
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
        target_include_directories(main.out
            PRIVATE
                ${CMAKE_BINARY_DIR}/third_party/matplot/source/matplot
        )

    endif()

    # target_link_libraries(main.out PRIVATE neurotensor)

    # Optional RPATH
    # set_target_properties(main.out PROPERTIES
    #     INSTALL_RPATH_USE_LINK_PATH TRUE
    # )

    install(TARGETS main.out DESTINATION bin)
endif()

option(BUILD_TESTS "Build NeuroTensor unit tests" ON)

if(BUILD_TESTS)
    file(GLOB_RECURSE TEST_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/tests/src/*.cpp)

    message(STATUS "Test sources are ${TEST_SOURCES}")

    add_executable(nt_tests ${TEST_SOURCES})

    target_include_directories(nt_tests PRIVATE
        ${PROJECT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/test
    )

    target_compile_options(nt_tests PRIVATE -fsanitize=address -g)
    target_link_options(nt_tests PRIVATE -fsanitize=address -g)

    if(NEUROTENSOR_ALL_LIBRARY_FILES_EXIST)
        target_link_libraries(nt_tests PRIVATE neurotensor)
    else()
        target_link_libraries(nt_tests PRIVATE ${TBB_LIB} ${MATPLOT_LIB} pthread znz niftiio ${NEUROTENSOR_DEPS})
    endif()

    install(TARGETS nt_tests DESTINATION bin)
endif()



