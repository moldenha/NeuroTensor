

message(STATUS "building TBB from source")
set(TBB_TEST OFF CACHE BOOL "Disable TBB tests") # Must be above add_subdirectory to take effect
# set(TBB_BUILD_TBBMALLOC OFF CACHE BOOL "" FORCE)
# set(TBB_BUILD_TBBMALLOC_PROXY OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Disable shared build of TBB" FORCE)
set(TBB_INSTALL_DIR_SKIP ON CACHE BOOL "Skip TBB install" FORCE) # No reason to install TBB
add_compile_definitions(__TBB_DYNAMIC_LOAD_ENABLED=0)
set(TBB_ENABLE_IPO OFF CACHE BOOL "Disable IPO to compile tbb statically")
add_subdirectory(third_party/tbb)
#Needed for GNU compatibility on linux
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_compile_options(tbb PRIVATE -Wno-stringop-overflow)
endif()
set(TBB_LIB tbb)  # Use the built TBB target, when built from source just called tbb
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/tbb/include)


#if(BUILD_MAIN)
## Find TBB
#find_package(TBB QUIET)
## if (TBB_FOUND)
##     # If TBB is found, link it
##     message(STATUS "Found system-installed TBB")
##     message(STATUS "TBB_INCLUDE_DIRS = ${TBB_INCLUDE_DIRS}")
##     message(STATUS "TBB_LIBRARIES = ${TBB_LIBRARIES}")
##     message(STATUS "TBB_LIBRARY = ${TBB_LIBRARY}")
##     message(STATUS "TBB_DIR = ${TBB_DIR}")
##     include_directories(${TBB_INCLUDE_DIRS})
##     set(TBB_LIB TBB::tbb TBB::tbbmalloc)
## else()
#    # If TBB is not found, build it from the third-party directory
#    message(STATUS "System-installed TBB not found, building from source")
#    set(TBB_TEST OFF CACHE BOOL "Disable TBB tests") # Must be above add_subdirectory to take effect
#    # set(TBB_BUILD_TBBMALLOC OFF CACHE BOOL "" FORCE)
#    # set(TBB_BUILD_TBBMALLOC_PROXY OFF CACHE BOOL "" FORCE)
#    set(BUILD_SHARED_LIBS OFF CACHE BOOL "Disable shared build of TBB" FORCE)
#    set(TBB_INSTALL_DIR_SKIP ON CACHE BOOL "Skip TBB install" FORCE) # No reason to install TBB
#    add_compile_definitions(__TBB_DYNAMIC_LOAD_ENABLED=0)
#    set(TBB_ENABLE_IPO OFF CACHE BOOL "Disable IPO to compile tbb statically")
#    add_subdirectory(third_party/tbb)
#    #Needed for GNU compatibility on linux
#    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
#      target_compile_options(tbb PRIVATE -Wno-stringop-overflow)
#    endif()
#    set(TBB_LIB tbb tbbmalloc)  # Use the built TBB target, when built from source just called tbb
#    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/tbb/include)

## endif()
#else()
#    # If TBB is not found, build it from the third-party directory
#    message(STATUS "building TBB from source")
#    set(TBB_TEST OFF CACHE BOOL "Disable TBB tests") # Must be above add_subdirectory to take effect
#    # set(TBB_BUILD_TBBMALLOC OFF CACHE BOOL "" FORCE)
#    # set(TBB_BUILD_TBBMALLOC_PROXY OFF CACHE BOOL "" FORCE)
#    set(BUILD_SHARED_LIBS OFF CACHE BOOL "Disable shared build of TBB" FORCE)
#    set(TBB_INSTALL_DIR_SKIP ON CACHE BOOL "Skip TBB install" FORCE) # No reason to install TBB
#    add_compile_definitions(_SILENCE_CXX17_C_HEADER_DEPRECATION_WARNING)
#    add_compile_definitions(__TBB_DYNAMIC_LOAD_ENABLED=0)
#    set(TBB_ENABLE_IPO OFF CACHE BOOL "Disable IPO to compile tbb statically")
#    add_subdirectory(third_party/tbb)
#    #Needed for GNU compatibility on linux
#    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
#      target_compile_options(tbb PRIVATE -Wno-stringop-overflow)
#    endif()
#    set(TBB_LIB tbb tbbmalloc)  # Use the built TBB target, when built from source just called tbb
#    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/tbb/include)

#    # set(TBB_CORE_LIB_FINAL_PATH "$<TARGET_FILE:TBB::tbb>")
#    # message(STATUS "TBB core library will be found at: ${TBB_CORE_LIB_FINAL_PATH}")
#endif()
