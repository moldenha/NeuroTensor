# Find TBB
find_package(TBB QUIET)

if (TBB_FOUND)
    # If TBB is found, link it
    message(STATUS "Found system-installed TBB")
    message(STATUS "TBB_INCLUDE_DIRS = ${TBB_INCLUDE_DIRS}")
    message(STATUS "TBB_LIBRARIES = ${TBB_LIBRARIES}")
    message(STATUS "TBB_LIBRARY = ${TBB_LIBRARY}")
    message(STATUS "TBB_DIR = ${TBB_DIR}")
    include_directories(${TBB_INCLUDE_DIRS})
    set(TBB_LIB TBB::tbb)
else()
    # If TBB is not found, build it from the third-party directory
    message(STATUS "System-installed TBB not found, building from source")
    set(TBB_TEST OFF CACHE BOOL "Disable TBB tests") # Must be above add_subdirectory to take effect
    add_subdirectory(third_party/tbb)
    set(TBB_LIB tbb)  # Use the built TBB target, when built from source just called tbb
    include_directories(${CMAKE_SOURCE_DIR}/third_party/tbb/include)
endif()
