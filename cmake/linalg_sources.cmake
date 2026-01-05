include(${CMAKE_CURRENT_LIST_DIR}/base_dir.cmake)

#split linalg into different functions
#this reduces compile times when specific functions need to be changed
#would like to do this for all general function in the future and current 
set(LINALG_SOURCES
    ${BASE_DIR}/nt/linalg/adjugate.cpp
    ${BASE_DIR}/nt/linalg/determinant.cpp
    ${BASE_DIR}/nt/linalg/norm.cpp
    ${BASE_DIR}/nt/linalg/pivots.cpp
    ${BASE_DIR}/nt/linalg/eye.cpp
    ${BASE_DIR}/nt/linalg/independent.cpp
)
