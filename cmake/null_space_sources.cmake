include(${CMAKE_CURRENT_LIST_DIR}/base_dir.cmake)

set(NULL_SPACE_SOURCES
    ${BASE_DIR}/nt/linalg/null_space.cpp

    # nt/linalg/process/null_space.cpp
    # null_space broken up into different types for compilation reasons
    ${BASE_DIR}/nt/linalg/process/null_space/null_space_complex_double.cpp 
    ${BASE_DIR}/nt/linalg/process/null_space/null_space_double.cpp         
    ${BASE_DIR}/nt/linalg/process/null_space/null_space_int16.cpp          
    ${BASE_DIR}/nt/linalg/process/null_space/null_space_int64.cpp          
    ${BASE_DIR}/nt/linalg/process/null_space/null_space_uint16.cpp         
    ${BASE_DIR}/nt/linalg/process/null_space/null_space_uint8.cpp
    ${BASE_DIR}/nt/linalg/process/null_space/null_space_complex_float.cpp  
    ${BASE_DIR}/nt/linalg/process/null_space/null_space_float.cpp          
    ${BASE_DIR}/nt/linalg/process/null_space/null_space_int32.cpp          
    ${BASE_DIR}/nt/linalg/process/null_space/null_space_int8.cpp           
    ${BASE_DIR}/nt/linalg/process/null_space/null_space_uint32.cpp
)
