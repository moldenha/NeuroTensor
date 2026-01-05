include(${CMAKE_CURRENT_LIST_DIR}/base_dir.cmake)

set(COLUMN_SPACE_SOURCES
    ${BASE_DIR}/nt/linalg/column_space.cpp

    # nt/linalg/process/column_space.cpp
    # column_space broken up into different types for compilation reasons
    ${BASE_DIR}/nt/linalg/process/column_space/column_space_complex_double.cpp 
    ${BASE_DIR}/nt/linalg/process/column_space/column_space_double.cpp         
    ${BASE_DIR}/nt/linalg/process/column_space/column_space_int16.cpp          
    ${BASE_DIR}/nt/linalg/process/column_space/column_space_int64.cpp          
    ${BASE_DIR}/nt/linalg/process/column_space/column_space_uint16.cpp         
    ${BASE_DIR}/nt/linalg/process/column_space/column_space_uint8.cpp
    ${BASE_DIR}/nt/linalg/process/column_space/column_space_complex_float.cpp  
    ${BASE_DIR}/nt/linalg/process/column_space/column_space_float.cpp          
    ${BASE_DIR}/nt/linalg/process/column_space/column_space_int32.cpp          
    ${BASE_DIR}/nt/linalg/process/column_space/column_space_int8.cpp           
    ${BASE_DIR}/nt/linalg/process/column_space/column_space_uint32.cpp

)

