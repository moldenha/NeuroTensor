include(${CMAKE_CURRENT_LIST_DIR}/base_dir.cmake)

set(INV_SOURCES
    ${BASE_DIR}/nt/linalg/inv.cpp

    # nt/linalg/process/inv.cpp
    # inv broken up into different types for compilation reasons
    ${BASE_DIR}/nt/linalg/process/inv/inv_complex_double.cpp 
    ${BASE_DIR}/nt/linalg/process/inv/inv_double.cpp         
    ${BASE_DIR}/nt/linalg/process/inv/inv_int16.cpp          
    ${BASE_DIR}/nt/linalg/process/inv/inv_int64.cpp          
    ${BASE_DIR}/nt/linalg/process/inv/inv_uint16.cpp         
    ${BASE_DIR}/nt/linalg/process/inv/inv_uint8.cpp
    ${BASE_DIR}/nt/linalg/process/inv/inv_complex_float.cpp  
    ${BASE_DIR}/nt/linalg/process/inv/inv_float.cpp          
    ${BASE_DIR}/nt/linalg/process/inv/inv_int32.cpp          
    ${BASE_DIR}/nt/linalg/process/inv/inv_int8.cpp           
    ${BASE_DIR}/nt/linalg/process/inv/inv_uint32.cpp

)

