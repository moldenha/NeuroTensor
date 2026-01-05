include(${CMAKE_CURRENT_LIST_DIR}/base_dir.cmake)

set(SVD_SOURCES
    ${BASE_DIR}/nt/linalg/svd.cpp

    # nt/linalg/process/svd.cpp
    # svd broken up into different types for compilation reasons
    ${BASE_DIR}/nt/linalg/process/svd/svd_complex_double.cpp 
    ${BASE_DIR}/nt/linalg/process/svd/svd_double.cpp         
    ${BASE_DIR}/nt/linalg/process/svd/svd_int16.cpp          
    ${BASE_DIR}/nt/linalg/process/svd/svd_int64.cpp          
    ${BASE_DIR}/nt/linalg/process/svd/svd_uint16.cpp         
    ${BASE_DIR}/nt/linalg/process/svd/svd_uint8.cpp
    ${BASE_DIR}/nt/linalg/process/svd/svd_complex_float.cpp  
    ${BASE_DIR}/nt/linalg/process/svd/svd_float.cpp          
    ${BASE_DIR}/nt/linalg/process/svd/svd_int32.cpp          
    ${BASE_DIR}/nt/linalg/process/svd/svd_int8.cpp           
    ${BASE_DIR}/nt/linalg/process/svd/svd_uint32.cpp

)

