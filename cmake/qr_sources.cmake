include(${CMAKE_CURRENT_LIST_DIR}/base_dir.cmake)

set(QR_SOURCES
    ${BASE_DIR}/nt/linalg/qr.cpp

    # nt/linalg/process/qr.cpp
    # qr broken up into different types for compilation reasons
    ${BASE_DIR}/nt/linalg/process/qr/qr_complex_double.cpp 
    ${BASE_DIR}/nt/linalg/process/qr/qr_double.cpp         
    ${BASE_DIR}/nt/linalg/process/qr/qr_int16.cpp          
    ${BASE_DIR}/nt/linalg/process/qr/qr_int64.cpp          
    ${BASE_DIR}/nt/linalg/process/qr/qr_uint16.cpp         
    ${BASE_DIR}/nt/linalg/process/qr/qr_uint8.cpp
    ${BASE_DIR}/nt/linalg/process/qr/qr_complex_float.cpp  
    ${BASE_DIR}/nt/linalg/process/qr/qr_float.cpp          
    ${BASE_DIR}/nt/linalg/process/qr/qr_int32.cpp          
    ${BASE_DIR}/nt/linalg/process/qr/qr_int8.cpp           
    ${BASE_DIR}/nt/linalg/process/qr/qr_uint32.cpp

)

