include(${CMAKE_CURRENT_LIST_DIR}/base_dir.cmake)

set(CORE_SOURCES
    ${BASE_DIR}/nt/refs/ArrayRef.cpp
    ${BASE_DIR}/nt/refs/SizeRef.cpp
    ${BASE_DIR}/nt/utils/utils.cpp
    ${BASE_DIR}/nt/utils/optional.cpp
    ${BASE_DIR}/nt/utils/CommaOperator.cpp
    ${BASE_DIR}/nt/Tensor.cpp
    ${BASE_DIR}/nt/nn/TensorGrad.cpp
    ${BASE_DIR}/nt/dtype/compatible/DTypeCheckFunctions.cpp
)
