
include(ExternalProject)

# Cling source
set(CLING_REPO https://github.com/root-project/cling.git)
set(CLING_TAG master)

# Cling-compatible LLVM source (with cling patches)
set(LLVM_PROJECT_REPO https://github.com/root-project/llvm-project.git)
set(LLVM_PROJECT_TAG cling-latest)

# Install path
set(CLING_INSTALL_DIR ${CMAKE_BINARY_DIR}/cling-install)




# Download Cling source
ExternalProject_Add(cling_source
  GIT_REPOSITORY ${CLING_REPO}
  GIT_TAG ${CLING_TAG}
  PREFIX ${CMAKE_BINARY_DIR}/cling-source
  SOURCE_DIR ${CMAKE_BINARY_DIR}/cling-source/src
  UPDATE_COMMAND ""
  GIT_SHALLOW TRUE
  UPDATE_DISCONNECTED TRUE  # Prevent CMake from updating if not needed
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  GIT_SHALLOW TRUE
)

# Download and build LLVM with cling
ExternalProject_Add(cling
  GIT_REPOSITORY ${LLVM_PROJECT_REPO}
  GIT_TAG ${LLVM_PROJECT_TAG}
  PREFIX ${CMAKE_BINARY_DIR}/cling-llvm
  SOURCE_DIR ${CMAKE_BINARY_DIR}/cling-llvm/src
  BINARY_DIR ${CMAKE_BINARY_DIR}/cling-llvm/build
  UPDATE_COMMAND ""
  GIT_SHALLOW TRUE
  UPDATE_DISCONNECTED TRUE  # Prevent CMake from updating if not needed
  DEPENDS cling_source
  SOURCE_SUBDIR llvm
  CMAKE_ARGS
    -DCMAKE_INSTALL_PREFIX=${CLING_INSTALL_DIR}
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    -DLLVM_ENABLE_PROJECTS=clang
    -DLLVM_TARGETS_TO_BUILD=host
    -DLLVM_EXTERNAL_PROJECTS=cling
    -DLLVM_EXTERNAL_CLING_SOURCE_DIR=${CMAKE_BINARY_DIR}/cling-source/src
    -DLLVM_ENABLE_RTTI=ON
    -DLLVM_ENABLE_ASSERTIONS=ON
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
  BUILD_COMMAND ${CMAKE_COMMAND} --build . --target cling
  INSTALL_COMMAND ${CMAKE_COMMAND} --build . --target install
  BUILD_BYPRODUCTS ${CLING_INSTALL_DIR}/bin/cling
  BUILD_ALWAYS OFF
)

# Add cling to the NeuroTensor REPL
add_executable(neurotensor-repl neurotensor_repl.cpp)
add_dependencies(neurotensor-repl neurotensor cling)

target_include_directories(neurotensor-repl PRIVATE
  ${CLING_INSTALL_DIR}/include
  ${CLING_INSTALL_DIR}/include/cling
  ${CLING_INSTALL_DIR}/include/clang
  ${CLING_INSTALL_DIR}/include/llvm
  $<TARGET_PROPERTY:neurotensor,INTERFACE_INCLUDE_DIRECTORIES>
)

target_link_directories(neurotensor-repl PRIVATE
  ${CLING_INSTALL_DIR}/lib
)

target_link_libraries(neurotensor-repl
  neurotensor
  clingInterpreter
  clingMetaProcessor
  clingUtils
  clangFrontend
  clangDriver
  clangSerialization
  clangParse
  clangSema
  clangAnalysis
  clangAST
  clangLex
  clangBasic
  LLVMCore
  LLVMSupport
  LLVMExecutionEngine
  LLVMInterpreter
  LLVMOrcJIT
  LLVMRuntimeDyld
)

target_compile_features(neurotensor-repl PRIVATE cxx_std_17)

target_compile_options(neurotensor-repl PRIVATE
  $<$<CXX_COMPILER_ID:Clang>:-Wno-deprecated-declarations -Wno-unused-command-line-argument>
)
