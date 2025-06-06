if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    message(STATUS "Compiler is ${CMAKE_CXX_COMPILER_ID}, enabling -march=native")
    add_compile_options(-march=native)
endif()
