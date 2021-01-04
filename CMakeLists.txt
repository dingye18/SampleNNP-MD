CMAKE_MINIMUM_REQUIRED(VERSION 3.5)
SET(CMAKE_VERBOSE_MAKEFILE ON)

# We need to know where TensorFlow is installed so we can access the headers and libraries.
SET(TENSORFLOW_ROOT "/home/dingye/.local/tensorflow/r1.14" CACHE PATH "Where TensorFlow is installed")
set(OP_CXX_ABI 1)
set (OP_CXX_FLAG -D_GLIBCXX_USE_CXX11_ABI=${OP_CXX_ABI})

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wno-ignored-attributes -Wl,--allow-multiple-definition -Wl,--whole-archive  -Wl,--no-as-needed")
set(TensorFlowFramework_LIBRARY "${TENSORFLOW_ROOT}/lib/libtensorflow_framework.so")
set(TensorFlowCore_LIBRARY "${TENSORFLOW_ROOT}/lib/libtensorflow_cc.so")
set (TensorFlowFramework_INCLUDE "${TENSORFLOW_ROOT}/include")
INCLUDE_DIRECTORIES("${TENSORFLOW_ROOT}/include")
INCLUDE_DIRECTORIES("${TENSORFLOW_ROOT}/include/tensorflow/c")

# Include the deepmd-kit.
SET(DEEPMD_DIR "/home/dingye/.local/deepmd-kit-1.2.0/" CACHE PATH "Where deepmd-kit is installed")
INCLUDE_DIRECTORIES("${DEEPMD_DIR}/include/deepmd")
LINK_DIRECTORIES("${DEEPMD_DIR}/lib")

add_executable(main main.cpp )

target_link_libraries(main ${TensorFlowFramework_LIBRARY} ${TensorFlowCore_LIBRARY})
TARGET_LINK_LIBRARIES(main deepmd)
target_include_directories(main PUBLIC ${TensorFlowFramework_INCLUDE} ${PROJECT_BINARY_DIR})

set_target_properties(
    main
    PROPERTIES 
    COMPILE_FLAGS ${OP_CXX_FLAG}
    )