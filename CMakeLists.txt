cmake_minimum_required(VERSION 3.22)
project(cuda_talib LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

option(BUILD_TESTING "Enable tests" ON)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

file(GLOB INDICATOR_SOURCES
    src/indicators/*.cu
)

add_library(cuda_talib_shared SHARED
    ${INDICATOR_SOURCES}
    src/api/api.cpp
)

find_package(CUDAToolkit REQUIRED)
target_link_libraries(cuda_talib_shared PRIVATE CUDA::cudart)
target_compile_features(cuda_talib_shared PRIVATE cxx_std_17)
target_compile_options(cuda_talib_shared PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(example_app src/main.cpp)
target_link_libraries(example_app PRIVATE cuda_talib_shared)

# Tests (simple CTest with a small program using the C API)
if (BUILD_TESTING)
  enable_testing()
  add_executable(test_basic tests/cpp/test_basic.cpp)
  target_link_libraries(test_basic PRIVATE cuda_talib_shared)
  add_test(NAME run_test_basic COMMAND test_basic)
endif()

# Install (optional)
install(TARGETS cuda_talib_shared example_app
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)
