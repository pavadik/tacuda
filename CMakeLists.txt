cmake_minimum_required(VERSION 3.22)
project(tacuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

option(BUILD_TESTING "Enable tests" ON)

file(GLOB INDICATOR_SOURCES
    src/indicators/*.cu
)

add_library(tacuda SHARED
    ${INDICATOR_SOURCES}
    src/api/api.cpp
)

target_include_directories(tacuda
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

find_package(CUDAToolkit REQUIRED)
target_link_libraries(tacuda PRIVATE CUDA::cudart)
target_compile_features(tacuda PRIVATE cxx_std_17)
target_compile_options(tacuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(example_app src/main.cpp)
target_link_libraries(example_app PRIVATE tacuda)

# Tests (simple CTest with a small program using the C API)
if (BUILD_TESTING)
  enable_testing()
  include(FetchContent)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
  )
  FetchContent_MakeAvailable(googletest)

  add_executable(test_basic tests/cpp/test_basic.cpp)
  target_link_libraries(test_basic PRIVATE tacuda GTest::gtest_main)
  include(GoogleTest)
  gtest_discover_tests(test_basic)
endif()

# Install (optional)
install(TARGETS tacuda example_app
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)
