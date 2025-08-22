cmake_minimum_required(VERSION 3.22)
project(tacuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

option(BUILD_TESTING "Enable tests" ON)

#Explicitly list indicator sources instead of globbing
set(INDICATOR_SOURCES
    src/indicators/MACD.cu
    src/indicators/MACDEXT.cu
    src/indicators/Momentum.cu
    src/indicators/ROC.cu
    src/indicators/SMA.cu
    src/indicators/EMA.cu
    src/indicators/DEMA.cu
    src/indicators/TEMA.cu
    src/indicators/TRIX.cu
    src/indicators/MA.cu
    src/indicators/MAX.cu
    src/indicators/MAMA.cu
    src/indicators/WMA.cu
    src/indicators/RSI.cu
    src/indicators/KAMA.cu
    src/indicators/BBANDS.cu
    src/indicators/ATR.cu
    src/indicators/Stochastic.cu
    src/indicators/CCI.cu
    src/indicators/OBV.cu
    src/indicators/SAR.cu
    src/indicators/ADX.cu
    src/indicators/Aroon.cu
    src/indicators/ADOSC.cu
    src/indicators/MFI.cu
    src/indicators/ULTOSC.cu
    src/indicators/AD.cu
    src/indicators/ADXR.cu
    src/indicators/APO.cu
    src/indicators/AroonOscillator.cu
    src/indicators/AvgPrice.cu
    src/indicators/MIN.cu
    src/indicators/MedPrice.cu
    src/indicators/Beta.cu
    src/indicators/BOP.cu
    src/indicators/CMO.cu
    src/indicators/Correl.cu
    src/indicators/DX.cu
    src/indicators/PlusDM.cu
    src/indicators/MinusDM.cu
    src/indicators/PlusDI.cu
    src/indicators/MinusDI.cu
    src/indicators/HT_DCPERIOD.cu
    src/indicators/HT_DCPHASE.cu
    src/indicators/HT_PHASOR.cu
    src/indicators/HT_SINE.cu
    src/indicators/HT_TRENDMODE.cu
    src/indicators/LINEARREG.cu
    src/indicators/LINEARREG_ANGLE.cu
    src/indicators/LINEARREG_INTERCEPT.cu
    src/indicators/LINEARREG_SLOPE.cu
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
target_compile_options(tacuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(example_app src/main.cpp)
target_link_libraries(example_app PRIVATE tacuda CUDA::cudart)

#Tests(simple CTest with a small program using the C API)
if (BUILD_TESTING)
  enable_testing()
  include(FetchContent)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
  )
  FetchContent_MakeAvailable(googletest)

  file(GLOB TEST_SOURCES tests/cpp/*.cpp)
  add_executable(test_basic ${TEST_SOURCES})
  target_include_directories(test_basic PRIVATE
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
  target_link_libraries(test_basic PRIVATE tacuda GTest::gtest_main)
  include(GoogleTest)
  gtest_discover_tests(test_basic)
endif()

# Install and export
install(TARGETS tacuda EXPORT tacudaTargets
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)
install(TARGETS example_app RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)
install(EXPORT tacudaTargets
        FILE tacudaTargets.cmake
        NAMESPACE tacuda::
        DESTINATION lib/cmake/tacuda)
install(FILES cmake/tacudaConfig.cmake DESTINATION lib/cmake/tacuda)
