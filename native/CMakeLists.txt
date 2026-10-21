cmake_minimum_required(VERSION 3.18)
project(hospital_policy_runner CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Prebuilt ONNX Runtime C++ release (headers in include/, library in lib/):
#   cmake -S native -B native/build -DONNXRUNTIME_ROOT=/path/to/onnxruntime-<os>-<arch>-<ver> \
#         -Dpybind11_DIR=$(python -m pybind11 --cmakedir) -DPython_EXECUTABLE=$(which python)
set(ONNXRUNTIME_ROOT "" CACHE PATH "unpacked ONNX Runtime release")
if(NOT ONNXRUNTIME_ROOT)
  message(FATAL_ERROR "set -DONNXRUNTIME_ROOT=<unpacked ONNX Runtime release>")
endif()
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
find_library(ORT_LIB onnxruntime PATHS ${ONNXRUNTIME_ROOT}/lib NO_DEFAULT_PATH REQUIRED)

pybind11_add_module(hospital_policy_runner bindings.cpp policy_runner.cpp)
target_include_directories(hospital_policy_runner PRIVATE ${ONNXRUNTIME_ROOT}/include)
target_link_libraries(hospital_policy_runner PRIVATE ${ORT_LIB})
set_target_properties(hospital_policy_runner PROPERTIES BUILD_RPATH ${ONNXRUNTIME_ROOT}/lib INSTALL_RPATH ${ONNXRUNTIME_ROOT}/lib)
