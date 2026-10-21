cmake_minimum_required(VERSION 3.18)
project(hospital_policy_runner CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Prebuilt ONNX Runtime C++ release (headers in include/, library in lib/), macOS or Linux:
#   cmake -S native -B native/build -DONNXRUNTIME_ROOT=/path/to/onnxruntime-<os>-<arch>-<ver> \
#         -DPython_EXECUTABLE=$(which python)
# pybind11 is found through `python -m pybind11 --cmakedir` unless -Dpybind11_DIR is given.
set(ONNXRUNTIME_ROOT "$ENV{ONNXRUNTIME_ROOT}" CACHE PATH "unpacked ONNX Runtime release")
if(NOT ONNXRUNTIME_ROOT)
  message(FATAL_ERROR "ONNX Runtime C++ release not given: pass -DONNXRUNTIME_ROOT=<dir> (or set $ONNXRUNTIME_ROOT). "
          "Download onnxruntime-<os>-<arch>-<ver>.tgz from github.com/microsoft/onnxruntime/releases; the pip wheel has no headers.")
endif()
if(NOT EXISTS "${ONNXRUNTIME_ROOT}/include/onnxruntime_cxx_api.h")
  message(FATAL_ERROR "ONNXRUNTIME_ROOT=${ONNXRUNTIME_ROOT} has no include/onnxruntime_cxx_api.h (is it the unpacked release root?)")
endif()
find_library(ORT_LIB onnxruntime PATHS ${ONNXRUNTIME_ROOT}/lib NO_DEFAULT_PATH)
if(NOT ORT_LIB)
  message(FATAL_ERROR "no libonnxruntime in ${ONNXRUNTIME_ROOT}/lib (wrong OS/arch release? this host is ${CMAKE_HOST_SYSTEM_NAME}/${CMAKE_HOST_SYSTEM_PROCESSOR})")
endif()

find_package(Python COMPONENTS Interpreter Development.Module)
if(NOT Python_FOUND)
  message(FATAL_ERROR "Python development files not found; pass -DPython_EXECUTABLE=<python> (Linux: install python3-dev)")
endif()
if(NOT pybind11_DIR)
  execute_process(COMMAND ${Python_EXECUTABLE} -m pybind11 --cmakedir OUTPUT_VARIABLE pybind11_DIR
                  OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET RESULT_VARIABLE _pb_rc)
  if(NOT _pb_rc EQUAL 0)
    message(FATAL_ERROR "pybind11 not importable by ${Python_EXECUTABLE}: pip install pybind11 (or pass -Dpybind11_DIR)")
  endif()
endif()
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(hospital_policy_runner bindings.cpp policy_runner.cpp)
target_include_directories(hospital_policy_runner PRIVATE ${ONNXRUNTIME_ROOT}/include)
target_link_libraries(hospital_policy_runner PRIVATE ${ORT_LIB})
set_target_properties(hospital_policy_runner PROPERTIES BUILD_RPATH ${ONNXRUNTIME_ROOT}/lib INSTALL_RPATH ${ONNXRUNTIME_ROOT}/lib)
