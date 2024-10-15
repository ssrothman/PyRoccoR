cmake_minimum_required(VERSION 3.15...3.30)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_roccor MODULE src/main.cpp src/roccor/RoccoR.cc)
install(TARGETS _roccor DESTINATION ${SKBUILD_PROJECT_NAME})

find_package(Python3 REQUIRED 
    COMPONENTS Interpreter NumPy Development)
include_directories(${Python3_INCLUDE_DIRS})
include_directories(${Python3_NumPy_INCLUDE_DIRS})
Python3_add_library(_ufunc MODULE src/ufunc.cpp src/roccor/RoccoR.cc)
install(TARGETS _ufunc DESTINATION ${SKBUILD_PROJECT_NAME})
