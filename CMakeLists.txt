cmake_minimum_required(VERSION 3.18)
project(ponca_kdtree LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(PYBIND11_FINDPYTHON ON)

include(FetchContent)

FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
)

FetchContent_MakeAvailable(pybind11)

pybind11_add_module(${PROJECT_NAME}
    src/bindings/main.cpp
    src/bindings/module.cpp
)

target_include_directories(${PROJECT_NAME}
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/ponca"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/ponca/external/eigen"
)

install(TARGETS ${PROJECT_NAME} DESTINATION ponca_kdtree)
install(
    FILES
        python/ponca_kdtree/__init__.py
        python/ponca_kdtree/__init__.pyi
    DESTINATION ponca_kdtree
)
