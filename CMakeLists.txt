cmake_minimum_required(VERSION 3.18)
project(ganit LANGUAGES CXX)

option(GANIT_CUDA "Build the CUDA GPU backend if a CUDA compiler is found" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenMP)

add_library(ganit_core STATIC
  src/lp.cpp
  src/mps.cpp
  src/scaling.cpp
  src/backend_cpu.cpp
  src/simplex.cpp
  src/mip.cpp)
target_include_directories(ganit_core PUBLIC include src)
if(OpenMP_CXX_FOUND)
  target_link_libraries(ganit_core PUBLIC OpenMP::OpenMP_CXX)
endif()

add_executable(ganit src/main.cpp)
target_link_libraries(ganit PRIVATE ganit_core)

if(GANIT_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      # Turing, Ampere, Ada, Hopper. Override with -DCMAKE_CUDA_ARCHITECTURES=native
      set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90)
    endif()
    set(CMAKE_CUDA_STANDARD 17)
    add_library(ganit_gpu STATIC src/backend_cuda.cu)
    target_include_directories(ganit_gpu PUBLIC include src)
    target_compile_definitions(ganit_gpu PUBLIC GANIT_HAVE_CUDA)
    target_compile_options(ganit_gpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
    target_link_libraries(ganit_gpu PUBLIC ganit_core)
    target_link_libraries(ganit PRIVATE ganit_gpu)
    message(STATUS "GANIT: CUDA backend enabled (${CMAKE_CUDA_COMPILER})")
  else()
    message(STATUS "GANIT: no CUDA compiler found, building CPU backend only")
  endif()
endif()
