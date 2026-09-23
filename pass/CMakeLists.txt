cmake_minimum_required(VERSION 3.20)
project(quant-opt LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Point MLIR_DIR at the MLIR that torch-mlir was built with, e.g.
#   -DMLIR_DIR=$TORCH_MLIR_BUILD/lib/cmake/mlir
find_package(MLIR REQUIRED CONFIG)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}" "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

set(LIBS
  ${dialect_libs}
  ${conversion_libs}
  ${extension_libs}
  MLIROptLib
  MLIRPass
  MLIRTransforms
)
# Newer LLVM splits the registration helpers into their own libraries.
foreach(lib MLIRRegisterAllDialects MLIRRegisterAllExtensions MLIRRegisterAllPasses)
  if(TARGET ${lib})
    list(APPEND LIBS ${lib})
  endif()
endforeach()

add_llvm_executable(quant-opt
  tools/quant-opt.cpp
  lib/QuantizeLinear.cpp
)
llvm_update_compile_flags(quant-opt)
target_link_libraries(quant-opt PRIVATE ${LIBS})
mlir_check_all_link_libraries(quant-opt)
