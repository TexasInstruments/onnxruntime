SET(CMAKE_SYSTEM_NAME Linux)
SET(CMAKE_SYSTEM_PROCESSOR aarch64)
SET(CMAKE_SYSTEM_VERSION 1)
SET(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
SET(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
SET(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
SET(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# Set the gcc and g++ toolchain
SET(sdk_path <PATH_TO_SDK>) # Update this
SET(CMAKE_C_COMPILER  ${sdk_path}/toolchain/sysroots/x86_64-arago-linux/usr/bin/aarch64-oe-linux/aarch64-oe-linux-gcc)
SET(CMAKE_CXX_COMPILER  ${sdk_path}/toolchain/sysroots/x86_64-arago-linux/usr/bin/aarch64-oe-linux/aarch64-oe-linux-g++)
SET(CMAKE_SYSROOT ${sdk_path}/targetfs)
include_directories (${CMAKE_SYSROOT}/usr/lib/python3.12/site-packages/numpy/core/include)

if(NOT TARGET Python::Module)
  add_library(Python::Module INTERFACE IMPORTED)
  set_target_properties(Python::Module PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_SYSROOT}/usr/include/python3.12"
  )
endif()

# Create the Python::NumPy target
if(NOT TARGET Python::NumPy)
  add_library(Python::NumPy INTERFACE IMPORTED)
  set_target_properties(Python::NumPy PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_SYSROOT}/usr/lib/python3.12/site-packages/numpy/core/include"
  )
endif()
