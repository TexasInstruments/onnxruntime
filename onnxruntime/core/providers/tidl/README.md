# TIDL Execution Provider

The TIDL Execution Provider enables ONNX Runtime to leverage Texas Instruments Deep Learning (TIDL) hardware accelerators for optimized inference performance on TI SoCs.

## Overview

The TIDL Execution Provider interfaces with TI's deep learning library to accelerate neural network operations on TI devices. It allows ONNX models to be executed efficiently on TI SoCs by utilizing the dedicated hardware accelerators.

## Key Features

- Hardware-accelerated inference on TI SoCs
- Support for common deep learning operations
- Integration with ONNX Runtime's graph optimization pipeline
- Improved performance for deep learning workloads on TI devices

## Supported Devices

The TIDL Execution Provider supports various Texas Instruments SoCs, including:

- AM62A
- AM68A / J721S2 / TDA4VL
- AM69A / J784S4 / TDA4VH
- AM68PA / J721E / TDA4VM
- AM67A / J722S / TDA4AEN


## Build

The build instructions for both x86 and aarch64 build is carried out on x86 PC.

### Prerequisite
```bash
pip install --no-input --upgrade pip setuptools numpy packaging pybind11 wheel numpy
pip3 install cmake==3.28.0
```

### x86 build

#### Build instructions

```bash
$ cd <ONNXRUNTIME_REPO_DIR>    # Repo base directory
$ git submodule init && git submodule update --init --recursive

$ rm -rf build_x86_64

$ python3 tools/ci_build/build.py --compile_no_warning_as_error --build_dir build_x86_64 --config Release --build_shared_lib --parallel `nproc` --skip_tests --use_tidl --build_wheel --cmake_extra_defines "-Wno-unused-variable PYTHON_INCLUDE_DIR=/usr/include/python3.10;/usr/include/python3.10/numpy/core/include" "PYTHON_LIBRARY=/usr/lib/python3.10" "CMAKE_VERBOSE_MAKEFILE:BOOL=ON"

# Notes:
# - The wheel will be built for same python3 version as being used by default python3.
#   You can also use virtual enviromnent to build for specific python version
#
# - Make sure PYTHON_INCLUDE_DIR has include path to numpy.
#   By default it is /usr/include/python3.10/numpy/core/include but might change depending on python version
#
```

#### Packaging instructions

These instructions packages necessary components from the build output in a single directory making it easier to install on the filesystem

```bash
$ mkdir -p ~/onnx_1.23.0_x86_u22
$ cd ~/onnx_1.23.0_x86_u22

# Python wheel
$ cp -rp <ONNXRUNTIME_REPO_DIR>/build_x86_64/Release/dist/*.whl ./

# CPP dependencies
$ cp -rp <ONNXRUNTIME_REPO_DIR>/build_x86_64/Release/libonnxruntime.so* ./  # Static lib
$ cd <ONNXRUNTIME_REPO_DIR>/../ && rm -rf onnxruntime.tar.gz
$ find ./onnxruntime/ -name "*.h" | tar -cf onnxruntime.tar.gz -T -
$ cd ~/onnx_1.23.0_x86_u22
$ cp <ONNXRUNTIME_REPO_DIR>/../onnxruntime.tar.gz ./
$ tar -xf onnxruntime.tar.gz && rm -rf onnxruntime.tar.gz                    # Onnxruntime headers

$ cd ../
$ tar -czf onnx_1.23.0_x86_u22.tar.gz onnx_1.23.0_x86_u22/
```

### aarch64 build

#### Requirements
`protobuf-3.21.12` is required to be build for aarch64 build

``` bash
$ cd ${HOME}
$ sudo apt install autoconf libtool
$ wget https://github.com/protocolbuffers/protobuf/archive/refs/tags/v3.21.12.tar.gz --no-check-certificate
tar xf v3.21.12.tar.gz
cd protobuf-3.21.12/
./autogen.sh
./configure CXXFLAGS=-fPIC --enable-shared=no LDFLAGS="-static"
make -j
```


#### Update tool.cmake

> [NOTE]
> TI SDK is needed to be present and setup on the system for aarch64 build

For aarch64 build, `tool.cmake` under <ONNXRUNTIME_REPO_DIR> is used. Update it with proper sdk paths
```bash
# tool.cmake
SET(CMAKE_SYSTEM_NAME Linux)
SET(CMAKE_SYSTEM_PROCESSOR aarch64)
SET(CMAKE_SYSTEM_VERSION 1)
SET(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
SET(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
SET(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
SET(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# Set the gcc and g++ toolchain
SET(sdk_path <PATH_TO_SDK>)
SET(CMAKE_C_COMPILER  ${sdk_path}/toolchain/sysroots/x86_64-arago-linux/usr/bin/aarch64-oe-linux/aarch64-oe-linux-gcc)
SET(CMAKE_CXX_COMPILER  ${sdk_path}/toolchain/sysroots/x86_64-arago-linux/usr/bin/aarch64-oe-linux/aarch64-oe-linux-g++)
SET(CMAKE_SYSROOT ${sdk_path}/targetfs)

# Include NumPy headers for Python support
include_directories (${CMAKE_SYSROOT}/usr/lib/python3.12/site-packages/numpy/core/include)

# Create Python targets required for the build
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
```

> [NOTE]
> - <PATH_TO_SDK> is full path to sdk base directory. Ex: /home/user/ti-processor-sdk-rtos-j721s2-evm-11_01_01_01/
> - The python version might change between SDKs, hence make sure to update the python version in all paths (both for Python and NumPy)
> - The additional Python targets are required to properly detect Python during cross-compilation

#### Build instructions

```bash
$ cd <ONNXRUNTIME_REPO_DIR>    # Repo base directory
$ git submodule init && git submodule update --init --recursive

$ rm -rf build_aarch64

$  python3 tools/ci_build/build.py --compile_no_warning_as_error --build_dir build_aarch64 --config Release --build_shared_lib --parallel `nproc` --skip_tests --skip_onnx_tests --use_tidl --build_wheel --path_to_protoc_exe ${PROTOBUF_SRC_PATH}/src/protoc --cmake_extra_defines "CMAKE_TOOLCHAIN_FILE=<ONNXRUNTIME_REPO_DIR>/tool.cmake" "CMAKE_VERBOSE_MAKEFILE:BOOL=OFF"

# Notes:
# - The wheel will be built for same python3 version as being used by default python3.
#   Make sure it is the same version as being used by the SDK you are building for
#
# - ${PROTOBUF_SRC_PATH} is the base path of protobuf-3.21.12 built as part of requirement
#
```

#### Packaging instructions

These instructions packages necessary components from the build output in a single directory making it easier to install on the filesystem

```bash
$ mkdir -p ~/onnx_1.23.0_aragoj7
$ cd ~/onnx_1.23.0_aragoj7

# Python wheel
$ cp -rp <ONNXRUNTIME_REPO_DIR>/build_aarch64/Release/dist/*.whl ./
$ find . -name '*.whl' -exec bash -c ' mv $0 ${0/\linux_x86_64/linux_aarch64}' {} \; # IMPORTANT: This replaces linux_x86_64 to linux_aarch64 in the wheel name which is required for installation on aarch64 systems


# CPP dependencies
$ cp -rp <ONNXRUNTIME_REPO_DIR>/build_aarch64/Release/libonnxruntime.so* ./  # Static lib
$ cd <ONNXRUNTIME_REPO_DIR>/../ && rm -rf onnxruntime.tar.gz
$ find ./onnxruntime/ -name "*.h" | tar -cf onnxruntime.tar.gz -T -
$ cd ~/onnx_1.23.0_aragoj7
$ cp <ONNXRUNTIME_REPO_DIR>/../onnxruntime.tar.gz ./
$ tar -xf onnxruntime.tar.gz && rm -rf onnxruntime.tar.gz                    # Onnxruntime headers

$ cd ../
$ tar -czf onnx_1.23.0_aragoj7.tar.gz onnx_1.23.0_aragoj7/
```

## Additional Resources

For more detailed information about onnxruntime interface with TIDL, usage examples, and best practices, please visit the [EdgeAI-TIDL-Tools GitHub repository](https://github.com/TexasInstruments/edgeai-tidl-tools).
