#!/bin/bash

################################################################################
# Ensure script is only run sourced
if [[ $_ == $0 ]]; then
    echo "!!!ERROR!!! - PLEASE RUN WITH 'source'"
    exit 1
fi

################################################################################
# Set environment variables
#ORT_LIB_PATH=../../build/Linux/RelWithDebInfo/
ORT_LIB_PATH=../../build/Linux/Debug/

LD_LIBRARY_PATH_LOCAL=$ORT_LIB_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$LD_LIBRARY_PATH_LOCAL
export TIDL_BASE_PATH=../../../tidl_it_onnxRt/c7x-mma-tidl
