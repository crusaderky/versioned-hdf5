#!/bin/bash
set -o errexit
set -o nounset
set -o pipefail

EXTRA="-DHDF5_ENABLE_ZLIB_SUPPORT=ON -DHDF5_ENABLE_SZIP_SUPPORT=ON"

export CC="clang"
export CMAKE_BUILD_PARALLEL_LEVEL="4"
export CTEST_PARALLEL_LEVEL="4"
# export CMAKE_CONFIG_TYPE="Debug"
# export CMAKE_BUILD_TYPE="Debug"
# export HDF5_CONFIG_ARGS="-DHDF5_ENABLE_DEBUG_APIS=ON -DHDF5_ENABLE_TRACE=ON -DHDF5_ENABLE_COVERAGE=ON $EXTRA"
export CMAKE_CONFIG_TYPE="Release"
export CMAKE_BUILD_TYPE="Release"
export HDF5_CONFIG_ARGS="$EXTRA"
# export VERBOSE="1"

cmake -S . -B "$BUILD_DIR/build" $HDF5_CONFIG_ARGS
cmake --build "$BUILD_DIR/build"
cmake --install "$BUILD_DIR/build" --prefix "$PREFIX"
