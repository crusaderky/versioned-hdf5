#!/bin/bash

set -eo pipefail

if [[ "$1" == "" ]] ; then
    echo "Usage: $0 <PROJECT_PATH>"
    exit 1
fi
PROJECT_PATH="$1"

if [[ "$ARCH" == "ARM64" ]]; then
    export ZLIB_ROOT="$PROJECT_PATH/zlib-win-arm64"
    export HDF5_VSVERSION="17-arm64"
elif [[ "$ARCH" == "AMD64" ]]; then
    export ZLIB_ROOT="$PROJECT_PATH/zlib-win-x64"
    export HDF5_VSVERSION="17-64"
else
    echo "Got unexpected arch '$ARCH'"
    exit 1
fi

echo "Building zlib into $ZLIB_ROOT"
./ci/get_zlib_windows.sh "$ZLIB_ROOT"

EXTRA_PATH="$ZLIB_ROOT/bin"
export CL="/I$ZLIB_ROOT/include"
export LINK="/LIBPATH:$ZLIB_ROOT/lib"

export PATH="$PATH:$EXTRA_PATH"

# HDF5
export HDF5_VERSION="2.0.0"
export HDF5_DIR="$PROJECT_PATH/cache/hdf5/$HDF5_VERSION"

pip install requests
python $PROJECT_PATH/ci/get_hdf5_win.py

# OpenSSL (libcrypto) for versioned_hdf5/hash.pyx.
# `choco install openssl` (see .github/workflows/wheels.yml) installs the Shining Light
# distribution but ships no pkg-config files, and its default path contains spaces,
# which pkg-config handles poorly. Copy the bits we need into a space-free directory and
# synthesise the .pc files so that meson's dependency('openssl') resolves it.
# delvewheel vendors libcrypto-*.dll into the wheel during the repair step.
OPENSSL_SRC=""
for cand in "/c/Program Files/OpenSSL-Win64" "/c/Program Files/OpenSSL" ; do
    if [[ -d "$cand/include/openssl" ]]; then OPENSSL_SRC="$cand"; break; fi
done
if [[ "$OPENSSL_SRC" == "" ]]; then
    echo "Could not locate the choco-installed OpenSSL" >&2
    exit 1
fi

export OPENSSL_DIR="$PROJECT_PATH/openssl"
mkdir -p "$OPENSSL_DIR"
cp -r "$OPENSSL_SRC/include" "$OPENSSL_DIR/"
cp -r "$OPENSSL_SRC/lib" "$OPENSSL_DIR/"
cp -r "$OPENSSL_SRC/bin" "$OPENSSL_DIR/"

OPENSSL_WIN=$(cygpath -m "$OPENSSL_DIR")  # forward-slash Windows path, no spaces
mkdir -p "$OPENSSL_DIR/lib/pkgconfig"
cat > "$OPENSSL_DIR/lib/pkgconfig/libcrypto.pc" <<EOF
prefix=$OPENSSL_WIN
libdir=\${prefix}/lib
includedir=\${prefix}/include

Name: OpenSSL-libcrypto
Description: OpenSSL cryptography library
Version: 3.0.0
Libs: -L\${libdir} -llibcrypto
Cflags: -I\${includedir}
EOF
cat > "$OPENSSL_DIR/lib/pkgconfig/openssl.pc" <<EOF
Name: OpenSSL
Description: Secure Sockets Layer and cryptography libraries
Version: 3.0.0
Requires: libcrypto
EOF

if [[ "$GITHUB_ENV" != "" ]] ; then
    # PATH on windows is special
    echo "$EXTRA_PATH" | tee -a $GITHUB_PATH
    echo "CL=$CL" | tee -a $GITHUB_ENV
    echo "LINK=$LINK" | tee -a $GITHUB_ENV
    echo "ZLIB_ROOT=$ZLIB_ROOT" | tee -a $GITHUB_ENV
    echo "HDF5_DIR=$HDF5_DIR" | tee -a $GITHUB_ENV
    echo "OPENSSL_DIR=$OPENSSL_DIR" | tee -a $GITHUB_ENV
    # cmake fallback for meson's dependency('openssl') if pkg-config is bypassed
    echo "OPENSSL_ROOT_DIR=$OPENSSL_DIR" | tee -a $GITHUB_ENV
fi
