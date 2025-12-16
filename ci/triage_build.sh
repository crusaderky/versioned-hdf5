#!/bin/bash

set -eo pipefail

if [[ "$1" == "" ]] || [[ "$2" == "" ]]; then
    echo "Usage: $0 <ARCH> <PYTHON>"
    exit 1
fi

ARCH=$1
PYTHON=$2
if [[ "$PYTHON" == "" ]]; then
    PYTHON="*"
fi

# strip '-dev' suffix for pre-releases Pythons
PYTHON="${PYTHON%-dev*}"

# replace dots in PYTHON with nothing, e.g., 3.8->38
CIBW_BUILD="cp${PYTHON//./}-*_$ARCH"
echo "CIBW_BUILD=$CIBW_BUILD" | tee -a $GITHUB_ENV
