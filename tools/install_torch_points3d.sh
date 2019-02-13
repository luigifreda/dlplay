#!/usr/bin/env bash

SCRIPT_DIR_=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) # get script dir
SCRIPT_DIR_=$(readlink -f $SCRIPT_DIR_)  # this reads the actual path if a symbolic directory is used

ROOT_DIR="$SCRIPT_DIR_/.."

STARTING_DIR=`pwd`  
cd "$ROOT_DIR"  


cd thirdparty

if [ ! -d "torch-points3d" ]; then
    git clone https://github.com/nicolas-chaulet/torch-points3d.git torch-points3d
    cd torch-points3d

    git fetch --tags
    git checkout 1.3.0
    
    git apply ../torch-points3d.patch
    pip install -e .
    
    # manage conflicts
    pip install torch-geometric --force
    
    pip install "numpy<2.0"
fi


cd "$STARTING_DIR"