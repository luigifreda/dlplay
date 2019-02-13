#!/usr/bin/env bash



SCRIPT_DIR_=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) # get script dir
SCRIPT_DIR_=$(readlink -f $SCRIPT_DIR_)  # this reads the actual path if a symbolic directory is used

ROOT_DIR="$SCRIPT_DIR_/.."

STARTING_DIR=`pwd`  
cd "$ROOT_DIR"  

OS="$(uname)"
IS_MACOS=$("$OS" == "Darwin")
IS_LINUX=$("$OS" == "Linux")

# create pyenv environment "dlplay" and activate it
./tools/pyenv-create.sh
source ~/.bashrc
pyenv activate dlplay

pip install --upgrade pip

# Install first some pre-requisites to avoid circular dependency issues
pip install wheel setuptools build
pip install torch torchvision torchaudio

# install the package in development mode 
pip install -e .  

# install torch-points3d
./tools/install_torch_points3d.sh


if $IS_MACOS; then
    # from https://stackoverflow.com/questions/76161237/tensorflow-2-11-0-on-mac-crashes-when-setting-the-seed-for-generating-random-ten
    pip install tensorflow==2.13rc0
fi

pip install "numpy<2.0" --force-reinstall

cd "$STARTING_DIR"