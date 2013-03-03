#!/bin/bash

VIRTUALENV_PREFIX=$PWD/external/virtualenv
source $VIRTUALENV_PREFIX/bin/activate
export LD_LIBRARY_PATH=$VIRTUALENV_PREFIX/lib
