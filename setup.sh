#!/bin/bash

VIRTUALENV_PREFIX=$PWD/external/virtualenv
mkdir -p $VIRTUALENV_PREFIX/
virtualenv $VIRTUALENV_PREFIX/
source $VIRTUALENV_PREFIX/bin/activate
pip install numpy 

echo 'Fetching submodules...'
git submodule init
git submodule update

echo -e '\n\nCompiling flann...'
cd external/flann
mkdir build/
cd build/
cmake -DCMAKE_INSTALL_PREFIX:PATH=$VIRTUALENV_PREFIX/ ../
make
make install

echo -e '\n\nCompiling igraph...'
cd ../../igraph
./bootstrap.sh
./configure --prefix=$VIRTUALENV_PREFIX/
make 
make install

echo -e '\n\nCompiling python-igraph...'
cd ../python-igraph
cp setup.py setup.py.orig

sed -i 's/pkg-config igraph/pkg-config uselocaligraph/g' setup.py
sed -i "s,/usr/include'\, '/usr/local/include","$VIRTUALENV_PREFIX"'/include/igraph,g' setup.py
sed -i "s,\['igraph',['igraph'\,'"$VIRTUALENV_PREFIX"/lib',g" setup.py

export LD_LIBRARY_PATH=$VIRTUALENV_PREFIX/lib
python setup.py build
python setup.py install

cd ../../
