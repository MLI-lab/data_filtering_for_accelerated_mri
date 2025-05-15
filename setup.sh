#!/bin/bash
apt-get update
apt-get install wget
apt-get install tar
apt-get install make gcc libfftw3-dev liblapacke-dev libpng-dev libopenblas-dev -y
wget https://github.com/mrirecon/bart/archive/v0.6.00.tar.gz
tar xzvf v0.6.00.tar.gz
cd bart-0.6.00
make
cd ..
rm v0.6.00.tar.gz
pip install -r setup/requirements.txt