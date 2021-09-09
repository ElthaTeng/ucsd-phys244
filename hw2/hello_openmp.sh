#! /bin/bash
#
g++ -c -Wall -fopenmp hello_openmp.cpp
if [ $? -ne 0 ]; then
  echo "Compile error."
  exit
fi
#
g++ -fopenmp hello_openmp.o -lm
if [ $? -ne 0 ]; then
  echo "Load error."
  exit
fi
mv a.out $HOME/bincpp/hello_openmp
#
echo "Normal end of execution."
