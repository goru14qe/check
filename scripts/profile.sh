#!/bin/bash

if [[ $# -le 1 ]]; then
  echo "Illegal number of parameters. Expected at least 2."
  echo "Usage: bash profile.sh <num_processes> <config_path>"
  exit 2
fi

cd  ../build/
# enable build with instrumentalization
cmake -DCMAKE_CXX_FLAGS=-pg -DCMAKE_EXE_LINKER_FLAGS=-pg -DCMAKE_SHARED_LINKER_FLAGS=-pg ..

export GMON_OUT_PREFIX=gmon.out
mpirun -x GMON_OUT_PREFIX --oversubscribe -np $1 alborz "$2"