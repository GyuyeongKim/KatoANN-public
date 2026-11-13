#!/usr/bin/env bash
# Builds the CABANA baseline against oneMKL. MKLROOT must hold include/mkl.h and
# lib/libmkl_rt.so; `pip install mkl mkl-include` into a virtual environment and
# MKLROOT=<that environment> works.
set -euo pipefail
cd "$(dirname "$0")"
MKLROOT=${MKLROOT:?set MKLROOT}
lib=$(ls -d "$MKLROOT"/lib "$MKLROOT"/lib64 2>/dev/null | head -1)
rt=$(ls "$lib"/libmkl_rt.so* | head -1)
[ -e "$lib/libmkl_rt.so" ] || ln -sf "$(basename "$rt")" "$lib/libmkl_rt.so"
${CXX:-g++} -O3 -march=native -fopenmp -std=c++17 -Wall -I"$MKLROOT/include" cabana.cpp -o cabana \
  -L"$lib" -Wl,-rpath,"$lib" -lmkl_rt -lgomp -lpthread -lm -ldl
echo "built $(pwd)/cabana"
