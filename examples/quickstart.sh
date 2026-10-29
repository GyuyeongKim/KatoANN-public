#!/usr/bin/env bash
# Builds an index over 200K synthetic 768-dimensional vectors and reports
# recall@10 and throughput for a few nprobe values.
set -euo pipefail
cd "$(dirname "$0")/.."
data=${DATA:-build/quickstart}
python3 examples/make_synthetic.py "$data"
build/katoann build --data "$data/base.fbin" --out "$data/index" --nlist 512 --seed 1
build/katoann search --index "$data/index" --queries "$data/query.fbin" --gt "$data/gt.bin" \
  --k 10 --nprobe 8,16,32,64
