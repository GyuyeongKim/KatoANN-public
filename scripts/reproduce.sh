#!/usr/bin/env bash
# Prepares the datasets, builds the indexes of KatoANN and the baselines,
# measures recall and throughput, and draws the figure. A finished build or
# measurement is kept, so running the script again only does what is missing.
#
#   DATA          dataset and index directory (default data)
#   OUT           results directory (default results)
#   DATASETS      subset of "openai3072-1m bioasq-10m"
#   SYSTEMS       subset of "katoann svs-leanvec scann cabana faiss-ivf"
#   THREADS       search and build threads (default: all CPUs)
#   PYTHON        Python with numpy, pyarrow, huggingface_hub, faiss-cpu, scalable-vs,
#                 and matplotlib (default python3)
#   SCANN_PYTHON  Python with scann (default $PYTHON); scann pins numpy and protobuf,
#                 so it may need its own virtual environment
#   MKLROOT       oneMKL for CABANA, needed only when baselines/cabana/cabana is missing
set -euo pipefail
cd "$(dirname "$0")/.."
DATA=${DATA:-data}
OUT=${OUT:-results}
DATASETS=${DATASETS:-"openai3072-1m bioasq-10m"}
SYSTEMS=${SYSTEMS:-"katoann svs-leanvec scann cabana faiss-ivf"}
THREADS=${THREADS:-$(nproc)}
PYTHON=${PYTHON:-python3}
SCANN_PYTHON=${SCANN_PYTHON:-$PYTHON}
run=()
if command -v numactl >/dev/null && [ "$(numactl --hardware | awk '/available:/ {print $2}')" -gt 1 ]; then
  run=(numactl --interleave=all)
fi
want() { [[ " $SYSTEMS " == *" $1 "* ]]; }
measure() {  # dataset system command... -> $OUT/<dataset>/<system>.jsonl
  local out=$OUT/$1/$2.jsonl; shift 2
  [ -s "$out" ] && return
  echo "== $out" >&2
  "$@" > "$out.tmp"
  mv "$out.tmp" "$out"
}
scann_grid() {  # leaves... -> leaves:reorder pairs
  local g= l r
  for l in "$@"; do for r in 100 200 400; do g="$g${g:+,}$l:$r"; done; done
  echo "$g"
}

if want cabana && [ ! -x baselines/cabana/cabana ]; then
  bash baselines/cabana/build.sh
fi

for ds in $DATASETS; do
  case $ds in
    openai3072-1m)
      nlist=2048 prefix=768
      knp=32,64,128,256,512,768,1024,1280,1536,1792,2048
      svs_dims=768 svs_w=25,50,100,200,300,400,600,800,1000,1200,1600
      scann_p=$(scann_grid 50 100 200 400 500 575 650 1000 1300 1600 2048)
      cnp=64,128,256,512,1024,2048
      fnp=32,64,128,256,512,1024,2048 ;;
    bioasq-10m)
      nlist=8192 prefix=512
      knp=128,256,512,1024,2048,2560,3072,3584,4096,6144,8192
      svs_dims=256 svs_w=400,800,1600,3200,6400,9600,12800,19200,25600,38400,51200,76800,102400
      scann_p=$(scann_grid 200 400 650 1000 2000 2500 3000 3500 4000 5000 6000 8192)
      cnp=256,512,1024,2048,4096,6144,8192
      fnp=128,256,512,1024,2048,4096 ;;
    *) echo "unknown dataset $ds" >&2; exit 2 ;;
  esac
  d=$DATA/$ds
  mkdir -p "$d" "$OUT/$ds"
  [ -s "$d/gt.bin" ] || "$PYTHON" scripts/datasets.py "$ds" "$d"
  q=(--queries "$d/query.fbin" --gt "$d/gt.bin" --k 10 --batch 4096 --repeat 3 --threads "$THREADS")

  # KatoANN's index also serves CABANA, which scans the same lists in bfloat16.
  if want katoann || want cabana; then
    [ -s "$d/index.dimc" ] || "${run[@]}" build/katoann build --data "$d/base.fbin" \
      --out "$d/index" --nlist $nlist --seed 1 --threads "$THREADS" > "$OUT/$ds/katoann.build.json"
  fi
  want katoann && measure $ds katoann "${run[@]}" build/katoann search --index "$d/index" "${q[@]}" \
    --nprobe $knp --prefix-dims $prefix --survivors 32
  want cabana && measure $ds cabana "${run[@]}" baselines/cabana/cabana --index "$d/index" "${q[@]}" \
    --nprobe $cnp

  if want svs-leanvec; then
    [ -d "$d/svs-leanvec/graph" ] || "${run[@]}" "$PYTHON" baselines/bench.py build --system svs-leanvec \
      --data "$d/base.fbin" --out "$d/svs-leanvec" --dims $svs_dims --threads "$THREADS" \
      > "$OUT/$ds/svs-leanvec.build.json"
    measure $ds svs-leanvec "${run[@]}" "$PYTHON" baselines/bench.py search --system svs-leanvec \
      --index "$d/svs-leanvec" "${q[@]}" --param $svs_w
  fi
  if want scann; then
    [ -s "$d/scann/scann_config.pb" ] || "${run[@]}" "$SCANN_PYTHON" baselines/bench.py build --system scann \
      --data "$d/base.fbin" --out "$d/scann" --nlist $nlist --threads "$THREADS" > "$OUT/$ds/scann.build.json"
    measure $ds scann "${run[@]}" "$SCANN_PYTHON" baselines/bench.py search --system scann \
      --index "$d/scann" "${q[@]}" --param "$scann_p"
  fi
  if want faiss-ivf; then
    [ -s "$d/faiss-ivf/faiss_ivf.index" ] || "${run[@]}" "$PYTHON" baselines/bench.py build --system faiss-ivf \
      --data "$d/base.fbin" --out "$d/faiss-ivf" --nlist $nlist --threads "$THREADS" \
      > "$OUT/$ds/faiss-ivf.build.json"
    measure $ds faiss-ivf "${run[@]}" "$PYTHON" baselines/bench.py search --system faiss-ivf \
      --index "$d/faiss-ivf" "${q[@]}" --param $fnp
  fi
done
"$PYTHON" scripts/plot.py "$OUT"
