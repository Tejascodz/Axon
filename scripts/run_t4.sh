#!/usr/bin/env bash
# Full comparison on Kaggle / Colab GPUs (T4 x1 or T4 x2).
# Same pretrained backbone and steps for every variant; only the representation + aggregator differ.
# With 2 GPUs the variants are split across them and run in parallel.
#   BACKBONE=BAAI/bge-small-en-v1.5 STEPS=1000 bash scripts/run_t4.sh     # quick pass
set -euo pipefail

BACKBONE=${BACKBONE:-BAAI/bge-base-en-v1.5}
STEPS=${STEPS:-3000}
VARIANTS=${VARIANTS:-"dense colbert colbert_grouped nyaya_tok nyaya_tok_or"}
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
NGPU=$(( NGPU > 0 ? NGPU : 1 ))
mkdir -p runs results logs

pip install -q "transformers>=4.48" pypdf

# LIMIT (Google DeepMind) for an out-of-distribution check
[ -d limit ] || git clone --depth 1 https://github.com/google-deepmind/limit

run_one() {   # gpu variant
  CUDA_VISIBLE_DEVICES=$1 python -m nyaya.train --backbone "$BACKBONE" --variant "$2" --steps "$STEPS" \
      --batch 32 --lr 1e-3 --backbone_lr 3e-5 --fp16 --grad_ckpt --save "runs/$2.pt"
  CUDA_VISIBLE_DEVICES=$1 python scripts/eval_ckpt.py --ckpt "runs/$2.pt" --andor \
      --beir limit/data/limit-small --out "results/$2.json"
}

i=0
declare -a QUEUE
for V in $VARIANTS; do
  g=$(( i % NGPU )); QUEUE[$g]="${QUEUE[$g]:-} $V"; i=$(( i + 1 ))
done
for g in $(seq 0 $(( NGPU - 1 ))); do
  echo "GPU $g:${QUEUE[$g]:-}  (log: logs/gpu$g.log)"
  ( for V in ${QUEUE[$g]:-}; do echo "== $V"; run_one "$g" "$V"; done ) > "logs/gpu$g.log" 2>&1 &
done
wait
echo "all variants finished"

VARIANTS="$VARIANTS" python - <<'EOF'
import json, os
for v in os.environ["VARIANTS"].split():
    try:
        r = json.load(open(f"results/{v}.json"))
    except FileNotFoundError:
        print(f"{v:16s}  missing (see logs/)"); continue
    a, l = r["andor"], r["limit-small"]["all"]
    ws = "  ".join(f"{k} {a[k]['recall@10']:.3f}" for k in sorted(a) if k.startswith("w"))
    print(f"{v:16s}  ANDOR-style R@10 {a['all']['recall@10']:.3f}  nDCG@10 {a['all']['ndcg@10']:.3f}   "
          f"LIMIT-small R@2 {l['recall@2']:.3f}  R@20 {l['recall@20']:.3f}")
    print(f"{'':16s}  by width: {ws}")
EOF
