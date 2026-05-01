#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="/home/eggplant/anaconda3/envs/minimind/bin/python"

cd "$ROOT_DIR"

echo "[formal_12h] start: $(date '+%Y-%m-%d %H:%M:%S')"
echo "[formal_12h] checking required files"
test -f dataset/pretrain_t2t_mini.jsonl
test -f dataset/sft_t2t_mini.jsonl
test -f out/formal_limited_pretrain_768.pth

mkdir -p logs outputs out checkpoints

echo "[formal_12h] pretrain start: $(date '+%Y-%m-%d %H:%M:%S')"
cd "$ROOT_DIR/trainer"
env HF_HOME=/tmp/huggingface XDG_CACHE_HOME=/tmp "$PYTHON_BIN" train_pretrain.py \
  --data_path ../dataset/pretrain_t2t_mini.jsonl \
  --from_weight formal_limited_pretrain \
  --save_dir ../out \
  --save_weight formal_12h_pretrain \
  --from_resume 1 \
  --epochs 1 \
  --batch_size 1 \
  --learning_rate 5e-4 \
  --device cuda:0 \
  --dtype float16 \
  --num_workers 0 \
  --accumulation_steps 32 \
  --log_interval 500 \
  --save_interval 5000 \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --max_seq_len 768 \
  --use_moe 0 \
  --loss_log_path ../logs/formal_12h_train_loss.csv \
  --max_steps 300000

echo "[formal_12h] pretrain end: $(date '+%Y-%m-%d %H:%M:%S')"
test -f "$ROOT_DIR/out/formal_12h_pretrain_768.pth"

echo "[formal_12h] sft start: $(date '+%Y-%m-%d %H:%M:%S')"
env HF_HOME=/tmp/huggingface XDG_CACHE_HOME=/tmp "$PYTHON_BIN" train_full_sft.py \
  --data_path ../dataset/sft_t2t_mini.jsonl \
  --from_weight formal_12h_pretrain \
  --save_dir ../out \
  --save_weight formal_12h_full_sft \
  --from_resume 1 \
  --epochs 1 \
  --batch_size 1 \
  --learning_rate 1e-5 \
  --device cuda:0 \
  --dtype float16 \
  --num_workers 0 \
  --accumulation_steps 16 \
  --log_interval 500 \
  --save_interval 5000 \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --max_seq_len 768 \
  --use_moe 0 \
  --loss_log_path ../logs/formal_12h_train_loss.csv \
  --max_steps 300000

echo "[formal_12h] sft end: $(date '+%Y-%m-%d %H:%M:%S')"
test -f "$ROOT_DIR/out/formal_12h_full_sft_768.pth"

cd "$ROOT_DIR"

echo "[formal_12h] plotting loss"
python scripts/plot_loss.py \
  --input logs/formal_12h_train_loss.csv \
  --output outputs/formal_12h_loss_curve.png \
  --output-dir outputs \
  --stage-prefix formal_12h_

echo "[formal_12h] comparing generated answers"
env HF_HOME=/tmp/huggingface XDG_CACHE_HOME=/tmp "$PYTHON_BIN" scripts/compare_models.py \
  --device cuda \
  --self-weight formal_12h_full_sft \
  --self-hidden-size 768 \
  --self-num-hidden-layers 8 \
  --finetune-base-weight formal_12h_full_sft \
  --finetune-lora-weight formal_lora_hust \
  --finetune-hidden-size 768 \
  --finetune-num-hidden-layers 8 \
  --raw-output outputs/formal_12h_model_compare_raw.jsonl \
  --markdown-output outputs/formal_12h_model_compare.md \
  --max-new-tokens 128 \
  --temperature 0.7 \
  --top-p 0.9

echo "[formal_12h] done: $(date '+%Y-%m-%d %H:%M:%S')"
echo "[formal_12h] outputs:"
echo "  logs/formal_12h_train_loss.csv"
echo "  out/formal_12h_pretrain_768.pth"
echo "  out/formal_12h_full_sft_768.pth"
echo "  outputs/formal_12h_loss_curve.png"
echo "  outputs/formal_12h_pretrain_loss_curve.png"
echo "  outputs/formal_12h_sft_loss_curve.png"
echo "  outputs/formal_12h_model_compare.md"
