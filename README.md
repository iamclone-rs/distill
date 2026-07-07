# CoPrompt SBIR — DFN5B benchmark

The maintained training path uses a frozen DFN5B teacher and a CoPrompt
ViT-B/32 student. The student objective is intentionally limited to:

```text
CE(photo, class) + CE(sketch, class)
+ Triplet(sketch, positive photo, negative photo)
+ lambda_kd * KL(DFN5B sketch-photo relations || student relations)
```

## Reproduce the Sketchy-1 benchmark

From the repository root, run both stages with:

```bash
bash scripts/run_sketchy1.sh /path/to/Sketchy
```

On the Colab layout used by the benchmark, the default root already resolves
to `/content/sketchy/Sketchy`, so `bash scripts/run_sketchy1.sh` is sufficient.

The commands executed by the script are shown below.

### 1. Train the residual DFN5B modality adapter

```bash
python -m src.finetune_dfn5b_adapter \
  --root /path/to/dataset \
  --dataset sketchy_1 \
  --epochs 1 \
  --batch_size 128 \
  --bottleneck_dim 32 \
  --lr 1e-4 \
  --temperature 0.07 \
  --lambda_retrieval 1 \
  --lambda_semantic 0.5 \
  --fp16_backbone \
  --output_dir teacher_adapter_runs/dfn5b_sketchy1
```

DFN5B (`ViT-H-14-quickgelu`, pretrained `dfn5b`) and residual adapter mode
are fixed by the benchmark code.

### 2. Train the CoPrompt student

```bash
python -m src.main_train \
  --root /path/to/dataset \
  --dataset sketchy_1 \
  --epochs 3 \
  --teacher_adapter_ckpt teacher_adapter_runs/dfn5b_sketchy1/best.pt \
  --batch_size 64 \
  --lr 4e-5 \
  --quantize_fp16 \
  --lambda_kd 3 \
  --kd_temperature 0.07 \
  --n_ctx 1 \
  --lambda_cls 1 \
  --lambda_triplet 1 \
  --exp_name sketchy1_dfn5b_kd
```

The recorded Sketchy-1 configuration uses `lambda_kd=3`, `n_ctx=1`, seed 42,
and selects the best student checkpoint by unseen `mAP@all`.
