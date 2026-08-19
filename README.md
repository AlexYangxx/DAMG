# DAMG: Degradation-Aware Multi-Prior Guidance for Extreme Low-Light Image Enhancement

<p align="center">
  <b>Official code</b> for <em>DAMG</em> (Degradation-Aware Multi-Prior Guidance).
</p>

## Highlights

- **Degradation-aware fusion (local + source level).** DAMG formulates extreme LLIE as a spatially adaptive fusion between HVI photometric features and heterogeneous vision-foundation priors, so dark regions can receive stronger semantic/geometric conditioning while recoverable regions stay anchored to observation-supported cues.
- **Two-branch restoration.** At the branch level, a learned local-guidance map mixes an **input-consistent restoration** path with a **prior-guided enhancement** path.
- **Spatially adaptive source fusion.** At the source level, learned fusion weights allocate contributions among **photometric / semantic / geometric** streams, guided by prior gating signals.
- **Region-aware objectives.** Dual-branch decoding and region-aware losses specialize the two pathways to reduce interference from a shared decoder serving both regimes.

The method is instantiated as **DAMG** and described in *MANUSCRIPT SUBMITTED TO IEEE TRANSACTIONS ON MULTIMEDIA*.

## What’s included in this repo

This GitHub project contains the **DAMG-LLIE code modules and evaluation utilities**:

- `damg_network/`: DAMG architecture implementation
- `damg_data/`: dataset + prior loading/indexing utilities
- `restoration_objectives/`: restoration losses
- `eval.py`: inference / evaluation entry
- `measure.py`: PSNR/SSIM/LPIPS measurement script
- `experiments/`: helper scripts (note: training entry requires `train.py`)

**Note (training scripts):** per your earlier request, this repo does **not** include `train.py` / `TRAIN_DAMG.md`. The PowerShell scripts under `experiments/train_damg_*.ps1` call `train.py`, so training cannot run out-of-the-box from this repo alone.

## Installation

This code is designed for Python + PyTorch (CUDA recommended).

1. Install dependencies (minimum set used by `eval.py` / `measure.py`):

```bash
pip install torch torchvision opencv-python lpips numpy pillow tqdm
```

2. (Optional) If you want GPU acceleration:

- Make sure your CUDA-enabled PyTorch is installed.

## Data & prior preparation

### Dataset layout (for `eval.py`)

`eval.py` loads evaluation inputs from these default roots (relative to the repo root):

- LOLv1: `./datasets/LOLdataset/eval15/low`
- LOLv2-real: `./datasets/LOLv2/Real_captured/Test/Low`
- LOLv2-syn: `./datasets/LOLv2/Synthetic/Test/Low`
- SICE_grad: `./datasets/SICE/SICE_Grad`
- SICE_mix: `./datasets/SICE/SICE_Mix`
- FiveK: `./datasets/FiveK/test/input`

For unpaired benchmarks, `eval.py` can also output: `DICM/`, `LIME/`, `MEF/`, `NPE/`, `VV/` from the corresponding folders under `./datasets/`.

### Prior directories (for semantic / depth / normal guidance)

DAMG can optionally load **semantic / depth / normal** priors from three directories:

- `--semantic_prior_dir`
- `--depth_prior_dir`
- `--normal_prior_dir`

Optional quality priors (for gating regularization / validity):

- `--semantic_quality_dir`
- `--depth_quality_dir`
- `--normal_quality_dir`

Supported prior file extensions:

`png`, `jpg`, `jpeg`, `bmp`, `npy`, `npz`, `pt`, `pth`.

**How priors are matched to each input image:** prior indexing searches the directory tree and tries to resolve a prior by

1. relative path key (under the prior root), or
2. filename stem key,

with case/extension normalized.

If a prior is **not found**, DAMG falls back to defaults:

- semantic prior defaults to a clone of the input low image tensor
- depth defaults to the mean over input channels
- normal defaults to a pseudo-normal derived from the depth gradient

## Run evaluation

### 1) Run a single dataset

Example (LOLv2-real, with priors and a custom checkpoint):

```bash
python eval.py \
  --lol_v2_real \
  --weights_path ./weights/LOLv2_real/wo_perc.pth \
  --lite_mode safe \
  --semantic_in_channels 384 \
  --semantic_prior_dir ./priors/lolv2_real_eval/semantic \
  --depth_prior_dir ./priors/lolv2_real_eval/depth \
  --normal_prior_dir ./priors/lolv2_real_eval/normal \
  --strict_prior_loading
```

Useful options in `eval.py`:

- `--tta_mode none|hflip|flip4`
- `--amp` (mixed precision, CUDA)
- `--tile_size` / `--tile_overlap` (sliding-window inference)
- `--dump_aux_maps` (save auxiliary maps)

### 2) Reproduce the repo’s default evaluation sweep

`experiments/reproduce_all.py` runs multiple `eval.py` configurations using the following expected directory conventions:

- checkpoints under `./weights/`
- priors under `./priors/` (e.g. `./priors/lolv1_eval/semantic`, `.../depth`, `.../normal`)

Run:

```bash
python experiments/reproduce_all.py
```

### 3) Measure metrics

`measure.py` evaluates PSNR/SSIM/LPIPS using images written by `eval.py`.

Example:

```bash
python measure.py --lol
python measure.py --lol_v2_real
python measure.py --lol_v2_syn
```

## Citation

If you use this repository, please cite the DAMG paper:

```bibtex
@article{chen2026damg,
  title   = {Degradation-Aware Multi-Prior Guidance for Extreme Low-Light Image Enhancement},
  author  = {Chen, Jiayuan and Chen, Yi and Lin, Yuxiang and Zhong, Bingbing and Shi, Xinjie and Yang, Xingxing},
  year    = {2026}
}
```

## License

MIT (see `LICENSE`).

