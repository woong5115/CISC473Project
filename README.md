# CISC 473 Project — P02: Zero-Reference Low-Light Image Enhancement

Reproduction of Zero-DCE and Zero-DCE++ on SICE, an analysis of our midterm
reproduction gap, and two new training losses for Zero-DCE++ (Max-RGB-guided
Dynamic Exposure Control and Ratio-Locking Spatial-Color Consistency) aimed at
extremely dark and underwater scenes.

## Reproduce all final-report results (one click)

Open [`final/CISC473_final_experiments.ipynb`](final/CISC473_final_experiments.ipynb)
in Google Colab and:

1. **Runtime → Change runtime type → T4 GPU**
2. Put the SICE data in your Google Drive (see *Data* below)
3. **Runtime → Run all**, and allow Google Drive access

Everything runs from that single "Run all": data preparation, evaluation of the
midterm and released checkpoints, 12 training runs (4 configurations × 3 seeds),
evaluation, and the final tables. It takes about 25 min of data preparation plus
~2 h 20 min on a free Colab T4. Results are written to
`My Drive/CISC473_final/` (`results/*.json`, `models/`, `figures/`, `tables.tex`).
If Colab disconnects, press **Run all** again: finished experiments are skipped and
training resumes from the last checkpoint (saved every 5 epochs).

Dependencies: [`requirements.txt`](requirements.txt) (installed by the notebook).

## Data

SICE ([official page](https://github.com/csjcai/SICE)):

| Part | Use | Google Drive |
|---|---|---|
| Part 1 (360 sequences, 3,021 images) | training, references unused | [link](https://drive.google.com/file/d/1HiLtYiyT9R7dR9DRTLRlUUrAicC4zzWN/view) |
| Part 2 (229 sequences + `Label`) | testing (767 images) | [link](https://drive.google.com/file/d/16VoHNPAZ5Js19zspjFOsKiGRrfkDgHoN/view) |

The notebook first looks for a file whose name contains `Dataset_Part1` /
`Dataset_Part2` anywhere in *My Drive* (up to 3 folders deep) and otherwise tries to
download it with `gdown`. The public links are often over Google's download limit,
so the reliable way is to add a **copy** of each file to your own Drive first.

Test-set rule: the darkest 3 (4) images of each 7-image (9-image) sequence, and the
darkest 2 (3) images of the 5/6-image (8-image) sequences, giving the 767 images of
the Zero-DCE paper. Images and references are resized to 1200×900 (Lanczos).

## Experimental setting

| Setting | Value |
|---|---|
| Model | Zero-DCE++ (architecture unchanged) |
| Training | SICE Part 1 at 256×256, 30 epochs, batch 8, Adam (lr 1e-4, wd 1e-4), grad-clip 0.1, mixed precision |
| Seeds | 42, 123, 999 (fixed in `final/exp.py`; each configuration trained 3× from scratch) |
| Metrics | PSNR, SSIM against the SICE reference (SSIM matches `skimage` exactly) |
| Proposed losses | `E_min = 0.4`, `E_max = 0.7`, `W_RL = 50` |

The released Zero-DCE++ trains at 512×512 for 100 epochs; that would take ~26
GPU-hours for 12 runs, so all our runs (baseline and proposed) use the reduced
budget above.

## Results (SICE Part 2, mean ± std over 3 seeds)

| Configuration | PSNR (dB) ↑ | SSIM ↑ |
|---|---|---|
| Zero-DCE++ baseline (our training) | 15.00 ± 0.06 | 0.504 ± <0.001 |
| + Ratio-Lock | **15.24 ± 0.04** | 0.493 ± 0.002 |
| + Dynamic Exposure | 13.53 ± 0.03 | 0.441 ± 0.002 |
| Full model (both) | 13.19 ± 0.05 | 0.420 ± 0.003 |
| Zero-DCE++ released checkpoint | 16.00 | 0.508 |
| Zero-DCE released checkpoint | 15.93 | 0.497 |

Midterm reproduction gap: our midterm notebook scored outputs against the
low-light **input** instead of the reference. Re-scoring the same checkpoints
correctly raises PSNR from 12.35 → 15.93 dB (Zero-DCE) and 9.24 → 15.68 dB
(Zero-DCE++). See Section 4.3 of the final report.

## Checkpoints

- Midterm checkpoints: `models/Zero-DCE/Zero-DCE_code/snapshots/Epoch99.pth`
  (identical to the authors' release) and
  `models/Zero-DCE_extension/Zero-DCE++/snapshots_Zero_DCE++/Epoch90.pth`.
- Final-report models are saved by the notebook to `My Drive/CISC473_final/models/`
  (e.g. `prop_rl_s42_e30_r256_ep30.pth`, ~50 KB each; copies are also in `final/models/` if uploaded). To regenerate them, run the
  notebook (about 10 min per model on a T4). To evaluate a saved model:

```python
import exp
m = exp.load_trained("dcepp", "path/to/model.pth")
print(exp.evaluate(m, "dcepp", exp.sice_pairs(exp.prepare_sice())))
```

## Repository structure

```
final/        final-report experiments: exp.py (all code) + Colab notebook
models/       original Zero-DCE and Zero-DCE++ code (midterm), with our fixes
notebooks/    midterm notebooks
images/       figures used in the reports
```

Paths: `final/exp.py` uses Colab locations (`/content/work`, `/content/drive/...`)
as defaults; all of them can be changed with `exp.configure(WORK=..., DRIVE=...)`.
