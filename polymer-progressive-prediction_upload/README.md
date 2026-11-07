# Polymer Progressive Prediction

Polymer property prediction with CNN, multi-level sequence and progressive regression models.

## Repository layout

```text
models/       Neural networks and digit prediction heads
data/        Dataset readers, tokenizers, scalers and CSV datasets
train/        Training and evaluation entry points
utils/        Initialization, plotting, splitting and audit helpers
configs/      Public Tg configuration and loss protocol
provenance/   Dataset and source records
tests/       Standard-library layout and data checks
```

## Installation

Python 3.10 or newer is recommended. Install a PyTorch build appropriate for your hardware, then:

```bash
python -m pip install -r requirements.txt
# For the Chemprop public Tg model:
python -m pip install -r requirements-public-tg.txt
```

## Training

Run commands from the repository root:

```bash
python -m train.theory_cnn --out_dir outputs/theory_cnn
python -m train.theory_multilevel --out_dir outputs/theory_multilevel
python -m train.theory_ablation --out_dir outputs/theory_ablation
python -m train.theory_progressive --no_grid_search --out_dir outputs/theory_progressive
python -m train.experimental --out_dir outputs/experimental
```

The optional distributed theoretical entry point is `train.theory_progressive_ddp`.

For public Tg, set `CUBLAS_WORKSPACE_CONFIG=:4096:8`. A GPU is required by this entry point.

```bash
# TASK is 0 through 99: fold=TASK//10, seed_index=TASK%10.
python -m train.public_tg train --task TASK
# After all 100 training tasks finish:
python -m train.public_tg finalize
```

## Data

- `data/theory`: 1574 training, 190 validation and 196 test records.
- `data/experimental/experimental_dataset.csv`: all 872 experimental records; group splits are generated at runtime.
- `data/public_tg`: 683 development and 201 test records. Existing data-processing records are in `provenance/public_tg`.

## Check files without running a model

```bash
python -m unittest discover -s tests -v
```
