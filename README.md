# Polymer Property Prediction with Multi-Level Encoding and Progressive Regression

Code and datasets for **Adaptive Multi-Level Chain-Length Encoding and Progressive Regression for Polymer Property Prediction**, by Zhiguo Xu and Jianfeng Li.

This project studies how polymer property models can represent both repeat-unit chemistry and chain length. It combines a multi-level sequence representation with a progressive regression head that predicts ordered digit groups. The repository also includes a Chemprop-based directed message-passing neural network (D-MPNN) for evaluating progressive prediction on a public glass-transition-temperature benchmark.

## Contents

- [Models and tasks](#models-and-tasks)
- [Repository structure](#repository-structure)
- [Installation](#installation)
- [Datasets](#datasets)
- [Training and evaluation](#training-and-evaluation)
- [Public Tg model and ensemble](#public-tg-model-and-ensemble)
- [Outputs and reproducibility](#outputs-and-reproducibility)
- [Repository checks](#repository-checks)
- [References and contact](#references-and-contact)

## Models and tasks

| Task | Included implementations |
| --- | --- |
| Theoretical polymer property prediction | CNN baseline, multi-level direct regression, an uncompressed-sequence ablation, and multi-level progressive regression |
| Experimental polymer property prediction | Multi-level multi-task prediction of glass-transition temperature, melt viscosity, and an auxiliary molecular-weight target |
| Public polymer Tg benchmark | D-MPNN encoder with a progressive prediction head, evaluated through ten folds and ten fixed training seeds |

The multi-level representation uses three related inputs: repeat-unit chemistry, a repeat-unit sequence or group definition, and a sequence of groups for longer chains. The progressive head predicts successive digit groups conditioned on preceding predictions during inference. The theoretical and public-Tg implementations use different target encodings; the public model's log-space three-digit encoding is described below.

## Repository structure

The executable project is inside **`polymer-progressive-prediction_upload/`**. Run the Python commands in this README from that directory.

```text
mufeng/
├── README.md
├── LICENSE
└── polymer-progressive-prediction_upload/
    ├── models/                    # Neural networks and progressive prediction heads
    ├── data/
    │   ├── theory/                # Fixed theoretical train/validation/test CSVs
    │   ├── experimental/          # Complete experimental dataset
    │   ├── public_tg/             # Public Tg development and test CSVs
    │   └── *.py                   # Data readers, tokenizers, and dataset classes
    ├── train/                     # Training and evaluation entry points
    ├── utils/                     # Splitting, plotting, metrics, and inspection helpers
    ├── configs/                   # Public Tg architecture and training protocol
    ├── provenance/                # Data, split, source, and packaging records
    ├── tests/                     # Standard-library repository integrity checks
    ├── requirements.txt
    └── requirements-public-tg.txt
```

## Installation

Python 3.10 or newer is recommended. Create a dedicated environment and install a PyTorch build compatible with your hardware before installing the remaining dependencies.

```bash
git clone https://github.com/mufeng-cloud/mufeng.git
cd mufeng/polymer-progressive-prediction_upload

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Install an appropriate PyTorch build, then install project dependencies.
python -m pip install -r requirements.txt
```

The core requirements include PyTorch, NumPy, pandas, scikit-learn, Matplotlib, and RDKit. For the public D-MPNN benchmark, install the additional Chemprop and Lightning dependencies:

```bash
python -m pip install -r requirements-public-tg.txt
```

The public benchmark requires **Chemprop 2.x** and a CUDA GPU with the current training entry point. The optional distributed theoretical implementation also requires CUDA GPUs and the NCCL backend. The other entry points select CUDA when available and otherwise use the CPU. The requirements files specify dependencies rather than a fully pinned environment lock; record the versions used for a new run.

## Datasets

All paths in this section are relative to `polymer-progressive-prediction_upload/`.

| Dataset | Files | Records | Split protocol |
| --- | --- | --- | --- |
| Theoretical properties | `data/theory/train.csv`, `val.csv`, `test.csv` | 1,574 training; 190 validation; 196 test | Fixed partitions with disjoint RDKit-canonical graph identities |
| Experimental properties | `data/experimental/experimental_dataset.csv` | 872 total | Polymer-group holdout and five group-disjoint development folds, generated at runtime |
| Public Tg benchmark | `data/public_tg/training_data.csv`, `test_data.csv` | 683 development; 201 test | Frozen ten-fold development split and a separate input-disjoint test set |

### Theoretical data

The theoretical CSVs contain `smiles`, `num_unit`, `Mw`, `Re`, and `polymer_seq`. The `Re` column stores the coarse-grained mean-square end-to-end-distance target. The fixed partitions contain 66, 8, and 8 canonical graph groups for training, validation, and testing, respectively. Dataset origin and split records are available in `provenance/theory/`.

### Experimental data

The complete experimental CSV is provided without pre-exported split CSVs. The training entry point reads `polymer`, `num_unit`, `Mw`, `Glass transition temperature`, and `viscosity`; unavailable task labels are handled with masks.

Splitting uses the exact `polymer` input string as the group identity. Twenty percent of polymer types are held out for testing using `GroupShuffleSplit`, and the remaining polymer types are divided with five-fold `GroupKFold`. Consequently, the held-out fraction refers to polymer types, not necessarily 20% of rows. The actual split is saved as `polymer_group_split.json` in the output directory. Target scaling and the character vocabulary are fitted using the training fold.

### Public Tg data

The public pipeline reads molecular inputs from `SMILES_clean` and continuous targets from `logTg`, expressed as `log10(Tg/K)`. Eighteen test records whose `SMILES_clean` inputs also occurred in the development data were excluded, leaving 201 test records. The loader checks this exact-string disjointness and the recorded dataset hashes.

The packaged dataset is a derived version: before the input-overlap exclusion, same-input groups with a Kelvin label spread above 10 K were assigned their group median and stored again as four-decimal log labels. The procedure used both original partitions. Its processing history is retained in `provenance/public_tg/harmonization_manifest.json` and `test_input_exclusion.json`; the supplied version should be identified when reporting new evaluations.

## Training and evaluation

Run each command from **`mufeng/polymer-progressive-prediction_upload/`**. Use a separate output directory for each experiment.

### Theoretical-property models

```bash
# CNN direct-regression baseline
python -m train.theory_cnn --out_dir outputs/theory_cnn

# Multi-level direct regression
python -m train.theory_multilevel --out_dir outputs/theory_multilevel

# Uncompressed-sequence ablation
python -m train.theory_ablation --out_dir outputs/theory_ablation

# Multi-level progressive regression, without the built-in grid search
python -m train.theory_progressive --no_grid_search \
  --out_dir outputs/theory_progressive
```

The progressive theoretical entry point enables grid search by default. Keep `--no_grid_search` for a single configured run. Inspect each entry point's `--help` output for its supported options and defaults.

An optional distributed implementation is available for multi-GPU training. For example, on a machine with two CUDA GPUs:

```bash
torchrun --standalone --nproc_per_node=2 \
  -m train.theory_progressive_ddp --no_grid_search \
  --out_dir outputs/theory_progressive_ddp
```

### Experimental multi-task model

```bash
python -m train.experimental \
  --data_csv data/experimental/experimental_dataset.csv \
  --out_dir outputs/experimental
```

This command creates the polymer-group test holdout, trains the five development folds, selects checkpoints using validation loss, and evaluates the selected models on the held-out test set. The supplied CSV values are standardized within each training fold; the script does not apply an additional logarithm to those values.

### Public Tg benchmark

Set the CUDA workspace configuration before starting Python:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Train one model: task IDs range from 0 to 99.
python -m train.public_tg train --task 0
```

Each task corresponds to one fold and one training seed:

```text
fold       = task // 10
seed_index = task % 10
```

To train all 100 models sequentially in a POSIX shell:

```bash
for task in $(seq 0 99); do
  python -m train.public_tg train --task "$task" || exit 1
done
```

The tasks can instead be assigned to separate scheduler jobs, with one task per GPU job. Start final evaluation only after all 100 tasks have completed successfully:

```bash
python -m train.public_tg finalize
```

Finalization verifies training identities, effective configurations, source and data hashes, and checkpoint hashes. It then freezes the test manifest, evaluates all validation-selected checkpoints, and writes the ensemble results. Training creates a per-model claim directory to reject duplicate launches; it does not implement automatic resume or retry. Keep results from different experiments in separate project copies.

## Public Tg model and ensemble

The fixed public model configuration is recorded in [`configs/public_tg.json`](polymer-progressive-prediction_upload/configs/public_tg.json):

| Component | Configuration |
| --- | --- |
| Molecular encoder | Chemprop bond message passing; hidden dimension 300; depth 3; mean aggregation; batch normalization |
| Thousands head | `300 → 1` |
| Hundreds head | `316 → 64 → PReLU → 1` |
| Tens head | `316 → 16 → PReLU → 1` |
| Previous-digit encoder | Shared numeric MLP: `digit/9 → Linear(1,16) → PReLU → Linear(16,16)` |
| Head input dropout | 0.3 for each head |
| Digit output | `9 × sigmoid`, followed by half-up rounding during decoding |
| Training objective | Unnormalized digit MSE weights `100:80:20`; no reconstruction loss |
| Optimizer | AdamW; initial learning rate 0.002; weight decay 0 |
| Learning-rate schedule | Epoch-wise cosine annealing; `T_max=300`; minimum learning rate `1e-7`; no warmup |
| Batch size and stopping | Batch size 64; maximum 300 epochs; early-stopping patience 50 |
| Checkpoint selection | Lowest autoregressive validation RMSE in log space |

The three heads predict the thousands, hundreds, and tens digits of the scaled log target. For example, `logTg = 2.5637` produces the digit labels `[2, 5, 6]`. A model's decoded prediction is

```text
pred_logTg = (1000 × d_thousands + 100 × d_hundreds + 10 × d_tens) / 1000
```

Single-model predictions therefore have a resolution of 0.01 in log space. The first digit is learned rather than fixed. Training uses the true preceding digit as a condition (teacher forcing); validation and testing use the preceding predicted digit.

The frozen development folds use split seed 42. Ten fixed training seeds, listed in `configs/public_tg.py`, give **10 folds × 10 seeds = 100 models**. Every model contributes equally:

```text
ensemble_log_prediction = mean(decoded_log_prediction from all 100 models)
```

MAE, RMSE, and R² are calculated after averaging predictions. Digits are not averaged and rounded again, and predictions are not converted to Kelvin before ensemble averaging. The outputs also report each seed's ten-model ensemble separately. New runs use the supplied 201-row test set; historical results from a different test version must be identified separately.

## Outputs and reproducibility

The theoretical and experimental commands write to the directory supplied with `--out_dir`; output contents depend on the selected entry point and include checkpoints, predictions, metrics, and plots.

The public benchmark writes to fixed directories within the executable project:

| Directory | Contents |
| --- | --- |
| `results/fold_XX/seed_SEED/` | Selected checkpoint, training logs, validation predictions, effective configuration, and completion metadata |
| `final_test/seed_SEED/fold_XX/` | Individual-model test predictions and metrics |
| `aggregate/` | Ensemble predictions, out-of-fold predictions, per-seed tables, and `summary.json` |
| `provenance/public_tg/frozen_test_manifest.json` | Checkpoint, source, split, and dataset identities frozen before testing |

The public prediction CSVs include row identity, molecular input, true and predicted log targets, and raw digit outputs. Preserve the CSVs, checkpoints, logs, configuration, and provenance records together when archiving an experiment. Source and data integrity checks are enforced by the public pipeline; intentional changes require a separately documented experiment with matching provenance records.

## Repository checks

These checks use the Python standard library and do not train or run a model:

```bash
python -m unittest discover -s tests -v
```

They check Python syntax, local import paths, dataset counts and hashes, frozen development folds, public test-input disjointness, and preservation of recorded model definitions and command-line defaults. They are packaging and integrity checks, not a substitute for evaluating a trained model.

## References and contact

- Project manuscript: *Adaptive Multi-Level Chain-Length Encoding and Progressive Regression for Polymer Property Prediction*, Zhiguo Xu and Jianfeng Li.
- Public benchmark source: *Polymer property prediction using ensemble of directed message-passing neural networks with uncertainty estimation and applicability domain analysis*. [DOI: 10.1557/s43579-026-00948-5](https://doi.org/10.1557/s43579-026-00948-5).
- For questions about the implementation, open an [issue](https://github.com/mufeng-cloud/mufeng/issues). Corresponding author: Jianfeng Li, `lijf@fudan.edu.cn`.
- See the repository's [LICENSE](LICENSE) for licensing terms.
