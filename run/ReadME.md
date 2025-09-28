# FLP Generation and Analysis (WGAN)

This repo trains a WGAN to generate Frustrated Lewis Pairs (FLPs) for CO₂ reuse and then analyzes the results.

## Contents

- `acid.csv` — Lewis acid dataset
- `zinc.csv` — Lewis base dataset
- `checkpoint_code_lr.py` — main training/generation script
- `run_model.sh` — batch runner that executes the model 5 times
- `Model_analysis.ipynb` — notebook to analyse outputs and make figures
- `vocab.json` — optional SMILES token vocab (speeds up training if present)
- *(not included)* `tokenized_dataset.pt` — optional pretokenised dataset (large file)

## Environment

The code uses the following Python packages (from imports):
- `torch`, `torchvision` not needed, but `torch` required
- `pandas`, `numpy`, `matplotlib`
- `rdkit`
- `selfies`
- `tqdm`
- Standard libs: `argparse`, `pathlib`, `itertools`, `json`, `os`, `random`
