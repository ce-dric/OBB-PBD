# Training environment

| File | Contents |
|---|---|
| `pip_freeze.txt` | `pip freeze` of the training container |
| `python_version.txt` | `python --version` |
| `gpu_inventory.csv` | GPU model and driver version (3 × NVIDIA RTX 6000 Ada Generation, driver 570.211.01) |

`pip_freeze.txt` and `python_version.txt` were written by `scripts/run_obb_pbd.sh` at the start of the seed-1 OBB-PBD run (2026-08-12), before training. They are identical to the package inventory captured from the same retained container on the same day, which the paper's Data Availability statement refers to. The container ran Ubuntu 22.04.5 LTS when that inventory was taken; no `/etc/os-release` copy was kept.

This is an inventory, not a lockfile: many entries are conda-forge build paths (`@ file:///home/conda/feedstock_root/...`), so `pip install -r pip_freeze.txt` will not work elsewhere. To set up an environment, use the `Dockerfile` and `requirements/base.txt`, and check the key versions against this file: Python 3.11.10, `torch==2.5.1+cu121`, `transformers==4.57.1`, `peft==0.20.0`.
