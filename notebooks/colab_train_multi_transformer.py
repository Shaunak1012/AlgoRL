# =============================================================================
# COLAB TRAINING — MVP Step 6: Multi-Asset PPO + Causal Transformer encoder
# =============================================================================
# Workflow:
#   1. Locally: zip the whole algo-trading-rl/ folder → algo-trading-rl.zip
#      (data/*.parquet must be inside — no redownload needed on Colab)
#   2. Colab: Runtime → Change runtime type → GPU (T4 is fine).
#   3. New notebook → paste each CELL below into its own cell in order.
#   4. Cell 2 will prompt you to upload algo-trading-rl.zip.
#   5. Cell 4 runs training (~100k steps). Watch the stdout stream.
#   6. Cell 5 packages results as colab_output.zip.
#   7. Download colab_output.zip from the Colab file browser and drop it into
#      the local workspace root. Claude Code reads runs/<timestamp>_r006_transformer/
#      (learning_curve.csv, config.yaml, model_final.zip, + stdout you paste back).
# =============================================================================


# ── CELL 1: Install dependencies ─────────────────────────────────────────────
!pip install -q gymnasium stable-baselines3 yfinance ta pyyaml


# ── CELL 2: Upload and extract project zip ───────────────────────────────────
import os, zipfile, sys
from google.colab import files

print("Select algo-trading-rl.zip when the dialog appears...")
uploaded = files.upload()

zip_name = next(iter(uploaded))
with zipfile.ZipFile(zip_name, "r") as z:
    z.extractall(".")

# Handle both flat and nested extraction
for candidate in ("algo-trading-rl", "algo-trading-rl-main"):
    if os.path.isdir(candidate):
        os.chdir(candidate)
        break

sys.path.insert(0, os.getcwd())
print(f"Working dir: {os.getcwd()}")
print("Contents:", sorted(os.listdir(".")))
assert os.path.exists("scripts/train_multi_asset_transformer.py"), "train script missing"
assert os.path.exists("data/SPY.parquet"), "SPY.parquet missing — re-zip with data/"


# ── CELL 3: Verify GPU ────────────────────────────────────────────────────────
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")


# ── CELL 4: Run training (100k steps, r006_transformer) ──────────────────────
# Uses configs/transformer.yaml as-is. Change --steps / --tag if you want.
!python scripts/train_multi_asset_transformer.py \
    --config configs/transformer.yaml \
    --tag r006_transformer


# ── CELL 5: Package results → colab_output.zip ───────────────────────────────
import shutil, os
shutil.make_archive("colab_output", "zip", "runs")
size_kb = os.path.getsize("colab_output.zip") / 1024
print(f"colab_output.zip ready ({size_kb:.0f} KB)")
print("Download it from the Colab file browser (folder icon → right-click → Download).")
print("Drop it into your local algo-trading-rl/ workspace root.")
