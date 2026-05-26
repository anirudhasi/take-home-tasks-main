# Setup & Execution Guide — VSCode

## Prerequisites
- Python 3.11+ installed
- VSCode installed
- Git installed (optional but recommended)

---

## Step 1: Open in VSCode

```
File → Open Folder → select the `codvo_pump_failure` folder
```

---

## Step 2: Create virtual environment

Open the VSCode integrated terminal (`Ctrl+`` ` on Windows/Linux, `Cmd+`` ` on Mac):

```bash
# Create venv
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Step 3: Select Python interpreter in VSCode

- Press `Ctrl+Shift+P` → `Python: Select Interpreter`
- Choose the one inside `./venv/`

---

## Step 4: Place data files

Copy the following files into the `data/` folder:
```
data/
  train.csv
  train_units.csv
  sample_test_input.csv
```

---

## Step 5: Train the model

**Option A — VSCode Run button:**
- Go to `Run & Debug` (Ctrl+Shift+D)
- Select `Train pipeline` from the dropdown
- Press the green play button

**Option B — Terminal:**
```bash
python train.py
```

**With Optuna hyperparameter tuning (takes ~1 hour):**
```bash
python train.py --tune
```

Training output will be saved to `submission/model_artifacts/`.

---

## Step 6: Run the notebooks (for EDA / presentation)

- Open any notebook in `notebooks/`
- Click `Select Kernel` → choose the venv Python
- Run cells top to bottom with `Shift+Enter`

Recommended order:
1. `01_EDA.ipynb`
2. `02_LabelEngineering.ipynb`
3. `03_FeatureEngineering.ipynb`
4. `04_ModelSelection.ipynb`
5. `05_ErrorAnalysis.ipynb`

---

## Step 7: Run inference

```bash
# On sample test data
python predict.py --input data/sample_test_input.csv

# On sample test data WITH SHAP explanation
python predict.py --input data/sample_test_input.csv --shap

# PRESENTATION DAY COMMAND
python predict.py --input <their_test_file>.csv --output live_predictions.csv --shap
```

Results are saved to `submission/predictions/predictions.csv` by default.

---

## Presentation Day Checklist

- [ ] `python train.py` completes without errors
- [ ] `submission/model_artifacts/` contains 5 `.pkl` files
- [ ] `python predict.py --input data/sample_test_input.csv` runs in < 3 min
- [ ] `submission/predictions/predictions.csv` has all required columns
- [ ] SHAP plot displays (test with `--shap` flag)
- [ ] You know the live inference command by heart

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `ModuleNotFoundError: src` | Make sure you are running from the project root |
| `FileNotFoundError: data/train.csv` | Copy your data files into `data/` folder |
| `Artefact not found` | Run `python train.py` first |
| SHAP import error | `pip install shap matplotlib` |
| Slow training | Remove `--tune` flag; default params are already good |
