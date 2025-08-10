# STEM Probabilistic Test Simulation Generator

Reproducible generator for **confidence‑based, multiple‑choice** STEM assessments. For each item, it emits a **ground‑truth vector** over options A–G (1–3 correct choices; truth mass conserved to four decimal places) and **archetype‑specific response distributions** (six behaviour profiles; probability mass conserved exactly at 4 d.p. via a largest‑remainder scheme). The CSV output (one row per *(Test, Question, Archetype)*) is designed for downstream **strictly proper scoring** (e.g., log‑loss/ignorance), **Brier** scoring, calibration diagnostics, and grading experiments.

---

## Companion repository (scoring & examples)

For end‑to‑end evaluation (Brier/log‑loss scoring, calibration plots, grade mappings) see the companion codebase:

**PAS_Brier_Clim** → <https://github.com/antonioclim/PAS_Brier_Clim>

Use this generator to create datasets, then follow the **PAS_Brier_Clim** README for analysis scripts/notebooks that compute scores, diagnostics, and visualisations on the same CSV schema.

> Quick integration sketch (Windows/PowerShell):
> ```powershell
> # 1) Generate a dataset here
> python .\generate_stem_sim.py --tests 20 --questions 25 --seed 42 --out .\data\stem_20x25.csv --validate
>
> # 2) Clone companion repo and set up its environment
> git clone https://github.com/antonioclim/PAS_Brier_Clim.git
> cd .\PAS_Brier_Clim
> python -m venv .venv
> .\.venv\Scripts\Activate.ps1
> python -m pip install -r requirements.txt
>
> # 3) Follow PAS_Brier_Clim docs to run scoring/plots on .\data\stem_20x25.csv
> ```
> Alternatively, add `PAS_Brier_Clim` as a submodule in this project:
> ```powershell
> git submodule add https://github.com/antonioclim/PAS_Brier_Clim external/PAS_Brier_Clim
> ```

---

## Highlights

- **Deterministic design** with a single RNG seed → identical datasets across machines.  
- **Item structure:** 4–7 presented options (A–G); at least as many incorrect as correct.  
- **Truth vectors:** k=1 → 1.0; k=2 → 0.5/0.5; k=3 → 0.3333/0.3333/0.3334 (sum = 1.0000).  
- **Archetypes (one‑line reminders; full detail in code):**  
  **Zota** (well‑calibrated strong), **Clim** (cautious 60/40), **Tolomacea** (overconfident distractor),  
  **Cristina** (good but erratic), **Gabriela** (underconfident), **Dan** (mixed; sometimes flips).  
- **Validation switch** (`--validate`) confirms exact mass conservation per row.  
- **Reproducibility helper** (`--save_requirements`) writes `requirements.txt` with pinned versions.

---

## Installation (Windows 11 / PowerShell)

Two standard paths. In both, **quote any paths containing spaces or `#`** (PowerShell treats `#` as a comment).

### Path A — Virtual environment (recommended)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

### Path B — Direct install into the active interpreter
```powershell
python -m pip install --upgrade pip
python -m pip install numpy pandas

      # NOTES: I prefer Path A for long‑term, publication‑grade reproducibility

## Usage

# Basic
```powershell
python .\generate_stem_sim.py --tests 20 --questions 25 --seed 42 --out .\data\stem_20x25.csv

# Synonyms supported:
#   --tests / --num_tests
#   --questions / --num_questions
#   --out / --output

# Optional switches
#   --zip                # also write a ZIP archive with the CSV inside
#   --validate           # run mass-conservation checks and report counts
#   --save_requirements  # write requirements.txt with pinned numpy/pandas


## Typical study batches
# 20 tests × {5,10,15,20,25,35,50,75,100} questions
```powershell
python .\generate_stem_sim.py --tests 20 --questions 5   --out .\data\stem_20x05.csv
python .\generate_stem_sim.py --tests 20 --questions 10  --out .\data\stem_20x10.csv
python .\generate_stem_sim.py --tests 20 --questions 15  --out .\data\stem_20x15.csv
python .\generate_stem_sim.py --tests 20 --questions 20  --out .\data\stem_20x20.csv
python .\generate_stem_sim.py --tests 20 --questions 25  --out .\data\stem_20x25.csv
python .\generate_stem_sim.py --tests 20 --questions 35  --out .\data\stem_20x35.csv
python .\generate_stem_sim.py --tests 20 --questions 50  --out .\data\stem_20x50.csv
python .\generate_stem_sim.py --tests 20 --questions 75  --out .\data\stem_20x75.csv
python .\generate_stem_sim.py --tests 20 --questions 100 --out .\data\stem_20x100.csv
```
      # VS Code tip: Use Terminal → New Terminal in the project folder, or create a launch.json to run with fixed arguments from Run and Debug.


# Output schema (CSV)

## One row per (Test, Question, Archetype):
    Indexing: Test (e.g., Test7), Question (e.g., Q12), Domain (cycled from a STEM list).
    Truth (sum = 1 over the correct set): TruthA … TruthG.
    Archetype label: Archetype ∈ {Zota, Clim, Tolomacea, Cristina, Gabriela, Dan}.
    Probabilities (sum = 1 over present options only): ProbA … ProbG.
    Present options: A contiguous prefix A… up to the item’s n_opts (4–7). Slots beyond n_opts are 0.0000.

## Invariants (enforced and validated):
    For each row, ProbA…G over presented options sums to 1.0000 at 4 d.p.
    For each row, TruthA…G positive entries sum to 1.0000 at 4 d.p.
    Items always have ≥ as many incorrect as correct choices.

## Using this data with PAS_Brier_Clim
    Purpose: score the generated responses with Brier/log‑loss, build calibration curves, and apply grade mappings.
    Workflow: generate CSVs here → process them in PAS_Brier_Clim (scripts/notebooks).
    Interoperability: column names and formats are intentionally simple (pandas‑friendly). A minimal ad‑hoc analysis (outside PAS_Brier_Clim) could look like:

      ```python

      import pandas as pd, numpy as np
      df = pd.read_csv(r'.\data\stem_20x25.csv')
      # Example: per-row log-loss against the correct set (multi-correct allowed)
      truth = df[[f'Truth{c}' for c in 'ABCDEFG']].to_numpy(float)
      pred  = df[[f'Prob{c}'  for c in 'ABCDEFG']].to_numpy(float)
      # Probability mass on the correct set (supports 1–3 correct options)
      p_correct = (truth * pred).sum(axis=1)  # since truth mass over corrects sums to 1
      # Numerical safety for log
      eps = 1e-12
      logloss = -(np.log(np.clip(p_correct, eps, 1.0)))
      df['logloss'] = logloss
      ```
```
      ### NOTE: For publication‑ready pipelines, prefer the curated implementations and plots in PAS_Brier_Clim.

## Reproducibility

    Seeded generation: --seed controls the entire pipeline (structure + probabilities).
    Pinned environment: --save_requirements writes requirements.txt with the exact numpy/pandas versions in use.
    Archival: --zip produces a single archive convenient for supplement upload.
    Validation: --validate prints counts of (i) probability‑mass errors and (ii) truth‑mass errors (both should be 0).

## Extending the tool (modular pathways)

    Testing: Add unit tests for your scoring/grade functions against canonical truth/probability vectors.
    Pipelines: Use this generator as stage 1 in a reproducible workflow (e.g., Snakemake/Make, notebooks).
    Alternative archetypes: Implement new behavioural models by following existing function signatures and re‑using the mass‑conserving rounding utility.
    Different scoring rules: Keep the dataset fixed and swap in CRPS/Brier/log‑loss variants; compare sensitivity and incentives.
    File adapters: Export to JSONL/Parquet for large‑scale experiments or cloud pipelines.

## Minimal dependencies

    Python ≥ 3.8 (tested on Windows 11).
    numpy, pandas.
    Create requirements.txt automatically via:

      ```powershell
         python .\generate_stem_sim.py --save_requirements
      ```
# References

    Brier, G. W. (1950). Verification of forecasts expressed in terms of probability. Monthly Weather Review, 78(1), 1–3. https://doi.org/10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2
    Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. Journal of the American Statistical Association, 102(477), 359–378. https://doi.org/10.1198/016214506000001437
    Good, I. J. (1952). Rational decisions. Journal of the Royal Statistical Society: Series B (Methodological), 14(1), 107–114. https://doi.org/10.1111/j.2517-6161.1952.tb00104.x
    Roulston, M. S., & Smith, L. A. (2002). Evaluating probabilistic forecasts using information theory. Monthly Weather Review, 130(6), 1653–1660. https://doi.org/10.1175/1520-0493(2002)130<1653:EPFUIT>2.0.CO;2

```
python -m pip install --upgrade pip
python -m pip install numpy pandas
