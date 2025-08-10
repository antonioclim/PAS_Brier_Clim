#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_stem_sim.py — Synthetic STEM test generator for probabilistic assessment studies
========================================================================================

Purpose
-------
This script programmatically synthesises multiple-choice STEM assessments with
probabilistic ground truth and simulated learner response profiles (archetypes).
It is intended for reproducible experimentation in psychometrics, including:
calibration diagnostics, strictly proper scoring (e.g., log-loss; Brier score),
and penalty strategies for maladaptive behaviours (e.g., overconfident errors).

Design summary
--------------
• Items present 4–7 options (slots A–G).                                          
• Each item has 1–3 correct options; truth mass is conserved (sums to 1.0) to
  four decimal places (k=3 encoded as 0.3333, 0.3333, 0.3334).
• Six archetypes produce probability vectors over the presented options; vectors
  are mass-conserving to exactly 1.0000 at 4 d.p. via a largest‑remainder scheme.
• Behaviour is deterministic given (test_idx, question_idx) and the global RNG seed.
• Output schema (CSV): one row per (Test, Question, Archetype) with:
    Test, Question, Domain,
    TruthA..TruthG,
    Archetype,
    ProbA..ProbG

Archetypes (behavioural profiles)
---------------------------------
Zota       — highly knowledgeable and well‑calibrated (≈0.90 mass on correct set).
Clim       — moderately strong, cautious (≈0.60 on correct, 0.40 on incorrect).
Tolomacea  — overconfident on one distractor (0.80 on a single incorrect option).
Cristina   — good but erratic (base 0.70 on correct; on some items shifts 0.20 to a distractor).
Gabriela   — underconfident (0.55 on correct; 0.45 hedged across incorrect).
Dan        — mixed calibration: usually 0.70 on one distractor; sometimes flips to 0.60/0.40 favouring correct.

Reproducibility affordances
---------------------------
• Global RNG seed controls the entire generation.
• Mass conservation is enforced after rounding (4 d.p.) using Hamilton’s largest‑remainder method.
• Optional writing of a `requirements.txt` capturing pinned numpy/pandas versions.

Command-line (synonyms supported)
---------------------------------
--tests / --num_tests           number of tests (quiz forms), default 20
--questions / --num_questions   number of questions per test, default 5
--seed                          RNG seed (int), default 42
--out / --output                output CSV filename, default "stem_sim_20tests_5q.csv"
--zip                           if set, also write ZIP archive containing the CSV
--save_requirements             if set, write requirements.txt (numpy & pandas, pinned)
--validate                      if set, run internal mass‑conservation checks and report

Windows/PowerShell note
-----------------------
If your path contains spaces or '#', quote the script path and output path.
Example:
    python "D:\\#___MY_SPACE\\Downloads\\generate_stem_sim.py" --tests 20 --questions 25 --out "stem_20x25.csv"
"""

from __future__ import annotations

import argparse
import os
import zipfile
from typing import Dict, List

import numpy as np
import pandas as pd


# ----------------------------- Configuration constants --------------------------------

DECIMALS: int = 4  # fixed reporting precision for probabilities (and truth) in CSV
OPTION_SLOTS: int = 7  # A..G
DOMAINS: List[str] = [
    "calculus", "algebra", "probability", "logic", "bash", "python", "c", "physics",
    "chemistry", "statistics", "operating_systems", "algorithms", "data_structures"
]
ARCHETYPES: List[str] = ["Zota", "Clim", "Tolomacea", "Cristina", "Gabriela", "Dan"]


# ----------------------------- Combinatorial structure --------------------------------

def choose_k(test_idx: int, q_idx: int) -> int:
    """
    Number of correct options k ∈ {1,2,3} for (test_idx, q_idx).
    Deterministic hash on indices yields approximate frequencies:
        P(k=1)=0.60, P(k=2)=0.30, P(k=3)=0.10.
    """
    r = (test_idx * 31 + q_idx * 17) % 100
    return 1 if r < 60 else (2 if r < 90 else 3)


def choose_options_count(k: int, test_idx: int, q_idx: int) -> int:
    """
    Total presented options n_opts ∈ {4,…,7} with constraint:
        n_opts ≥ max(4, 2*k)  (ensures at least as many incorrect as correct).
    The base count cycles deterministically with indices.
    """
    min_opts = max(4, 2 * k)                          # ← explicit multiplication (Windows ‘debug.py’ pitfall fixed)
    base = 4 + ((test_idx + q_idx) % 4)               # cycles through 4,5,6,7
    return max(min(base, 7), min_opts)


def choose_correct_indices(n_opts: int, k: int, test_idx: int, q_idx: int) -> List[int]:
    """
    Choose k distinct indices in [0, n_opts-1] via a rotation of 0..n_opts-1
    by a deterministic shift. Sorting is applied for stable ordering.
    """
    idxs = list(range(n_opts))
    shift = (test_idx * 7 + q_idx * 13) % n_opts
    rotated = idxs[shift:] + idxs[:shift]
    return sorted(rotated[:k])


def make_truth_vector(n_opts: int, k: int, correct_idxs: List[int]) -> np.ndarray:
    """
    Construct the ground‑truth vector over A..G (length=7), with mass conserved to 4 d.p.
    k=1 → [1.0];  k=2 → [0.5, 0.5];  k=3 → [0.3333, 0.3333, 0.3334].
    Non‑present slots (indices ≥ n_opts) remain 0.0.
    """
    t = np.zeros(OPTION_SLOTS, dtype=float)
    if k == 1:
        t[correct_idxs[0]] = 1.0
    elif k == 2:
        t[correct_idxs[0]] = 0.5
        t[correct_idxs[1]] = 0.5
    else:  # k == 3
        t[correct_idxs[0]] = 0.3333
        t[correct_idxs[1]] = 0.3333
        t[correct_idxs[2]] = 0.3334
    return t


# ----------------------------- Simplex utilities --------------------------------------

def normalise(vec: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Normalise `vec` so entries where `mask` is True sum to 1.0; others set to 0.0.
    If the masked total is zero (degenerate), distribute uniformly over mask.
    """
    out = np.zeros_like(vec, dtype=float)
    total = float(vec[mask].sum())
    out[mask] = (vec[mask] / total) if total > 0 else (1.0 / int(mask.sum()))
    return out


def mass_conserve_round(vec: np.ndarray, present_mask: np.ndarray, decimals: int = DECIMALS) -> np.ndarray:
    """
    Round to `decimals` while preserving total mass (sum=1.0) over present options.
    Procedure (Hamilton / largest‑remainder):
        1) normalise on present_mask
        2) scale by 10^decimals
        3) floor each component
        4) distribute the remaining units to the largest fractional parts
        5) rescale back
    Returns a vector with exact mass conservation at the specified precision.
    """
    p = normalise(vec, present_mask)
    idxs = np.where(present_mask)[0]
    if idxs.size == 0:
        return np.zeros_like(vec, dtype=float)
    scale = 10 ** decimals
    scaled = p[idxs] * scale
    floors = np.floor(scaled)
    remainder = int(round(scale - floors.sum()))
    # Assign +1 to largest fractional parts
    frac = scaled - floors
    order = np.argsort(-frac)
    increments = np.zeros_like(floors)
    if remainder > 0:
        increments[order[:remainder]] = 1
    rounded = np.zeros_like(vec, dtype=float)
    rounded[idxs] = (floors + increments) / scale
    rounded[~present_mask] = 0.0
    return rounded


# ----------------------------- Archetype probability models ---------------------------

def probs_zota(truth: np.ndarray, present_mask: np.ndarray) -> np.ndarray:
    """
    Zota — well‑calibrated strong performer.
    Single‑correct: 0.92 on the correct option; remainder uniform on incorrect.
    Multi‑correct:  0.90 distributed ∝ truth over correct; 0.10 uniform over incorrect.
    """
    p = np.zeros(OPTION_SLOTS, dtype=float)
    cm = (truth > 0) & present_mask
    im = present_mask & (~cm)
    if cm.sum() == 1:
        p[cm] = 0.92
        if im.sum():
            p[im] = 0.08 / im.sum()
    else:
        if cm.sum():
            p[cm] = 0.90 * truth[cm] / truth[cm].sum()
        if im.sum():
            p[im] = 0.10 / im.sum()
    return mass_conserve_round(p, present_mask)


def probs_clim(truth: np.ndarray, present_mask: np.ndarray) -> np.ndarray:
    """
    Clim — moderate & cautious.
    0.60 over correct (∝ truth), 0.40 uniform over incorrect.
    """
    p = np.zeros(OPTION_SLOTS, dtype=float)
    cm = (truth > 0) & present_mask
    im = present_mask & (~cm)
    if cm.sum():
        p[cm] = 0.60 * truth[cm] / truth[cm].sum()
    if im.sum():
        p[im] = 0.40 / im.sum()
    return mass_conserve_round(p, present_mask)


def probs_tolomacea(truth: np.ndarray, present_mask: np.ndarray, attract_idx: int | None) -> np.ndarray:
    """
    Tolomacea — overconfident miscalibration.
    0.80 on one selected incorrect option; 0.20 spread uniformly across the remainder.
    If no incorrect options exist (edge), distribute uniformly.
    """
    p = np.zeros(OPTION_SLOTS, dtype=float)
    cm = (truth > 0) & present_mask
    im = present_mask & (~cm)
    if im.sum():
        if (attract_idx is None) or (truth[attract_idx] > 0) or (not present_mask[attract_idx]):
            attract_idx = int(np.where(im)[0][0])
        p[attract_idx] = 0.80
        other = present_mask.copy()
        other[attract_idx] = False
        if other.sum():
            p[other] = 0.20 / other.sum()
    else:
        p[present_mask] = 1.0 / present_mask.sum()
    return mass_conserve_round(p, present_mask)


def probs_cristina(truth: np.ndarray, present_mask: np.ndarray, erratic: bool) -> np.ndarray:
    """
    Cristina — generally good; sporadically erratic.
    Base: 0.70 over correct (∝ truth), 0.30 uniform over incorrect.
    Erratic items: shift up to 0.20 from correct to a single distractor.
    """
    p = np.zeros(OPTION_SLOTS, dtype=float)
    cm = (truth > 0) & present_mask
    im = present_mask & (~cm)
    if cm.sum():
        p[cm] = 0.70 * truth[cm] / truth[cm].sum()
    if im.sum():
        p[im] = 0.30 / im.sum()
    if erratic and im.sum():
        d = int(np.where(im)[0][0])
        shift = min(0.20, float(p[cm].sum()))
        if shift > 0:
            scale = float(p[cm].sum())
            if scale > 0:
                p[cm] *= (1 - shift / scale)
            p[d] += shift
    return mass_conserve_round(p, present_mask)


def probs_gabriela(truth: np.ndarray, present_mask: np.ndarray) -> np.ndarray:
    """
    Gabriela — underconfident hedging.
    0.55 over correct (∝ truth), 0.45 uniform over incorrect.
    """
    p = np.zeros(OPTION_SLOTS, dtype=float)
    cm = (truth > 0) & present_mask
    im = present_mask & (~cm)
    if cm.sum():
        p[cm] = 0.55 * truth[cm] / truth[cm].sum()
    if im.sum():
        p[im] = 0.45 / im.sum()
    return mass_conserve_round(p, present_mask)


def probs_dan(truth: np.ndarray, present_mask: np.ndarray, biased_wrong_idx: int | None, flip_to_correct: bool) -> np.ndarray:
    """
    Dan — mixed strategy.
    Normal mode (flip_to_correct=False): 0.70 on one incorrect option; remainder uniform.
    Flipped mode: 0.60 on correct (∝ truth), 0.40 uniform on incorrect.
    """
    p = np.zeros(OPTION_SLOTS, dtype=float)
    cm = (truth > 0) & present_mask
    im = present_mask & (~cm)
    if (not flip_to_correct) and im.sum():
        if (biased_wrong_idx is None) or (truth[biased_wrong_idx] > 0) or (not present_mask[biased_wrong_idx]):
            biased_wrong_idx = int(np.where(im)[0][0])
        p[biased_wrong_idx] = 0.70
        other = present_mask.copy()
        other[biased_wrong_idx] = False
        if other.sum():
            p[other] = 0.30 / other.sum()
    else:
        if cm.sum():
            p[cm] = 0.60 * truth[cm] / truth[cm].sum()
        if im.sum():
            p[im] = 0.40 / im.sum()
    return mass_conserve_round(p, present_mask)


# ----------------------------- Data set generation ------------------------------------

def generate(n_tests: int = 20,
             n_questions: int = 5,
             seed: int = 42,
             out: str = "stem_sim.csv") -> pd.DataFrame:
    """
    Synthesis of a full panel: tests × questions × archetypes.

    For each (test, question):
      1) Assign a STEM domain (cycled).
      2) Choose k and n_opts subject to constraints; pick correct indices.
      3) Construct truth vector (A..G) with mass conservation at 4 d.p.
      4) For each archetype, compute probability vector; round mass‑conservatively.
      5) Emit one row per archetype.

    Returns the DataFrame and writes it to `out` (CSV) with float_format='%.4f'.
    """
    np.random.seed(seed)

    tests = [f"Test{i}" for i in range(1, n_tests + 1)]
    questions = [f"Q{i}" for i in range(1, n_questions + 1)]

    rows: List[Dict[str, float | str]] = []

    for t_idx, t in enumerate(tests, start=1):
        for q_idx, q in enumerate(questions, start=1):
            domain = DOMAINS[(q_idx - 1) % len(DOMAINS)]

            k = choose_k(t_idx, q_idx)
            n_opts = choose_options_count(k, t_idx, q_idx)
            correct_idxs = choose_correct_indices(n_opts, k, t_idx, q_idx)
            truth = make_truth_vector(n_opts, k, correct_idxs)

            present_mask = np.zeros(OPTION_SLOTS, dtype=bool)
            present_mask[:n_opts] = True

            # First incorrect index to serve as an “attractor” for some archetypes
            inc_mask = present_mask & (truth == 0.0)
            inc_idxs = np.where(inc_mask)[0]
            attract_idx = int(inc_idxs[0]) if inc_idxs.size > 0 else None

            # Structured variability toggles (deterministic in indices)
            erratic = ((t_idx + q_idx) % 5 == 0)           # Cristina’s occasional lapse
            flip_to_correct = ((t_idx + 2 * q_idx) % 3 == 0)  # Dan’s mode switch

            arche_probs = {
                "Zota": probs_zota(truth, present_mask),
                "Clim": probs_clim(truth, present_mask),
                "Tolomacea": probs_tolomacea(truth, present_mask, attract_idx),
                "Cristina": probs_cristina(truth, present_mask, erratic),
                "Gabriela": probs_gabriela(truth, present_mask),
                "Dan": probs_dan(truth, present_mask, attract_idx, flip_to_correct),
            }

            for a in ARCHETYPES:
                p = arche_probs[a]
                rows.append({
                    "Test": t,
                    "Question": q,
                    "Domain": domain,
                    "TruthA": float(truth[0]),
                    "TruthB": float(truth[1]),
                    "TruthC": float(truth[2]),
                    "TruthD": float(truth[3]),
                    "TruthE": float(truth[4]),
                    "TruthF": float(truth[5]),
                    "TruthG": float(truth[6]),
                    "Archetype": a,
                    "ProbA": float(p[0]),
                    "ProbB": float(p[1]),
                    "ProbC": float(p[2]),
                    "ProbD": float(p[3]),
                    "ProbE": float(p[4]),
                    "ProbF": float(p[5]),
                    "ProbG": float(p[6]),
                })

    cols = [
        "Test", "Question", "Domain",
        "TruthA", "TruthB", "TruthC", "TruthD", "TruthE", "TruthF", "TruthG",
        "Archetype", "ProbA", "ProbB", "ProbC", "ProbD", "ProbE", "ProbF", "ProbG"
    ]
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(out, index=False, float_format=f"%.{DECIMALS}f")
    return df


# ----------------------------- Validation (mass checks) -------------------------------

def validate_mass(df: pd.DataFrame) -> dict:
    """
    Validate mass conservation:
      • For each row, probabilities over PRESENT options sum to 1.0000 exactly.
      • For each row, positive truth entries sum to 1.0000 exactly.

    Returns a dict with counts of violations (ideally zeros).
    """
    p_errors = 0
    t_errors = 0

    for _, r in df.iterrows():
        tnum = int(str(r["Test"]).replace("Test", ""))
        qnum = int(str(r["Question"]).replace("Q", ""))

        k = choose_k(tnum, qnum)
        n_opts = choose_options_count(k, tnum, qnum)

        mask = np.zeros(OPTION_SLOTS, dtype=bool)
        mask[:n_opts] = True

        probs = np.array([r["ProbA"], r["ProbB"], r["ProbC"], r["ProbD"], r["ProbE"], r["ProbF"], r["ProbG"]], dtype=float)
        if round(float(probs[mask].sum()), DECIMALS) != 1.0:
            p_errors += 1

        truth = np.array([r["TruthA"], r["TruthB"], r["TruthC"], r["TruthD"], r["TruthE"], r["TruthF"], r["TruthG"]], dtype=float)
        if round(float(truth[truth > 0].sum()), DECIMALS) != 1.0:
            t_errors += 1

    return {"prob_mass_errors": p_errors, "truth_mass_errors": t_errors}


# ----------------------------- Requirements writer -----------------------------------

def write_requirements(path: str = "requirements.txt") -> None:
    """
    Write a minimal requirements file pinning numpy and pandas versions present at runtime.
    This supports precise environment reproduction for reviewers.
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"numpy=={np.__version__}\n")
        f.write(f"pandas=={pd.__version__}\n")


# ----------------------------- CLI ----------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate probabilistic STEM simulations (truth + archetype probabilities)."
    )
    # Accept synonyms to avoid friction with documentation variants.
    parser.add_argument("--tests", "--num_tests", dest="tests", type=int, default=20,
                        help="Number of tests (quiz forms).")
    parser.add_argument("--questions", "--num_questions", dest="questions", type=int, default=5,
                        help="Number of questions per test (e.g., 5,10,15,20,25,35,50,75,100).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument("--out", "--output", dest="out", type=str, default="stem_sim_20tests_5q.csv",
                        help="Output CSV filename.")
    parser.add_argument("--zip", action="store_true",
                        help="Also write a ZIP archive containing the CSV.")
    parser.add_argument("--save_requirements", action="store_true",
                        help="Write requirements.txt with pinned numpy/pandas versions.")
    parser.add_argument("--validate", action="store_true",
                        help="Run mass‑conservation checks and report counts.")
    args = parser.parse_args()

    if args.save_requirements:
        write_requirements()
        print("Wrote requirements.txt (pinned numpy/pandas).")

    df = generate(n_tests=args.tests,
                  n_questions=args.questions,
                  seed=args.seed,
                  out=args.out)

    if args.validate:
        report = validate_mass(df)
        print("Validation report:", report)

    if args.zip:
        zip_path = args.out.replace(".csv", ".zip")
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(args.out, arcname=os.path.basename(args.out))
        print("Wrote:", args.out, "and ZIP:", zip_path)
    else:
        print("Wrote:", args.out)


if __name__ == "__main__":
    main()
