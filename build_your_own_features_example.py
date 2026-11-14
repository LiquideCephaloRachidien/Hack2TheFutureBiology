"""
build_your_features_example.py

TEMPLATE: how to write your own feature extraction for the
anticancer peptide (ACP) challenge.

The only rule: your `extract_features(seq)` function must take a single
amino acid sequence (str) and return a dict of {feature_name: number}.
Every sequence must produce a dict with the SAME keys, in any order,
so they line up into a table when we build the feature matrix.

Copy this file, rename it, and edit the parts marked TODO.
"""

import re
import numpy as np
import pandas as pd
from collections import Counter

AA = list("ACDEFGHIKLMNPQRSTVWY")

# Example lookup table - swap in whatever scale you want
# (hydrophobicity, charge, volume, your own scores, etc.)
HYDROPHOBICITY = {
    'A': 0.62, 'R': -2.53, 'N': -0.78, 'D': -0.90, 'C': 0.29,
    'Q': -0.85, 'E': -0.74, 'G': 0.48, 'H': -0.40, 'I': 1.38,
    'L': 1.06, 'K': -1.50, 'M': 0.64, 'F': 1.19, 'P': 0.12,
    'S': -0.18, 'T': -0.05, 'W': 0.81, 'Y': 0.26, 'V': 1.08
}


def clean_sequence(seq: str) -> str:
    """Keep only standard amino acid letters, uppercase. Always do this first."""
    seq = seq.strip().upper()
    return re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', '', seq)


# ---------------------------------------------------------------
# EXAMPLE 1: a couple of simple, hand-picked features
# ---------------------------------------------------------------
def simple_features(seq: str) -> dict:
    n = len(seq) if len(seq) > 0 else 1
    return {
        "length": len(seq),
        "frac_hydrophobic": sum(seq.count(a) for a in "AILMFVWC") / n,
        "avg_hydrophobicity": sum(HYDROPHOBICITY.get(a, 0.0) for a in seq) / n,
    }


# ---------------------------------------------------------------
# EXAMPLE 2: amino acid composition (one feature per letter)
# This pattern - loop over AA, build a dict - is how you scale up
# to bigger feature sets (dipeptides, k-mers, etc.)
# ---------------------------------------------------------------
def aa_composition(seq: str) -> dict:
    n = len(seq) if len(seq) > 0 else 1
    counts = Counter(seq)
    return {f"aac_{a}": counts.get(a, 0) / n for a in AA}


# ---------------------------------------------------------------
# TODO: add your own feature function(s) here.
# Ideas: charge at physiological pH, isoelectric point, motif counts,
# k-mer frequencies, a score from a custom lookup table, etc.
# ---------------------------------------------------------------
def hydrophobic_moment(seq: str, angle: float = 100.0) -> float:
    """Mean Eisenberg moment. Large when greasy residues cluster on one helical face."""
    n = len(seq)
    if n == 0:
        return 0.0
    theta = np.deg2rad(angle)
    cos_sum = 0.0
    sin_sum = 0.0
    for i, residue in enumerate(seq):
        h = HYDROPHOBICITY.get(residue, 0.0)
        cos_sum += h * np.cos(i * theta)
        sin_sum += h * np.sin(i * theta)
    return float(np.sqrt(cos_sum ** 2 + sin_sum ** 2) / n)


def my_custom_features(seq: str) -> dict:
    n = len(seq) if len(seq) > 0 else 1
    return {
        "sequence_length": len(seq),
        "composition": sum(seq.count(a) for a in AA) / n,
        "hydrophobicity": sum(HYDROPHOBICITY.get(a, 0.0) for a in seq) / n,
        "charge": (sum(seq.count(a) for a in "KR") - sum(seq.count(a) for a in "DE")) / n,
        "hydrophobic_moment": hydrophobic_moment(seq),
    }


def dipeptide_composition(seq: str) -> dict:
    """Share of each adjacent amino-acid pair. One column per pair, 400 total."""
    pairs = [seq[i:i + 2] for i in range(len(seq) - 1)]
    n_pairs = len(pairs) if pairs else 1
    counts = Counter(pairs)
    return {f"dpc_{a}{b}": counts.get(a + b, 0) / n_pairs for a in AA for b in AA}


def extract_features(seq: str) -> dict:
    """
    Combine all feature groups into one flat dict.
    This is the function the training script calls per sequence.
    """
    seq = clean_sequence(seq)
    if len(seq) == 0:
        seq = "A"  # guard against fully-invalid rows

    feats = {}
    feats.update(simple_features(seq))
    feats.update(aa_composition(seq))
    feats.update(my_custom_features(seq))
    feats.update(dipeptide_composition(seq))
    return feats


def build_feature_matrix(sequences) -> pd.DataFrame:
    """Turn a list/Series of sequences into a feature table (one row per sequence)."""
    return pd.DataFrame([extract_features(s) for s in sequences])


if __name__ == "__main__":
    # Quick sanity check on a few example sequences
    example_seqs = ["KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK", "GIGKFLHSAKKFGKAFVGEIMNS", "AAAAA"]
    df = build_feature_matrix(example_seqs)
    print(df)
