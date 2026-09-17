# KD-GAMS-Net: Knowledge Distillation and a Leakage-Audit Case Study for Brain Tumor MRI Classification

This repository accompanies the manuscript **"KD-GAMS-Net: Knowledge Distillation Narrows
the Pretraining Gap for a Lightweight Brain Tumor MRI Classifier, and Uncovers Pervasive
Near-Duplication in a Merged Kaggle Corpus"** (Divyakant Meva, Kalpesh Popat — Marwadi
University), and contains the full experimental pipeline, raw result artifacts, and the
manuscript draft, released for reproducibility and editorial/reviewer access.

This is a follow-up to a companion study on **GAMS-Net**, a from-scratch lightweight CNN
that trailed pretrained lightweight backbones on the same task. That study's own ablation
diagnosed the likely cause (no pretraining, not the architecture) and this repository's
experiment tests the most direct fix: knowledge distillation.

## Summary

**KD-GAMS-Net** (3.164M parameters) keeps GAMS-Net's Ghost modules, doubles its Multi-Scale
Dilated Fusion (MSDF) block, replaces channel attention (ECA) with CBAM-style spatial
attention, and is trained with a focal-loss-plus-knowledge-distillation objective against a
frozen MobileNetV2 + EfficientNet-B0 teacher ensemble.

**Two findings, reported with equal weight:**

1. **Knowledge distillation works, modestly.** Across 3 seeds, KD-GAMS-Net (96.70% ± 0.43%
   accuracy) edged out a faithful from-scratch GAMS-Net reproduction (96.21% ± 1.68%) with
   roughly **4× lower run-to-run variance**, narrowing the previously reported ~5–6-point
   gap to pretrained MobileNetV2 (97.96% ± 0.75%) / EfficientNet-B0 (97.56% ± 0.26%) down to
   ~1–1.3 points. A genuine paired McNemar's test confirmed KD's own ablation contribution
   is statistically significant (p = 0.045); the headline comparisons against the baseline
   and both teachers were not significant at a single seed.
2. **A leakage audit found the constructed test split was 59.1% near-duplicate content.**
   1,184 of 2,003 test images were flagged against the training/validation pool by
   perceptual hashing, with the no-tumor class alone losing 94.9% of its test images
   (leaving only 19). This is reported as a central result, not a footnote — see
   `results/leakage_audit.csv` and Section 3.2/4.1 of the manuscript.

A parameter-counting artifact discovered during analysis (both pretrained teachers showed
0 parameters in the raw efficiency output, because freezing them for use as KD sources set
`requires_grad = False` on every parameter before the counting step ran later in the same
session) is disclosed and corrected in the manuscript and in
`results/efficiency_results_corrected.csv`.

## Repository contents

```
.
├── notebooks/
│   └── KD-GAMS-Net_Brain_Tumor_MRI_Experiment.ipynb   # Full experimental pipeline (48 cells)
├── results/
│   ├── ablation_results.csv              # KD / double-MSDF / spatial-attention ablation
│   ├── seed_robustness.csv               # 3-seed (42, 123, 2024) mean ± std
│   ├── mcnemar_results.csv               # Real, paired, per-sample McNemar's test
│   ├── leakage_audit.csv                 # Every flagged near-duplicate pair (3,722 rows)
│   ├── efficiency_results.csv            # Raw output (contains the 0-params artifact)
│   ├── efficiency_results_corrected.csv  # Corrected params for MobileNetV2/EfficientNet-B0
│   ├── run_summary.json                  # Seed-42 headline accuracy + config
│   ├── predictions/                      # Cached y_true/y_pred/y_prob per model (.npz)
│   └── figures/
│       ├── confusion_grid_headline.png       # 4-model confusion matrix grid
│       ├── confusion_<model>-seed42.png      # Individual confusion matrices (7 files)
│       ├── class_distribution.png            # Train/val/test class balance (pre-audit)
│       └── gradcam_kd_gams_net.png           # Grad-CAM, 7 examples incl. 1 misclassification
├── manuscript/
│   └── KD-GAMS-Net_Manuscript.docx
├── requirements.txt
├── LICENSE
├── CITATION.cff
└── README.md
```

## Dataset

Same corpus as the companion GAMS-Net study: **Brain Tumor MRI Dataset (Merged)**, Kaggle:
[`sabersakin/brainmri`](https://www.kaggle.com/datasets/sabersakin/brainmri). Four classes
(glioma, meningioma, no-tumor, pituitary), 13,351 images in the labeled pool. **Not
redistributed here** — download directly from Kaggle.

The archive's shipped "test" folder holds only four demonstration images and is not used;
the notebook auto-detects this and self-splits the full labeled pool (70/15/15,
stratified, seed 42).

**Important caveat carried over from the manuscript:** this experiment's leakage audit
found a far higher near-duplicate rate (59.1%) than the companion GAMS-Net study's audit of
a nominally similar split of the same archive (0 pairs found). We do not have a confirmed
explanation — see `results/leakage_audit.csv` for the full flagged-pair list and Section
3.2/5 of the manuscript for discussion. Anyone reusing this corpus should run their own
audit rather than relying on either study's specific count.

## Reproducing the experiment

1. Open `notebooks/KD-GAMS-Net_Brain_Tumor_MRI_Experiment.ipynb` on
   [Kaggle](https://www.kaggle.com/) (GPU accelerator) or another CUDA-capable environment.
2. Attach the `sabersakin/brainmri` dataset as a notebook input.
3. Run all cells top to bottom. Section 2's diagnostic output should be checked against the
   Kaggle "Data" panel before proceeding.
4. Set `SKIP_MULTISEED = True` in the Section 1 config cell for a faster (~2.5 hour)
   single-seed run; leave it `False` for the full ~7.5–8.5 hour, 3-seed protocol reported in
   the manuscript.
5. Every model's predictions are cached to `predictions/<tag>.npz` immediately after that
   model is evaluated — if you only need to regenerate confusion matrices or the McNemar's
   test from an existing run, you can load these directly without retraining (see the
   `real_mcnemar()` and `evaluate_and_cache()` functions in the notebook).

## Key results

| Model | Params (M) | Accuracy (mean ± SD, 3 seeds) | F1 macro (mean ± SD) |
|---|---|---|---|
| KD-GAMS-Net (proposed) | 3.164 | 0.9670 ± 0.0043 | 0.9579 ± 0.0075 |
| GAMS-Net baseline (from scratch) | 2.956 | 0.9621 ± 0.0168 | 0.9556 ± 0.0152 |
| MobileNetV2 (pretrained, KD teacher) | 2.229 | 0.9796 ± 0.0075 | 0.9807 ± 0.0052 |
| EfficientNet-B0 (pretrained, KD teacher) | 4.013 | 0.9756 ± 0.0026 | 0.9735 ± 0.0063 |

See `results/ablation_results.csv` and `results/mcnemar_results.csv` for the component
ablation and the real, paired significance testing, and the manuscript for full discussion
including the appropriate low-power caveats on the 3-seed comparisons.

## What is *not* included in this repository

- **Trained model checkpoints** (`.pt` weight files) — not included due to size; retrain
  from the notebook to reproduce them exactly (fixed seeds are used throughout).
- **The raw dataset** — redistribute-by-reference only; see "Dataset" above.
- **Seed-123/2024 individual predictions** — only seed-42 predictions are cached to
  `.npz` per the notebook's design (`save=False` for the extra multi-seed runs, to limit
  disk usage); only their aggregated accuracy/F1 contribute to `seed_robustness.csv`.

## Status and how to cite

This manuscript is a draft prepared for journal submission and has **not yet been peer
reviewed**. If this repository is made public before acceptance, check your target
journal's preprint/embargo policy first. See `CITATION.cff` for citation metadata (update
the `doi`/`url`/publication fields once assigned by the journal).

## License

The code in this repository (Jupyter notebook) is released under the MIT License — see
`LICENSE`. The manuscript text and figures in `manuscript/` and `results/figures/` are
**not** covered by that license; copyright in the manuscript typically transfers to the
publisher upon acceptance, and it is included here for editorial/reviewer reference only
unless your target journal's policy states otherwise.

## Related repository

The companion GAMS-Net study (the from-scratch baseline this work builds on and improves
upon) has its own repository: `GAMS-Net-Brain-Tumor-MRI` (link to be added once both
repositories are hosted).

## Contact

Divyakant Meva, Kalpesh Popat — Faculty of Computer Applications (FOCA), Marwadi
University, Rajkot, Gujarat, India.
