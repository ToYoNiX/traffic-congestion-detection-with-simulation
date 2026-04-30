# Traffic Congestion Detection with Sensor-Noise Robustness

A multi-model machine learning system for traffic congestion classification that evaluates model robustness under simulated sensor degradation (occlusion noise + dropout). Five classifiers — Logistic Regression, Random Forest, Extra Trees, Histogram Gradient Boosting, and a Neural Network — are trained on two months of real vehicle-count data and benchmarked on both clean and synthetically noisy test sets. A robustness decay metric (ΔF1) quantifies each model's resilience.

The accompanying paper (`paper.tex` / `paper.pdf`) documents the full methodology, related work, and results in IEEE conference format.

---

## Project Structure

```
.
├── models_train.py        # Full ML pipeline (train, evaluate, save artifacts)
├── TrafficTwoMonth.csv    # Raw dataset (15-min intervals, ~5,952 records)
├── paper.tex              # IEEE-format LaTeX paper
├── paper.pdf              # Compiled paper (pre-built)
├── paper.md               # Markdown draft of the paper
├── pyproject.toml         # Python dependencies (Poetry)
└── outputs/               # Generated after running the pipeline
    └── traffic_final_research/
        ├── full_experiment_results.csv
        ├── traffic_nn_model.keras
        ├── traffic_scaler.pkl
        ├── traffic_encoder.pkl
        ├── cm_*.png               # Confusion matrices (10 total)
        └── robustness_decay.png
```

---

## Setup

Requires Python 3.11+ and [Poetry](https://python-poetry.org/).

```bash
poetry install
poetry add tensorflow   # install separately if needed
```

---

## Running the Experiment

```bash
poetry run python models_train.py
```

Results and plots are saved to `outputs/traffic_final_research/`.

---

## Compiling the Paper

Requires a LaTeX distribution with `IEEEtran` (e.g. TeX Live with `texlive-publishers`).

```bash
pdflatex paper.tex
pdflatex paper.tex   # second pass resolves cross-references
```

The compiled `paper.pdf` is also committed to the repository for convenience.

---

## Authors

**Students** — Faculty of Information Technology, MUST  
Assem Mohamed Saad · Omar Ahmed Asaad · Habiba Hosam Eldin · Seif Adel Elbasha · Owida Elsayed

**Supervisors**  
Dr. Amr Ibrahim · Dr. Walid Hamdy
