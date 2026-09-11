# Exploratory Insulin Variant Classification

This directory contains a prototype for parsing known human insulin variants and exploring classification of submitted clinical-significance labels.

## Data

`P01308.json` is an exported UniProt variant dataset for accession `P01308`, the human insulin protein encoded by `INS`.

The included snapshot contains 138 variant records distributed across pathogenic, likely pathogenic, benign, likely benign, uncertain-significance, and unknown labels. More than half of the records do not have a clinical-significance label, and the benign classes contain only a few observations.

## Workflow

The script:

1. parses variant records from JSON;
2. extracts sequence, position, consequence, prediction, and clinical-significance fields;
3. one-hot encodes categorical variables;
4. fits a random-forest classifier;
5. explores a small hyperparameter grid; and
6. reports exploratory holdout, bootstrap, and cross-validation outputs.

## Run

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 "Predicting the Effects of Variant Mutations/Prediction Model.py"
```

## Limitations

This is a learning and prototyping project, not a validated pathogenicity predictor or clinical tool. The small, imbalanced dataset and the use of existing effect-prediction fields such as PolyPhen and SIFT limit what can be concluded from accuracy values. A stronger follow-up would use a larger multi-gene dataset, a clearly defined target label, leakage-resistant features, stratified nested validation, and class-appropriate metrics.
