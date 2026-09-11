# Genetic Variant Analysis Experiments

Two small Python projects exploring genetic classification: a mitochondrial haplogroup U lookup tool and an exploratory human insulin variant analysis.

## Scientific scope

This repository contains two distinct analyses and should not be interpreted as one mitochondrial pathogenicity model.

| Project | Biological system | Purpose |
|---|---|---|
| Haplogroup Analysis Tool | Human mitochondrial DNA, haplogroup U | Match an entered set of defining mutations to a haplogroup U subclade |
| Predicting the Effects of Variant Mutations | Human nuclear `INS` gene and insulin protein | Explore whether features recorded for known insulin variants can classify their submitted clinical-significance labels |

The second project uses UniProt accession `P01308` (`INS_HUMAN`). It is not a mitochondrial dataset and is not a clinical diagnostic model.

## Project 1: mitochondrial haplogroup U lookup

The command-line tool compares user-supplied mutations with defining mutation sets compiled from the PhyloTree haplogroup U tree.

### Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 "Haplogroup Analysis Tool/Haplogroup Analysis Tool.py"
```

Enter mutations separated by spaces when prompted.

### Boundary

The lookup returns the first subclade whose defining mutations are all present. It does not perform sequence alignment, probabilistic classification, quality control, or clinical ancestry interpretation.

## Project 2: exploratory insulin variant classification

This experiment parses variant records for human insulin from a UniProt export and constructs an exploratory random-forest classification workflow.

The source file contains 138 variant records. In the included snapshot, 73 records have no submitted clinical-significance label, and benign categories are represented by only a few observations. PolyPhen and SIFT values are also recorded alongside clinical significance.

### Run

```bash
python3 "Predicting the Effects of Variant Mutations/Prediction Model.py"
```

### Interpretation boundary

The current experiment is useful as a record-parsing and model-prototyping exercise, but its accuracy values should not be treated as evidence of clinical predictive validity because:

- the dataset is small and highly imbalanced;
- many records have unknown or uncertain labels;
- PolyPhen and SIFT are themselves variant-effect predictions and may introduce circularity when used to predict clinical significance;
- the bootstrap loop and fixed test set are exploratory rather than a nested validation design; and
- the data represents one protein rather than a broad population of independent genes or patients.

For those reasons, this repository presents the workflow as exploratory analysis rather than a deployable pathogenicity classifier.

## Data sources

- Haplogroup definitions: [PhyloTree mtDNA tree, haplogroup U](https://www.phylotree.org/tree/U.htm)
- Insulin variants: [UniProt P01308 variant viewer](https://www.uniprot.org/uniprotkb/P01308/variant-viewer)
- Clinical-significance fields embedded in the UniProt export reference sources including ClinVar.

## Repository structure

```text
Mitochondrial-Haplogroup-Mutations/
├── Haplogroup Analysis Tool/
│   ├── Haplogroup Analysis Tool.py
│   ├── Haplogroups.csv
│   └── ReadMe.md
├── Predicting the Effects of Variant Mutations/
│   ├── P01308.json
│   ├── Prediction Model.py
│   ├── analysis.py
│   └── ReadMe.md
├── requirements.txt
└── README.md
```

## Tools

Python, pandas, scikit-learn, JSON, and CSV.
