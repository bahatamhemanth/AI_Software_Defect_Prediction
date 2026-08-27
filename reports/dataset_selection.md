# Dataset Selection

## Selected Dataset

The primary dataset is **PC1**, one of the NASA Metrics Data Program (MDP)
software defect datasets distributed through the PROMISE repository and
catalogued by OpenML.

- Source: [OpenML PC1](https://www.openml.org/d/1068)
- Direct source file: [pc1.arff](https://openml.org/data/v1/download/53951/pc1.arff)
- Original-data reference: [PROMISE PC1 record](http://openscience.us/repo/defect/mccabehalsted/pc1.html)
- OpenML dataset ID: `1068`
- License/visibility: Public, according to the OpenML metadata
- Original data author: Mike Chapman, NASA
- Collection year: 2004
- Target variable: `defects` (`true` or `false`)
- Raw project copy: `data/raw/pc1.arff`

The OpenML metadata cites Sayyad Shirabad and Menzies (2005), *The PROMISE
Repository of Software Engineering Databases*, as the dataset citation.

## Measured Statistics

The statistics below were obtained from the downloaded PC1 ARFF file through
the OpenML catalog loader. No EDA or preprocessing has been performed yet.

| Property | Value |
| --- | ---: |
| Rows | 1,109 |
| Predictor features | 21 numeric |
| Target columns | 1 (`defects`) |
| Total columns | 22 |
| Defective modules | 77 (`true`) |
| Non-defective modules | 1,032 (`false`) |
| Defective percentage | 6.94% |
| Non-defective percentage | 93.06% |
| Missing values | 0 |

The target is substantially imbalanced. This makes recall, F1-score, ROC-AUC,
and PR-AUC more informative than accuracy alone for later experiments.

## Feature Names

The 21 predictors are:

`loc`, `v(g)`, `ev(g)`, `iv(g)`, `n`, `v`, `l`, `d`, `i`, `e`, `b`, `t`,
`lOCode`, `lOComment`, `lOBlank`, `lOCodeAndComment`, `uniq_Op`, `uniq_Opnd`,
`total_Op`, `total_Opnd`, and `branchCount`.

They represent McCabe complexity, Halstead measures, lines of code/comments/
blank lines, operator and operand counts, and control-flow branching.

## Candidate Comparison

NASA MDP and PROMISE datasets are closely related: PROMISE distributes
software-engineering datasets, including NASA projects, while OpenML provides
a versioned machine-readable catalog entry. Candidate families were assessed
for public availability, software-metric relevance, binary defect labels,
reproducibility, and suitability for risk ranking.

PC1 was selected because it has a public, versioned OpenML record; a stable
direct ARFF download; a clearly documented binary target; 21 interpretable
software metrics; and enough modules for stratified evaluation while remaining
manageable for a reproducible implementation. The available metric and defect label
also support a defensible risk-based module ranking. No unavailable business
priority or historical test-cost field will be invented.

## Limitations

- PC1 represents one NASA software context, so results may not generalize to
  other languages, teams, or development processes.
- The target records reported defects, not every defect that existed.
- The strong class imbalance can make naive accuracy misleading.
- The dataset does not provide business criticality, test execution cost, or
  complete historical test-suite information; test prioritization will
  therefore use only measurable model risk and documented software metrics.
- Dataset quality and possible project-level dependencies will be examined
  during EDA and preprocessing rather than assumed away.

## Reproducibility Rule

The original downloaded ARFF file remains unchanged under `data/raw/`.
Processed data, derived features, figures, and measured results will be saved
under their respective `data/processed/`, `reports/`, and `models/` locations.