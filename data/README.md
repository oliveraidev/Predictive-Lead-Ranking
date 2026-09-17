# Data documentation

## Source and attribution

The unchanged `bank-full.csv` is the older, 45,211-row, 16-input Bank Marketing dataset
from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/222/bank+marketing).
It describes telephone marketing at a Portuguese bank; `y` records term-deposit subscription.
This project does not use `bank-additional-full.csv`.

On 2026-09-16, the existing CSV was verified byte-for-byte against `bank-full.csv` inside
`bank.zip` from the [official download](https://archive.ics.uci.edu/static/public/222/bank+marketing.zip).
SHA-256: `d1513ec63b385506f7cfce9f2c5caa9fe99e7ba4e8c3fa264b3aaf0f849ed32d`.
The original local download date and acquisition process are unknown.
Variable meanings below were checked against `bank-names.txt` in that archive.

Dataset citation: Moro, S., Rita, P., & Cortez, P. (2014). *Bank Marketing* [Dataset].
UCI Machine Learning Repository. [DOI: 10.24432/C5K306](https://doi.org/10.24432/C5K306).
The older archive also requests citation of Moro, S., Laureano, R., & Cortez, P. (2011),
*Using Data Mining for Bank Direct Marketing: An Application of the CRISP-DM Methodology*,
ESM'2011, pp. 117–121. [Paper record](https://hdl.handle.net/1822/14838).

The UCI dataset page identifies the dataset license as
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), checked 2026-09-16.
The raw file is unmodified. Top-lead scores and evaluation files are derived by this project;
they are not original bank predictions. This attribution does not assign a license to the project code.

## Decision point and fields

Scenario: rank records before contact in the current campaign. Use client attributes assumed to
be held before contact and documented prior-campaign history. No timestamped client snapshots
are provided, so that availability assumption cannot be independently verified for each row.

| Fields | Meaning / treatment |
|---|---|
| `age`, `job`, `marital`, `education` | Client demographic/profile fields; included. `divorced` also covers widowed in the source. |
| `default`, `housing`, `loan` | Credit-default and loan indicators; included as recorded categories. |
| `balance` | Average annual account balance in euros; included under the pre-existing-profile assumption. |
| `previous` | Contact count before the current campaign; included as prior history. |
| `poutcome` | Earlier campaign outcome; included. Unknown does not mean failure. |
| `contact`, `day`, `month`, `duration` | Last-contact characteristics of the current campaign; excluded. |
| `campaign` | Current-campaign contact count, including the last contact; excluded. |
| `pdays` | Elapsed days since an earlier campaign contact; `-1` means no previous contact. Excluded because the interval's reference time is not established at our decision point. |
| `y` | Target: `yes` maps to 1, `no` to 0. Never a feature. |

The row ID is never a feature. We do not infer a full date from month/day or use row order as a
predictor. The source describes chronological ordering, but this project uses a random stratified
split and does not claim temporal validation.

## Data quality

- 45,211 rows; 5,289 positives (11.70%).
- No null cells and no exact duplicate full rows in the raw file; the notebook asserts both.
- Explicit `unknown` values: job 288; education 1,857; contact 13,020; poutcome 36,959.
- `pdays = -1`: 36,954 rows. No imputation is needed for this excluded variable.
- Unknown included categories are preserved. One-hot categories and numeric scaling are learned
  inside each training fold; unseen categories are accepted at evaluation time.
- Numeric extremes are retained, not automatically labeled errors. The notebook shows their
  distribution. There is no data-driven outlier filtering before splitting.
- Matching modeled feature profiles may exist across splits. Without a true customer identifier,
  neither customer uniqueness nor independent customer-level partitioning can be established.
  The generated JSON records feature-profile duplication counts.

## Generated files

Run `notebooks/01_eda.ipynb` from top to bottom to regenerate both artifacts together:

- `top_leads.csv`: exactly 100 holdout records, descending score then ascending row ID.
- `model_evaluation.json`: training CV comparison, selected model, final holdout/capacity metrics,
  feature choices, data checks, source/export checksums and package versions.

CSV schema:

| Field | Meaning |
|---|---|
| `source_row_id` | Zero-based position in the unchanged source CSV; not a proven unique customer ID. |
| `model_score` | Uncalibrated ranking score from the selected model; not a validated individual probability. |
| `priority` | `Top 100`: capacity selection, not a probability-based High/Medium/Low category. |
| `actual_outcome` | Historical target, available only afterward; displayed solely for evaluation. |

Lift is precision@100 divided by positive prevalence in the same holdout. It measures concentration,
not causal uplift or extra conversions caused by a contact. Historical sample performance does not
establish current operational usefulness, fairness, future performance or return on investment.
