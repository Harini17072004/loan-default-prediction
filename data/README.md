# Dataset

## Credit Risk Dataset

A public dataset of loan applicants with financial and credit-bureau information,
used to predict whether a borrower will default on a loan.

- **Source:** Kaggle — [Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)
- **Uploader:** `laotse`
- **License:** **CC0: Public Domain** (redistribution permitted)
- **File in this repo:** `data/credit_risk_dataset.csv`
- **SHA-256:** `ce3c6d2167717bf1627d1c0c81cbccd28323cd4aa7b96d542599366d5ff6aac8`
- **Rows / columns:** 32,581 / 12
- **Target:** `loan_status` — `1` = defaulted, `0` = did not default

> This is a synthetic/simulated credit-bureau dataset published for learning and
> portfolio work. It is **not** real customer data and must not be used for real
> lending decisions.

## Columns

| Column | Type | Description |
|---|---|---|
| `person_age` | numeric | Applicant age (years) |
| `person_income` | numeric | Annual income |
| `person_home_ownership` | categorical | `RENT`, `MORTGAGE`, `OWN`, `OTHER` |
| `person_emp_length` | numeric | Years in current employment |
| `loan_intent` | categorical | Purpose of the loan |
| `loan_grade` | categorical (ordinal) | Lender risk grade, `A` (best) to `G` (worst) |
| `loan_amnt` | numeric | Requested loan amount |
| `loan_int_rate` | numeric | Interest rate (%) |
| `loan_percent_income` | numeric | Loan amount as a fraction of income |
| `cb_person_default_on_file` | categorical | Prior default on credit file (`Y`/`N`) |
| `cb_person_cred_hist_length` | numeric | Credit history length (years) |
| `loan_status` | **target** | `1` = default, `0` = no default |

## How to download it yourself

The CSV is already committed, so you do **not** need to download anything to run
the project. To refresh it from the original source:

```bash
# option 1: project helper (no Kaggle account required for this public dataset)
python data/download_data.py

# option 2: Kaggle CLI (requires `pip install kaggle` and ~/.kaggle/kaggle.json)
kaggle datasets download -d laotse/credit-risk-dataset -p data --unzip
```

If the download fails, download the dataset manually from the Kaggle page linked
above, unzip it, and place `credit_risk_dataset.csv` in this `data/` directory.
