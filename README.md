# Bayesian Insurance Decision Network

A Streamlit-based interactive tool for predicting whether a customer should receive a vehicle insurance promotion, using a Bayesian Belief Network built from a real-world dataset of 381,109 health insurance customers.

---

## What the Program Does

The program models the **cross-sell prediction problem**: given an existing health insurance customer's profile, should the company send them a vehicle insurance promotion?

It uses **Bayesian probability weights** derived directly from the dataset to estimate the probability that a customer will respond positively to an offer. That probability is then fed into a **Decision Network** that computes the **Expected Utility** of sending or not sending the promotion.

### Core features

- **Probability estimation** — estimates P(Response = 1) for a customer profile using conditional probabilities from the dataset across four key variables
- **Expected Utility calculator** — computes EU(Offer) and EU(No Offer) given configurable profit and marketing cost assumptions
- **Interactive Bayesian Decision Network graph** — visualised using Graphviz, updates live as you change customer inputs
- **Evidence probability table** — shows which conditional probability is firing for each selected variable
- **Decision recommendation** — recommends Send Promotion or Do Not Send based on EU comparison
- **Value of Perfect Information (VPI)** — computes the population-level VPI for the `Previously_Insured` variable, showing how much the decision improves when that information is known

---

## Project Structure

```
bayesian_insurance_ui.py    # Main Streamlit application
```

The entire program is a single self-contained Python file. All probability weights are hardcoded from the dataset — no external data file is needed at runtime.

---

## Requirements

- Python 3.8 or higher
- pip

### Install dependencies

```bash
pip install streamlit pandas
pip install graphviz
```

> **Note:** The `graphviz` Python package also requires the Graphviz system binaries.
> Install them depending on your OS:
>
> - **Windows:** Download and install from https://graphviz.org/download/, then add `bin/` to your PATH
> - **macOS:** `brew install graphviz`
> - **Ubuntu/Debian:** `sudo apt-get install graphviz`

---

## How to Run

```bash
streamlit run bayesian_insurance_ui.py
```

Streamlit will open a browser tab automatically at `http://localhost:8501`.

---

## How to Use the App

The app is divided into four sections:

### 1 — Customer Information
Select the customer profile using the four dropdowns:

| Input | Options |
|---|---|
| Vehicle Damage | No / Yes |
| Previously Insured | No / Yes |
| Vehicle Age | < 1 Year / 1–2 Year / > 2 Years |
| Age Group | 18–24 / 25–34 / 35–44 / 45–54 / 55–64 / 65+ |

### 2 — Business Utility Assumptions
Set the financial parameters:

| Parameter | Default | Description |
|---|---|---|
| Profit if customer buys | 500 | Revenue gained from a successful sale |
| Marketing cost of sending offer | 50 | Cost of sending the promotion |

### 3 — Outputs
The app displays:

- **P(Response = 1)** — the estimated probability this customer responds positively
- **EU(Offer Promotion)** — expected financial value of sending the offer
- **EU(No Promotion)** — always 0 (no action = no cost, no gain)
- **Bayesian Decision Network graph** — live DAG showing the network with current values
- **Evidence probability table** — the conditional probability weight used for each variable
- **Decision recommendation** — green (send) or red (do not send)
- **VPI for Previously_Insured** — computed at population level, shown at the bottom

---

## How the Probability Is Calculated

The model uses conditional probabilities extracted from cross-tabulations of the dataset. For a given customer profile, the posterior probability of response is estimated as:

```
P(R=1 | evidence) =
    prior × [P(R=1|VehicleDamage) / prior]
           × [P(R=1|PrevInsured)  / prior]
           × [P(R=1|AgeGroup)     / prior]
           × P(VehicleDamage | VehicleAge)
```

The last term accounts for the structural dependency **Vehicle_Age → Vehicle_Damage** captured in the DAG.

The **Expected Utility** formula is:

```
EU(Offer) = P(Response=1) × Profit − Marketing Cost
EU(No Offer) = 0
```

The app recommends sending the promotion whenever `EU(Offer) > EU(No Offer)`.

---

## Key Probability Weights (from dataset)

### P(Response = 1 | Vehicle_Damage)
| Vehicle_Damage | P(Response = 1) |
|---|---|
| No | 0.0052 |
| Yes | 0.2377 |

### P(Response = 1 | Previously_Insured)
| Previously_Insured | P(Response = 1) |
|---|---|
| No | 0.2255 |
| Yes | 0.0009 |

### P(Vehicle_Damage | Vehicle_Age)
| Vehicle_Age | P(VD = No) | P(VD = Yes) |
|---|---|---|
| < 1 Year | 0.7075 | 0.2925 |
| 1–2 Year | 0.3599 | 0.6401 |
| > 2 Years | 0.0009 | 0.9991 |

### P(Response = 1 | Age_Group)
| Age_Group | P(Response = 1) |
|---|---|
| 18–24 | 0.0353 |
| 25–34 | 0.0889 |
| 35–44 | 0.2159 |
| 45–54 | 0.2018 |
| 55–64 | 0.1458 |
| 65+ | 0.0862 |

Prior P(Response = 1) from dataset: **0.1226**

---

## Dataset

**Health Insurance Cross-Sell Prediction**
https://www.kaggle.com/datasets/anmolkumar/health-insurance-cross-sell-prediction

- 381,109 customer records
- Target variable: `Response` (1 = interested, 0 = not interested)

The probability weights in the code were derived from this dataset using `pd.crosstab()` with `normalize="index"`.

---

## Notes

- This is a simplified Bayesian model. It does not use a full joint probability table — it approximates posterior inference using a naive Bayes-style product of likelihood ratios combined with one structural conditional dependency (`Vehicle_Age → Vehicle_Damage`).
- The VPI calculation at the bottom of the app is computed at the **population level** (using population-wide priors), not for the specific customer profile selected in the UI.
- Changing the Profit or Marketing Cost sliders will dynamically update the EU values and VPI.
