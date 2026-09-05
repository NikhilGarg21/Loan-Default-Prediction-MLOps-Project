<div align="center">

# 🏦 Loan Default Prediction — MLOps Project

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)
![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-47A248?style=for-the-badge&logo=mongodb&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data-150458?style=for-the-badge&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![CatBoost](https://img.shields.io/badge/CatBoost-Model-FFCC00?style=for-the-badge)
![imbalanced--learn](https://img.shields.io/badge/imbalanced--learn-SMOTEENN-9146FF?style=for-the-badge)
![Status](https://img.shields.io/badge/status-in%20progress-yellow?style=for-the-badge)

A project I built while learning MLOps — going beyond just training a model in a notebook and actually structuring it as a proper pipeline: data ingestion, validation, transformation, and training, each as its own component with its own config and artifacts, the way a real production ML system would be organized.

</div>

---

## 📖 Table of Contents

- [Overview](#-overview)
- [What I Learned](#-what-i-learned-building-this)
- [Project Structure](#-project-structure)
- [The Pipeline](#-the-pipeline)
- [Results](#-results)
- [Setup](#-setup)
- [Running It](#-running-it)
- [What's Next](#-whats-next)
- [Tech Stack](#-tech-stack)

---

## 📌 Overview

Predicting whether a borrower will default on a loan, using a dataset of ~255,000 loan records with a fairly realistic challenge baked in — only about **11.6%** of loans actually default, so a big part of this project was learning how to handle that imbalance properly instead of just chasing accuracy.

## 🎓 What I Learned Building This

<details>
<summary><b>Click to expand</b></summary>

- How to structure an ML pipeline into separate stages (ingestion → validation → transformation → training) that pass artifacts to each other instead of one big script
- Why accuracy is a misleading metric on imbalanced data, and how to actually think about precision/recall/PR-AUC instead
- The difference between fixing class imbalance at the data level (SMOTE/SMOTEENN), the algorithm level (class weighting), and the decision level (threshold tuning) — and that threshold tuning + class weighting usually beats resampling on real-world data
- Why you should never let any resampling technique touch your test set — it has to stay untouched to mean anything
- How to pull data from MongoDB into a pipeline instead of just reading a static CSV
- Why `.gitignore` won't un-track files that were already committed before you added the rule

</details>

## 🗂️ Project Structure

```
├── src/
│   ├── components/
│   │   ├── __init__.py
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   ├── model_evaluation.py       # coming soon
│   │   └── model_pusher.py           # coming soon
│   ├── configuration/
│   │   ├── __init__.py
│   │   ├── mongo_db_connection.py
│   │   └── aws_connection.py         # coming soon
│   ├── cloud_storage/
│   │   ├── __init__.py
│   │   └── aws_storage.py            # coming soon
│   ├── data_access/
│   │   ├── __init__.py
│   │   └── load_default_data.py      # LoanDefaultData — pulls from MongoDB
│   ├── constants/
│   │   └── __init__.py
│   ├── entity/
│   │   ├── __init__.py
│   │   ├── config_entity.py
│   │   ├── artifact_entity.py
│   │   ├── estimator.py              # MyModel — bundles preprocessing + model + tuned threshold
│   │   └── s3_estimator.py           # coming soon
│   ├── exception/
│   │   └── __init__.py
│   ├── logger/
│   │   └── __init__.py
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── training_pipeline.py
│   │   └── prediction_pipeline.py    # coming soon
│   └── utils/
│       ├── __init__.py
│       └── main_utils.py
├── config/
│   ├── schema.yaml                   # column definitions, types, encoding groups
│   └── model.yaml                    # coming soon
├── notebook/                         # EDA + model comparison (gitignored) — adding this next
├── artifact/                         # timestamped pipeline run outputs (gitignored)
├── app.py                            # coming soon
├── demo.py                           # entry point — runs the full pipeline
├── setup.py
├── pyproject.toml
├── Dockerfile                        # coming soon
├── .dockerignore                     # coming soon
├── .gitignore
└── requirements.txt
```

## 🔄 The Pipeline

| Stage | What it does |
|---|---|
| **1. Data Ingestion** | Pulls the raw collection from MongoDB, exports it to a feature-store CSV, and splits it into train/test sets |
| **2. Data Validation** | Checks the ingested data against `config/schema.yaml` (right number of columns, all expected numeric/categorical columns present) before anything downstream touches it |
| **3. Data Transformation** | Encodes categorical features (ordinal for Education, binary mapping for the yes/no columns, one-hot for the rest), scales numeric features with StandardScaler, and applies SMOTEENN to the training data only — the test set stays a true, untouched reflection of the real class distribution |
| **4. Model Training** | Trains a CatBoost classifier (it came out ahead of Logistic Regression, Decision Tree, Random Forest, and Extra Trees during my notebook experiments) and tunes the classification threshold to maximize F1 instead of just using the default 0.5 cutoff |

## 📊 Results

<div align="center">

| Metric | Score |
|:---:|:---:|
| ROC-AUC | ~0.76 |
| PR-AUC | ~0.33 |
| Accuracy | ~0.81 |
| Precision | ~0.30 |
| Recall | ~0.50 |

</div>

Honestly, these numbers plateaued no matter what I tried — more feature engineering, different resampling strategies, hyperparameter tuning — which taught me that sometimes the ceiling is the data itself, not the model. That was a useful lesson on its own.

## ⚙️ Setup

```bash
python -m venv venv
venv\Scripts\activate      # source venv/bin/activate on Mac/Linux
pip install -r requirements.txt
```

Set your MongoDB connection string as an environment variable:

```bash
# Bash
export MONGODB_URL="mongodb+srv://<username>:<password>...."

# PowerShell
$env:MONGODB_URL = "mongodb+srv://<username>:<password>...."
```

## ▶️ Running It

```bash
python demo.py
```

This runs the full pipeline end to end — ingestion, validation, transformation, and training — and writes a timestamped run folder under `artifact/`.

## 🚧 What's Next

- [ ] Model evaluation stage — compare a newly trained model against whatever's currently deployed before replacing it
- [ ] Push trained models to S3
- [ ] Build a simple prediction API
- [ ] Dockerize the pipeline
- [ ] Set up CI/CD with GitHub Actions so pushes automatically retrain/redeploy

## 🛠️ Tech Stack

<div align="center">

![MongoDB](https://img.shields.io/badge/MongoDB-47A248?style=flat-square&logo=mongodb&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![scikit--learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![CatBoost](https://img.shields.io/badge/CatBoost-ML-FFCC00?style=flat-square)
![imbalanced--learn](https://img.shields.io/badge/imbalanced--learn-9146FF?style=flat-square)

</div>

---

<div align="center">

If you found this project helpful or have any questions, feel free to reach out! ⭐

</div>
