<div align="center">

# Loan Default Prediction — MLOps Project

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-47A248?style=for-the-badge&logo=mongodb&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data-150458?style=for-the-badge&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![CatBoost](https://img.shields.io/badge/CatBoost-Model-FFCC00?style=for-the-badge)
![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-SMOTEENN-9146FF?style=for-the-badge)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Model%20Hub-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-CI%2FCD-2088FF?style=for-the-badge&logo=githubactions&logoColor=white)
![Render](https://img.shields.io/badge/Render-Deployed-46E3B7?style=for-the-badge&logo=render&logoColor=white)
![Status](https://img.shields.io/badge/status-completed-success?style=for-the-badge)

An end-to-end MLOps project for predicting loan defaults. The project goes beyond training a model in a notebook by implementing a modular machine learning pipeline for data ingestion, validation, transformation, training, evaluation, model storage, prediction, containerization, CI/CD, and deployment.

</div>

---

## 📖 Table of Contents

- [Overview](#-overview)
- [What I Learned](#-what-i-learned-building-this)
- [Project Structure](#-project-structure)
- [The Pipeline](#-the-pipeline)
- [Prediction Application](#-prediction-application)
- [Results](#-results)
- [Setup](#️-setup)
- [Running It](#️-running-it)
- [Testing](#-testing)
- [Docker](#-docker)
- [CI/CD](#-cicd)
- [Deployment](#-deployment)
- [Tech Stack](#️-tech-stack)
- [Future Improvements](#-future-improvements)

---

## 📌 Overview

Predicting whether a borrower will default on a loan using a dataset of approximately **255,000 loan records**.

The dataset presents a realistic challenge: only about **11.6%** of loans default. This makes the problem imbalanced, so simply optimizing accuracy would be misleading.

The goal of this project was to build more than a notebook model. The system is structured as a production-style machine learning pipeline where data ingestion, validation, transformation, training, evaluation, and model pushing are separate components with their own configurations and artifacts.

The project also includes a FastAPI prediction application, Docker containerization, automated testing, GitHub Actions CI/CD, GitHub Container Registry, and deployment on Render.

---

## 🧠 What I Learned Building This

This project helped me understand how machine learning projects move beyond experimentation.

Some of the main areas explored were:

- Designing modular ML pipelines
- Data ingestion from MongoDB Atlas
- Schema-based data validation
- Feature transformation and preprocessing
- Handling class imbalance with SMOTEENN
- Model training and threshold tuning
- Comparing a newly trained model with a production model
- Model storage and retrieval using Hugging Face Hub
- FastAPI application development
- Docker containerization
- Automated testing with Pytest
- CI/CD with GitHub Actions
- Docker image storage with GitHub Container Registry
- Cloud deployment using Render
- Environment variables and secret management

---

## 🗂️ Project Structure

```text
.
├── .github/
│   └── workflows/
│       └── cicd.yaml
│
├── src/
│   ├── components/
│   │   ├── __init__.py
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   ├── model_evaluation.py
│   │   └── model_pusher.py
│   │
│   ├── configuration/
│   │   ├── __init__.py
│   │   ├── mongo_db_connection.py
│   │   └── hf_connection.py
│   │
│   ├── cloud_storage/
│   │   ├── __init__.py
│   │   └── hf_storage.py
│   │
│   ├── data_access/
│   │   ├── __init__.py
│   │   └── load_default_data.py
│   │
│   ├── constants/
│   │   └── __init__.py
│   │
│   ├── entity/
│   │   ├── __init__.py
│   │   ├── config_entity.py
│   │   ├── artifact_entity.py
│   │   ├── estimator.py
│   │   └── hf_estimator.py
│   │
│   ├── exception/
│   │   └── __init__.py
│   │
│   ├── logger/
│   │   └── __init__.py
│   │
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── training_pipeline.py
│   │   └── prediction_pipeline.py
│   │
│   └── utils/
│       ├── __init__.py
│       └── main_utils.py
│
├── config/
│   └── schema.yaml
│
├── notebook/
│   ├── LoanDefault.csv
│   ├── notebook.ipynb
│   └── mongodb_demo.ipynb
│
├── tests/
│   ├── test_config_entity.py
│   ├── test_estimator.py
│   ├── test_hf_connection.py
│   ├── test_imports.py
│   └── test_schema.py
│
├── static/
│
├── templates/
│   ├── index.html
│   └── result.html
│                
│
├── app.py                    
├── demo.py                    
├── Dockerfile
├── .dockerignore
├── .gitignore
├── requirements.txt
├── setup.py
└── pyproject.toml
```

---

## 🔄 The Pipeline

| Stage | What it does |
|---|---|
| **1. Data Ingestion** | Pulls the raw collection from MongoDB, exports it to a feature-store CSV, and splits it into train/test sets |
| **2. Data Validation** | Checks the ingested data against `config/schema.yaml` before downstream processing |
| **3. Data Transformation** | Encodes categorical features, scales numerical features, and applies SMOTEENN to training data only |
| **4. Model Training** | Trains a CatBoost classifier with tuned hyperparameters and a calibrated classification threshold |
| **5. Model Evaluation** | Compares the newly trained model with the currently deployed model |
| **6. Model Pusher** | Uploads an accepted model to Hugging Face Hub |

### 1. Data Ingestion

The ingestion component pulls the raw loan data from MongoDB Atlas, exports it to a feature store, and creates training and testing datasets.

```text
MongoDB Atlas
      ↓
Feature Store
      ↓
Train / Test Split
```

### 2. Data Validation

The ingested data is validated against `config/schema.yaml`.

Validation checks include:

- Expected columns
- Numerical columns
- Categorical columns
- Dataset structure

This prevents invalid data from silently reaching downstream components.

### 3. Data Transformation

The transformation component handles:

- Ordinal encoding for education-related features
- Binary mapping for yes/no features
- One-hot encoding for remaining categorical features
- Numerical feature scaling using `StandardScaler`
- Class imbalance handling using `SMOTEENN`

`SMOTEENN` is applied only to the training data. The test set remains untouched so evaluation reflects the original data distribution.

### 4. Model Training

The project trains a `CatBoostClassifier` using hyperparameters tuned during experimentation.

A calibrated classification threshold is used instead of relying only on the default `0.5` cutoff.

### 5. Model Evaluation

The newly trained model is compared with the currently deployed model.

The new model is accepted only when its performance improves beyond the configured threshold.

```text
New Model
    ↓
Evaluation
    ↓
Compare with Production Model
    ↓
Improved?
 ┌────┴────┐
 Yes       No
 ↓          ↓
Push       Reject
```

### 6. Model Pusher

Accepted models are uploaded to Hugging Face Hub.

This keeps model artifacts separate from the application code:

```text
Application Code → GitHub

Model Artifacts → Hugging Face Hub
```

---

## 🌐 Prediction Application

The project includes a FastAPI-based prediction application.

The application:

1. Accepts loan information through a web form.
2. Converts the submitted values into structured input.
3. Creates a Pandas DataFrame.
4. Passes the data through the prediction pipeline.
5. Generates a prediction.
6. Displays whether the applicant is predicted to **Will Default** or **Will Repay**.

```text
User Input
    ↓
FastAPI
    ↓
LoanApplicantData
    ↓
Prediction Pipeline
    ↓
Model Prediction
    ↓
Will Default / Will Repay
```

---

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

These results plateaued despite experimenting with feature engineering, resampling strategies, hyperparameter tuning, and classification thresholds.

One useful lesson from the project was that model performance is not always limited by the algorithm. Sometimes the available data itself establishes a performance ceiling.

---

## ⚙️ Setup

Clone the repository:

```bash
git clone <repository-url>
cd Loan-Default-Prediction-MLOps-Project
```

Create and activate a virtual environment:

```bash
python -m venv venv
```

### Windows

```powershell
venv\Scripts\activate
```

### Mac/Linux

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Set the required environment variables.

### Bash

```bash
export MONGODB_URL="mongodb+srv://<username>:<password>...."
export HF_TOKEN="hf_xxxxxxxxxxxx"
export HF_REPO_ID="your-username/your-model-repo-name"
```

### PowerShell

```powershell
$env:MONGODB_URL = "mongodb+srv://<username>:<password>...."
$env:HF_TOKEN = "hf_xxxxxxxxxxxx"
$env:HF_REPO_ID = "your-username/your-model-repo-name"
```

> Never commit credentials or API tokens to GitHub.

---

## ▶️ Running It

Run the full training pipeline:

```bash
python demo.py
```

This runs:

```text
Data Ingestion
      ↓
Data Validation
      ↓
Data Transformation
      ↓
Model Training
      ↓
Model Evaluation
      ↓
Model Pushing
```

Pipeline outputs are written to timestamped directories under `artifact/`.

### Running the Prediction Application

Start the FastAPI application:

```bash
python app.py
```

The application runs on:

```text
http://localhost:5000
```

---

## 🧪 Testing

Run tests locally:

```bash
pytest tests/ -v
```

Tests are also executed automatically by GitHub Actions before the Docker image is built.

---

## 🐳 Docker

Build the Docker image:

```bash
docker build -t loan-default-app .
```

Run the container:

```bash
docker run -p 5000:5000 loan-default-app
```

The application will be available at:

```text
http://localhost:5000
```

---

## 🔁 CI/CD

The project uses GitHub Actions for Continuous Integration and Continuous Deployment.

The pipeline is triggered when code is pushed to the `main` branch.

```text
git push
    ↓
GitHub Actions
    ↓
Continuous Integration
    ├── Install Dependencies
    ├── Run Tests
    └── Check Python Syntax
    ↓
Build Docker Image
    ↓
Push Image to GHCR
    ↓
Trigger Render Deployment
    ↓
Live Application
```

### Continuous Integration

The CI stage:

- Installs project dependencies
- Installs test dependencies
- Runs automated tests with Pytest
- Checks Python syntax using `compileall`

The Docker build begins only after the CI stage succeeds.

### Build and Push Docker Image

After CI passes:

1. GitHub Actions builds the Docker image.
2. The image is pushed to GitHub Container Registry.
3. Images are tagged with:
   - `latest`
   - Git commit SHA

This provides both a current image and a version-specific image.

### Continuous Deployment

After the Docker image is successfully pushed to GHCR, the pipeline triggers the Render deployment hook.

Render then deploys the configured Docker image.

---

## 🚀 Deployment

The application is deployed on **Render**.

Deployment flow:

```text
GitHub Push
      ↓
GitHub Actions
      ↓
CI Checks
      ↓
Docker Build
      ↓
GitHub Container Registry (GHCR)
      ↓
Render
      ↓
Live Application
```

The application has been tested after deployment and the full CI/CD pipeline has been verified successfully.

---

## 🛠️ Tech Stack

<div align="center">

| Category | Technologies |
|---|---|
| Programming Language | Python 3.10 |
| Machine Learning | Scikit-learn, CatBoost |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| Imbalanced Learning | imbalanced-learn, SMOTEENN |
| Database | MongoDB Atlas |
| Model Storage | Hugging Face Hub |
| API Framework | FastAPI |
| Web Server | Uvicorn |
| Containerization | Docker |
| Testing | Pytest |
| CI/CD | GitHub Actions |
| Container Registry | GitHub Container Registry |
| Deployment | Render |

</div>

---

## 🔮 Future Improvements

Although the end-to-end pipeline, containerization, CI/CD, and deployment are complete, possible future improvements include:

- [ ] Model monitoring
- [ ] Data drift detection
- [ ] Experiment tracking
- [ ] Improved test coverage
- [ ] Automated retraining
- [ ] Secret scanning
- [ ] Infrastructure improvements for production workloads

---

<div align="center">

## ⭐ End-to-End MLOps Project

**From raw data to a deployed machine learning application.**

If you found this project helpful, consider giving it a ⭐!

</div>
