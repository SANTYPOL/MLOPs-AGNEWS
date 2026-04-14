# Fake News Detection using AG News Dataset

**An End-to-End MLOps Pipeline with CI/CD, Model Deployment, and Cloud Readiness**

---

## 📌 Table of Contents
- [Project Objectives](#project-objectives)
- [Problem Definition & ML Use Case](#problem-definition--ml-use-case)
- [System Architecture & Pipeline Flow](#system-architecture--pipeline-flow)
- [Folder Structure](#folder-structure)
- [MLOps Components](#mlops-components)
  - [Data Versioning (DVC)](#data-versioning-dvc)
  - [Experiment Tracking (MLflow)](#experiment-tracking-mlflow)
  - [Modular Pipeline Design](#modular-pipeline-design)
  - [CI/CD Implementation](#cicd-implementation)
  - [Model Deployment Strategy](#model-deployment-strategy)
  - [Cloud Deployment & Infrastructure](#cloud-deployment--infrastructure)
  - [Monitoring, Logging & Governance](#monitoring-logging--governance)
- [Reproducibility & GitHub](#reproducibility--github)
- [Technical Report Summary](#technical-report-summary)
- [How to Run the Project](#how-to-run-the-project)
- [Results & Model Performance](#results--model-performance)
- [References & Tools](#references--tools)

---

## 🎯 Project Objectives

This project demonstrates a **production-grade MLOps pipeline** for fake news detection. The key objectives are:

| Objective | Implementation |
|-----------|----------------|
| **Define ML problem** | Multi-class text classification (World, Sports, Business, Sci/Tech) using AG News dataset. |
| **Data versioning** | DVC tracks `data/raw/agnews.csv` to ensure reproducibility. |
| **Experiment tracking** | MLflow logs all parameters, metrics, and artifacts for every model run. |
| **Modular pipeline** | Separate components (ingestion, preprocessing, training, evaluation) with clean interfaces. |
| **CI/CD** | GitHub Actions automatically tests, trains, and deploys on every push to `main`. |
| **Deployment strategy** | Real‑time REST API (FastAPI) justified by low‑latency inference requirements. |
| **Cloud deployment** | Docker + AWS EC2 (or alternative) for scalable, managed infrastructure. |
| **Monitoring & logging** | Structured logging, MLflow model registry, and drift detection stubs. |
| **Reproducibility** | Locked dependencies (`requirements.txt`), DVC, and a clear README. |

---

## 🧠 Problem Definition & ML Use Case

**Problem:**  
Classify news articles into **four categories**:  
- `0` – World  
- `1` – Sports  
- `2` – Business  
- `3` – Science/Technology  

**Use Case:**  
A media monitoring company wants to automatically route incoming news feeds to topic‑specific editors. This reduces manual effort and speeds up content curation.  

**ML Formulation:**  
Supervised multi‑class text classification.  
- Input: Raw news article text  
- Output: Predicted class label (0–3)  

**Business Relevance:**  
- Reduces human error in topic assignment  
- Scalable to millions of articles per day  
- Provides real‑time predictions for automated workflows  

**MLOps Lifecycle Alignment:**  
From data ingestion → preprocessing → training → evaluation → deployment → monitoring, every phase is automated and tracked.

---

## 🏗️ System Architecture & Pipeline Flow

![Pipeline Diagram](https://via.placeholder.com/800x400?text=Data+Ingestion+→+Preprocessing+→+Training+→+Evaluation+→+Deployment)

```mermaid
graph LR
    A[AG News Dataset] --> B(Data Ingestion)
    B --> C(Preprocessing)
    C --> D{Training}
    D --> E[Classical Models]
    D --> F[BERT]
    E & F --> G[Model Evaluation]
    G --> H[Best Classical Model]
    H --> I[Model Registry MLflow]
    H --> J[FastAPI Deployment]
    J --> K[Prediction Endpoint]
Flow Description:

Data Ingestion – Loads AG News from Hugging Face, saves a local CSV, optionally subsamples for testing.

Preprocessing – Lowercases and removes non‑alphabetic characters.

Training – Trains 4 classical models (TF‑IDF + LogReg, NB, SVM, RF) and fine‑tunes BERT. Logs all runs to MLflow.

Evaluation – Computes accuracy, F1, confusion matrix, and classification report. Saves plots as artifacts.

Deployment – Exports the best classical model as model.pkl and serves it via FastAPI.

CI/CD – GitHub Actions runs the pipeline on every commit and deploys if tests pass.

📁 Folder Structure
text
MLOPS_V004/
├── .github/workflows/
│   └── mlops.yml                # CI/CD pipeline definition
├── app/
│   └── app.py                   # FastAPI inference server
├── config/
│   └── config.yaml              # Configuration (test_mode, mlflow exp)
├── src/
│   ├── components/
│   │   ├── data_ingestion.py    # Load & version dataset
│   │   ├── preprocessing.py     # Text cleaning
│   │   ├── trainer.py           # Classical + BERT training
│   │   └── evaluator.py         # Metrics & plots
│   ├── pipeline/
│   │   └── training_pipeline.py # Orchestrates the whole pipeline
│   └── utils/
│       └── logger.py            # Logging configuration
├── tests/
│   └── test_basic.py            # Basic CI test
├── Dockerfile                   # Containerization for deployment
├── requirements.txt             # Version‑locked dependencies
└── README.md                    # This file
Justification:

Separation of concerns – Each component has a single responsibility.

Reusability – Components can be imported and tested individually.

Config‑driven – config.yaml controls behavior without code changes.

🔧 MLOps Components
Data Versioning (DVC)
Why DVC? The AG News dataset is ~120k rows. DVC stores the CSV file (data/raw/agnews.csv) in remote storage (e.g., S3, GCS) and keeps a lightweight pointer in Git.

Implementation:

bash
dvc init
dvc add data/raw/agnews.csv
dvc remote add myremote s3://mybucket/agnews
Benefit: Anyone can reproduce the exact dataset version by running dvc pull.

Experiment Tracking (MLflow)
Why MLflow? Tracks hyperparameters (model name, TF‑IDF settings, BERT epochs), metrics (accuracy, F1, training time), and artifacts (confusion matrix, comparison plot).

Implementation: Each model run is an MLflow run under the experiment agnews-exp.

Benefit: Compare multiple runs, identify the best model, and share results.

Modular Pipeline Design
Pipeline entry point: src/pipeline/training_pipeline.py calls load_data() → preprocess() → train() → evaluate().

Clean code principles:

Type hints (e.g., def train(df: pd.DataFrame, config: dict) -> tuple).

Docstrings for every function.

Error handling via logging.

Orchestration: Simple sequential execution; can be extended to Airflow or Prefect.

CI/CD Implementation
GitHub Actions workflow (.github/workflows/mlops.yml):

yaml
name: MLOps Pipeline
on: [push]
jobs:
  build-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with: { python-version: '3.10' }
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run pipeline (test mode)
        run: python src/pipeline/training_pipeline.py
      - name: Deploy to EC2 (if on main)
        if: github.ref == 'refs/heads/main'
        run: ./deploy.sh
What it does:

Runs the entire pipeline in test mode (small subset) on every push.

On merge to main, deploys the model to a cloud VM.

Model Deployment Strategy
Chosen approach: Real‑time REST API (FastAPI).

Justification:

Low latency required – The use case (automatic news routing) expects sub‑second responses.

Synchronous interaction – Downstream systems (e.g., content management system) call the API with each new article.

Simple scaling – Stateless API can be horizontally scaled with a load balancer.

Alternatives considered & rejected:

Batch – Too slow for real‑time feeds.

Streaming – Overkill for the current volume; Kafka would add unnecessary complexity.

Cloud Deployment & Infrastructure
Platform: AWS (or any cloud)

Services used:

EC2 (t2.micro for demo, larger for production) – hosts the FastAPI app.

S3 – stores DVC‑tracked dataset and MLflow artifacts.

GitHub Actions – pushes Docker image to Amazon ECR, then deploys to EC2.

Infrastructure as Code (optional):
A terraform/ folder can provision the EC2 instance and security groups.

Why this architecture?

Cost‑effective for a capstone project.

Cloud‑agnostic – can switch to GCP/Azure with minimal changes.

Dockerised – Dockerfile ensures the same environment everywhere.

Monitoring, Logging & Governance
Logging:

Python logging module with timestamps and levels (INFO, ERROR).

Logs are printed to console and can be aggregated (e.g., CloudWatch).

Monitoring (planned):

Model performance – Log predictions to a file; compute weekly accuracy if ground truth is available.

Drift detection – Use evidently to compare input text distributions over time.

Governance:

MLflow model registry marks the “best” model.

Only approved models are deployed.

Access to the API can be controlled via API keys or OAuth.

🔁 Reproducibility & GitHub
Repository structure follows standard MLOps conventions.

Version control – All code, config, and DVC metafiles are under Git.

Dependencies – Exact versions in requirements.txt (pinned).

Environment – Use a virtual environment (venv/conda) and Docker.

Reproduce the pipeline:

bash
git clone https://github.com/your-repo/MLOPS_V004.git
cd MLOPS_V004
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
dvc pull          # if you have remote storage configured
python src/pipeline/training_pipeline.py
📄 Technical Report Summary
System Architecture Diagram
(Refer to the Mermaid diagram above.)

Pipeline Flow (Step‑by‑Step)
Load config – config.yaml controls test mode, MLflow experiment name.

Data ingestion – Downloads AG News, saves CSV, logs to DVC.

Preprocessing – Cleans text (lowercase, remove non‑letters).

Training – Splits data (stratified), trains 4 classical models + BERT, logs to MLflow.

Evaluation – Computes metrics, generates confusion matrix, picks best classical model.

Model saving – Pickles the best model to models/model.pkl.

Deployment – FastAPI loads the model and serves predictions.

Tools Used & Justification
Tool	Role	Justification
Python 3.10	Core language	Rich ecosystem for ML & MLOps.
scikit‑learn	Classical models	Reliable, fast, production‑ready.
Transformers (BERT)	Deep learning	SOTA for text classification.
MLflow	Experiment tracking	Industry standard, easy integration.
DVC	Data versioning	Lightweight, works with Git.
FastAPI	API framework	High performance, async support.
GitHub Actions	CI/CD	Native to GitHub, simple YAML config.
Docker	Containerisation	Ensures consistency across environments.
Experiments & Results (sample)
Model	Accuracy	F1 (weighted)
Logistic Regression	0.920	0.920
SVM	0.925	0.925
Random Forest	0.904	0.904
BERT	0.941	0.941
Best classical model: SVM (saved as model.pkl).
BERT achieves higher accuracy but requires GPU; classical models are chosen for cost‑efficient deployment.

Technical Viva (Implementation Defense)
CI/CD – Explain how the GitHub Action triggers on push, runs tests, and deploys.

DVC – Show how dvc add and dvc push work, and how to reproduce data.

Deployment – Demonstrate curl request to the FastAPI endpoint.

Cloud setup – Describe EC2 security groups, Docker run command, and reverse proxy (nginx).

Design decisions – Why classical model over BERT for deployment? (Latency, cost, sufficient accuracy.)