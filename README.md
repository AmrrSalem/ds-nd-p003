# End-to-End NLP Pipeline with FastAPI Dashboard
---
## **Project Description**

This project delivers an **end-to-end Natural Language Processing (NLP) pipeline** with a built-in **FastAPI dashboard** for monitoring, evaluation, and prediction. It automates the complete workflow of text-based machine learning tasks, from raw data ingestion to trained model deployment.

### **Key Features**

* **Automated Preprocessing**

  * Detects text, categorical, and numeric columns automatically.
  * Cleans and normalizes text (regex-based or spaCy-based lemmatization).
  * Supports TF–IDF vectorization, optional character n-grams, POS/NER counts, and basic text statistics.

* **Flexible Training**

  * Auto-detects classification vs regression targets.
  * Configurable pipeline with cross-validated hyperparameter tuning (`RandomizedSearchCV`).
  * Saves all training artifacts: model (`model.pkl`), metrics (`metrics.json`), and run configuration (`run_config.json`).

* **Evaluation & Reporting**

  * Calculates standard metrics (Accuracy, F1, ROC-AUC, R², MAE, RMSE, etc.).
  * Generates plots (confusion matrix, PR curve, etc.).
  * Logs step-wise runtime and total training time.

* **FastAPI Dashboard**

  * Serves artifacts and metrics through a lightweight, dark-themed UI.
  * Displays inline metrics, plots, and predictions from 20 random samples of the training CSV.
  * Provides a one-click option to re-sample and predict on new subsets.

* **CLI Integration**

  * `quick_main_full.py` enables a **one-command, full-feature run** with lemmatization, POS/NER features, and text statistics enabled.
  * Simple configuration via command-line arguments (`--csv`, `--target`, `--artifacts`, etc.).

### **Technology Stack**

* **ML/NLP**: scikit-learn, spaCy, joblib, numpy, pandas, matplotlib, scipy
* **Web/API**: FastAPI, Uvicorn
* **Utilities**: argparse, dataclasses, json, regex
---

## **Getting Started**

Follow these instructions to set up the project locally and run both the **training pipeline** and the **FastAPI dashboard**.

---

### **1. Clone the Repository**

```bash
git clone <your-repo-url>
cd <your-repo-folder>
```

---

### **2. Create and Activate a Virtual Environment**

It is recommended to use a virtual environment for dependency isolation.

```bash
# Create virtual environment (Python 3.10+ recommended)
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

---

### **3. Install Dependencies**

Install all required Python packages:

```bash
pip install -r requirements.txt
```

If you don’t have `requirements.txt` yet, install manually:

```bash
pip install fastapi uvicorn python-multipart scikit-learn spacy joblib numpy pandas matplotlib scipy
```

Then download the spaCy English model:

```bash
python -m spacy download en_core_web_sm
```

---

### **4. Prepare Your Data**

Place your dataset (CSV file) in the project root or provide its path.
Example:

```
data.csv
```

---

### **5. Train the Model (Full Pipeline)**

Run the **one-command training script**:

```bash
python quick_main_full.py --csv data.csv --target <target_column> --artifacts artifacts
```

Options:

* `--target` : Specify target column (omit to auto-detect).
* `--artifacts` : Directory to save model and metrics (default: `artifacts/`).
* `--clean` : Wipe old artifacts before training.
* `--cv` : Number of cross-validation folds (default: 3).
* `--n-iter` : Number of randomized search iterations (default: 10).

---

### **6. Launch the FastAPI Dashboard**

Once training is complete, start the dashboard:

```bash
DATA_CSV=./data.csv ARTIFACTS_DIR=artifacts uvicorn dashboard:app --reload --port 7860
```

Then open in your browser:

```
http://127.0.0.1:7860
```

The dashboard will show:

* Metrics from training
* Artifacts (plots, confusion matrix, PR curve)
* Predictions on 20 random samples from your CSV

---

### **7. Verify Installation**

Check health endpoint:

```bash
curl http://127.0.0.1:7860/healthz
```

Expected response:

```
ok
```
### Dependencies

```
Examples here
```

### Installation

Step by step explanation of how to get a dev environment running.

List out the steps

```
Give an example here
```

## Testing

Explain the steps needed to run any automated tests

### Break Down Tests

Explain what each test does and why

```
Examples here
```

## Project Instructions

This section should contain all the student deliverables for this project.

## Built With

* [Item1](www.item1.com) - Description of item
* [Item2](www.item2.com) - Description of item
* [Item3](www.item3.com) - Description of item

Include all items used to build project.

## License

[License](LICENSE.txt)
