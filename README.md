# End-to-End NLP Pipeline with FastAPI Dashboard  

This project is an **end-to-end Natural Language Processing (NLP) pipeline** designed to take a raw dataset, automatically preprocess it, train a machine learning model, evaluate results, and serve everything in a clean, dark-themed **FastAPI dashboard**.  

It’s built to be flexible, modular, and easy to run — whether you’re experimenting with text data, testing models locally, or showcasing results interactively.  

---

## Getting Started  

Follow these steps to set up the project locally.  

### Dependencies  

- **Core ML/NLP**  
  - scikit-learn  
  - spacy  *(plus download: `python -m spacy download en_core_web_sm`)*  
  - joblib  
  - numpy  
  - pandas  
  - matplotlib  
  - scipy  

- **Web Framework (Dashboard)**  
  - fastapi  
  - uvicorn  
  - python-multipart  

- **Standard Library** (Python ≥3.9)  
  - argparse, dataclasses, json, re, os, time, inspect, shutil, etc.  

### Installation  

```bash
# Clone repository
git clone https://github.com/AmrrSalem/ds-nd-p003.git
cd ds-nd-p003

# Create virtual environment
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scriptsctivate         # Windows

# Install dependencies
pip install -r requirements.txt

# Or install manually
pip install fastapi uvicorn python-multipart scikit-learn spacy joblib numpy pandas matplotlib scipy
python -m spacy download en_core_web_sm
```

---

## Testing  

### Quick Run (Full Training)  

```bash
python quick_main_full.py --csv data.csv --target target_column --artifacts artifacts
```

This will:  
- Train a **full-feature pipeline** (TF–IDF, lemmatization, POS/NER counts, basic stats).  
- Perform cross-validation with randomized hyperparameter search.  
- Save artifacts: `model.pkl`, `metrics.json`, and `run_config.json`.  

### Dashboard  

```bash
DATA_CSV=./data.csv ARTIFACTS_DIR=artifacts uvicorn dashboard:app --reload --port 7860
```

Open in your browser:  
```
http://127.0.0.1:7860
```

You’ll see:  
- Inline metrics  
- Artifacts (confusion matrix, PR curve)  
- Predictions on 20 random rows from your CSV  

### Health Check  

```bash
curl http://127.0.0.1:7860/healthz
```
Response:  
```
ok
```

---

## Project Instructions  

- Place your dataset as `data.csv` in the project root (or specify via `--csv`).  
- Specify target column with `--target` (if omitted, auto-detection heuristics are applied).  
- Use `--clean` to reset artifacts before a new run.  
- All results are stored in the `artifacts/` directory.  

---

## Built With  

- [FastAPI](https://fastapi.tiangolo.com/) – lightweight web framework  
- [Uvicorn](https://www.uvicorn.org/) – ASGI server  
- [scikit-learn](https://scikit-learn.org/stable/) – ML models & pipelines  
- [spaCy](https://spacy.io/) – text processing, lemmatization, POS/NER  
- [pandas](https://pandas.pydata.org/) – data handling  
- [numpy](https://numpy.org/) – numerical operations  
- [matplotlib](https://matplotlib.org/) – evaluation plots  
- [joblib](https://joblib.readthedocs.io/) – model persistence  

---

## References  

This project was inspired by and built upon concepts and tools from:  
- [FastAPI Documentation](https://fastapi.tiangolo.com/) – for API and dashboard development.  
- [scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html) – for pipelines, transformers, and model training.  
- [spaCy Documentation](https://spacy.io/usage) – for text cleaning, tokenization, and lemmatization.  
- [Uvicorn](https://www.uvicorn.org/) – lightweight ASGI server to serve FastAPI apps.  
- [Kaggle NLP Projects](https://www.kaggle.com/datasets?tags=13204-NLP) – inspiration for dataset-driven pipeline design.  
- [Udacity Data Science Nanodegree](https://www.udacity.com/course/data-scientist-nanodegree--nd025) – project structure and rubric-style evaluation ideas.  

---

## License  

[MIT License](LICENSE.txt)  

