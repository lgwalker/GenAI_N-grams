# GenAI_N-grams Language Model Project

This project implements an **N-gram language model** for Java source code. The model predicts Java tokens and evaluates code using **perplexity**. The pipeline includes data collection, preprocessing, training, and evaluation.

---

## Table of Contents
- [Installation and Dependencies](#installation-and-dependencies)
- [Workflow](#workflow)
  - [Data Preprocessing](#1-data-preprocessing)
  - [Model Training](#2-model-training)
  - [Testing & Evaluation](#3-testing--evaluation)
- [Hyperparameters and Tuning](#hyperparameters-and-tuning)
- [Output Files](#output-files)

---

## Installation and Dependencies

### Requirements
- Python 3.12+

### Install Dependencies

Using requirements file:
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install javalang gitpython pandas requests nltk
```

### External Tools
- **Git** must be installed and available in your system path.
- The preprocessing step uses Git to clone repositories.

---

## Workflow

### 1. Data Preprocessing
**File:** `PreprocessingData.ipynb`

This step collects high-quality Java repositories and extracts individual methods.

**Execution**
Run all cells in the notebook.

**Process**
- Fetches top **700 Java repositories**
  - Stars > 500
  - Updated after 2020
- Performs shallow clones (`--depth 1`)
- Randomly samples up to **20 `.java` files** per repository
- Extracts method bodies using brace-counting
- Filters:
  - Non-ASCII content
  - Methods with fewer than 10 tokens
  - Duplicate methods
- Tokenizes methods into space-separated tokens

**Output Location**
```
dataset/ngram_dataset/
```

Data splits:
- `train_15k.txt`
- `train_25k.txt`
- `train_35k.txt`
- `validation.txt`
- `test_self_mined.txt`

---

### 2. Model Training
**File:** `Training.ipynb`

Trains and evaluates multiple N-gram models using **NLTK**.

**Execution**
Run all cells after preprocessing is complete.

**Process**
- Loads datasets:
  - T1: 15k
  - T2: 25k
  - T3: 35k
- Trains:
  - 3-gram
  - 5-gram
  - 7-gram models
- Evaluates models using **validation perplexity**
- Selects the best-performing configuration

**Saved Outputs**
- `best_model.pkl`
- `best_vocab.pkl`

---

### 3. Testing & Evaluation
**File:** `test_script.py`

Command-line tool for generating predictions and perplexity.

**Usage**
```bash
python test_script.py --test_file ./path/to/test.txt --test_output results.json
```

**Arguments**
- `--test_file`  
  Path to the test dataset  
  Default: `./dataset/ngram_dataset/test.txt`

- `--test_output`  
  Output JSON filename

- `--mined_file`  
  Path to a self-mined test dataset

**Process**
- Maps unseen tokens to `<UNK>`
- Uses sliding window prediction
- Computes log-probabilities
- Calculates overall perplexity
- Saves predictions and metrics to JSON

---

## Hyperparameters and Tuning

| Parameter | Value | Description |
|----------|------|-------------|
| N-gram Order | 3, 5, 7 | Evaluated across datasets (3-gram typically performed best) |
| Smoothing (Alpha) | 0.01 | Lidstone smoothing for sparsity handling |
| UNK Cutoff | 3 | Tokens appearing fewer than 3 times mapped to `<UNK>` |
| Max Tokens | 512 | Methods longer than this are excluded |
| Min Unique Tokens | 10 | Ensures sufficient method complexity |

---

## Output Files

```
dataset/java_repos/
```
Local clones of GitHub repositories

```
dataset/ngram_dataset/*.txt
```
Preprocessed and tokenized dataset splits

```
best_model.pkl
```
Serialized `nltk.lm.Lidstone` model

```
best_vocab.pkl
```
Serialized vocabulary

```
results-*.json
```
Evaluation output containing:
- Token predictions
- Log-probabilities
- Overall perplexity
