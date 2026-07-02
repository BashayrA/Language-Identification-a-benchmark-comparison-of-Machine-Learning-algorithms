# Language-Identification-a-benchmark-comparison-of-Machine-Learning-algorithms

## Language Identification: ML vs LLM Benchmark
This project performs language identification using two approaches:

Traditional Machine Learning (TF‑IDF + Logistic Regression / Linear SVM)

Large Language Model (LLM) using DeepSeek‑V4‑Pro

The goal is to compare performance, accuracy, and behavior across both methods.

## 📂 Dataset
The dataset used in this project is a multilingual text dataset from Kaggle.

Since the file is too large to display here, you can download it from Kaggle:
🔗 Kaggle Dataset:  
https://www.kaggle.com/datasets/rtatman/language-identification-datasets (kaggle.com in Bing)  
(This is the standard dataset used for multilingual language identification tasks.)

## How to Access the Dataset
Download the dataset from Kaggle
Upload it to Google Colab
Load it

## 🧪 Project Workflow
Below is a simple breakdown of each step in your notebook.

1️⃣ Upload & Load Dataset
Upload CSV file in Colab.

Read into pandas DataFrame.

2️⃣ Exploratory Data Analysis (EDA)
Check dataset shape, columns, info
Remove duplicate text entries
Class imbalance visualization
Countplot of languages

Text length distribution

Histograms per language

Noise detection
You detect:

- URLs

- Emojis

- Numbers

- Punctuation

Then remove URL + emoji samples

3️⃣ Data Preprocessing
Lowercasing
Stripping whitespace


4️⃣ Traditional Machine Learning Approach
Train/Test Split
70% training, 30% testing.

TF‑IDF Vectorization
Character‑level n‑grams (1–3).

Models Trained
Logistic Regression

Linear SVM (LinearSVC)

Evaluation
Accuracy

Classification report

Confusion matrix

“sns.heatmap(cm, annot=False, cmap='Blues')”

5️⃣ LLM Approach — DeepSeek‑V4‑Pro
Prompt Engineering
A multi‑shot prompt with strict rules:

“Do NOT translate the text. Do NOT explain your reasoning. Output ONLY the language label.”

API Call
Using HuggingFace Router + OpenAI client.

Batch Prediction
LLM predicts language for a subset of test samples.

Evaluation
Accuracy

Classification report

Confusion matrix

Strict Prompt Fix
To prevent predicting languages not in the dataset (e.g., “German”).

📊 Results
The project compares:

ML accuracy vs LLM accuracy

Confusion matrices

Latency per LLM prediction

This allows you to evaluate:

Speed

Reliability

Error patterns

Generalization ability

## 🚀 How to Run the Project
1. Upload dataset

2. Run EDA cells
Visualize class distribution, text length, noise, etc.

3. Preprocess text
Lowercase + clean.

4. Train ML models
Run TF‑IDF → Logistic Regression → SVM.

5. Evaluate ML models
Print accuracy + confusion matrix.

6. Run LLM predictions
Call DeepSeek‑V4‑Pro using your API key.

7. Compare results
Check accuracy, confusion matrix, and latency.

## 🙌 Final Notes
This project demonstrates a full pipeline for multilingual language identification using both traditional ML and modern LLMs. It is ideal for benchmarking, academic work, or practical NLP applications.
A requirements.txt

A short abstract for your internship report
