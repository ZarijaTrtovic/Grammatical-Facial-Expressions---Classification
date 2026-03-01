# Grammatical Facial Expressions Classification

Classification of grammatical facial expressions in Brazilian Sign Language (LIBRAS) using 3D facial landmarks. 

**Course project for Data Mining (Istraživanje podataka 2)**  
Faculty of Mathematics, University of Belgrade

---

## Overview

This project analyzes and classifies 9 types of grammatical facial expressions using machine learning algorithms:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Support Vector Machine (SVM)
- Neural Network (MLP)

**Best result:** 96.78% accuracy with Neural Network (MLP) using PCA reduction.

---

## Dataset

**Grammatical Facial Expressions Dataset** from [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/317/grammatical+facial+expressions)

- 27,936 frames from 18 videos
- 100 3D facial landmarks per frame (300 features)
- 9 expression types
- 2 users (A and B)

---

## Data Files

Due to GitHub file size limitations, the preprocessed CSV files are **not included** in this repository.

**To reproduce the analysis:**

**Option 1: Download preprocessed data**
- [Download from Google Drive](#) *(you can upload there later)*

**Option 2: Generate from raw dataset**
1. Download raw dataset from [UCI Repository](https://archive.ics.uci.edu/dataset/317/grammatical+facial+expressions)
2. Run the preprocessing cells in the Jupyter notebook
3. Preprocessed files will be saved in `data/` folder

## Project Structure
```
├── notebooks/           # Jupyter notebook with full analysis
├── data/                # Preprocessed datasets (CSV)
├── figures/             # Visualizations (PNG)
├── outputs/             # Model results (CSV)
├── saved_models/        # Trained models (pickle files)
└── requirements.txt     # Python dependencies
```

---

## Installation
```bash
# Clone the repository
git clone https://github.com/[YOUR_USERNAME]/grammatical-facial-expressions.git
cd grammatical-facial-expressions

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Usage
```bash
jupyter notebook notebooks/facial_expressions.ipynb
```

---

## Results

| Model | Feature Set | Accuracy | F1-Score |
|-------|-------------|----------|----------|
| Neural Network (MLP) | PCA (50) | **96.78%** | 96.78% |
| K-Nearest Neighbors | PCA (50) | 94.08% | 94.07% |
| Support Vector Machine | All (300) | 90.53% | 90.50% |
| Decision Tree | KBest (100) | 88.01% | 87.96% |
| Logistic Regression | All (300) | 83.73% | 83.34% |

---

## Author

**Zarija Trtović**  
Faculty of Mathematics, University of Belgrade  
February and March 2026.

---