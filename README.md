# Grammatical Facial Expressions Classification

Classification of grammatical facial expressions in Brazilian Sign Language (LIBRAS) using 3D facial landmarks. 

**Course Project:** Data Mining (Istraživanje podataka 2)  
**Author:** Zarija Trtović  
**Institution:** Faculty of Mathematics, University of Belgrade  
**School year:** 2025/2026

---

##  Project Overview

This project tackles a **multiclass classification problem** for recognizing grammatical facial expressions in Brazilian Sign Language (LIBRAS). The goal is to classify facial configurations into 9 expression types and predict their activation state (binary classification).

The workflow includes:
* Preprocessing and normalization of 3D facial landmark data
* Feature reduction using PCA and SelectKBest
* Training and evaluation of 5 machine learning algorithms
* Comprehensive performance comparison across 30 model variants
* Model persistence for reproducibility

---

##  Dataset

**Source:** [Grammatical Facial Expressions Dataset](https://archive.ics.uci.edu/dataset/317/grammatical+facial+expressions) (UCI Machine Learning Repository)

**Characteristics:**
- 27,936 frames extracted from 18 video recordings
- 100 3D facial landmarks per frame (X, Y coordinates in pixels, Z depth in millimeters)
- 300 numerical features total
- 2 participants (A and B)
- 9 expression types: affirmative, conditional, doubt_question, emphasis, negative, relative, topics, wh_question, yn_question
- Binary labels: 0 (inactive), 1 (active)

**Class distribution:**
- Inactive frames: 64.6%
- Active frames: 35.4%

---

##  Feature Variants

Models were trained and evaluated on three different feature representations:

* **All_Features** – Full set of 300 attributes (100 landmarks × 3 coordinates)
* **PCA_50** – Principal components preserving 99.38% variance (50 components)
* **KBest_100** – Top 100 features selected using ANOVA F-test

---

##  Implemented Algorithms

The following classifiers were evaluated:

* **Logistic Regression** – Baseline linear model
* **K-Nearest Neighbors (KNN)** – Instance-based learning (k=5)
* **Decision Tree** – Interpretable rule-based classifier (max_depth=10)
* **Support Vector Machine (SVM)** – RBF kernel for non-linear boundaries
* **Neural Network (MLP)** – Multi-layer perceptron with 2 hidden layers (100, 50 neurons)

**Total experiments:** 30 models (5 algorithms × 3 feature sets × 2 targets)

---

##  Evaluation Metrics

Models were evaluated using:

* **Accuracy** – Overall classification correctness
* **Precision** – Exactness of positive predictions
* **Recall** – Completeness of positive class detection
* **F1-Score** – Harmonic mean of precision and recall (primary metric)

For multiclass classification (9 expressions), **weighted average** was used to account for class imbalance.

---

##  Best Model

The best overall performance was achieved using:

**Neural Network (MLP) with PCA_50**
* Target: Expression classification (9 classes)
* Test Accuracy: **96.78%**
* Test F1-Score: **96.78%**

**Key findings:**
- PCA reduction (50 components) **outperforms** full feature set (300) for Neural Network
- Expression classification (9 classes) achieves higher accuracy than binary label classification
- K-Nearest Neighbors shows strong robustness across all feature sets
- Logistic Regression confirms the problem is highly non-linear

---

##  Repository Structure
```
├── notebooks/
│   └── facial_expressions.ipynb       # Complete analysis workflow
├── data/
│   ├── dataset_original.csv.gz        # Raw data (compressed)
│   └── dataset_preprocessed.csv.gz    # Cleaned data (compressed)
├── figures/
│   ├── 3d_landmarks.png               # 3D visualization of facial landmarks
│   ├── distribucija.png               # Class distribution charts
│   ├── confusion_matrix_best.png      # Best model confusion matrix
│   └── model_comparison.png           # Algorithm comparison plots
├── outputs/
│   └── model_results.csv              # Performance metrics for all 30 models
├── saved_models/                       # Trained models and preprocessing objects
│   ├── Neural_Network_MLP_PCA_50_Expression.pkl
│   ├── scaler_expression.pkl
│   ├── pca_expression.pkl
│   ├── label_encoder.pkl
│   └── ...
├── zapisnik.pdf                        # Project report (Serbian)
├── requirements.txt                    # Python dependencies
└── README.md
```

---

##  Reproducibility

### 1. Clone the repository
```bash
git clone https://github.com/ZarijaTrtovic/Grammatical-Facial-Expressions---Classification.git
cd Grammatical-Facial-Expressions---Classification
```

### 2. Extract compressed datasets
```bash
gunzip data/dataset_original.csv.gz
gunzip data/dataset_preprocessed.csv.gz
```

### 3. Install dependencies
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Run the analysis
```bash
jupyter notebook notebooks/facial_expressions.ipynb
```

---

## ⚙️ Loading Saved Models
```python
import pickle

# Load best model
with open('saved_models/Neural_Network_MLP_PCA_50_Expression.pkl', 'rb') as f:
    model = pickle.load(f)

# Load preprocessing objects
with open('saved_models/scaler_expression.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('saved_models/pca_expression.pkl', 'rb') as f:
    pca = pickle.load(f)

with open('saved_models/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Prepare new data
X_scaled = scaler.transform(X_new)
X_pca = pca.transform(X_scaled)

# Make predictions
predictions = model.predict(X_pca)
expression_names = label_encoder.inverse_transform(predictions)
```

---

##  Results Summary

### Top 5 Models (by Accuracy)

| Rank | Model | Feature Set | Target | Accuracy | F1-Score |
|------|-------|-------------|--------|----------|----------|
| 1 | Neural Network (MLP) | PCA_50 | Expression | **96.78%** | 96.78% |
| 2 | Neural Network (MLP) | All_Features | Expression | 96.31% | 96.32% |
| 3 | Neural Network (MLP) | KBest_100 | Expression | 94.22% | 94.23% |
| 4 | K-Nearest Neighbors | PCA_50 | Expression | 94.08% | 94.07% |
| 5 | K-Nearest Neighbors | All_Features | Expression | 94.06% | 94.05% |

### Algorithm Comparison (Average Performance)

| Algorithm | Avg Accuracy | Avg F1-Score |
|-----------|--------------|--------------|
| Neural Network (MLP) | 94.41% | 92.87% |
| K-Nearest Neighbors | 91.89% | 89.76% |
| Support Vector Machine | 86.54% | 83.33% |
| Decision Tree | 80.72% | 77.21% |
| Logistic Regression | 76.17% | 70.67% |

---

## Technical Notes

* **Train/Test Split:** 80% / 20% stratified split (22,348 train, 5,588 test samples)
* **Normalization:** StandardScaler applied (fit on training data only to prevent data leakage)
* **Feature Reduction:** Applied after scaling
* **Missing Values:** Z=0 depth measurements replaced with median values
* **Reproducibility:** All models trained with `random_state=42`
* **Class Imbalance:** Present in binary classification (64:36 ratio), not addressed with resampling

---

##  Required Libraries
```
numpy >= 1.24.0
pandas >= 2.0.0
matplotlib >= 3.7.0
seaborn >= 0.12.0
scikit-learn >= 1.3.0
jupyter >= 1.0.0
```

---

##  Documentation

Full project documentation (in Serbian) is available in `zapisnik.pdf`, including:
- Detailed methodology
- Algorithm descriptions
- Complete results analysis
- Discussion and conclusions

---

##  Academic Context

This work was completed as a course project for **Data Mining (Istraživanje podataka 2)** at the Faculty of Mathematics, University of Belgrade, under the supervision of Prof. Mirjana Maljković Ružičić.

**Educational Use Only** – This project is for academic purposes as part of coursework.

---

## 🔗 Links

* [UCI Dataset](https://archive.ics.uci.edu/dataset/317/grammatical+facial+expressions)
* [GitHub Repository](https://github.com/ZarijaTrtovic/Grammatical-Facial-Expressions---Classification)