# Breast Cancer Classification: KNN vs CART vs ID3

Lab project comparing three supervised classifiers on the **Breast Cancer Wisconsin (Diagnostic)** dataset to predict whether a tumor is **malignant** or **benign**.

Department of CSE, East West University.

## Dataset

Loaded with `sklearn.datasets.load_breast_cancer()`.

| Property | Value |
|---|---|
| Samples | 569 |
| Features | 30 (mean, standard error, worst of 10 nucleus measurements) |
| Classes | Malignant (212), Benign (357) |
| Missing values | 0 |

## Method

1. 80/20 stratified train-test split (`random_state=42`): 455 train, 114 test.
2. `StandardScaler` fitted on the training set only (used for KNN).
3. Models:
   - **KNN** (k = 5, scaled features)
   - **CART** (`DecisionTreeClassifier`, `criterion="gini"`)
   - **ID3** (`DecisionTreeClassifier`, `criterion="entropy"`; scikit-learn's entropy-based approximation of ID3)
4. Evaluation: accuracy, precision, recall, F1-score, confusion matrix, classification report.

## Results (test set, benign = positive class)

| Algorithm | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| KNN | 95.61% | 95.89% | 97.22% | 96.55% |
| CART | 91.23% | 95.59% | 90.28% | 92.86% |
| ID3 | 91.23% | 96.97% | 88.89% | 92.75% |

KNN performed best overall. CART and ID3 scored the same accuracy; ID3 had slightly higher precision but lower recall.

![Performance comparison](images/performance_comparison.png)

![Confusion matrices](images/confusion_matrices.png)

## Project structure

```
.
├── notebooks/
│   └── cancer_classification.ipynb
├── images/
│   ├── performance_comparison.png
│   └── confusion_matrices.png
├── report/                 # exported lab report (PDF/DOCX)
├── requirements.txt
├── LICENSE
└── README.md
```

## How to run

```bash
git clone https://github.com/<your-username>/breast-cancer-knn-cart-id3.git
cd breast-cancer-knn-cart-id3
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/cancer_classification.ipynb
```

No dataset download is needed; it ships with scikit-learn. The notebook also runs directly in Google Colab.

## Limitations

- Single train-test split with default hyperparameters.
- Trees are unpruned; cross-validation, tuning k, and `max_depth` / `min_samples_leaf` would give a fairer comparison.

## Author

Anik, CSE, East West University.
