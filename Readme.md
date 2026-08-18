#  Wine Quality Classification

---

## a. Problem Statement
The objective of this project is to apply multiple machine learning classification models to the **UCI Red Wine Quality dataset** and evaluate their performance.  
The task is to predict wine quality based on physicochemical properties and compare models using standard evaluation metrics.

---

## b. Dataset Description
- **Source:** UCI Machine Learning Repository  
- **Dataset:** Red Wine Quality  
- **Samples:** 1,599 wines  
- **Features:** 11 numeric attributes (fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol)  
- **Target Variable:** `quality` (integer score from 3–8)

---

## c. GitHub Repository Link 
 https://github.com/2025da04139-source/202504139-Repo

---

## d. Models Used
The following models were implemented:
1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbor (KNN)  
4. Naive Bayes (Gaussian)  
5. Random Forest (Ensemble)

###  Comparison Table of Evaluation Metrics
| ML Model Name       | Accuracy | AUC  | Precision | Recall | F1   | MCC  |
|---------------------|----------|------|-----------|--------|------|------|
| Logistic Regression | 0.55     | 0.72 | 0.56      | 0.55   | 0.55 | 0.32 |
| Decision Tree       | 0.61     | 0.68 | 0.62      | 0.61   | 0.61 | 0.39 |
| KNN                 | 0.58     | 0.70 | 0.59      | 0.58   | 0.58 | 0.35 |
| Naive Bayes         | 0.53     | 0.65 | 0.54      | 0.53   | 0.53 | 0.28 |
| Random Forest       | 0.67     | 0.80 | 0.68      | 0.67   | 0.67 | 0.45 |

*(Values may vary slightly depending on random seed.)*

---

## e. Observations on Model Performance

| ML Model Name       | Observation about model performance |
|---------------------|-------------------------------------|
| Logistic Regression | Performs moderately; struggles with complex decision boundaries. |
| Decision Tree       | Better than Logistic Regression; captures non-linear patterns but prone to overfitting. |
| KNN                 | Reasonable performance; sensitive to scaling and choice of k. |
| Naive Bayes         | Weakest performer; assumes feature independence which doesn’t hold well here. |
| Random Forest       | Best performer; ensemble approach reduces overfitting and improves accuracy and AUC. |

**Overall Winner for this dataset:**  
 **Random Forest (Ensemble)** — highest accuracy, AUC, and balanced performance across all metrics.

---

##  Notes
- All models were trained and saved as `.pkl` files in the `model/` folder.  
- Results are stored in `test_data.csv`.  
