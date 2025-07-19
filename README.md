# Credit Card Fraud Detection using Machine Learning

This project demonstrates a complete machine learning workflow to detect fraudulent credit card transactions from a highly imbalanced dataset. It explores various techniques including data preprocessing, handling class imbalance with SMOTE, and hyperparameter tuning with `RandomizedSearchCV` to build a high-performance `LightGBM` classification model.

---

## 📋 Table of Contents
* [Dataset](#-dataset)
* [Project Workflow](#-project-workflow)
* [Key Technologies and Libraries](#-key-technologies-and-libraries)
* [How to Run the Code](#-how-to-run-the-code)
* [Final Model Performance](#-final-model-performance)
* [Conclusion](#-conclusion)

---

## Dataset

The project utilizes the "Credit Card Fraud Detection" dataset available on Kaggle.

* **Source**: [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
* **Characteristics**:
    * Contains transactions made by European cardholders in September 2013.
    * Highly imbalanced: Features **492 frauds** out of **284,807 transactions** (only 0.172%).
    * Features `V1` through `V28` are the result of a PCA transformation to protect user identity. The only features that have not been transformed with PCA are `Time` and `Amount`.

---

## 🚀 Project Workflow

The project follows a structured machine learning pipeline:

1.  **Data Loading and Preprocessing**: The dataset is loaded, and the `Time` and `Amount` columns are standardized using `StandardScaler` to bring all features to a comparable scale.

2.  **Handling Class Imbalance**: To address the severe class imbalance, the **Synthetic Minority Over-sampling Technique (SMOTE)** is applied. SMOTE is used **only on the training data** to prevent data leakage, creating synthetic samples for the minority (fraud) class to create a balanced training set.

3.  **Model Selection**: Several models were considered, with the final implementation using **LightGBM** (`lgb.LGBMClassifier`) due to its high performance and exceptional speed on large datasets, outperforming `RandomForestClassifier`.

4.  **Hyperparameter Tuning**: Instead of using a fixed set of parameters, `RandomizedSearchCV` is employed to efficiently search through a distribution of hyperparameters and find the optimal combination. The search is optimized for **recall**, as the primary goal is to identify as many fraudulent transactions as possible.

5.  **Evaluation**: The final model is evaluated on the untouched test set. Performance is measured using metrics suitable for imbalanced classification:
    * **Classification Report**: Provides precision, recall, and F1-score for each class.
    * **Confusion Matrix**: Visualizes the model's predictions versus the actual classes.
    * **Precision-Recall (PR) Curve**: Shows the trade-off between precision and recall, with the Area Under the Curve (PR-AUC) serving as a key summary metric.

---

## 🛠️ Key Technologies and Libraries

* **Python 3.13.5**
* **Pandas**: For data manipulation and analysis.
* **Scikit-learn**: For data preprocessing, splitting, and model tuning (`StandardScaler`, `train_test_split`, `RandomizedSearchCV`).
* **Imbalanced-learn**: For handling class imbalance with `SMOTE`.
* **LightGBM**: As the primary classification algorithm.
* **Matplotlib & Seaborn**: For data visualization.

---

## ⚙️ How to Run the Code

To replicate the project, follow these steps:

1.  **Clone the repository.**
    ```bash
    git clone <https://github.com/sh-kartik18/Credit-Card-Fraud-Detection>
    ```

2.  **Download the dataset** from the Kaggle link above and place `creditcard.csv` in the root project directory.

3.  **Install the required libraries.** It is recommended to use a virtual environment.
    ```bash
    pip install notebook jupyterlab pandas scikit-learn imbalanced-learn lightgbm matplotlib seaborn
    ```

4.  **Launch Jupyter Lab or Notebook.**
    ```bash
    jupyter lab
    ```

5.  **Open and run the notebook.**
    * Navigate to and open the `CreditCardFraudDetection.ipynb` file.
    * Run the cells sequentially from top to bottom.
    

---

## 📊 Final Model Performance

The final `LightGBM` model, tuned with `RandomizedSearchCV`, achieved excellent performance on the test set.

* **PR-AUC Score**: **0.8689**

#### Classification Report:
```
               precision    recall  f1-score   support

Non-Fraud (0)       1.00      1.00      1.00     56864
    Fraud (1)       0.81      0.86      0.83        98

     accuracy                           1.00     56962
    macro avg       0.90      0.93      0.92     56962
 weighted avg       1.00      1.00      1.00     56962
```

The key result is a **recall of 0.86** for the fraud class, meaning the model successfully identified 86% of all fraudulent transactions while maintaining high precision.

---

## ✅ Conclusion

This project successfully demonstrates the creation of an effective fraud detection system. By using `SMOTE` to handle class imbalance and `LightGBM` for its speed and performance, the final model is both fast and highly capable of identifying fraudulent transactions, making it a powerful tool for this real-world problem.
