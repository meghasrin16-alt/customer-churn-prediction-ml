# Customer Churn Prediction Using Machine Learning

## 1. Project Overview
This project predicts whether a telecom customer is likely to leave the service using machine learning. It uses the IBM Telco Customer Churn dataset.

## 2. Dataset
- Dataset: IBM Telco Customer Churn
- Total records: 7,043
- Input features: 19
- Target column: Churn
- Target values: Yes (churn), No (no churn)

## 3. Data Preprocessing
- Removed the customer ID from model inputs.
- Cleaned text values and handled missing values.
- Converted TotalCharges to a numeric data type.
- Separated input features and the target variable.
- Split the data into training and testing sets.

## 4. Model Evaluation
The model was evaluated on 1,409 test records.

| Metric | Result |
|---|---:|
| Accuracy | 80.55% |
| Churn Precision | 65.72% |
| Churn Recall | 55.88% |
| Churn F1-score | 60.40% |
| ROC-AUC | 84.19% |
| Majority-class baseline accuracy | 73.46% |

## 5. Confusion Matrix

| Actual / Predicted | No Churn | Churn |
|---|---:|---:|
| No Churn | 926 | 109 |
| Churn | 165 | 209 |

## 6. Key Findings
- The model achieved 80.55% accuracy.
- It correctly identified 209 customers who churned.
- It missed 165 customers who actually churned.
- Churn recall is an important area for improvement.
- The model performed better in accuracy than the majority-class baseline.

## 7. Limitations
- Predictions are estimates and are not guaranteed outcomes.
- The model may miss customers who are likely to churn.
- Performance should be tested on new, unseen data before real-world use.
- Model thresholds and alternative algorithms can be evaluated to improve churn detection.

## 8. Future Improvements
- Compare Logistic Regression with other classification algorithms.
- Tune the classification threshold.
- Improve churn recall while monitoring false positives.
- Build an interactive dashboard or web application.
- Monitor model performance on new customer data.

## 9. Conclusion
This project demonstrates a machine learning workflow for telecom customer churn prediction, including data preprocessing, model training, evaluation, and prediction for new customer records.