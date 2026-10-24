# Customer Churn Prediction Using Machine Learning

A machine learning project that predicts whether a telecom customer is likely to leave the service using customer account and service information.

## Project Overview

Customer churn is an important business problem for telecom companies. Identifying customers who may leave can help businesses plan customer-retention strategies.

This project uses the IBM Telco Customer Churn dataset to train a classification model and generate predictions for new customer records.

## Results

| Evaluation Metric | Score |
|---|---:|
| Accuracy | 80.55% |
| Churn Precision | 65.72% |
| Churn Recall | 55.88% |
| Churn F1-score | 60.40% |
| ROC-AUC | 84.19% |
| Majority-class baseline accuracy | 73.46% |

Evaluation was performed on 1,409 test records.

### Confusion Matrix

| Actual / Predicted | No Churn | Churn |
|---|---:|---:|
| No Churn | 926 | 109 |
| Churn | 165 | 209 |

The model performed better in accuracy than the majority-class baseline. However, improving churn recall is an important next step because some customers who actually churned were missed.

## Dataset

- Source: IBM Telco Customer Churn dataset
- Records: 7,043
- Input features: 19
- Target: `Churn` (`Yes` or `No`)

The dataset contains information about customer tenure, contract type, monthly charges, internet services, payment methods, and other account features.

## Technologies Used

- Python
- pandas
- NumPy
- scikit-learn
- Matplotlib
- Seaborn
- joblib

## Project Structure

```text
customer-churn-prediction-ml/
├── data/
│   └── raw/
├── models/
│   ├── churn_pipeline.joblib
│   └── model_info.json
├── outputs/
│   ├── figures/
│   └── metrics/
├── src/
│   ├── train.py
│   └── predict.py
├── report.md
├── requirements.txt
└── README.md
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/meghasrin16-alt/customer-churn-prediction-ml.git
cd customer-churn-prediction-ml
```

### 2. Create and activate a virtual environment

Windows:

```bat
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the model

```bash
python src/train.py
```

### 5. Predict churn for a customer

```bash
python src/predict.py
```

The prediction script asks for customer details and displays a churn prediction and, when available, a churn probability estimate.

## Output Files

- `models/churn_pipeline.joblib` — trained model pipeline
- `models/model_info.json` — model feature and configuration information
- `outputs/metrics/metrics.json` — evaluation metrics
- `outputs/metrics/confusion_matrix.csv` — confusion matrix
- `outputs/metrics/classification_report.txt` — classification report
- `report.md` — detailed project report

## Limitations and Future Improvements

- Improve recall for the churn class.
- Compare different classification algorithms.
- Tune the prediction threshold.
- Evaluate performance on new data.
- Build an interactive dashboard or web application.

## Disclaimer

Predictions are estimates and are not guarantees of customer behavior. Model performance may vary on new data.# customer-churn-prediction-ml