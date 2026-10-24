import json
from pathlib import Path

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "churn_pipeline.joblib"
INFO_PATH = BASE_DIR / "models" / "model_info.json"


def get_customer_input(info):
    customer = {}

    print("\nEnter the customer details.")
    print("For categorical fields, choose one of the listed values.\n")

    for column in info["feature_columns"]:
        if column in info["numeric_features"]:
            while True:
                try:
                    value = float(input(f"{column}: "))
                    if column in ["SeniorCitizen", "tenure"] and not value.is_integer():
                        print("Please enter a whole number.")
                        continue
                    if column == "SeniorCitizen" and value not in [0, 1]:
                        print("Enter 0 or 1.")
                        continue
                    if column == "tenure" and not 0 <= value <= 100:
                        print("Please enter a valid tenure in months.")
                        continue
                    if column == "MonthlyCharges" and value < 0:
                        print("Charges cannot be negative.")
                        continue
                    if column == "TotalCharges" and value < 0:
                        print("Charges cannot be negative.")
                        continue

                    if column in ["SeniorCitizen", "tenure"]:
                        value = int(value)

                    customer[column] = value
                    break
                except ValueError:
                    print("Please enter a valid number.")

        else:
            allowed = info["allowed_categorical_values"][column]
            print(f"\n{column} options: {' / '.join(allowed)}")

            while True:
                value = input(f"{column}: ").strip()
                if value in allowed:
                    customer[column] = value
                    break
                print("Invalid option. Please type one of the listed values.")

    return customer


def main():
    if not MODEL_PATH.exists() or not INFO_PATH.exists():
        print("Model files are missing. Check the models folder.")
        return

    with open(INFO_PATH, "r", encoding="utf-8") as file:
        info = json.load(file)

    model = joblib.load(MODEL_PATH)

    print("=" * 45)
    print("       CUSTOMER CHURN PREDICTION")
    print("=" * 45)

    customer = get_customer_input(info)

    customer_df = pd.DataFrame(
        [customer],
        columns=info["feature_columns"]
    )

    prediction = int(model.predict(customer_df)[0])

    print("\n" + "=" * 45)
    print("             PREDICTION RESULT")
    print("=" * 45)

    if prediction == 1:
        print("Predicted result: CHURN")
        print("The model predicts this customer may leave.")
    else:
        print("Predicted result: NO CHURN")
        print("The model predicts this customer may stay.")

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(customer_df)[0]
        classes = list(model.classes_)

        if 1 in classes:
            churn_probability = probabilities[classes.index(1)] * 100
            print(f"Predicted churn probability: {churn_probability:.2f}%")

    print("\nNote: This is a model prediction, not a guarantee.")
    print("=" * 45)


if __name__ == "__main__":
    main()