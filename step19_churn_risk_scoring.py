import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


print("=" * 70)
print("STEP 8.4 - CHURN PROBABILITY & RISK SCORING")
print("=" * 70)


# ------------------------------------------------------------
# 1. Load training data
# ------------------------------------------------------------

X_train = pd.read_csv("data/X_train.csv")
X_test = pd.read_csv("data/X_test.csv")

y_train = pd.read_csv(
    "data/y_train.csv"
).squeeze("columns")

y_test = pd.read_csv(
    "data/y_test.csv"
).squeeze("columns")


# ------------------------------------------------------------
# 2. Load model results
# ------------------------------------------------------------

results = pd.read_csv(
    "data/churn_model_results.csv"
)

print("\nMODEL RESULTS")
print("-" * 70)

print(results)


# ------------------------------------------------------------
# 3. Select model using F1-score
# ------------------------------------------------------------

best_model_name = (
    results
    .sort_values("F1_Score", ascending=False)
    .iloc[0]["Model"]
)

print("\nSelected model based on F1-score:")
print(best_model_name)


# ------------------------------------------------------------
# 4. Create selected model
# ------------------------------------------------------------

if best_model_name == "Logistic Regression":

    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

elif best_model_name == "Decision Tree":

    model = DecisionTreeClassifier(
        max_depth=6,
        random_state=42
    )

elif best_model_name == "Random Forest":

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

else:

    raise ValueError(
        "Unknown model selected."
    )


# ------------------------------------------------------------
# 5. Train selected model
# ------------------------------------------------------------

model.fit(
    X_train,
    y_train
)


# ------------------------------------------------------------
# 6. Load customer feature dataset
# ------------------------------------------------------------

customers = pd.read_csv(
    "data/customer_features.csv"
)

print("\nCustomer feature dataset:")
print(customers.shape)


# ------------------------------------------------------------
# 7. Prepare scoring data
# ------------------------------------------------------------

customer_ids = customers["customer_id"].copy()

X_customers = customers.drop(
    columns=["customer_id"],
    errors="ignore"
)


# ------------------------------------------------------------
# 8. Match training features
# ------------------------------------------------------------

training_features = X_train.columns.tolist()

missing_features = [
    column
    for column in training_features
    if column not in X_customers.columns
]

extra_features = [
    column
    for column in X_customers.columns
    if column not in training_features
]

print("\nMissing training features:")
print(missing_features)

print("\nExtra customer features:")
print(extra_features)


# Add missing features as zero
for column in missing_features:
    X_customers[column] = 0


# Keep exactly the training columns
X_customers = X_customers[
    training_features
]


# Convert to numeric
X_customers = X_customers.apply(
    pd.to_numeric,
    errors="coerce"
).fillna(0)


# ------------------------------------------------------------
# 9. Predict churn probability
# ------------------------------------------------------------

churn_probability = model.predict_proba(
    X_customers
)[:, 1]


# ------------------------------------------------------------
# 10. Create risk score
# ------------------------------------------------------------

risk_scores = pd.DataFrame({

    "customer_id": customer_ids,

    "churn_probability":
        churn_probability

})


# ------------------------------------------------------------
# 11. Create risk categories
# ------------------------------------------------------------

def risk_category(probability):

    if probability >= 0.75:
        return "Very High Risk"

    elif probability >= 0.50:
        return "High Risk"

    elif probability >= 0.25:
        return "Medium Risk"

    else:
        return "Low Risk"


risk_scores["risk_category"] = (
    risk_scores["churn_probability"]
    .apply(risk_category)
)


# ------------------------------------------------------------
# 12. Add customer information
# ------------------------------------------------------------

customer_columns = [
    "customer_id",
    "customer_segment",
    "total_orders",
    "total_revenue",
    "total_profit",
    "total_interactions",
    "RFM_score",
    "customer_segment_rfm"
]

available_columns = [
    column
    for column in customer_columns
    if column in customers.columns
]

risk_scores = risk_scores.merge(
    customers[available_columns],
    on="customer_id",
    how="left"
)


# ------------------------------------------------------------
# 13. Sort by risk
# ------------------------------------------------------------

risk_scores = risk_scores.sort_values(
    "churn_probability",
    ascending=False
)


# ------------------------------------------------------------
# 14. Display results
# ------------------------------------------------------------

print("\nTOP 20 HIGH-RISK CUSTOMERS")
print("-" * 70)

print(
    risk_scores.head(20).to_string(
        index=False
    )
)


# ------------------------------------------------------------
# 15. Risk distribution
# ------------------------------------------------------------

print("\nRISK DISTRIBUTION")
print("-" * 70)

print(
    risk_scores["risk_category"]
    .value_counts()
)


# ------------------------------------------------------------
# 16. Save scored customers
# ------------------------------------------------------------

risk_scores.to_csv(
    "data/churn_scored_customers.csv",
    index=False
)


print("\nSaved:")
print("data/churn_scored_customers.csv")


print("\n" + "=" * 70)
print("CHURN RISK SCORING COMPLETED")
print("=" * 70)