import pandas as pd

print("=" * 70)
print("STEP 7 - FEATURE ENGINEERING")
print("=" * 70)

# ------------------------------------------------------------
# 1. Load datasets
# ------------------------------------------------------------

customers = pd.read_csv(
    "data/customer_analytics.csv"
)

rfm = pd.read_csv(
    "data/customer_rfm.csv"
)

transactions = pd.read_csv(
    "data/clean_transactions.csv"
)

# ------------------------------------------------------------
# 2. Convert dates
# ------------------------------------------------------------

customers["signup_date"] = pd.to_datetime(
    customers["signup_date"],
    errors="coerce"
)

transactions["transaction_date"] = pd.to_datetime(
    transactions["transaction_date"],
    errors="coerce"
)

# ------------------------------------------------------------
# 3. Analysis date
# ------------------------------------------------------------

analysis_date = transactions["transaction_date"].max()

print("\nAnalysis Date:")
print(analysis_date)

# ------------------------------------------------------------
# 4. Customer lifetime
# ------------------------------------------------------------

customers["customer_lifetime_days"] = (
    analysis_date - customers["signup_date"]
).dt.days

customers["customer_lifetime_days"] = (
    customers["customer_lifetime_days"]
    .clip(lower=0)
)

# ------------------------------------------------------------
# 5. Purchase frequency
# ------------------------------------------------------------

customers["purchase_frequency"] = (
    customers["total_orders"]
    / customers["customer_lifetime_days"].replace(0, 1)
)

# ------------------------------------------------------------
# 6. Revenue per interaction
# ------------------------------------------------------------

customers["revenue_per_interaction"] = (
    customers["total_revenue"]
    / customers["total_interactions"].replace(0, 1)
)

# ------------------------------------------------------------
# 7. Select RFM features
# ------------------------------------------------------------

rfm_features = rfm[
    [
        "customer_id",
        "Recency",
        "Frequency",
        "Monetary",
        "R_Score",
        "F_Score",
        "M_Score",
        "RFM_Score",
        "RFM_Segment"
    ]
].copy()

# Rename RFM columns to standard lowercase names
rfm_features = rfm_features.rename(
    columns={
        "Recency": "recency",
        "Frequency": "frequency",
        "Monetary": "monetary",
        "R_Score": "R_score",
        "F_Score": "F_score",
        "M_Score": "M_score",
        "RFM_Score": "RFM_score",
        "RFM_Segment": "customer_segment_rfm"
    }
)

# ------------------------------------------------------------
# 8. Merge RFM with customer data
# ------------------------------------------------------------

features = customers.merge(
    rfm_features,
    on="customer_id",
    how="left"
)

# ------------------------------------------------------------
# 9. Recency category
# ------------------------------------------------------------

def recency_category(days):

    if pd.isna(days):
        return "No Purchase"

    elif days <= 30:
        return "Recent"

    elif days <= 90:
        return "Moderate"

    elif days <= 180:
        return "Inactive"

    else:
        return "Long Inactive"


features["recency_category"] = (
    features["recency"]
    .apply(recency_category)
)

# ------------------------------------------------------------
# 10. Customer value category
# ------------------------------------------------------------

revenue_median = features["total_revenue"].median()

features["value_category"] = features[
    "total_revenue"
].apply(
    lambda x:
    "High Value"
    if x > revenue_median
    else "Standard Value"
)

# ------------------------------------------------------------
# 11. Fill missing numeric values
# ------------------------------------------------------------

numeric_columns = features.select_dtypes(
    include="number"
).columns

features[numeric_columns] = (
    features[numeric_columns]
    .fillna(0)
)

# ------------------------------------------------------------
# 12. Display results
# ------------------------------------------------------------

print("\nFEATURE DATASET SHAPE")
print("-" * 70)

print("Rows:", features.shape[0])
print("Columns:", features.shape[1])

print("\nFEATURE COLUMNS")
print("-" * 70)

print(features.columns.tolist())

# ------------------------------------------------------------
# 13. Display important features
# ------------------------------------------------------------

print("\nSAMPLE FEATURE DATA")
print("-" * 70)

print(
    features[
        [
            "customer_id",
            "total_orders",
            "total_revenue",
            "total_profit",
            "total_interactions",
            "recency",
            "frequency",
            "monetary",
            "RFM_score",
            "customer_segment_rfm"
        ]
    ].head(10)
)

# ------------------------------------------------------------
# 14. Save
# ------------------------------------------------------------

features.to_csv(
    "data/customer_features.csv",
    index=False
)

print("\nFeature dataset saved:")
print("data/customer_features.csv")

print("\n" + "=" * 70)
print("FEATURE ENGINEERING COMPLETED")
print("=" * 70)