import pandas as pd
import os

# --------------------------------------------------
# STEP 20: RETENTION & BUSINESS ANALYSIS
# --------------------------------------------------

INPUT_FILE = "data/churn_scored_customers.csv"
OUTPUT_FILE = "data/retention_analysis.csv"
PRIORITY_FILE = "data/retention_priority_customers.csv"

# --------------------------------------------------
# 1. Load churn scored customers
# --------------------------------------------------

df = pd.read_csv(INPUT_FILE)

print("Dataset loaded successfully.")
print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

# --------------------------------------------------
# 2. Basic churn risk summary
# --------------------------------------------------

print("\n========== CHURN RISK DISTRIBUTION ==========")

risk_summary = (
    df.groupby("risk_category")
    .agg(
        customers=("customer_id", "count"),
        total_revenue=("total_revenue", "sum"),
        total_profit=("total_profit", "sum"),
        avg_revenue=("total_revenue", "mean"),
        avg_profit=("total_profit", "mean")
    )
    .reset_index()
)

print(risk_summary)

# --------------------------------------------------
# 3. High-risk customer analysis
# --------------------------------------------------

high_risk = df[
    df["risk_category"].isin(["Very High Risk", "High Risk"])
].copy()

print("\n========== HIGH-RISK CUSTOMERS ==========")
print("Number of high-risk customers:", len(high_risk))

print(
    "Revenue associated with high-risk customers:",
    round(high_risk["total_revenue"].sum(), 2)
)

print(
    "Profit associated with high-risk customers:",
    round(high_risk["total_profit"].sum(), 2)
)

# --------------------------------------------------
# 4. High-value customers at risk
# --------------------------------------------------

# Use median revenue as the high-value threshold
revenue_median = df["total_revenue"].median()

df["value_category"] = df["total_revenue"].apply(
    lambda x: "High Value" if x >= revenue_median else "Standard Value"
)

high_value_at_risk = df[
    (df["value_category"] == "High Value") &
    (df["risk_category"].isin(["Very High Risk", "High Risk"]))
].copy()

print("\n========== HIGH-VALUE AT-RISK CUSTOMERS ==========")

print(
    "Number of high-value at-risk customers:",
    len(high_value_at_risk)
)

print(
    "Revenue at risk:",
    round(high_value_at_risk["total_revenue"].sum(), 2)
)

print(
    "Profit at risk:",
    round(high_value_at_risk["total_profit"].sum(), 2)
)

# --------------------------------------------------
# 5. RFM Segment vs Churn Risk
# --------------------------------------------------

print("\n========== RFM SEGMENT VS CHURN RISK ==========")

rfm_risk = pd.crosstab(
    df["customer_segment_rfm"],
    df["risk_category"]
)

print(rfm_risk)

# --------------------------------------------------
# 6. Customer Segment vs Churn Risk
# --------------------------------------------------

print("\n========== CUSTOMER SEGMENT VS CHURN RISK ==========")

segment_risk = pd.crosstab(
    df["customer_segment"],
    df["risk_category"]
)

print(segment_risk)

# --------------------------------------------------
# 7. Risk group revenue analysis
# --------------------------------------------------

print("\n========== REVENUE BY RISK CATEGORY ==========")

revenue_by_risk = (
    df.groupby("risk_category")
    .agg(
        customers=("customer_id", "count"),
        revenue=("total_revenue", "sum"),
        profit=("total_profit", "sum")
    )
    .reset_index()
)

revenue_by_risk["revenue_percentage"] = (
    revenue_by_risk["revenue"]
    / revenue_by_risk["revenue"].sum()
    * 100
)

print(revenue_by_risk)

# --------------------------------------------------
# 8. Create retention priority score
# --------------------------------------------------

# Higher churn probability + higher revenue = higher priority

df["retention_priority_score"] = (
    df["churn_probability"] *
    (df["total_revenue"] / df["total_revenue"].max())
)

# --------------------------------------------------
# 9. Assign retention priority
# --------------------------------------------------

def assign_priority(score):
    if score >= 0.50:
        return "Critical"
    elif score >= 0.30:
        return "High"
    elif score >= 0.15:
        return "Medium"
    else:
        return "Low"


df["retention_priority"] = df[
    "retention_priority_score"
].apply(assign_priority)

# --------------------------------------------------
# 10. Sort customers by priority
# --------------------------------------------------

priority_customers = df.sort_values(
    by="retention_priority_score",
    ascending=False
).copy()

# --------------------------------------------------
# 11. Save full retention analysis
# --------------------------------------------------

os.makedirs("data", exist_ok=True)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

# --------------------------------------------------
# 12. Save top priority customers
# --------------------------------------------------

priority_columns = [
    "customer_id",
    "customer_segment",
    "customer_segment_rfm",
    "total_orders",
    "total_revenue",
    "total_profit",
    "total_interactions",
    "RFM_score",
    "churn_probability",
    "risk_category",
    "value_category",
    "retention_priority_score",
    "retention_priority"
]

# Keep only columns that actually exist
priority_columns = [
    col for col in priority_columns
    if col in priority_customers.columns
]

priority_customers[
    priority_columns
].head(200).to_csv(
    PRIORITY_FILE,
    index=False
)

# --------------------------------------------------
# 13. Final summary
# --------------------------------------------------

print("\n========== RETENTION PRIORITY ==========")

priority_summary = (
    df.groupby("retention_priority")
    .agg(
        customers=("customer_id", "count"),
        revenue=("total_revenue", "sum"),
        profit=("total_profit", "sum")
    )
    .reset_index()
)

print(priority_summary)

print("\n========== TOP PRIORITY CUSTOMERS ==========")

print(
    priority_customers[
        priority_columns
    ].head(10)
)

print("\nAnalysis completed successfully.")

print("\nFiles created:")
print("1.", OUTPUT_FILE)
print("2.", PRIORITY_FILE)