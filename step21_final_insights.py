import pandas as pd

# --------------------------------------------------
# STEP 21: FINAL INSIGHTS & RECOMMENDATIONS
# --------------------------------------------------

FILE = "data/retention_analysis.csv"

df = pd.read_csv(FILE)

print("=" * 70)
print("CUSTOMER CHURN & RETENTION - FINAL BUSINESS ANALYSIS")
print("=" * 70)

# --------------------------------------------------
# 1. Overall business metrics
# --------------------------------------------------

total_customers = df["customer_id"].nunique()
total_revenue = df["total_revenue"].sum()
total_profit = df["total_profit"].sum()

print("\n1. OVERALL BUSINESS METRICS")
print("-" * 40)

print(f"Total Customers : {total_customers:,}")
print(f"Total Revenue   : {total_revenue:,.2f}")
print(f"Total Profit    : {total_profit:,.2f}")

# --------------------------------------------------
# 2. Risk distribution
# --------------------------------------------------

print("\n2. CHURN RISK DISTRIBUTION")
print("-" * 40)

risk_summary = (
    df.groupby("risk_category")
    .agg(
        customers=("customer_id", "count"),
        revenue=("total_revenue", "sum"),
        profit=("total_profit", "sum")
    )
    .reset_index()
)

risk_summary["customer_percentage"] = (
    risk_summary["customers"] / total_customers * 100
)

risk_summary["revenue_percentage"] = (
    risk_summary["revenue"] / total_revenue * 100
)

print(risk_summary.to_string(index=False))

# --------------------------------------------------
# 3. High-risk analysis
# --------------------------------------------------

high_risk = df[
    df["risk_category"].isin(
        ["High Risk", "Very High Risk"]
    )
]

high_risk_customers = high_risk["customer_id"].nunique()
high_risk_revenue = high_risk["total_revenue"].sum()
high_risk_profit = high_risk["total_profit"].sum()

print("\n3. HIGH-RISK CUSTOMER ANALYSIS")
print("-" * 40)

print(f"High-Risk Customers : {high_risk_customers:,}")
print(f"Revenue at Risk     : {high_risk_revenue:,.2f}")
print(f"Profit at Risk      : {high_risk_profit:,.2f}")

print(
    f"Customer Percentage : "
    f"{high_risk_customers / total_customers * 100:.2f}%"
)

print(
    f"Revenue Percentage  : "
    f"{high_risk_revenue / total_revenue * 100:.2f}%"
)

# --------------------------------------------------
# 4. High-value customers at risk
# --------------------------------------------------

high_value_at_risk = df[
    (df["value_category"] == "High Value") &
    (df["risk_category"].isin(
        ["High Risk", "Very High Risk"]
    ))
]

print("\n4. HIGH-VALUE CUSTOMERS AT RISK")
print("-" * 40)

print(
    "Customers:",
    high_value_at_risk["customer_id"].nunique()
)

print(
    "Revenue:",
    f"{high_value_at_risk['total_revenue'].sum():,.2f}"
)

print(
    "Profit:",
    f"{high_value_at_risk['total_profit'].sum():,.2f}"
)

# --------------------------------------------------
# 5. RFM analysis
# --------------------------------------------------

print("\n5. RFM SEGMENT ANALYSIS")
print("-" * 40)

rfm_summary = (
    df.groupby("customer_segment_rfm")
    .agg(
        customers=("customer_id", "count"),
        revenue=("total_revenue", "sum"),
        profit=("total_profit", "sum"),
        avg_churn_probability=("churn_probability", "mean")
    )
    .reset_index()
)

rfm_summary = rfm_summary.sort_values(
    "revenue",
    ascending=False
)

print(rfm_summary.to_string(index=False))

# --------------------------------------------------
# 6. Retention priority
# --------------------------------------------------

print("\n6. RETENTION PRIORITY")
print("-" * 40)

priority_summary = (
    df.groupby("retention_priority")
    .agg(
        customers=("customer_id", "count"),
        revenue=("total_revenue", "sum"),
        profit=("total_profit", "sum"),
        avg_churn_probability=("churn_probability", "mean")
    )
    .reset_index()
)

print(priority_summary.to_string(index=False))

# --------------------------------------------------
# 7. Top customers requiring attention
# --------------------------------------------------

print("\n7. TOP RETENTION TARGETS")
print("-" * 40)

priority_columns = [
    "customer_id",
    "customer_segment_rfm",
    "total_revenue",
    "total_profit",
    "churn_probability",
    "risk_category",
    "retention_priority"
]

top_targets = df.sort_values(
    "retention_priority_score",
    ascending=False
)[priority_columns].head(15)

print(top_targets.to_string(index=False))

# --------------------------------------------------
# 8. Business recommendations
# --------------------------------------------------

print("\n8. BUSINESS RECOMMENDATIONS")
print("-" * 40)

print("""
1. PRIORITIZE HIGH-VALUE HIGH-RISK CUSTOMERS
   Focus retention campaigns on customers who combine
   high revenue contribution with elevated churn probability.

2. RE-ENGAGE INACTIVE CUSTOMERS
   Customers showing reduced purchasing activity can be
   targeted with personalized offers and engagement campaigns.

3. PROTECT HIGH-VALUE RFM SEGMENTS
   Champions and loyal customers should receive strategies
   designed to maintain their engagement and purchasing frequency.

4. USE PERSONALIZED RETENTION CAMPAIGNS
   Segment customers according to value, RFM behavior and
   predicted churn risk instead of using one campaign for everyone.

5. MONITOR REVENUE AT RISK
   Churn analysis should consider both the probability of churn
   and the financial value associated with each customer.

6. CONTINUOUSLY UPDATE THE MODEL
   Retrain the churn model periodically as new customer,
   transaction and interaction data become available.
""")

print("\n" + "=" * 70)
print("FINAL ANALYSIS COMPLETED")
print("=" * 70)