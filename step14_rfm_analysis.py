
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# STEP 6 - RFM ANALYSIS

# Create output folder
Path("outputs").mkdir(exist_ok=True)

# 1. Load customer analytics data
df = pd.read_csv("data/customer_analytics.csv")

df["last_purchase_date"] = pd.to_datetime(
    df["last_purchase_date"],
    errors="coerce"
)

# 2. Select customers who purchased
rfm = df[df["total_orders"] > 0].copy()

# 3. Calculate RFM metrics
# Use one day after the latest purchase as
# the analysis reference date.
reference_date = (
    rfm["last_purchase_date"].max()
    + pd.Timedelta(days=1)
)

rfm["Recency"] = (
    reference_date - rfm["last_purchase_date"]
).dt.days

rfm["Frequency"] = rfm["total_orders"]
rfm["Monetary"] = rfm["total_revenue"]

# 4. Assign scores from 1 to 5
# Rank first to handle repeated values.

rfm["R_Score"] = pd.qcut(
    rfm["Recency"].rank(method="first"),
    5,
    labels=[5, 4, 3, 2, 1]
).astype(int)

rfm["F_Score"] = pd.qcut(
    rfm["Frequency"].rank(method="first"),
    5,
    labels=[1, 2, 3, 4, 5]
).astype(int)

rfm["M_Score"] = pd.qcut(
    rfm["Monetary"].rank(method="first"),
    5,
    labels=[1, 2, 3, 4, 5]
).astype(int)

# 5. Calculate total RFM score
rfm["RFM_Score"] = (
    rfm["R_Score"]
    + rfm["F_Score"]
    + rfm["M_Score"]
)

# 6. Create customer segments
# These are illustrative segmentation rules.

def assign_segment(row):
    r = row["R_Score"]
    f = row["F_Score"]
    m = row["M_Score"]

    if r >= 4 and f >= 4 and m >= 4:
        return "Champions"
    elif r >= 3 and f >= 4:
        return "Loyal Customers"
    elif r <= 2 and f >= 4 and m >= 4:
        return "At Risk High Value"
    elif r <= 2 and f <= 2:
        return "Hibernating"
    elif r >= 4 and f <= 2:
        return "Potential Loyalists"
    else:
        return "Regular Customers"

rfm["RFM_Segment"] = rfm.apply(
    assign_segment,
    axis=1
)

# 7. Display results
print("\nRFM ANALYSIS")
print("-" * 60)

print(rfm[[
    "customer_id",
    "Recency",
    "Frequency",
    "Monetary",
    "RFM_Score",
    "RFM_Segment"
]].head(10))

# 8. Segment distribution
segment_counts = (
    rfm["RFM_Segment"].value_counts()
)

print("\nCUSTOMER SEGMENTS")
print(segment_counts)

# 9. Revenue by segment
segment_revenue = (
    rfm.groupby("RFM_Segment")["Monetary"]
    .sum()
    .sort_values(ascending=False)
)

print("\nREVENUE BY SEGMENT")
print(segment_revenue)

# 10. Visualize customer segments
plt.figure(figsize=(10, 5))

segment_counts.plot(kind="bar")

plt.title("Customer RFM Segmentation")
plt.xlabel("Customer Segment")
plt.ylabel("Number of Customers")
plt.xticks(rotation=35, ha="right")

plt.tight_layout()
plt.savefig(
    "outputs/rfm_segments.png",
    dpi=300
)
plt.show()

# 11. Revenue by segment chart
plt.figure(figsize=(10, 5))

segment_revenue.plot(kind="bar")

plt.title("Revenue by RFM Segment")
plt.xlabel("Customer Segment")
plt.ylabel("Total Revenue")
plt.xticks(rotation=35, ha="right")

plt.tight_layout()
plt.savefig(
    "outputs/rfm_revenue.png",
    dpi=300
)
plt.show()

# 12. Save RFM results
rfm.to_csv(
    "data/customer_rfm.csv",
    index=False
)

print("\nRFM ANALYSIS COMPLETED")
print("Saved: data/customer_rfm.csv")