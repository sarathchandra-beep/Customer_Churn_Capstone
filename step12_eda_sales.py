import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# STEP 5.2 - REVENUE & SALES ANALYSIS
# ============================================================

# Load transaction-product dataset
df = pd.read_csv("data/transaction_product_analysis.csv")

# Convert date
df["transaction_date"] = pd.to_datetime(
    df["transaction_date"]
)

print("=" * 60)
print("REVENUE & SALES ANALYSIS")
print("=" * 60)

# ------------------------------------------------------------
# 1. Total Revenue
# ------------------------------------------------------------

total_revenue = df["revenue"].sum()

print("\nTOTAL REVENUE")
print("-" * 60)
print("Total Revenue:", round(total_revenue, 2))


# ------------------------------------------------------------
# 2. Total Profit
# ------------------------------------------------------------

total_profit = df["profit"].sum()

print("\nTOTAL PROFIT")
print("-" * 60)
print("Total Profit:", round(total_profit, 2))


# ------------------------------------------------------------
# 3. Total Orders
# ------------------------------------------------------------

total_orders = df["transaction_id"].nunique()

print("\nTOTAL ORDERS")
print("-" * 60)
print("Total Orders:", total_orders)


# ------------------------------------------------------------
# 4. Total Quantity Sold
# ------------------------------------------------------------

total_quantity = df["quantity"].sum()

print("\nTOTAL QUANTITY SOLD")
print("-" * 60)
print("Units Sold:", total_quantity)


# ------------------------------------------------------------
# 5. Average Order Value
# ------------------------------------------------------------

average_order_value = (
    df.groupby("transaction_id")["revenue"]
    .sum()
    .mean()
)

print("\nAVERAGE ORDER VALUE")
print("-" * 60)
print(
    "Average Order Value:",
    round(average_order_value, 2)
)


# ------------------------------------------------------------
# 6. Revenue by Product Category
# ------------------------------------------------------------

category_revenue = (
    df.groupby("category")["revenue"]
    .sum()
    .sort_values(ascending=False)
)

print("\nREVENUE BY CATEGORY")
print("-" * 60)
print(category_revenue)


# ------------------------------------------------------------
# 7. Profit by Product Category
# ------------------------------------------------------------

category_profit = (
    df.groupby("category")["profit"]
    .sum()
    .sort_values(ascending=False)
)

print("\nPROFIT BY CATEGORY")
print("-" * 60)
print(category_profit)


# ------------------------------------------------------------
# 8. Revenue by Customer Segment
# ------------------------------------------------------------

segment_revenue = (
    df.groupby("customer_segment")["revenue"]
    .sum()
    .sort_values(ascending=False)
)

print("\nREVENUE BY CUSTOMER SEGMENT")
print("-" * 60)
print(segment_revenue)


# ------------------------------------------------------------
# 9. Monthly Revenue
# ------------------------------------------------------------

df["month"] = (
    df["transaction_date"]
    .dt.to_period("M")
    .astype(str)
)

monthly_revenue = (
    df.groupby("month")["revenue"]
    .sum()
)

print("\nMONTHLY REVENUE")
print("-" * 60)
print(monthly_revenue)


# ============================================================
# VISUALIZATIONS
# ============================================================

# ------------------------------------------------------------
# 10. Revenue by Category
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

category_revenue.plot(kind="bar")

plt.title("Revenue by Product Category")
plt.xlabel("Category")
plt.ylabel("Revenue")
plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "revenue_by_category.png",
    dpi=300
)

plt.show()


# ------------------------------------------------------------
# 11. Profit by Category
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

category_profit.plot(kind="bar")

plt.title("Profit by Product Category")
plt.xlabel("Category")
plt.ylabel("Profit")
plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "profit_by_category.png",
    dpi=300
)

plt.show()


# ------------------------------------------------------------
# 12. Revenue by Customer Segment
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

segment_revenue.plot(kind="bar")

plt.title("Revenue by Customer Segment")
plt.xlabel("Customer Segment")
plt.ylabel("Revenue")
plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "revenue_by_customer_segment.png",
    dpi=300
)

plt.show()


# ------------------------------------------------------------
# 13. Monthly Revenue Trend
# ------------------------------------------------------------

plt.figure(figsize=(10, 5))

monthly_revenue.plot(kind="line")

plt.title("Monthly Revenue Trend")
plt.xlabel("Month")
plt.ylabel("Revenue")
plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "monthly_revenue_trend.png",
    dpi=300
)

plt.show()


print("\n" + "=" * 60)
print("REVENUE & SALES EDA COMPLETED")
print("=" * 60)