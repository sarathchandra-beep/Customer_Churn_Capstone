import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# STEP 5.3 - CUSTOMER PURCHASE BEHAVIOR & CHURN ANALYSIS
# ============================================================

# Load customer analytics dataset
df = pd.read_csv("data/customer_analytics.csv")

print("=" * 60)
print("CUSTOMER PURCHASE BEHAVIOR & CHURN ANALYSIS")
print("=" * 60)

# ------------------------------------------------------------
# 1. Basic purchase statistics
# ------------------------------------------------------------

print("\nPURCHASE STATISTICS")
print("-" * 60)

print("Average orders per customer:",
      round(df["total_orders"].mean(), 2))

print("Average revenue per customer:",
      round(df["total_revenue"].mean(), 2))

print("Average profit per customer:",
      round(df["total_profit"].mean(), 2))

print("Average interactions per customer:",
      round(df["total_interactions"].mean(), 2))


# ------------------------------------------------------------
# 2. Top customers by revenue
# ------------------------------------------------------------

print("\nTOP 10 CUSTOMERS BY REVENUE")
print("-" * 60)

top_customers = (
    df[[
        "customer_id",
        "total_orders",
        "total_revenue",
        "total_profit"
    ]]
    .sort_values(
        "total_revenue",
        ascending=False
    )
    .head(10)
)

print(top_customers)


# ------------------------------------------------------------
# 3. Customers by purchase frequency
# ------------------------------------------------------------

print("\nPURCHASE FREQUENCY")
print("-" * 60)

print(
    df["total_orders"].describe()
)


# ------------------------------------------------------------
# 4. Customers with no purchases
# ------------------------------------------------------------

no_purchase = (
    df["total_orders"] == 0
).sum()

print("\nCUSTOMERS WITH NO PURCHASES")
print("-" * 60)

print("Number of customers:", no_purchase)


# ------------------------------------------------------------
# 5. Customers with high revenue
# ------------------------------------------------------------

revenue_median = df["total_revenue"].median()

high_value_customers = (
    df["total_revenue"] > revenue_median
).sum()

print("\nHIGH-VALUE CUSTOMERS")
print("-" * 60)

print(
    "Revenue median:",
    round(revenue_median, 2)
)

print(
    "Customers above revenue median:",
    high_value_customers
)


# ------------------------------------------------------------
# 6. Customer engagement
# ------------------------------------------------------------

print("\nCUSTOMER ENGAGEMENT")
print("-" * 60)

print(
    df["total_interactions"].describe()
)


# ------------------------------------------------------------
# 7. Revenue vs Orders
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    df["total_orders"],
    df["total_revenue"],
    alpha=0.5
)

plt.title(
    "Customer Orders vs Revenue"
)

plt.xlabel("Total Orders")
plt.ylabel("Total Revenue")

plt.tight_layout()

plt.savefig(
    "orders_vs_revenue.png",
    dpi=300
)

plt.show()


# ------------------------------------------------------------
# 8. Interactions vs Revenue
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    df["total_interactions"],
    df["total_revenue"],
    alpha=0.5
)

plt.title(
    "Customer Interactions vs Revenue"
)

plt.xlabel("Total Interactions")
plt.ylabel("Total Revenue")

plt.tight_layout()

plt.savefig(
    "interactions_vs_revenue.png",
    dpi=300
)

plt.show()


# ------------------------------------------------------------
# 9. Orders distribution
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.hist(
    df["total_orders"],
    bins=20
)

plt.title(
    "Customer Purchase Frequency"
)

plt.xlabel("Total Orders")
plt.ylabel("Number of Customers")

plt.tight_layout()

plt.savefig(
    "purchase_frequency.png",
    dpi=300
)

plt.show()


# ------------------------------------------------------------
# 10. Revenue distribution
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.hist(
    df["total_revenue"],
    bins=30
)

plt.title(
    "Customer Revenue Distribution"
)

plt.xlabel("Total Revenue")
plt.ylabel("Number of Customers")

plt.tight_layout()

plt.savefig(
    "customer_revenue_distribution.png",
    dpi=300
)

plt.show()


print("\n" + "=" * 60)
print("PURCHASE BEHAVIOR ANALYSIS COMPLETED")
print("=" * 60)