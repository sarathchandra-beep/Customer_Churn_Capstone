import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# STEP 5.1 - CUSTOMER EDA
# ============================================================

# Load integrated customer dataset
df = pd.read_csv("data/customer_analytics.csv")

print("=" * 60)
print("CUSTOMER EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# ------------------------------------------------------------
# 1. Basic information
# ------------------------------------------------------------

print("\nDATASET SHAPE")
print("-" * 60)
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# ------------------------------------------------------------
# 2. Customer segments
# ------------------------------------------------------------

print("\nCUSTOMER SEGMENTS")
print("-" * 60)

print(df["customer_segment"].value_counts())

# ------------------------------------------------------------
# 3. Gender distribution
# ------------------------------------------------------------

print("\nGENDER DISTRIBUTION")
print("-" * 60)

print(df["gender"].value_counts())

# ------------------------------------------------------------
# 4. Acquisition channels
# ------------------------------------------------------------

print("\nACQUISITION CHANNELS")
print("-" * 60)

print(df["acquisition_channel"].value_counts())

# ------------------------------------------------------------
# 5. Age statistics
# ------------------------------------------------------------

print("\nAGE ANALYSIS")
print("-" * 60)

print(df["age"].describe())

# ------------------------------------------------------------
# 6. Total customers
# ------------------------------------------------------------

print("\nTOTAL CUSTOMERS")
print("-" * 60)

print("Total customers:", df["customer_id"].nunique())

# ------------------------------------------------------------
# 7. Customers with purchases
# ------------------------------------------------------------

customers_with_purchases = (
    df["total_orders"] > 0
).sum()

print(
    "Customers with purchases:",
    customers_with_purchases
)

# ------------------------------------------------------------
# 8. Customers without purchases
# ------------------------------------------------------------

customers_without_purchases = (
    df["total_orders"] == 0
).sum()

print(
    "Customers without purchases:",
    customers_without_purchases
)


# ============================================================
# VISUALIZATIONS
# ============================================================

# ------------------------------------------------------------
# 9. Customer segment chart
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

df["customer_segment"].value_counts().plot(
    kind="bar"
)

plt.title("Customers by Segment")
plt.xlabel("Customer Segment")
plt.ylabel("Number of Customers")
plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "customer_segments.png",
    dpi=300
)

plt.show()


# ------------------------------------------------------------
# 10. Gender distribution
# ------------------------------------------------------------

plt.figure(figsize=(7, 5))

df["gender"].value_counts().plot(
    kind="bar"
)

plt.title("Customer Gender Distribution")
plt.xlabel("Gender")
plt.ylabel("Number of Customers")

plt.tight_layout()

plt.savefig(
    "customer_gender.png",
    dpi=300
)

plt.show()


# ------------------------------------------------------------
# 11. Acquisition channel
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

df["acquisition_channel"].value_counts().plot(
    kind="bar"
)

plt.title("Customers by Acquisition Channel")
plt.xlabel("Acquisition Channel")
plt.ylabel("Number of Customers")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "customer_acquisition.png",
    dpi=300
)

plt.show()


# ------------------------------------------------------------
# 12. Age distribution
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.hist(
    df["age"],
    bins=20
)

plt.title("Customer Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Customers")

plt.tight_layout()

plt.savefig(
    "customer_age_distribution.png",
    dpi=300
)

plt.show()


print("\n" + "=" * 60)
print("CUSTOMER EDA COMPLETED")
print("=" * 60)