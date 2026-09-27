import pandas as pd

# ============================================================
# STEP 3.2 - CLEAN TRANSACTIONS DATA
# ============================================================

# Load original dataset
transactions = pd.read_csv("data/transactions.csv")

print("=" * 60)
print("TRANSACTIONS DATA CLEANING")
print("=" * 60)

# ------------------------------------------------------------
# 1. Original shape
# ------------------------------------------------------------
print("\nOriginal Shape:")
print(transactions.shape)

# ------------------------------------------------------------
# 2. Remove duplicate rows
# ------------------------------------------------------------
duplicate_rows = transactions.duplicated().sum()

print("\nDuplicate rows found:", duplicate_rows)

transactions = transactions.drop_duplicates()

# ------------------------------------------------------------
# 3. Remove duplicate transaction IDs
# ------------------------------------------------------------
duplicate_ids = transactions["transaction_id"].duplicated().sum()

print("Duplicate transaction IDs:", duplicate_ids)

transactions = transactions.drop_duplicates(
    subset="transaction_id"
)

# ------------------------------------------------------------
# 4. Convert transaction date
# ------------------------------------------------------------
transactions["transaction_date"] = pd.to_datetime(
    transactions["transaction_date"],
    errors="coerce"
)

# ------------------------------------------------------------
# 5. Check missing values
# ------------------------------------------------------------
print("\nMissing values before cleaning:")
print(transactions.isnull().sum())

# ------------------------------------------------------------
# 6. Remove rows with missing important IDs
# ------------------------------------------------------------
transactions = transactions.dropna(
    subset=[
        "transaction_id",
        "customer_id",
        "product_id"
    ]
)

# ------------------------------------------------------------
# 7. Handle missing quantity
# ------------------------------------------------------------
transactions["quantity"] = transactions["quantity"].fillna(0)

# ------------------------------------------------------------
# 8. Handle missing price
# ------------------------------------------------------------
transactions["price"] = transactions["price"].fillna(
    transactions["price"].median()
)

# ------------------------------------------------------------
# 9. Handle missing discount
# ------------------------------------------------------------
transactions["discount"] = transactions["discount"].fillna(0)

# ------------------------------------------------------------
# 10. Remove invalid quantities
# ------------------------------------------------------------
transactions = transactions[
    transactions["quantity"] > 0
]

# ------------------------------------------------------------
# 11. Remove invalid prices
# ------------------------------------------------------------
transactions = transactions[
    transactions["price"] >= 0
]

# ------------------------------------------------------------
# 12. Correct discount values
# ------------------------------------------------------------
transactions.loc[
    transactions["discount"] < 0,
    "discount"
] = 0

# If discount is greater than 100%, cap it at 100%
transactions.loc[
    transactions["discount"] > 100,
    "discount"
] = 100

# ------------------------------------------------------------
# 13. Check invalid dates
# ------------------------------------------------------------
print("\nInvalid transaction dates:",
      transactions["transaction_date"].isna().sum())

# Remove transactions with invalid dates
transactions = transactions.dropna(
    subset=["transaction_date"]
)

# ------------------------------------------------------------
# 14. Create Revenue column
# ------------------------------------------------------------
transactions["revenue"] = (
    transactions["quantity"]
    * transactions["price"]
    * (1 - transactions["discount"] / 100)
)

# ------------------------------------------------------------
# 15. Final missing values
# ------------------------------------------------------------
print("\nMissing values after cleaning:")
print(transactions.isnull().sum())

# ------------------------------------------------------------
# 16. Final shape
# ------------------------------------------------------------
print("\nFinal Shape:")
print(transactions.shape)

# ------------------------------------------------------------
# 17. Revenue summary
# ------------------------------------------------------------
print("\nREVENUE SUMMARY")
print("-" * 60)

print("Total Revenue:",
      transactions["revenue"].sum())

print("Average Revenue per Transaction:",
      transactions["revenue"].mean())

# ------------------------------------------------------------
# 18. Save cleaned dataset
# ------------------------------------------------------------
transactions.to_csv(
    "data/clean_transactions.csv",
    index=False
)

print("\nCleaned dataset saved successfully!")
print("File: data/clean_transactions.csv")

print("=" * 60)
print("TRANSACTION CLEANING COMPLETED")
print("=" * 60)