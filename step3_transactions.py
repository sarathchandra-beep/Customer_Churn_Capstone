import pandas as pd

transactions = pd.read_csv("data/transactions.csv")

print("=" * 60)
print("Transaction Inspetion")
print("=" * 60)

print("\nFirst 5 Rows")
print("-" * 60)
print(transactions.head())

print("\nDataset Shape")
print("-" * 60)
print("Rows:", transactions.shape[0])
print("Columns:", transactions.shape[1])

# 3. Column names
print("\nColumn Names")
print("-" * 60)
print(transactions.columns.tolist())

# 4. Data types
print("\nData Types")
print("-" * 60)
print(transactions.dtypes)

# 5. Missing values
print("\nMissing Values")
print("-" * 60)
print(transactions.isnull().sum())

# 6. Duplicate rows
print("\nDuplicate Rows")
print("-" * 60)
print("Duplicate rows:", transactions.duplicated().sum())

# 7. Statistical summary
print("\nStatistical Summary")
print("-" * 60)
print(transactions.describe())

# 8. Convert transaction date
transactions["transaction_date"] = pd.to_datetime(
    transactions["transaction_date"],
    errors="coerce"
)

# 9. Transaction date range
print("\nTransaction Date")
print("-" * 60)
print("Minimum date:", transactions["transaction_date"].min())
print("Maximum date:", transactions["transaction_date"].max())

# 10. Unique customers
print("\nUnique Customers")
print("-" * 60)
print("Unique customers:",
      transactions["customer_id"].nunique())

# 11. Unique products
print("\nUnique Products")
print("-" * 60)
print("Unique products:",
      transactions["product_id"].nunique())

# 12. Unique transactions
print("\nUnique")
print("-" * 60)
print("Unique transactions:",
      transactions["transaction_id"].nunique())

# 13. Quantity analysis
print("\nQuantity")
print("-" * 60)
print(transactions["quantity"].describe())

# 14. Price analysis
print("\nPrice")
print("-" * 60)
print(transactions["price"].describe())

# 15. Discount analysis
print("\nDiscount")
print("-" * 60)
print(transactions["discount"].describe())

# 16. Check negative values
print("\n Negative check")
print("-" * 60)

print("Negative quantity:",
      (transactions["quantity"] < 0).sum())

print("Negative price:",
      (transactions["price"] < 0).sum())

print("Negative discount:",
      (transactions["discount"] < 0).sum())

# 17. Check invalid dates
print("\nInvalid Dates")
print("-" * 60)

print("Invalid transaction dates:",
      transactions["transaction_date"].isna().sum())

print("\n" + "=" * 60)
print("TRANSACTION INSPECTION COMPLETED")
print("=" * 60)