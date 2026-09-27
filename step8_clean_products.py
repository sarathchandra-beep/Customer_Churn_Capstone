import pandas as pd

# Load products dataset
products = pd.read_csv("data/products.csv")

print("=" * 60)
print("PRODUCTS DATA CLEANING")
print("=" * 60)

# 1. Original shape
print("\nOriginal Shape:")
print(products.shape)

# 2. Remove duplicate rows
products = products.drop_duplicates()

# 3. Remove duplicate product IDs
products = products.drop_duplicates(
    subset="product_id"
)

# 4. Convert launch date
products["launch_date"] = pd.to_datetime(
    products["launch_date"],
    errors="coerce"
)

# 5. Check missing values
print("\nMissing values before cleaning:")
print(products.isnull().sum())

# 6. Fill missing category
products["category"] = products["category"].fillna("Unknown")

# 7. Fill missing price and cost with median
products["price"] = products["price"].fillna(
    products["price"].median()
)

products["cost"] = products["cost"].fillna(
    products["cost"].median()
)

# 8. Remove invalid prices
products = products[
    products["price"] >= 0
]

# 9. Remove invalid costs
products = products[
    products["cost"] >= 0
]

# 10. Create profit column
products["profit_per_unit"] = (
    products["price"] - products["cost"]
)

# 11. Create profit margin
products["profit_margin"] = (
    products["profit_per_unit"]
    / products["price"]
) * 100

# Avoid division problems where price is zero
products.loc[
    products["price"] == 0,
    "profit_margin"
] = 0

# 12. Remove invalid launch dates
products = products.dropna(
    subset=["launch_date"]
)

# 13. Final missing values
print("\nMissing values after cleaning:")
print(products.isnull().sum())

# 14. Final shape
print("\nFinal Shape:")
print(products.shape)

# 15. Profit summary
print("\nPROFIT SUMMARY")
print("-" * 60)

print(
    "Total potential profit per unit:",
    products["profit_per_unit"].sum()
)

print(
    "Average profit per unit:",
    products["profit_per_unit"].mean()
)

print(
    "Average profit margin:",
    products["profit_margin"].mean()
)

# 16. Save cleaned dataset
products.to_csv(
    "data/clean_products.csv",
    index=False
)

print("\nCleaned dataset saved:")
print("data/clean_products.csv")

print("=" * 60)
print("PRODUCT CLEANING COMPLETED")
print("=" * 60)