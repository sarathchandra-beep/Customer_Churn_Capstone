import pandas as pd


products = pd.read_csv("data/products.csv")

print("=" * 60)
print("PRODUCTS DATA CLEANING")
print("=" * 60)


print("\nOriginal Shape:")
print(products.shape)


products = products.drop_duplicates()


products = products.drop_duplicates(
    subset="product_id"
)


products["launch_date"] = pd.to_datetime(
    products["launch_date"],
    errors="coerce"
)


print("\nMissing values before cleaning:")
print(products.isnull().sum())


products["category"] = products["category"].fillna("Unknown")


products["price"] = products["price"].fillna(
    products["price"].median()
)

products["cost"] = products["cost"].fillna(
    products["cost"].median()
)


products = products[
    products["price"] >= 0
]


products = products[
    products["cost"] >= 0
]


products["profit_per_unit"] = (
    products["price"] - products["cost"]
)


products["profit_margin"] = (
    products["profit_per_unit"]
    / products["price"]
) * 100


products.loc[
    products["price"] == 0,
    "profit_margin"
] = 0


products = products.dropna(
    subset=["launch_date"]
)


print("\nMissing values after cleaning:")
print(products.isnull().sum())


print("\nFinal Shape:")
print(products.shape)


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


products.to_csv(
    "data/clean_products.csv",
    index=False
)

print("\nCleaned dataset saved:")
print("data/clean_products.csv")

print("=" * 60)
print("PRODUCT CLEANING COMPLETED")
print("=" * 60)