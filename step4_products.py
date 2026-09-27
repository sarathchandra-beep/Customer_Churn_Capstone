import pandas as pd

products = pd.read_csv("data/products.csv")

print("=" * 40)
print("PRODUCTS DATASET INSPECTION")
print("=" * 50)

print("\nFIRST 4 ROWS")
print("-" * 60)
print(products.head())

print("\nDATASET SHAPE")
print("-" * 70)
print("Rows:", products.shape[0])
print("Columns:", products.shape[1])

print("\nCOLUMN NAMES")
print("-" * 40)
print(products.columns.tolist())

print("\nDATA TYPES")
print("-" * 50)
print(products.dtypes)

print("\nMISSING VALUES")
print("-" * 60)
print(products.isnull().sum())

print("\nDUPLICATE ROWS")
print("-" * 80)
print("Duplicate rows:", products.duplicated().sum())

print("\nSTATISTICAL SUMMARY")
print("-" * 70)
print(products.describe())

print("\nPRODUCT ID CHECK")
print("-" * 60)
print("Total product records:", len(products))
print("Unique product IDs:", products["product_id"].nunique())

print(
    "Duplicate product IDs:",
    products["product_id"].duplicated().sum()
)

print("\nPRODUCT CATEGORIES")
print("-" * 40)
print(products["category"].value_counts())

print("\nPRICE ANALYSIS")
print("-" * 50)
print(products["price"].describe())

print("\nCOST ANALYSIS")
print("-" * 50)
print(products["cost"].describe())

print("\nNEGATIVE VALUES CHECK")
print("-" * 40)

print(
    "Negative prices:",
    (products["price"] < 0).sum()
)

print(
    "Negative costs:",
    (products["cost"] < 0).sum()
)

print("\nPRICE vs COST CHECK")
print("-" * 80)

print(
    "Products where cost > price:",
    (products["cost"] > products["price"]).sum()
)

products["launch_date"] = pd.to_datetime(
    products["launch_date"],
    errors="coerce"
)

print("\nLAUNCH DATE RANGE")
print("-" * 60)

print(
    "Minimum launch date:",
    products["launch_date"].min()
)

print(
    "Maximum launch date:",
    products["launch_date"].max()
)

print(
    "Invalid launch dates:",
    products["launch_date"].isna().sum()
)

print("\n" + "=" * 40)
print("PRODUCT INSPECTION COMPLETED")
print("=" * 40)