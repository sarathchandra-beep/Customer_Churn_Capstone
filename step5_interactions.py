import pandas as pd

# Load interactions dataset
interactions = pd.read_csv("data/interactions.csv")

print("=" * 60)
print("INTERACTIONS DATASET INSPECTION")
print("=" * 60)

# 1. First 5 rows
print("\nFIRST 5 ROWS")
print("-" * 60)
print(interactions.head())

# 2. Dataset shape
print("\nDATASET SHAPE")
print("-" * 60)
print("Rows:", interactions.shape[0])
print("Columns:", interactions.shape[1])

# 3. Column names
print("\nCOLUMN NAMES")
print("-" * 60)
print(interactions.columns.tolist())

# 4. Data types
print("\nDATA TYPES")
print("-" * 60)
print(interactions.dtypes)

# 5. Missing values
print("\nMISSING VALUES")
print("-" * 60)
print(interactions.isnull().sum())

# 6. Duplicate rows
print("\nDUPLICATE ROWS")
print("-" * 60)
print("Duplicate rows:", interactions.duplicated().sum())

# 7. Statistical summary
print("\nSTATISTICAL SUMMARY")
print("-" * 60)
print(interactions.describe())

# 8. Unique interaction IDs
print("\nINTERACTION ID CHECK")
print("-" * 60)

print(
    "Total interaction records:",
    len(interactions)
)

print(
    "Unique interaction IDs:",
    interactions["interaction_id"].nunique()
)

print(
    "Duplicate interaction IDs:",
    interactions["interaction_id"].duplicated().sum()
)

# 9. Unique customers
print("\nUNIQUE CUSTOMERS")
print("-" * 60)

print(
    "Unique customers:",
    interactions["customer_id"].nunique()
)

# 10. Unique products
print("\nUNIQUE PRODUCTS")
print("-" * 60)

print(
    "Unique products:",
    interactions["product_id"].nunique()
)

# 11. Interaction types
print("\nINTERACTION TYPES")
print("-" * 60)

print(
    interactions["interaction_type"].value_counts()
)

# 12. Channels
print("\nCHANNELS")
print("-" * 60)

print(
    interactions["channel"].value_counts()
)

# 13. Convert timestamp
interactions["interaction_timestamp"] = pd.to_datetime(
    interactions["interaction_timestamp"],
    errors="coerce"
)

# 14. Date range
print("\nINTERACTION DATE RANGE")
print("-" * 60)

print(
    "Minimum timestamp:",
    interactions["interaction_timestamp"].min()
)

print(
    "Maximum timestamp:",
    interactions["interaction_timestamp"].max()
)

# 15. Invalid timestamps
print("\nINVALID TIMESTAMPS")
print("-" * 60)

print(
    "Invalid timestamps:",
    interactions["interaction_timestamp"].isna().sum()
)

print("\n" + "=" * 60)
print("INTERACTION INSPECTION COMPLETED")
print("=" * 60)