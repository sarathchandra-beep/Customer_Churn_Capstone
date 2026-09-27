import pandas as pd

interactions = pd.read_csv("data/interactions.csv")

print("=" * 60)
print("INTERACTIONS DATASET INSPECTION")
print("=" * 60)

print("\nFIRST 5 ROWS")
print("-" * 60)
print(interactions.head())


print("\nDATASET SHAPE")
print("-" * 60)
print("Rows:", interactions.shape[0])
print("Columns:", interactions.shape[1])


print("\nCOLUMN NAMES")
print("-" * 60)
print(interactions.columns.tolist())


print("\nDATA TYPES")
print("-" * 60)
print(interactions.dtypes)


print("\nMISSING VALUES")
print("-" * 60)
print(interactions.isnull().sum())


print("\nDUPLICATE ROWS")
print("-" * 60)
print("Duplicate rows:", interactions.duplicated().sum())


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


print("\nUNIQUE CUSTOMERS")
print("-" * 60)

print(
    "Unique customers:",
    interactions["customer_id"].nunique()
)


print("\nUNIQUE PRODUCTS")
print("-" * 60)

print(
    "Unique products:",
    interactions["product_id"].nunique()
)


print("\nINTERACTION TYPES")
print("-" * 60)

print(
    interactions["interaction_type"].value_counts()
)


print("\nCHANNELS")
print("-" * 60)

print(
    interactions["channel"].value_counts()
)


interactions["interaction_timestamp"] = pd.to_datetime(
    interactions["interaction_timestamp"],
    errors="coerce"
)


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

print("\nINVALID TIMESTAMPS")
print("-" * 60)

print(
    "Invalid timestamps:",
    interactions["interaction_timestamp"].isna().sum()
)

print("\n" + "=" * 60)
print("INTERACTION INSPECTION COMPLETED")
print("=" * 60)