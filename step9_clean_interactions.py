import pandas as pd

# Load interactions dataset
interactions = pd.read_csv("data/interactions.csv")

print("=" * 60)
print("INTERACTIONS DATA CLEANING")
print("=" * 60)

# 1. Original shape
print("\nOriginal Shape:")
print(interactions.shape)

# 2. Remove duplicate rows
interactions = interactions.drop_duplicates()

# 3. Remove duplicate interaction IDs
interactions = interactions.drop_duplicates(
    subset="interaction_id"
)

# 4. Convert timestamp
interactions["interaction_timestamp"] = pd.to_datetime(
    interactions["interaction_timestamp"],
    errors="coerce"
)

# 5. Check missing values
print("\nMissing values before cleaning:")
print(interactions.isnull().sum())

# 6. Remove rows with missing IDs
interactions = interactions.dropna(
    subset=[
        "interaction_id",
        "customer_id"
    ]
)

# 7. Fill missing interaction type
interactions["interaction_type"] = (
    interactions["interaction_type"]
    .fillna("Unknown")
)

# 8. Fill missing channel
interactions["channel"] = (
    interactions["channel"]
    .fillna("Unknown")
)

# 9. Remove invalid timestamps
interactions = interactions.dropna(
    subset=["interaction_timestamp"]
)

# 10. Final missing values
print("\nMissing values after cleaning:")
print(interactions.isnull().sum())

# 11. Final shape
print("\nFinal Shape:")
print(interactions.shape)

# 12. Interaction type summary
print("\nINTERACTION TYPES")
print("-" * 60)
print(
    interactions["interaction_type"].value_counts()
)

# 13. Channel summary
print("\nCHANNELS")
print("-" * 60)
print(
    interactions["channel"].value_counts()
)

# 14. Save cleaned dataset
interactions.to_csv(
    "data/clean_interactions.csv",
    index=False
)

print("\nCleaned dataset saved:")
print("data/clean_interactions.csv")

print("=" * 60)
print("INTERACTION CLEANING COMPLETED")
print("=" * 60)