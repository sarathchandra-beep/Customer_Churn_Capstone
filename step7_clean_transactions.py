import pandas as pd


transactions = pd.read_csv("data/transactions.csv")

print("=" * 60)
print("TRANSACTIONS DATA CLEANING")
print("=" * 60)


print("\nOriginal Shape:")
print(transactions.shape)


duplicate_rows = transactions.duplicated().sum()

print("\nDuplicate rows found:", duplicate_rows)

transactions = transactions.drop_duplicates()


duplicate_ids = transactions["transaction_id"].duplicated().sum()

print("Duplicate transaction IDs:", duplicate_ids)

transactions = transactions.drop_duplicates(
    subset="transaction_id"
)


transactions["transaction_date"] = pd.to_datetime(
    transactions["transaction_date"],
    errors="coerce"
)


print("\nMissing values before cleaning:")
print(transactions.isnull().sum())


transactions = transactions.dropna(
    subset=[
        "transaction_id",
        "customer_id",
        "product_id"
    ]
)


transactions["quantity"] = transactions["quantity"].fillna(0)


transactions["price"] = transactions["price"].fillna(
    transactions["price"].median()
)


transactions["discount"] = transactions["discount"].fillna(0)


transactions = transactions[
    transactions["quantity"] > 0
]


transactions = transactions[
    transactions["price"] >= 0
]


transactions.loc[
    transactions["discount"] < 0,
    "discount"
] = 0


transactions.loc[
    transactions["discount"] > 100,
    "discount"
] = 100


print("\nInvalid transaction dates:",
      transactions["transaction_date"].isna().sum())


transactions = transactions.dropna(
    subset=["transaction_date"]
)


transactions["revenue"] = (
    transactions["quantity"]
    * transactions["price"]
    * (1 - transactions["discount"] / 100)
)


print("\nMissing values after cleaning:")
print(transactions.isnull().sum())


print("\nFinal Shape:")
print(transactions.shape)


print("\nREVENUE SUMMARY")
print("-" * 60)

print("Total Revenue:",
      transactions["revenue"].sum())

print("Average Revenue per Transaction:",
      transactions["revenue"].mean())


transactions.to_csv(
    "data/clean_transactions.csv",
    index=False
)

print("\nCleaned dataset saved successfully!")
print("File: data/clean_transactions.csv")

print("=" * 60)
print("TRANSACTION CLEANING COMPLETED")
print("=" * 60)