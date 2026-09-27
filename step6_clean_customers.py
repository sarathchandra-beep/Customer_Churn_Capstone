import pandas as pd


customers = pd.read_csv("data/customers.csv")

print("=" * 50)
print("CUSTOMER DATA CLEANING")
print("=" * 60)


print("\nOriginal Shape:")
print(customers.shape)


customers = customers.drop_duplicates()


customers = customers.drop_duplicates(
    subset="customer_id"
)


customers["signup_date"] = pd.to_datetime(
    customers["signup_date"],
    errors="coerce"
)


print("\nMissing values before cleaning:")
print(customers.isnull().sum())


categorical_columns = [
    "customer_segment",
    "location",
    "acquisition_channel",
    "gender"
]

for column in categorical_columns:
    customers[column] = customers[column].fillna("Unknown")


customers["age"] = customers["age"].fillna(
    customers["age"].median()
)


customers.loc[
    (customers["age"] < 18) | (customers["age"] > 100),
    "age"
] = customers["age"].median()


print("\nMissing values after cleaning:")
print(customers.isnull().sum())


print("\nFinal Shape:")
print(customers.shape)


customers.to_csv(
    "data/clean_customers.csv",
    index=False
)

print("\nCleaned dataset saved successfully!")
print("File: data/clean_customers.csv")

print("=" * 80)
print("CUSTOMER CLEANING COMPLETED")
print("=" * 40)