import pandas as pd

# Load dataset
customers = pd.read_csv("data/customers.csv")

print("=" * 60)
print("CUSTOMER DATA CLEANING")
print("=" * 60)

# 1. Check original shape
print("\nOriginal Shape:")
print(customers.shape)

# 2. Remove duplicate rows
customers = customers.drop_duplicates()

# 3. Remove duplicate customer IDs
customers = customers.drop_duplicates(
    subset="customer_id"
)

# 4. Convert signup date
customers["signup_date"] = pd.to_datetime(
    customers["signup_date"],
    errors="coerce"
)

# 5. Handle missing values
print("\nMissing values before cleaning:")
print(customers.isnull().sum())

# Fill missing categorical values
categorical_columns = [
    "customer_segment",
    "location",
    "acquisition_channel",
    "gender"
]

for column in categorical_columns:
    customers[column] = customers[column].fillna("Unknown")

# Fill missing age with median
customers["age"] = customers["age"].fillna(
    customers["age"].median()
)

# 6. Check invalid ages
customers.loc[
    (customers["age"] < 18) | (customers["age"] > 100),
    "age"
] = customers["age"].median()

# 7. Check missing dates
print("\nMissing values after cleaning:")
print(customers.isnull().sum())

# 8. Final shape
print("\nFinal Shape:")
print(customers.shape)

# 9. Save cleaned dataset
customers.to_csv(
    "data/clean_customers.csv",
    index=False
)

print("\nCleaned dataset saved successfully!")
print("File: data/clean_customers.csv")

print("=" * 60)
print("CUSTOMER CLEANING COMPLETED")
print("=" * 60)