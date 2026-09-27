import pandas as pd
customers = pd.read_csv("data/customers.csv")
print("=" * 40)
print("Customers Inspection")
print("=" * 40)

print("\nFirst 5 Rows")
print("-" * 30)
print(customers.head())

print("\nData Shape")
print("-" * 70)
print("Rows:", customers.shape[0])
print("Columns:", customers.shape[1])

print("\nColumn Names")
print("-" * 40)
print(customers.columns.tolist())

print("\nData Types")
print("-" * 40)
print(customers.dtypes)

print("\nMissing Values")
print("-" * 60)
print(customers.isnull().sum())

print("\nDuplicate Rows")
print("-" * 80)
print("Duplicate rows:", customers.duplicated().sum())

print("\nNumerical Summary")
print("-" * 50)
print(customers.describe())

print("\nCategorical Values")
print("-" * 60)

for col in customers.select_dtypes(include="object").columns:
    print(f"\n--- {col} ---")
    print(customers[col].value_counts())

print("\nCustomer ID")
print("-" * 80)

print("Total customer records:", len(customers))
print("Unique customer IDs:", customers["customer_id"].nunique())

print("Duplicate customer IDs:",
      customers["customer_id"].duplicated().sum())

print("\nAge")
print("-" * 60)

print("Minimum age:", customers["age"].min())
print("Maximum age:", customers["age"].max())

print("Invalid ages:",
      ((customers["age"] < 0) | (customers["age"] > 100)).sum())

print("\nSignup Date")
print("-" * 50)

customers["signup_date"] = pd.to_datetime(
    customers["signup_date"],
    errors="coerce"
)

print("Minimum signup date:",
      customers["signup_date"].min())

print("Maximum signup date:",
      customers["signup_date"].max())

print("Invalid signup dates:",
      customers["signup_date"].isna().sum())

print("\n" + "=" * 60)
print("Customer Inspection Completed")
print("=" * 60)