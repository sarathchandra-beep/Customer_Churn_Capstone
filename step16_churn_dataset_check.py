import pandas as pd

print("=" * 70)
print("STEP 8 - CHURN MODELING DATASET CHECK")
print("=" * 70)

# Load churn modeling dataset
df = pd.read_csv(
    "data/churn_modeling_dataset.csv"
)

# ------------------------------------------------------------
# 1. Dataset shape
# ------------------------------------------------------------

print("\nDATASET SHAPE")
print("-" * 70)

print("Rows:", df.shape[0])
print("Columns:", df.shape[1])


# ------------------------------------------------------------
# 2. Column names
# ------------------------------------------------------------

print("\nCOLUMN NAMES")
print("-" * 70)

print(df.columns.tolist())


# ------------------------------------------------------------
# 3. Data types
# ------------------------------------------------------------

print("\nDATA TYPES")
print("-" * 70)

print(df.dtypes)


# ------------------------------------------------------------
# 4. Missing values
# ------------------------------------------------------------

print("\nMISSING VALUES")
print("-" * 70)

print(df.isnull().sum())


# ------------------------------------------------------------
# 5. Duplicate rows
# ------------------------------------------------------------

print("\nDUPLICATE ROWS")
print("-" * 70)

print("Duplicate rows:", df.duplicated().sum())


# ------------------------------------------------------------
# 6. First five rows
# ------------------------------------------------------------

print("\nFIRST 5 ROWS")
print("-" * 70)

print(df.head())


# ------------------------------------------------------------
# 7. Statistical summary
# ------------------------------------------------------------

print("\nSTATISTICAL SUMMARY")
print("-" * 70)

print(df.describe(include="all"))


# ------------------------------------------------------------
# 8. Check categorical columns
# ------------------------------------------------------------

print("\nCATEGORICAL COLUMNS")
print("-" * 70)

categorical_columns = df.select_dtypes(
    include="object"
).columns

for column in categorical_columns:

    print(f"\n--- {column} ---")
    print(df[column].value_counts())


# ------------------------------------------------------------
# 9. Look for possible churn columns
# ------------------------------------------------------------

print("\nPOSSIBLE CHURN COLUMNS")
print("-" * 70)

for column in df.columns:

    column_lower = column.lower()

    if (
        "churn" in column_lower
        or "attrition" in column_lower
        or "retained" in column_lower
    ):

        print(
            column,
            "->",
            df[column].unique()
        )


print("\n" + "=" * 70)
print("CHURN DATASET CHECK COMPLETED")
print("=" * 70)