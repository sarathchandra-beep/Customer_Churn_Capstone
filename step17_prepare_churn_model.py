import pandas as pd
from sklearn.model_selection import train_test_split

print("=" * 70)
print("STEP 8.2 - CHURN MODEL PREPARATION")
print("=" * 70)

# ------------------------------------------------------------
# 1. Load dataset
# ------------------------------------------------------------

df = pd.read_csv(
    "data/churn_modeling_dataset.csv"
)

print("\nDataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())


# ------------------------------------------------------------
# 2. Find churn target
# ------------------------------------------------------------

possible_targets = [
    "churn",
    "Churn",
    "churned",
    "Churned",
    "attrition",
    "Attrition"
]

target = None

for column in possible_targets:
    if column in df.columns:
        target = column
        break

if target is None:
    raise ValueError(
        "Churn target column was not found. "
        "Check the column names in your dataset."
    )

print("\nTarget column:")
print(target)


# ------------------------------------------------------------
# 3. Check target values
# ------------------------------------------------------------

print("\nTarget values:")
print(df[target].value_counts(dropna=False))


# ------------------------------------------------------------
# 4. Convert target to 0/1
# ------------------------------------------------------------

if df[target].dtype == "object":

    df[target] = (
        df[target]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({
            "yes": 1,
            "no": 0,
            "true": 1,
            "false": 0,
            "churn": 1,
            "not churn": 0,
            "churned": 1,
            "active": 0
        })
    )

else:

    # Make sure numeric target is integer
    df[target] = pd.to_numeric(
        df[target],
        errors="coerce"
    )


# Remove rows where target could not be interpreted
df = df.dropna(subset=[target])

df[target] = df[target].astype(int)


# ------------------------------------------------------------
# 5. Check target distribution
# ------------------------------------------------------------

print("\nCHURN DISTRIBUTION")
print("-" * 70)

print(df[target].value_counts())

print("\nChurn percentage:")

print(
    df[target].mean() * 100,
    "%"
)


# ------------------------------------------------------------
# 6. Remove ID columns
# ------------------------------------------------------------

id_columns = []

for column in df.columns:

    column_lower = column.lower()

    if (
        column_lower == "customer_id"
        or column_lower.endswith("_id")
        or column_lower == "id"
    ):
        id_columns.append(column)

print("\nID columns removed:")
print(id_columns)

df = df.drop(
    columns=id_columns,
    errors="ignore"
)


# ------------------------------------------------------------
# 7. Separate X and y
# ------------------------------------------------------------

X = df.drop(
    columns=[target]
)

y = df[target]


# ------------------------------------------------------------
# 8. Handle missing values
# ------------------------------------------------------------

numeric_columns = X.select_dtypes(
    include=["int64", "float64"]
).columns

categorical_columns = X.select_dtypes(
    include=["object", "category", "bool"]
).columns


# Numeric missing values
for column in numeric_columns:

    X[column] = X[column].fillna(
        X[column].median()
    )


# Categorical missing values
for column in categorical_columns:

    X[column] = X[column].fillna(
        "Unknown"
    )


# ------------------------------------------------------------
# 9. Convert categorical variables
# ------------------------------------------------------------

X = pd.get_dummies(
    X,
    columns=categorical_columns,
    drop_first=True
)


# ------------------------------------------------------------
# 10. Convert boolean columns
# ------------------------------------------------------------

X = X.astype(float)


# ------------------------------------------------------------
# 11. Train-test split
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ------------------------------------------------------------
# 12. Display results
# ------------------------------------------------------------

print("\nTRAINING DATA")
print("-" * 70)

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)


print("\nTEST DATA")
print("-" * 70)

print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


print("\nFinal number of features:")
print(X_train.shape[1])


# ------------------------------------------------------------
# 13. Save prepared datasets
# ------------------------------------------------------------

X_train.to_csv(
    "data/X_train.csv",
    index=False
)

X_test.to_csv(
    "data/X_test.csv",
    index=False
)

y_train.to_csv(
    "data/y_train.csv",
    index=False
)

y_test.to_csv(
    "data/y_test.csv",
    index=False
)


print("\nPrepared datasets saved:")
print("data/X_train.csv")
print("data/X_test.csv")
print("data/y_train.csv")
print("data/y_test.csv")


print("\n" + "=" * 70)
print("CHURN DATA PREPARATION COMPLETED")
print("=" * 70)