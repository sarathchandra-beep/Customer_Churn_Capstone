import pandas as pd

print("=" * 70)
print("STEP 4 - DATA INTEGRATION")
print("=" * 70)

# ------------------------------------------------------------
# 1. Load cleaned datasets
# ------------------------------------------------------------

customers = pd.read_csv("data/clean_customers.csv")
transactions = pd.read_csv("data/clean_transactions.csv")
products = pd.read_csv("data/clean_products.csv")
interactions = pd.read_csv("data/clean_interactions.csv")

print("\nDatasets loaded successfully!")

print("\nCustomers:", customers.shape)
print("Transactions:", transactions.shape)
print("Products:", products.shape)
print("Interactions:", interactions.shape)


# ------------------------------------------------------------
# 2. Convert date columns
# ------------------------------------------------------------

customers["signup_date"] = pd.to_datetime(
    customers["signup_date"]
)

transactions["transaction_date"] = pd.to_datetime(
    transactions["transaction_date"]
)

products["launch_date"] = pd.to_datetime(
    products["launch_date"]
)

interactions["interaction_timestamp"] = pd.to_datetime(
    interactions["interaction_timestamp"]
)


# ------------------------------------------------------------
# 3. Merge transactions with products
# ------------------------------------------------------------

transaction_product = transactions.merge(
    products,
    on="product_id",
    how="left",
    suffixes=("_transaction", "_product")
)

print("\nTransaction + Product:")
print(transaction_product.shape)


# ------------------------------------------------------------
# 4. Calculate transaction-level profit
# ------------------------------------------------------------

transaction_product["profit"] = (
    transaction_product["revenue"]
    - (
        transaction_product["quantity"]
        * transaction_product["cost"]
    )
)

print("\nProfit column created.")


# ------------------------------------------------------------
# 5. Merge customers with transaction data
# ------------------------------------------------------------

customer_transactions = customers.merge(
    transaction_product,
    on="customer_id",
    how="left"
)

print("\nCustomer + Transaction + Product:")
print(customer_transactions.shape)


# ------------------------------------------------------------
# 6. Create customer transaction summary
# ------------------------------------------------------------

customer_transaction_summary = (
    transaction_product
    .groupby("customer_id")
    .agg(
        total_orders=("transaction_id", "nunique"),
        total_quantity=("quantity", "sum"),
        total_revenue=("revenue", "sum"),
        total_profit=("profit", "sum"),
        average_order_value=("revenue", "mean"),
        last_purchase_date=("transaction_date", "max"),
        first_purchase_date=("transaction_date", "min")
    )
    .reset_index()
)

print("\nCustomer transaction summary:")
print(customer_transaction_summary.head())


# ------------------------------------------------------------
# 7. Create customer interaction summary
# ------------------------------------------------------------

customer_interaction_summary = (
    interactions
    .groupby("customer_id")
    .agg(
        total_interactions=("interaction_id", "count"),
        unique_products_viewed=("product_id", "nunique"),
        last_interaction_date=("interaction_timestamp", "max"),
        first_interaction_date=("interaction_timestamp", "min")
    )
    .reset_index()
)

print("\nCustomer interaction summary:")
print(customer_interaction_summary.head())


# ------------------------------------------------------------
# 8. Create interaction-type counts
# ------------------------------------------------------------

interaction_type_counts = pd.crosstab(
    interactions["customer_id"],
    interactions["interaction_type"]
).reset_index()

print("\nInteraction type summary:")
print(interaction_type_counts.head())


# ------------------------------------------------------------
# 9. Merge customer information
# ------------------------------------------------------------

customer_analytics = customers.merge(
    customer_transaction_summary,
    on="customer_id",
    how="left"
)

customer_analytics = customer_analytics.merge(
    customer_interaction_summary,
    on="customer_id",
    how="left"
)

customer_analytics = customer_analytics.merge(
    interaction_type_counts,
    on="customer_id",
    how="left"
)


# ------------------------------------------------------------
# 10. Fill missing transaction/interaction values
# ------------------------------------------------------------

numeric_columns = [
    "total_orders",
    "total_quantity",
    "total_revenue",
    "total_profit",
    "average_order_value",
    "total_interactions",
    "unique_products_viewed"
]

for column in numeric_columns:

    if column in customer_analytics.columns:

        customer_analytics[column] = (
            customer_analytics[column].fillna(0)
        )


# ------------------------------------------------------------
# 11. Display final dataset
# ------------------------------------------------------------

print("\nFINAL CUSTOMER ANALYTICS DATASET")
print("=" * 70)

print("Rows:", customer_analytics.shape[0])
print("Columns:", customer_analytics.shape[1])

print("\nColumns:")
print(customer_analytics.columns.tolist())

print("\nFirst 5 rows:")
print(customer_analytics.head())


# ------------------------------------------------------------
# 12. Save customer-level dataset
# ------------------------------------------------------------

customer_analytics.to_csv(
    "data/customer_analytics.csv",
    index=False
)

print("\nCustomer analytics dataset saved:")
print("data/customer_analytics.csv")


# ------------------------------------------------------------
# 13. Save transaction-level integrated dataset
# ------------------------------------------------------------

transaction_product.to_csv(
    "data/transaction_product_analysis.csv",
    index=False
)

print("\nTransaction-product dataset saved:")
print("data/transaction_product_analysis.csv")


print("\n" + "=" * 70)
print("DATA INTEGRATION COMPLETED")
print("=" * 70)