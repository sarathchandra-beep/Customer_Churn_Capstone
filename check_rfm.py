import pandas as pd

rfm = pd.read_csv("data/customer_rfm.csv")

print("=" * 60)
print("RFM DATASET CHECK")
print("=" * 60)

print("\nColumns in customer_rfm.csv:")
print(rfm.columns.tolist())

print("\nShape:")
print(rfm.shape)

print("\nFirst 5 rows:")
print(rfm.head())