import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Load the datasets
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

# =============================
# 🔍 Basic Preview of the Data
# =============================
print("✅ Preview of Train Data:\n", train_df.head())
print("\n✅ Preview of Test Data:\n", test_df.head())

# =============================
# 🔍 Null Value Check
# =============================
print("\n🧼 Null Values in Train:\n", train_df.isnull().sum())
print("\n🧼 Null Values in Test:\n", test_df.isnull().sum())

# =============================
# 🔍 Column Names
# =============================
print("\n🧾 Columns in Train:", train_df.columns.tolist())
print("\n🧾 Columns in Test:", test_df.columns.tolist())

# =============================
# 🔍 Shape of the Data
# =============================
print("\n🔢 Shape of Train:", train_df.shape)
print("🔢 Shape of Test:", test_df.shape)

# =============================
# 🔍 Data Types
# =============================
print("\n📌 Data Types in Train:\n", train_df.dtypes)
print("\n📌 Data Types in Test:\n", test_df.dtypes)

# =============================
# 🔍 Statistical Summary (Numerical Columns Only)
# =============================
print("\n📊 Statistical Summary of Train:\n", train_df.describe())
print("\n📊 Statistical Summary of Test:\n", test_df.describe())