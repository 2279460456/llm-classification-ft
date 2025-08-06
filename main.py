# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('data'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

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