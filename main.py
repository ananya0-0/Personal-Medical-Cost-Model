import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv("insurance.csv")          # put insurance.csv in the same folder
print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"Shape : {df.shape}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nStatistical Summary:\n{df.describe()}")

#EXPLORATORY DATA ANALYSIS  (EDA)
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Exploratory Data Analysis – Medical Cost Dataset", fontsize=14, fontweight="bold")
 
# Distribution of charges
axes[0, 0].hist(df["charges"], bins=40, color="steelblue", edgecolor="white")
axes[0, 0].set_title("Distribution of Medical Charges")
axes[0, 0].set_xlabel("Charges (USD)")
axes[0, 0].set_ylabel("Frequency")
 
# Charges by smoker status
df.boxplot(column="charges", by="smoker", ax=axes[0, 1],
           boxprops=dict(color="steelblue"))
axes[0, 1].set_title("Charges by Smoker Status")
axes[0, 1].set_xlabel("Smoker")
axes[0, 1].set_ylabel("Charges (USD)")
 
# Age vs Charges
axes[0, 2].scatter(df["age"], df["charges"],
                   c=df["smoker"].map({"yes": "red", "no": "steelblue"}),
                   alpha=0.5, s=20)
axes[0, 2].set_title("Age vs Charges  (red = smoker)")
axes[0, 2].set_xlabel("Age")
axes[0, 2].set_ylabel("Charges (USD)")
 
# BMI vs Charges
axes[1, 0].scatter(df["bmi"], df["charges"],
                   c=df["smoker"].map({"yes": "red", "no": "steelblue"}),
                   alpha=0.5, s=20)
axes[1, 0].set_title("BMI vs Charges  (red = smoker)")
axes[1, 0].set_xlabel("BMI")
axes[1, 0].set_ylabel("Charges (USD)")
 
# Charges by region
df.boxplot(column="charges", by="region", ax=axes[1, 1])
axes[1, 1].set_title("Charges by Region")
axes[1, 1].set_xlabel("Region")
axes[1, 1].set_ylabel("Charges (USD)")
 
# Average charges by number of children
avg_children = df.groupby("children")["charges"].mean()
axes[1, 2].bar(avg_children.index, avg_children.values, color="steelblue", edgecolor="white")
axes[1, 2].set_title("Avg Charges by No. of Children")
axes[1, 2].set_xlabel("Children")
axes[1, 2].set_ylabel("Avg Charges (USD)")
 
plt.tight_layout()
plt.savefig("eda_plots.png", dpi=150)
plt.show()
print("\n[EDA plots saved → eda_plots.png]")