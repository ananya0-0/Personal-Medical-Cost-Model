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

#PRE-PROCESSING
df_model = df.copy()
 
# Label Encoding for categorical columns
le_sex    = LabelEncoder()
le_smoker = LabelEncoder()
le_region = LabelEncoder()
 
df_model["sex"]    = le_sex.fit_transform(df_model["sex"])       # female=0, male=1
df_model["smoker"] = le_smoker.fit_transform(df_model["smoker"]) # no=0, yes=1
df_model["region"] = le_region.fit_transform(df_model["region"]) # 0-3
 
print("\n" + "=" * 60)
print("PRE-PROCESSING")
print("=" * 60)
print("Encoded dataset (first 5 rows):")
print(df_model.head())
 
# Correlation heat-map
plt.figure(figsize=(8, 6))
sns.heatmap(df_model.corr(), annot=True, fmt=".2f",
            cmap="coolwarm", linewidths=0.5)
plt.title("Feature Correlation Heat-Map")
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=150)
plt.show()
print("\n[Correlation heatmap saved → correlation_heatmap.png]")
 
# Feature / Target split
X = df_model.drop("charges", axis=1)
y = df_model["charges"]
 
# Train-Test split  (80 / 20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)
 
print(f"\nTraining samples : {X_train.shape[0]}")
print(f"Testing  samples : {X_test.shape[0]}")
 
# 4.  MODEL TRAINING  (5 algorithms)

models = {
    "Linear Regression"          : LinearRegression(),
    "Ridge Regression"           : Ridge(alpha=1.0),
    "Lasso Regression"           : Lasso(alpha=1.0),
    "Random Forest Regressor"    : RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=100, random_state=42),
}
 
results = {}
print("\n" + "=" * 60)
print("MODEL TRAINING & EVALUATION")
print("=" * 60)
 
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
 
    mae  = mean_absolute_error(y_test, y_pred)
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_test, y_pred)
    cv   = cross_val_score(model, X_train, y_train, cv=5, scoring="r2").mean()
 
    results[name] = {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2, "CV_R2": cv}
 
    print(f"\n{'─'*50}")
    print(f"  {name}")
    print(f"{'─'*50}")
    print(f"  MAE  : {mae:>12,.2f}")
    print(f"  RMSE : {rmse:>12,.2f}")
    print(f"  R²   : {r2:>12.4f}")
    print(f"  CV R²: {cv:>12.4f}  (5-fold)")

# BEST MODEL  (highest R²)
best_name = max(results, key=lambda k: results[k]["R2"])
best_model = models[best_name]
print(f"\n✅  Best Model → {best_name}  (R² = {results[best_name]['R2']:.4f})")
 
# 6.  RESULT VISUALIZATIONS
y_pred_best = best_model.predict(X_test)
 
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(f"Result Analysis – {best_name}", fontsize=13, fontweight="bold")
 
# 6a. Actual vs Predicted scatter
axes[0].scatter(y_test, y_pred_best, alpha=0.4, color="steelblue", s=20)
lim = [y_test.min(), y_test.max()]
axes[0].plot(lim, lim, "r--", lw=2)
axes[0].set_title("Actual vs Predicted Charges")
axes[0].set_xlabel("Actual Charges (USD)")
axes[0].set_ylabel("Predicted Charges (USD)")
 
# 6b. Residual plot
residuals = y_test - y_pred_best
axes[1].scatter(y_pred_best, residuals, alpha=0.4, color="coral", s=20)
axes[1].axhline(0, color="black", lw=1.5)
axes[1].set_title("Residual Plot")
axes[1].set_xlabel("Predicted Charges")
axes[1].set_ylabel("Residuals")
 
# 6c. R² bar comparison
model_names  = list(results.keys())
r2_scores    = [results[m]["R2"] for m in model_names]
short_labels = ["Lin.Reg", "Ridge", "Lasso", "Rand.Forest", "Grad.Boost"]
colors       = ["steelblue" if m != best_name else "darkorange" for m in model_names]
axes[2].barh(short_labels, r2_scores, color=colors, edgecolor="white")
axes[2].set_xlim(0, 1.0)
axes[2].set_title("R² Comparison (orange = best)")
axes[2].set_xlabel("R² Score")
 
plt.tight_layout()
plt.savefig("results_plots.png", dpi=150)
plt.show()
print("\n[Result plots saved → results_plots.png]")

#SAMPLE PREDICTION
print("\n" + "=" * 55)
print("SAMPLE PREDICTION")
print("=" * 55)
# age=35, sex=male(1), bmi=28.5, children=2, smoker=no(0), region=southwest(3)
sample = pd.DataFrame([[35, 1, 28.5, 2, 0, 3]], columns=X.columns)
sample_scaled = scaler.transform(sample)
predicted = model.predict(sample_scaled)[0]
print("Input  : age=35, sex=male, bmi=28.5, children=2, smoker=no, region=southwest")
print(f"Predicted Medical Cost : ${predicted:,.2f}")
 
print("\n[Script completed successfully!]")
 