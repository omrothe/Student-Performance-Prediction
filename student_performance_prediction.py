"""
Week 6 Final AI Assignment: Student Performance Prediction
Run from the project root: python src/student_performance_prediction.py
Dataset is synthetic and intended for educational demonstration only.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "student_performance.csv"
OUT = ROOT / "outputs"
FIG = OUT / "figures"
OUT.mkdir(exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(DATA_PATH)
print("First records:\\n", df.head())
print("Last records:\\n", df.tail())
print("Shape:", df.shape)
print("Dtypes:\\n", df.dtypes)
print("Missing values:\\n", df.isna().sum())
print("Duplicate rows:", df.duplicated().sum())

df = df.drop_duplicates().copy()
for c in df.select_dtypes(include=np.number).columns:
    df[c] = df[c].fillna(df[c].median())
for c in df.select_dtypes(exclude=np.number).columns:
    df[c] = df[c].fillna(df[c].mode().iloc[0])
df.to_csv(OUT / "cleaned_dataset.csv", index=False)
print("Cleaned shape:", df.shape, "| Missing cells:", df.isna().sum().sum())

# EDA charts
plt.figure(figsize=(7, 4))
plt.hist(df["final_score"], bins=20, edgecolor="black")
plt.title("Distribution of Final Scores"); plt.xlabel("Final score"); plt.ylabel("Students")
plt.tight_layout(); plt.savefig(FIG / "01_score_distribution.png", dpi=160); plt.close()

bins = pd.cut(df["hours_studied"], bins=[-0.1, 2, 4, 6, 8, 10, 12])
df.groupby(bins, observed=False)["final_score"].mean().plot(kind="bar", edgecolor="black", figsize=(7, 4))
plt.title("Average Final Score by Study-Hour Band"); plt.xlabel("Study hours band"); plt.ylabel("Mean score")
plt.xticks(rotation=35, ha="right"); plt.tight_layout()
plt.savefig(FIG / "02_score_by_study_hours.png", dpi=160); plt.close()

plt.figure(figsize=(7, 4))
plt.scatter(df["attendance_pct"], df["final_score"], alpha=0.55)
plt.title("Attendance vs Final Score"); plt.xlabel("Attendance (%)"); plt.ylabel("Final score")
plt.tight_layout(); plt.savefig(FIG / "03_attendance_vs_score.png", dpi=160); plt.close()

corr = df.select_dtypes(include=np.number).corr()
plt.figure(figsize=(7, 5)); plt.imshow(corr, aspect="auto"); plt.colorbar(label="Correlation")
plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
plt.yticks(range(len(corr.columns)), corr.columns); plt.title("Numeric Feature Correlation Heatmap")
plt.tight_layout(); plt.savefig(FIG / "04_correlation_heatmap.png", dpi=160); plt.close()

df.boxplot(column="final_score", by="tutoring", figsize=(7, 4))
plt.title("Final Score by Tutoring Participation"); plt.suptitle("")
plt.xlabel("Tutoring"); plt.ylabel("Final score"); plt.tight_layout()
plt.savefig(FIG / "05_tutoring_score_boxplot.png", dpi=160); plt.close()

X = df.drop(columns=["final_score"]); y = df["final_score"]
num = X.select_dtypes(include=np.number).columns.tolist()
cat = X.select_dtypes(exclude=np.number).columns.tolist()
prep = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat)
])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = Pipeline([("preprocess", prep), ("regressor", RandomForestRegressor(n_estimators=250, min_samples_leaf=2, random_state=42))])
model.fit(X_train, y_train)
pred = model.predict(X_test)
mae = mean_absolute_error(y_test, pred)
mse = mean_squared_error(y_test, pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, pred)
metrics = pd.DataFrame({"Metric": ["MAE", "MSE", "RMSE", "R2"], "Value": [mae, mse, rmse, r2]})
metrics.to_csv(OUT / "evaluation_metrics.csv", index=False)
print("\\nEvaluation metrics:\\n", metrics.round(3).to_string(index=False))

nplot = min(45, len(y_test))
plt.figure(figsize=(8, 4))
plt.plot(np.arange(nplot), y_test.iloc[:nplot].to_numpy(), marker="o", label="Actual")
plt.plot(np.arange(nplot), pred[:nplot], marker="x", label="Predicted")
plt.title("Actual vs Predicted Scores"); plt.xlabel("Test sample index"); plt.ylabel("Final score")
plt.legend(); plt.tight_layout(); plt.savefig(FIG / "06_actual_vs_predicted.png", dpi=160); plt.close()

names = model.named_steps["preprocess"].get_feature_names_out()
imp = model.named_steps["regressor"].feature_importances_
fi = pd.DataFrame({"feature": names, "importance": imp}).sort_values("importance", ascending=False)
fi.to_csv(OUT / "feature_importance.csv", index=False)
top = fi.head(10).sort_values("importance")
plt.figure(figsize=(7, 4)); plt.barh(top["feature"], top["importance"], edgecolor="black")
plt.title("Top 10 Feature Importances"); plt.xlabel("Importance")
plt.tight_layout(); plt.savefig(FIG / "07_feature_importance.png", dpi=160); plt.close()

results = X_test.copy()
results["actual_final_score"] = y_test.to_numpy()
results["predicted_final_score"] = np.round(pred, 2)
results.to_csv(OUT / "test_predictions.csv", index=False)
print("\\nSample predictions:\\n", results.head(10).to_string(index=False))
print("\\nAll outputs saved under:", OUT)
