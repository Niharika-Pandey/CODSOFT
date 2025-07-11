import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv(r"C:\Users\Dell\Downloads\dataset.csv", encoding="latin1")
df = df.dropna(subset=["Rating"])


df["Duration"] = df["Duration"].str.extract(r"(\d+)").astype(float)
df["Votes"]    = df["Votes"].str.replace(",", "").astype(float)

num_feats = ["Year", "Duration", "Votes"]
cat_feats = ["Genre", "Director", "Actor 1", "Actor 2", "Actor 3"]
target    = "Rating"
title_col = "Name"        

df = df.dropna(subset=num_feats + cat_feats + [title_col])

X = df[num_feats + cat_feats]
y = df[target]
titles = df[title_col]

num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer([
    ("num", num_pipe, num_feats),
    ("cat", cat_pipe, cat_feats)
])
model = Pipeline([
    ("prep", preprocessor),
    ("reg", GradientBoostingRegressor(random_state=42))
])

X_train, X_test, y_train, y_test, titles_train, titles_test = train_test_split(
    X, y, titles, test_size=0.2, random_state=42)

model.fit(X_train, y_train)
pred = model.predict(X_test)


mae  = mean_absolute_error(y_test, pred)
rmse = mean_squared_error(y_test, pred) ** 0.5
r2   = r2_score(y_test, pred)

print(" Model Evaluation:")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.2f}\n")

print("🎬 Sample Predictions (Predicted vs Actual Ratings):")
for name, actual, predicted in zip(titles_test[:10], y_test[:10], pred[:10]):
    print(f"📽️ {name}: Predicted = {predicted:.2f}, Actual = {actual:.2f}")
