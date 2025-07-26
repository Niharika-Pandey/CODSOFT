# 🚀 CODSOFT Internship Projects

This repository contains a collection of machine learning projects completed as part of the **CODSOFT Internship**. Each project demonstrates a different ML technique and problem-solving approach using real-world datasets.

---

## 🎯 1. Sales Prediction Using Linear Regression

### 📌 Overview
Predicts product sales based on advertising spend across TV, Radio, and Newspaper channels using a **Multiple Linear Regression** model.

### ✅ Features
- Reads and cleans real-world advertising data
- Visualizes data with pairplots and correlation heatmaps
- Trains a linear regression model
- Evaluates performance using R² and MSE
- Displays:
  - 📈 Actual vs Predicted Sales
  - 📉 Residual plot (Profit = Green, Loss = Red)

### 🧠 Technologies
- Python, Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn

### 🗃️ Dataset
- **File**: `advertising.csv`
- **Source**: [ISLR Advertising Dataset](https://www.statlearning.com)
- **Columns**: `TV`, `Radio`, `Newspaper`, `Sales`

---

## 🎬 2. Movie Rating Prediction Using Regression

### 📌 Overview
Predicts IMDb movie ratings using features like genre, director, actors, runtime, and vote count using a **Gradient Boosting Regressor**.

### ✅ Features
- Cleans messy real-world movie data
- Parses and converts columns like runtime ("142 min") and votes ("1,245,000")
- Encodes categorical features (Genre, Director, Actors)
- Trains a regression model and evaluates with MAE, RMSE, and R²
- Displays predicted vs actual ratings with movie names

### 🧠 Technologies
- Python, Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn

### 🗃️ Dataset
- **File**: `movies.csv`
- **Columns**: `Title`, `Genre`, `Director`, `Actors`, `Runtime`, `Votes`, `Rating`  
- **Source**: IMDb dataset or Kaggle variant

---

## 🌸 3. Iris Flower Classification Using KNN

### 📌 Overview
Classifies iris flowers into three species (*setosa*, *versicolor*, *virginica*) using **K-Nearest Neighbors** (KNN) algorithm.

### ✅ Features
- Loads standard `iris` dataset from scikit-learn
- Splits data into train/test sets
- Trains a KNN classifier
- Evaluates using confusion matrix, accuracy, and classification report
- Visualizes decision boundaries and flower clusters

### 🧠 Technologies
- Python, Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn

### 🗃️ Dataset
- **Source**: Built-in `sklearn.datasets.load_iris()`
- **Features**: `Sepal Length`, `Sepal Width`, `Petal Length`, `Petal Width`
- **Target**: `Species`

