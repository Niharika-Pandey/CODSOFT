import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset (✅ Use raw string to avoid unicode escape error)
df = pd.read_csv(r"C:\Users\Dell\Downloads\advertising (1).csv")

# Display the first few rows
print("First 5 rows of the dataset:")
print(df.head())

# Pairplot: Visual relationship between ad spends and sales
sns.pairplot(df, x_vars=['TV', 'Radio', 'Newspaper'], y_vars='Sales', height=5, aspect=0.7, kind='scatter')
plt.suptitle('Advertising Spend vs Sales', y=1.02)
plt.show()

# Heatmap of correlations between features
plt.figure(figsize=(6,4))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

# Features and target
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Print model performance
print("\nModel Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Plot: Actual vs Predicted Sales
plt.figure(figsize=(6,4))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs Predicted Sales')
plt.grid(True)
plt.tight_layout()
plt.show()

# Residual plot (Profit vs Loss)
residuals = y_pred - y_test
plt.figure(figsize=(8,5))
colors = ['red' if res > 0 else 'green' for res in residuals]
plt.scatter(y_test, residuals, c=colors, edgecolor='black')
plt.axhline(y=0, color='gray', linestyle='--')
plt.xlabel('Actual Sales')
plt.ylabel('Prediction Error (Residual)')
plt.title('Profit (Green) vs Loss (Red) Residual Plot')
plt.grid(True)
plt.tight_layout()
plt.show()
