🎬 Movie Rating Prediction Model

This project uses machine learning to predict movie ratings based on various features like genre, director, cast, release year, duration, and user votes. It is part of the **CODSOFT Internship** project series.

📌 Features

- Reads and cleans real-world movie data
- Extracts useful numeric features from messy strings (e.g., "142 min", "1,245,000 votes")
- Handles missing data with preprocessing pipelines
- Encodes categorical variables (e.g., Genre, Director, Actors)
- Trains a regression model (Gradient Boosting)
- Evaluates performance using MAE, RMSE, and R² score
- Displays predicted vs actual ratings with movie names

---

🧠 Technologies Used

- Python 3
- Pandas
- NumPy
- Scikit-learn

---

🗂️ Project Structure

```bash
CODSOFT/
├── Movie-Rating-Prediction/
│   ├── movierating.py         # Main model training and prediction script
│   ├── dataset.csv            # Movie dataset
│   └── README.md              # This file
