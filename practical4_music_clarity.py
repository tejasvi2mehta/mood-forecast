# Practical 4: Predicting Music Clarity Trends (Stock Market Style)
# Run in VS Code

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# -----------------------------
# Step 1: Sample dataset (Clarity score over time)
# -----------------------------
data = {
    "day": [1, 2, 3, 4, 5, 6, 7],
    "clarity_score": [50, 55, 60, 62, 65, 70, 72]  # Example clarity scores
}
df = pd.DataFrame(data)
print("Original Dataset:\n", df)

# -----------------------------
# Step 2: Prepare features
# -----------------------------
X = df["day"].values.reshape(-1,1)      # Feature = day
y = df["clarity_score"].values          # Target = clarity_score

# -----------------------------
# Step 3: Train Linear Regression Model
# -----------------------------
model = LinearRegression()
model.fit(X, y)

# -----------------------------
# Step 4: Predict future clarity scores
# -----------------------------
future_days = np.array([8, 9, 10, 11, 12]).reshape(-1,1)
predictions = model.predict(future_days)

# -----------------------------
# Step 5: Visualization
# -----------------------------
plt.figure(figsize=(8,5))
plt.plot(df["day"], y, marker="o", label="Actual Clarity Score")
plt.plot(future_days, predictions, marker="x", linestyle="--", label="Predicted Trend")
plt.xlabel("Day")
plt.ylabel("Clarity Score")
plt.title("Music Clarity Trend Prediction")
plt.legend()
plt.grid(True)
plt.show()

# -----------------------------
# Step 6: Print predictions
# -----------------------------
for d, p in zip(future_days.flatten(), predictions):
    print(f"Day {d} → Predicted Clarity Score: {p:.2f}")
