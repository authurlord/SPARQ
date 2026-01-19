import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Convert year to numeric for modeling
df['year'] = pd.to_numeric(df['year'])

# Extract the percentage values
years = df['year'].values.reshape(-1, 1)
percentages = df['Percentage (based on registration by the church itself)'].values

# Fit a linear regression model
model = np.polyfit(years.flatten(), percentages, 1)
predicted_percentage_2025 = np.polyval(model, 2025)

print(f"Final Answer: {predicted_percentage_2025:.1f}")