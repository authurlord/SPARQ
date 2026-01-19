import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Convert 'year' to integer and 'Percentage' to float
df['year'] = pd.to_numeric(df['year'])
df['Percentage (based on registration by the church itself)'] = pd.to_numeric(df['Percentage (based on registration by the church itself)'])

# Prepare data for linear regression
X = df['year'].values.reshape(-1, 1)
y = df['Percentage (based on registration by the church itself)'].values

# Fit linear regression model
model = np.polyfit(X.flatten(), y, 1)
predicted_percentage_2025 = np.polyval(model, 2025)

print(f"Final Answer: {predicted_percentage_2025:.1f}")