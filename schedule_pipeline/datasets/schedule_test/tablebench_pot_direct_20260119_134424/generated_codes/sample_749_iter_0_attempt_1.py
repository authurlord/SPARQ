import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Convert broadcast date to integer and viewers to float
df['broadcast date'] = pd.to_numeric(df['broadcast date'])
df['viewers (millions)'] = pd.to_numeric(df['viewers (millions)'])

# Extract x (years) and y (viewers)
x = df['broadcast date'].values.reshape(-1, 1)
y = df['viewers (millions)'].values

# Fit a linear regression model
model = np.polyfit(x.flatten(), y, 1)
predicted_2013 = np.polyval(model, 2013)

print(f"Final Answer: {predicted_2013:.2f}")