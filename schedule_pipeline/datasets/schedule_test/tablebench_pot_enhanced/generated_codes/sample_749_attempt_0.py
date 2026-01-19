import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Convert broadcast date to integer and sort by year
df['broadcast date'] = pd.to_numeric(df['broadcast date'])
df = df.sort_values('broadcast date')

# Extract year and viewers
years = df['broadcast date'].values.reshape(-1, 1)
viewers = df['viewers (millions)'].values

# Fit a linear regression model
model = np.polyfit(years.flatten(), viewers, 1)
predicted_viewers_2013 = np.polyval(model, 2013)

print(f"Final Answer: {predicted_viewers_2013:.2f}")