import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Convert broadcast date to integer year
df['broadcast date'] = pd.to_numeric(df['broadcast date'])

# Extract year and viewers
years = df['broadcast date'].values
viewers = df['viewers (millions)'].astype(float).values

# Fit a linear model: viewers ~ year
# Using numpy's polyfit to get slope and intercept
slope, intercept = np.polyfit(years, viewers, 1)

# Predict viewership for 2013
predicted_2013 = slope * 2013 + intercept

print(f"Final Answer: {predicted_2013:.2f}")