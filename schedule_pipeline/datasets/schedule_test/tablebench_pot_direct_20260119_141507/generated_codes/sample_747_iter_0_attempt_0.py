import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Convert year to numeric and extract urban percentage
df['year'] = pd.to_numeric(df['year (january)'], errors='coerce')
df['urban_percent'] = pd.to_numeric(df['urban , %'], errors='coerce')

# Remove any rows with missing values
df = df.dropna(subset=['year', 'urban_percent'])

# Fit a linear regression model: urban_percent ~ year
X = df['year'].values.reshape(-1, 1)
y = df['urban_percent'].values

# Use numpy to calculate slope and intercept
slope, intercept = np.polyfit(df['year'], df['urban_percent'], 1)

# Predict urban percentage for 2015
predicted_urban_2015 = slope * 2015 + intercept

print(f"Final Answer: {predicted_urban_2015:.1f}")