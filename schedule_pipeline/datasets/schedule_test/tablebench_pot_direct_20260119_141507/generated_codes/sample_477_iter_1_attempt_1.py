import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Convert 'points' to numeric, coercing errors to NaN
df['points'] = pd.to_numeric(df['points'], errors='coerce')

# Drop rows where points are NaN (invalid or missing)
df = df.dropna(subset=['points'])

# Calculate Q1, Q3, and IQR
Q1 = df['points'].quantile(0.25)
Q3 = df['points'].quantile(0.75)
IQR = Q3 - Q1

# Define outlier bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Find outliers
outliers = df[(df['points'] < lower_bound) | (df['points'] > upper_bound)]

# Extract song names of outliers
outlier_songs = outliers['song'].tolist()

# Print the result
print(f"Final Answer: {', '.join(outlier_songs)}")