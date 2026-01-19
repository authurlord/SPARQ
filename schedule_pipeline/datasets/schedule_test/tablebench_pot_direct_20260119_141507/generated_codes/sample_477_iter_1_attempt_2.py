import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Convert 'points' to numeric, coercing errors to NaN
df['points'] = pd.to_numeric(df['points'], errors='coerce')

# Remove rows where points are NaN
df_clean = df.dropna(subset=['points'])

# Calculate mean and standard deviation of points
mean_points = df_clean['points'].mean()
std_points = df_clean['points'].std()

# Identify outliers using 2 standard deviations rule
threshold = 2 * std_points
outliers = df_clean[(df_clean['points'] > mean_points + threshold) | (df_clean['points'] < mean_points - threshold)]

# Extract song names of outliers
outlier_songs = outliers['song'].tolist()

print(f"Final Answer: {', '.join(outlier_songs)}")