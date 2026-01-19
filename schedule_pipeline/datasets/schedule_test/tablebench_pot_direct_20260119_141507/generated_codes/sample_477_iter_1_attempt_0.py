import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Convert 'points' to numeric, coercing errors to NaN
df['points'] = pd.to_numeric(df['points'], errors='coerce')

# Remove rows with invalid points
df_clean = df.dropna(subset=['points'])

# Calculate mean and std of points
mean_points = df_clean['points'].mean()
std_points = df_clean['points'].std()

# Identify outliers: points more than 2 standard deviations from mean
threshold = 2 * std_points
outliers = df_clean[(df_clean['points'] < (mean_points - threshold)) | (df_clean['points'] > (mean_points + threshold))]

# Extract song names of these outliers
outlier_songs = outliers['song'].tolist()

print(f"Final Answer: {', '.join(outlier_songs)}")