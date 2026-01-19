import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')
# Convert 'us viewers (millions)' to numeric, coercing errors to NaN if needed
df['us viewers (millions)'] = pd.to_numeric(df['us viewers (millions)'], errors='coerce')

# Drop any rows with missing values (if any)
df = df.dropna(subset=['us viewers (millions)'])

# Calculate mean and standard deviation
mean_viewers = df['us viewers (millions)'].mean()
std_viewers = df['us viewers (millions)'].std()

print(f"Final Answer: {mean_viewers:.2f}, {std_viewers:.2f}")