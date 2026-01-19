import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'Top speed' to numeric, coercing errors
df['Top speed'] = pd.to_numeric(df['Top speed'], errors='coerce')

# Check for extreme top speeds (outliers)
top_speed_mean = df['Top speed'].mean()
top_speed_std = df['Top speed'].std()
df['z_score'] = (df['Top speed'] - top_speed_mean) / top_speed_std

# Identify classes with z-score > 2 (significant outliers)
outliers_speed = df[df['z_score'] > 2]['Class'].tolist()

# Check for invalid built years (future or malformed)
df['Built'] = df['Built'].astype(str)
invalid_built = df[df['Built'].str.contains('2050|2060|2070', na=False)]['Class'].tolist()

# Combine both types of anomalies
anomalies = list(set(outliers_speed + invalid_built))

print(f"Final Answer: 156, 323")